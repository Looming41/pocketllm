"""Expert-aware requantization: routed experts in few bits, everything else in 8 bits.

When a MoE model is streamed from a drive, nearly every byte read per token is a routed
expert, while attention, router and shared weights stay cached in RAM. So the bits are
best spent unevenly: low-bit experts shrink both the model (it fits a smaller drive, or a
bigger model fits the same drive) and the bytes read per token, while 8-bit attention and
router keep the parts every token passes through accurate. This is the recipe Pulsar uses
(`pulsar-quant --map "_exps.=q2_k" --default q8_0`) and what Unsloth's dynamic quants do,
done here with llama.cpp's own `llama-quantize --tensor-type`.

Quality: lower-bit experts change the model. Check every recipe with `pocketllm qualify`
(or any eval set) before handing it out. An importance matrix (`--imatrix`, from
llama-imatrix on representative text) is strongly recommended below 3 bits and required
by the IQ2 types.
"""
from __future__ import annotations

import os
import subprocess

from . import gguf, runtime

EXPERT_PATTERN = r"ffn_(gate|up|down|gate_up)_exps"

# bits per weight of llama.cpp tensor types (block size and bytes from ggml)
TYPE_BPW = {
    "f16": 16.0, "bf16": 16.0, "q8_0": 8.5, "q6_k": 6.5625, "q5_k": 5.5, "q4_k": 4.5, "q4_0": 4.5,
    "iq4_xs": 4.25, "mxfp4": 4.25, "q3_k": 3.4375, "iq3_xxs": 3.0625, "q2_k": 2.625,
    "iq2_xs": 2.3125, "iq2_xxs": 2.0625, "iq1_m": 1.75, "iq1_s": 1.5625,
}
NEEDS_IMATRIX = {"iq2_xs", "iq2_xxs", "iq1_s", "iq1_m"}

# named recipes: (expert type, everything-else type)
# Measured on Qwen3-30B-A3B, requantized from Q4_K_M, against Q4_K_M (RESULTS R17):
#   experts-q3: mean KLD 0.064, same top token 89.5%, HellaSwag 73% (ref 76%)  -> usable
#   experts-q2: mean KLD 0.336, same top token 77.1%, HellaSwag 65%             -> too lossy
# A published 2-bit quant made from BF16 with an importance matrix (unsloth UD-Q2_K_XL) measured
# KLD 0.127 / 86.2% / 72%: for 2-bit, download one of those instead of requantizing a Q4 file.
MAX_KLD_OK = 0.15  # above this mean KLD vs the reference, a recipe is not offered by default

RECIPES = {
    "experts-q4": ("q4_k", "q8_0"),
    "experts-q3": ("q3_k", "q8_0"),     # recommended when the source is already quantized
    "experts-q2": ("q2_k", "q8_0"),     # Pulsar's streaming recipe; from BF16 + imatrix only
    "experts-q2-small": ("q2_k", "q4_k"),  # smallest RAM footprint for the always-resident part
    "experts-iq2": ("iq2_xxs", "q8_0"),  # needs --imatrix
}


def _elements(t) -> int:
    n = 1
    for d in t.shape:
        n *= d
    return n


def estimate(m: gguf.GGUFModel, experts: str, rest: str) -> dict:
    """Predicted size of the requantized model. 1-D tensors (norms, biases) keep their bytes."""
    eb, rb = TYPE_BPW[experts], TYPE_BPW[rest]
    exp_bytes = rest_bytes = embd = 0
    for t in m.tensors:
        if len(t.shape) < 2 or (t.shape[0] if t.shape else 0) % 32:
            b = t.nbytes
        elif t.is_expert:
            exp_bytes += int(_elements(t) * eb / 8)
            continue
        else:
            b = int(_elements(t) * rb / 8)
        rest_bytes += b
        if t.name.startswith("token_embd."):
            embd += b  # a row lookup per token, not read in full
    frac = min(1.0, (m.n_expert_used or 1) / (m.n_expert or 1))
    src_exp = m.expert_bytes()
    return {"total_bytes": exp_bytes + rest_bytes, "expert_bytes": exp_bytes, "rest_bytes": rest_bytes,
            "active_bytes_per_token": int(rest_bytes - embd + exp_bytes * frac),
            "expert_bytes_per_token": int(exp_bytes * frac),
            "source_bytes": m.weight_bytes, "source_expert_bytes": src_exp,
            "source_rest_bytes": m.weight_bytes - src_exp,
            "source_active_bytes_per_token": m.active_bytes_per_token(),
            "source_expert_bytes_per_token": int(src_exp * frac)}


def command(src: str, dst: str, experts: str, rest: str, imatrix: str | None = None, threads: int | None = None):
    if experts not in TYPE_BPW or rest not in TYPE_BPW:
        raise ValueError(f"unknown type; choose from {sorted(TYPE_BPW)}")
    if experts in NEEDS_IMATRIX and not imatrix:
        raise ValueError(f"{experts} needs an importance matrix (--imatrix, from llama-imatrix)")
    cmd = [runtime.binary("llama-quantize")]
    m = gguf.load(src)
    if any(t.ggml_type not in (0, 1, 30) for t in m.tensors if len(t.shape) >= 2):
        # source is already quantized (f32=0, f16=1, bf16=30 are the clean inputs)
        cmd.append("--allow-requantize")
    if imatrix:
        cmd += ["--imatrix", imatrix]
    cmd += ["--tensor-type", f"{EXPERT_PATTERN}={experts}", src, dst, rest.upper()]
    if threads:
        cmd.append(str(threads))
    return cmd


def run(src: str, dst: str, experts: str, rest: str, imatrix: str | None = None, log=print) -> dict:
    cmd = command(src, dst, experts, rest, imatrix)
    if "--allow-requantize" in cmd:
        log("note: source is already quantized; requantizing loses more quality than starting from BF16/F16")
        if TYPE_BPW[experts] < 3 and not imatrix:
            log("warning: 2-bit experts from an already-quantized file measured far too lossy (KLD 0.34, "
                "HellaSwag 76% -> 65% on Qwen3-30B-A3B). Use --recipe experts-q3, or a published 2-bit "
                "quant made from BF16 with an importance matrix (e.g. unsloth UD-Q2_K_XL).")
    log(" ".join(cmd))
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    out = gguf.load(dst)
    return {"path": dst, "total_bytes": out.weight_bytes, "expert_bytes": out.expert_bytes(),
            "active_bytes_per_token": out.active_bytes_per_token()}

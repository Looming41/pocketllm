"""Model catalog used for *pre-download* planning.

Once a GGUF file is on the drive the planner reads exact numbers from its
header; these entries only exist so PocketLLM can say "will it run?" before a
user spends an hour downloading 60 GB.

Every architectural number below comes from the model's published config and
should be re-checked against the GGUF header after download (``pocketllm
inspect``). Size estimates use bits-per-weight derived from llama.cpp's own
quantize table (tools/quantize/quantize.cpp, measured on Llama-3-8B), so they
are approximations: e.g. the estimate for Llama 4 Scout Q4_K_M is ~66.8 GB,
while a published file is ~60.9 GiB (65.3 GB).
"""
from .planner import ModelSpec

# bits per weight, from llama.cpp quantize table sizes for Llama-3-8B (8.03B params)
BPW = {
    "Q8_0": 8.52,
    "Q6_K": 6.57,
    "Q5_K_M": 5.70,
    "Q4_K_M": 4.90,
    "IQ4_XS": 4.25,
    "MXFP4": 4.25,
    "Q3_K_M": 4.00,
    "Q2_K": 3.17,
    "IQ2_XXS": 2.06,
    "IQ1_S": 1.56,
}

# kv bytes/token = n_layer * n_kv_heads * (k_dim + v_dim) * 2 bytes (f16 cache)
CATALOG = [
    {
        "id": "llama-3.2-3b-instruct", "family": "Meta Llama 3.2", "params_b": 3.21, "active_b": 3.21,
        "n_layer": 28, "kv": 28 * 8 * 256 * 2, "expert_frac": 0.0, "ctx": 131072,
        "quants": ["Q4_K_M", "Q8_0"], "license": "Llama 3.2 Community License",
        "hf": "unsloth/Llama-3.2-3B-Instruct-GGUF", "role": "UX demo model (any laptop)",
    },
    {
        "id": "llama-3.1-8b-instruct", "family": "Meta Llama 3.1", "params_b": 8.03, "active_b": 8.03,
        "n_layer": 32, "kv": 32 * 8 * 256 * 2, "expert_frac": 0.0, "ctx": 131072,
        "quants": ["Q4_K_M", "Q8_0"], "license": "Llama 3.1 Community License",
        "hf": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "role": "default chat model (16 GB laptops)",
    },
    {
        "id": "llama-3.3-70b-instruct", "family": "Meta Llama 3.3", "params_b": 70.6, "active_b": 70.6,
        "n_layer": 80, "kv": 80 * 8 * 256 * 2, "expert_frac": 0.0, "ctx": 131072,
        "quants": ["Q4_K_M", "Q2_K"], "license": "Llama 3.3 Community License",
        "hf": "bartowski/Llama-3.3-70B-Instruct-GGUF", "role": "dense large-model reference",
    },
    {
        # 109B total / 17B active, 16 routed experts (top-1) + shared expert.
        # Routed-expert share solved from 109 = base + E and 17 = base + E/16 -> E ~ 98B.
        "id": "llama-4-scout-17b-16e-instruct", "family": "Meta Llama 4", "params_b": 109, "active_b": 17,
        "n_layer": 48, "kv": 48 * 8 * 256 * 2, "expert_frac": 0.90, "ctx": 10_000_000,
        "quants": ["Q4_K_M", "Q3_K_M", "Q2_K", "IQ2_XXS", "IQ1_S"], "license": "Llama 4 Community License",
        "hf": "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF", "role": "100B-class target (Meta, MoE)",
    },
    {
        # 116.8B total / 5.1B active, 128 experts top-4; ships natively in MXFP4.
        "id": "gpt-oss-120b", "family": "OpenAI gpt-oss", "params_b": 116.8, "active_b": 5.1,
        "n_layer": 36, "kv": 36 * 8 * 128 * 2, "expert_frac": 0.98, "ctx": 131072,
        "quants": ["MXFP4"], "license": "Apache 2.0",
        "hf": "ggml-org/gpt-oss-120b-GGUF", "role": "100B-class comparison (very sparse MoE)",
    },
    {
        "id": "llama-4-maverick-17b-128e-instruct", "family": "Meta Llama 4", "params_b": 400, "active_b": 17,
        "n_layer": 48, "kv": 48 * 8 * 256 * 2, "expert_frac": 0.96, "ctx": 1_000_000,
        "quants": ["Q2_K", "IQ1_S"], "license": "Llama 4 Community License",
        "hf": "unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF", "role": "out-of-reach reference",
    },
]


def specs(ids=None):
    out = []
    for e in CATALOG:
        if ids and e["id"] not in ids:
            continue
        for q in e["quants"]:
            s = ModelSpec.estimate(f'{e["id"]}:{q}', e["params_b"], e["active_b"], BPW[q], e["n_layer"],
                                   e["kv"], e["expert_frac"], e["ctx"])
            s.path = f'{e["hf"]}:{q}'
            out.append(s)
    return out


def entry(model_id):
    base = model_id.split(":")[0]
    return next((e for e in CATALOG if e["id"] == base), None)

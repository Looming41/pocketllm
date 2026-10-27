"""MLX models (for the SwiftLM backend on Apple Silicon).

An MLX model is a directory with config.json and *.safetensors. Sizes come from
the safetensors headers (no weights are read), so the planner gets the same
exact dense/expert split it gets from GGUF headers.
"""
from __future__ import annotations

import glob
import json
import os
import struct

from .planner import ModelSpec

# tensor-name markers of routed experts in MLX/HF checkpoints
EXPERT_MARKERS = (".switch_mlp.", ".experts.", ".block_sparse_moe.experts", ".mlp.experts")


def is_mlx_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json")) and bool(glob.glob(os.path.join(path, "*.safetensors")))


def find_mlx_models(root: str) -> list:
    out = []
    if not os.path.isdir(root):
        return out
    for dp, dns, _ in os.walk(root):
        if is_mlx_dir(dp):
            out.append(dp)
            dns[:] = []
    return sorted(out)


def _tensors(path):
    """(name, nbytes) for every tensor in a safetensors file, from its JSON header."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        yield name, b - a


def load_spec(path: str) -> ModelSpec:
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    n_layer = int(tc.get("num_hidden_layers", 0)) or 1
    n_expert = int(tc.get("num_experts") or tc.get("num_local_experts") or tc.get("n_routed_experts") or 0)
    n_used = int(tc.get("num_experts_per_tok") or tc.get("moe_topk") or 0)
    hidden = int(tc.get("hidden_size", 0))
    n_kv = int(tc.get("num_key_value_heads", tc.get("num_attention_heads", 0)) or 0)
    head_dim = int(tc.get("head_dim") or (hidden // max(1, int(tc.get("num_attention_heads", 1)))))
    total = experts = 0
    per_layer_dense = [0] * n_layer
    per_layer_exp = [0] * n_layer
    non_layer = 0
    for st in glob.glob(os.path.join(path, "*.safetensors")):
        for name, nb in _tensors(st):
            total += nb
            layer = None
            parts = name.split(".")
            if "layers" in parts:
                i = parts.index("layers")
                if i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer = int(parts[i + 1])
            is_exp = any(m in name for m in EXPERT_MARKERS)
            if layer is None or layer >= n_layer:
                non_layer += nb
            elif is_exp:
                per_layer_exp[layer] += nb
                experts += nb
            else:
                per_layer_dense[layer] += nb
    moe = n_expert > 1 and experts > 0
    frac = (n_used / n_expert) if moe and n_used else 1.0
    active = total - experts + int(experts * frac)
    return ModelSpec(name=os.path.basename(path.rstrip("/")), total_bytes=total, n_layer=n_layer,
                     layer_dense_bytes=per_layer_dense, layer_expert_bytes=per_layer_exp, non_layer_bytes=non_layer,
                     active_bytes=active, kv_bytes_per_token=n_layer * n_kv * head_dim * 2 * 2, is_moe=moe,
                     train_ctx=int(tc.get("max_position_embeddings", 0) or 0), path=path, source="mlx",
                     n_expert_used=n_used if moe else 0)

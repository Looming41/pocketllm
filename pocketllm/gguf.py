"""Minimal, dependency-free GGUF header reader.

Reads only the header, metadata and tensor table (never the weights), so it is
fast even for a 60 GB model sitting on a slow USB stick. Split models
(``*-00001-of-0000N.gguf``) are merged into one logical model.
"""
from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field

GGUF_MAGIC = b"GGUF"

# GGUF metadata value types
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64 = range(13)
_SCALAR = {
    _U8: "<B", _I8: "<b", _U16: "<H", _I16: "<h", _U32: "<I", _I32: "<i",
    _F32: "<f", _BOOL: "<?", _U64: "<Q", _I64: "<q", _F64: "<d",
}

# MoE expert tensors: blk.N.ffn_{up,down,gate,gate_up}_exps.weight (shared experts use _shexp)
EXPERT_RE = re.compile(r"\.ffn_(?:up|down|gate|gate_up)_exps\.")
LAYER_RE = re.compile(r"^blk\.(\d+)\.")
SPLIT_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")


@dataclass
class Tensor:
    name: str
    shape: tuple
    ggml_type: int
    offset: int
    nbytes: int = 0

    @property
    def layer(self) -> int | None:
        m = LAYER_RE.match(self.name)
        return int(m.group(1)) if m else None

    @property
    def is_expert(self) -> bool:
        return bool(EXPERT_RE.search(self.name))


@dataclass
class GGUFModel:
    paths: list
    metadata: dict = field(default_factory=dict)
    tensors: list = field(default_factory=list)

    # ---- convenience accessors -------------------------------------------------
    @property
    def arch(self) -> str:
        return self.metadata.get("general.architecture", "unknown")

    def arch_key(self, key, default=None):
        return self.metadata.get(f"{self.arch}.{key}", default)

    @property
    def name(self) -> str:
        return self.metadata.get("general.name") or os.path.basename(self.paths[0])

    @property
    def file_bytes(self) -> int:
        return sum(os.path.getsize(p) for p in self.paths)

    @property
    def weight_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    @property
    def n_params(self) -> int:
        n = 0
        for t in self.tensors:
            p = 1
            for d in t.shape:
                p *= d
            n += p
        return n

    @property
    def n_layer(self) -> int:
        return int(self.arch_key("block_count", 0) or 0)

    @property
    def n_expert(self) -> int:
        return int(self.arch_key("expert_count", 0) or 0)

    @property
    def n_expert_used(self) -> int:
        return int(self.arch_key("expert_used_count", 0) or 0)

    @property
    def is_moe(self) -> bool:
        return self.n_expert > 1 and any(t.is_expert for t in self.tensors)

    @property
    def context_length(self) -> int:
        return int(self.arch_key("context_length", 0) or 0)

    def expert_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors if t.is_expert)

    def layer_bytes(self, layer: int, experts: bool | None = None) -> int:
        """Bytes of one transformer block. experts=None: all, True: only routed experts, False: no experts."""
        total = 0
        for t in self.tensors:
            if t.layer != layer:
                continue
            if experts is None or t.is_expert == experts:
                total += t.nbytes
        return total

    def non_layer_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors if t.layer is None)

    def active_bytes_per_token(self) -> int:
        """Weight bytes touched to generate one token.

        Dense: every weight. MoE: everything except routed experts, plus the
        fraction of routed experts that the router selects (n_used / n_expert).
        Token embeddings are a row lookup, so they are excluded.
        """
        embd = sum(t.nbytes for t in self.tensors if t.name.startswith("token_embd."))
        if not self.is_moe:
            return self.weight_bytes - embd
        exp = self.expert_bytes()
        frac = self.n_expert_used / self.n_expert if self.n_expert else 1.0
        return int(self.weight_bytes - embd - exp + exp * frac)

    def kv_bytes_per_token(self, cache_type_bytes: float = 2.0) -> int:
        """KV-cache bytes per context token (f16 cache by default).

        Conservative: assumes every layer keeps full-length K and V, which
        overestimates models that use sliding-window attention.
        """
        n_layer = self.n_layer
        n_head = self.arch_key("attention.head_count", 0) or 0
        n_head_kv = self.arch_key("attention.head_count_kv", n_head) or n_head
        n_embd = self.arch_key("embedding_length", 0) or 0
        if isinstance(n_head, list):
            n_head = max(n_head)
        head_dim_default = (n_embd // n_head) if n_head else 0
        k_len = self.arch_key("attention.key_length", head_dim_default) or head_dim_default
        v_len = self.arch_key("attention.value_length", head_dim_default) or head_dim_default
        if isinstance(n_head_kv, list):  # per-layer head counts
            heads = sum(n_head_kv)
        else:
            heads = n_layer * n_head_kv
        return int(heads * (k_len + v_len) * cache_type_bytes)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "arch": self.arch,
            "files": [os.path.basename(p) for p in self.paths],
            "file_gb": round(self.file_bytes / 1e9, 2),
            "params_b": round(self.n_params / 1e9, 2),
            "n_layer": self.n_layer,
            "moe": self.is_moe,
            "n_expert": self.n_expert,
            "n_expert_used": self.n_expert_used,
            "expert_gb": round(self.expert_bytes() / 1e9, 2),
            "active_gb_per_token": round(self.active_bytes_per_token() / 1e9, 3),
            "kv_mb_per_1k_ctx": round(self.kv_bytes_per_token() * 1024 / 1e6, 1),
            "train_ctx": self.context_length,
        }


class _Reader:
    def __init__(self, f):
        self.f = f

    def read(self, fmt):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) != size:
            raise ValueError("unexpected end of GGUF header")
        return struct.unpack(fmt, data)[0]

    def string(self) -> str:
        n = self.read("<Q")
        return self.f.read(n).decode("utf-8", errors="replace")

    def value(self, vtype, keep_arrays=True):
        if vtype in _SCALAR:
            return self.read(_SCALAR[vtype])
        if vtype == _STR:
            return self.string()
        if vtype == _ARR:
            etype = self.read("<I")
            n = self.read("<Q")
            if etype in _SCALAR and not keep_arrays:
                self.f.seek(n * struct.calcsize(_SCALAR[etype]), os.SEEK_CUR)
                return f"<array {n}>"
            if etype == _STR and not keep_arrays:
                for _ in range(n):
                    self.f.seek(self.read("<Q"), os.SEEK_CUR)
                return f"<array {n}>"
            return [self.value(etype, keep_arrays) for _ in range(n)]
        raise ValueError(f"unknown GGUF value type {vtype}")


def _read_one(path: str):
    with open(path, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        r = _Reader(f)
        version = r.read("<I")
        if version < 2:
            raise ValueError(f"{path}: GGUF v{version} is too old")
        n_tensors = r.read("<Q")
        n_kv = r.read("<Q")
        meta = {"gguf.version": version}
        for _ in range(n_kv):
            key = r.string()
            vtype = r.read("<I")
            # tokenizer arrays are huge and irrelevant for planning; skip them
            keep = not key.startswith("tokenizer.ggml.")
            meta[key] = r.value(vtype, keep_arrays=keep)
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.read("<I")
            shape = tuple(r.read("<Q") for _ in range(n_dims))
            ggml_type = r.read("<I")
            offset = r.read("<Q")
            tensors.append(Tensor(name, shape, ggml_type, offset))
        align = int(meta.get("general.alignment", 32))
        data_start = (f.tell() + align - 1) // align * align
        data_len = os.path.getsize(path) - data_start
    # tensor size = distance to the next tensor (includes alignment padding, < 32 B each)
    order = sorted(tensors, key=lambda t: t.offset)
    for i, t in enumerate(order):
        end = order[i + 1].offset if i + 1 < len(order) else data_len
        t.nbytes = max(0, end - t.offset)
    return meta, tensors


def split_siblings(path: str) -> list:
    """Return all parts of a split GGUF given any one part (or [path] if not split)."""
    d, base = os.path.split(os.path.abspath(path))
    m = SPLIT_RE.match(base)
    if not m:
        return [os.path.abspath(path)]
    stem, _, total = m.groups()
    parts = [os.path.join(d, f"{stem}-{i:05d}-of-{total}.gguf") for i in range(1, int(total) + 1)]
    missing = [p for p in parts if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"split model is incomplete, missing: {', '.join(map(os.path.basename, missing))}")
    return parts


def load(path: str) -> GGUFModel:
    parts = split_siblings(path)
    model = GGUFModel(paths=parts)
    for i, p in enumerate(parts):
        meta, tensors = _read_one(p)
        if i == 0:
            model.metadata = meta
        model.tensors.extend(tensors)
    return model


def find_models(root: str) -> list:
    """List loadable models under root: plain .gguf files and the first part of split sets.

    Skips multimodal projector files (mmproj) and llama.cpp vocab-only files.
    """
    out = []
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith(".gguf") or fn.startswith(("mmproj", "ggml-vocab")) or fn.startswith("."):
                continue
            m = SPLIT_RE.match(fn)
            if m and m.group(2) != "00001":
                continue
            out.append(os.path.join(dirpath, fn))
    return out

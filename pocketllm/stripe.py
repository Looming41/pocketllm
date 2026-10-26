"""Stripe one GGUF model across several drives.

A MoE layer needs three expert tensors per token (gate, up, down). If they
sit on the same drive they are read one after another; if they sit on three
different drives, the expert streamer's read-ahead (ggml-pocket) fetches them
concurrently, so read bandwidth adds up across drives.

This tool rewrites a GGUF into llama.cpp's standard split format
(``name-0000i-of-0000N.gguf``) but chooses *which tensors go into which split*:
routed-expert tensors are dealt round-robin across the drives in tensor order
(gate -> drive 0, up -> drive 1, down -> drive 2, next layer continues), all
other (small, always-hot) tensors go to the first drive. Each split is written
directly onto its drive, and a directory of symlinks makes the set loadable
with a plain ``-m <dir>/name-00001-of-0000N.gguf``.

Stdlib only; tensor data is copied with large sequential reads.
"""
from __future__ import annotations

import os
import re
import struct

from .gguf import EXPERT_RE, GGUF_MAGIC, _Reader

KV_SPLIT_NO = "split.no"
KV_SPLIT_COUNT = "split.count"
KV_SPLIT_TENSORS = "split.tensors.count"
_U16, _I32, _STR = 2, 5, 8


def _scan(path):
    """Return header pieces of an unsplit GGUF: version, raw KV bytes, n_kv, tensor infos, data start, alignment."""
    with open(path, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        r = _Reader(f)
        version = r.read("<I")
        n_tensors = r.read("<Q")
        n_kv = r.read("<Q")
        kv_start = f.tell()
        align = 32
        for _ in range(n_kv):
            key = r.string()
            vtype = r.read("<I")
            val = r.value(vtype, keep_arrays=False)
            if key == "general.alignment":
                align = int(val)
            if key.startswith("split."):
                raise ValueError("input is already split; merge it first with llama-gguf-split --merge")
        kv_end = f.tell()
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.read("<I")
            dims = [r.read("<Q") for _ in range(n_dims)]
            ttype = r.read("<I")
            offset = r.read("<Q")
            tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})
        data_start = (f.tell() + align - 1) // align * align
        f.seek(kv_start)
        kv_raw = f.read(kv_end - kv_start)
    size = os.path.getsize(path) - data_start
    order = sorted(tensors, key=lambda t: t["offset"])
    for i, t in enumerate(order):
        end = order[i + 1]["offset"] if i + 1 < len(order) else size
        t["nbytes"] = end - t["offset"]   # includes alignment padding, harmless to copy
    return version, kv_raw, n_kv, tensors, data_start, align


def _str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key, vtype, fmt, value) -> bytes:
    return _str(key) + struct.pack("<I", vtype) + struct.pack(fmt, value)


def assign(tensors, n_drives: int) -> list:
    """Drive index for each tensor: experts round-robin, everything else on drive 0."""
    out, k = [], 0
    for t in tensors:
        if n_drives > 1 and EXPERT_RE.search(t["name"]):
            out.append(k % n_drives)
            k += 1
        else:
            out.append(0)
    return out


def stripe(src: str, drive_dirs: list, link_dir: str, progress=print) -> list:
    """Write len(drive_dirs) splits of `src`, one per drive, plus symlinks in link_dir.
    Returns the path of the first split inside link_dir (pass it to llama.cpp -m)."""
    version, kv_raw, n_kv, tensors, data_start, align = _scan(src)
    n = len(drive_dirs)
    where = assign(tensors, n)
    stem = re.sub(r"\.gguf$", "", os.path.basename(src))
    names = [f"{stem}-{i + 1:05d}-of-{n:05d}.gguf" for i in range(n)]
    os.makedirs(link_dir, exist_ok=True)
    with open(src, "rb") as fin:
        for i in range(n):
            mine = [t for t, d in zip(tensors, where) if d == i]
            split_kv = (_kv(KV_SPLIT_NO, _U16, "<H", i) + _kv(KV_SPLIT_COUNT, _U16, "<H", n)
                        + _kv(KV_SPLIT_TENSORS, _I32, "<i", len(tensors)))
            kv = (kv_raw if i == 0 else b"") + split_kv
            n_kv_out = (n_kv if i == 0 else 0) + 3
            ti, off = bytearray(), 0
            for t in mine:
                ti += _str(t["name"]) + struct.pack("<I", len(t["dims"]))
                ti += b"".join(struct.pack("<Q", d) for d in t["dims"])
                ti += struct.pack("<IQ", t["type"], off)
                t["_new_off"] = off
                off += (t["nbytes"] + align - 1) // align * align
            head = GGUF_MAGIC + struct.pack("<IQQ", version, len(mine), n_kv_out) + kv + bytes(ti)
            pad = (-len(head)) % align
            os.makedirs(drive_dirs[i], exist_ok=True)
            out_path = os.path.join(drive_dirs[i], names[i])
            with open(out_path, "wb") as fout:
                fout.write(head + b"\0" * pad)
                base = fout.tell()
                for t in mine:
                    fout.seek(base + t["_new_off"])
                    fin.seek(data_start + t["offset"])
                    left = t["nbytes"]
                    while left:
                        chunk = fin.read(min(left, 16 << 20))
                        fout.write(chunk)
                        left -= len(chunk)
                fout.truncate(base + off)
            gb = sum(t["nbytes"] for t in mine) / 1e9
            progress(f"split {i + 1}/{n}: {len(mine)} tensors, {gb:.2f} GB -> {out_path}")
            link = os.path.join(link_dir, names[i])
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(os.path.abspath(out_path), link)
    return os.path.join(link_dir, names[0])

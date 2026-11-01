"""Data for the dashboard: which PocketLLM drives are plugged in, how fast each one reads,
which model parts live where, and what each model is expected to do with 1 drive vs all of them.

Drive detection is cheap (directory scans and statvfs), so the UI can poll it every couple of
seconds to show drives appearing and disappearing. Bus/filesystem facts are cached per mount
point; read speed is measured once per drive in the background (a 256 MB cold read).
"""
from __future__ import annotations

import glob
import os
import shutil
import threading
import time

from . import drives, gguf, planner, profiler, runtime

# ggml tensor types (ggml.h)
TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
              10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
              17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
              29: "IQ1_M", 30: "BF16", 39: "MXFP4"}

_info_cache: dict = {}
_speed: dict = {}          # mount -> {"mbps": float, "file": str, "at": time} | {"measuring": True}
_lock = threading.Lock()


def _mount(path):
    return profiler._mount_point(path)


def _storage(mount):
    if mount not in _info_cache:
        try:
            _info_cache[mount] = profiler.storage_info(mount)
        except OSError:
            _info_cache[mount] = {}
    return _info_cache[mount]


def _pocket_dirs(root):
    return [d for d in glob.glob(os.path.join(root, "PocketLLM*")) if os.path.isdir(d)]


def drive_list() -> list:
    """Every mounted volume that carries PocketLLM files, the app's own drive first."""
    main_mount = _mount(runtime.ROOT)
    models_home = runtime.models_dir()
    roots = [main_mount] + [r for r in drives.volume_roots() if _mount(r) != main_mount]
    out, seen = [], set()
    for root in roots:
        mount = _mount(root)
        if mount in seen:
            continue
        model_dirs = [os.path.join(d, "models") for d in _pocket_dirs(root)]
        if os.path.isdir(models_home) and _mount(models_home) == mount:
            model_dirs.insert(0, models_home)
        if mount != main_mount and not model_dirs and not _pocket_dirs(root):
            continue
        if mount == main_mount and not model_dirs:
            continue  # app runs from the computer's own disk and keeps no models there
        seen.add(mount)
        files, seen_files = [], set()
        for md in model_dirs:
            for p in sorted(glob.glob(os.path.join(md, "*.gguf"))):
                rp = os.path.realpath(p)
                if os.path.isfile(p) and not os.path.islink(p) and rp not in seen_files:
                    seen_files.add(rp)
                    files.append({"file": os.path.basename(p), "gb": round(os.path.getsize(p) / 1e9, 2), "path": p})
        try:
            du = shutil.disk_usage(root)
            free, total = du.free, du.total
        except OSError:
            free = total = 0
        st = _storage(mount)
        if mount == main_mount and not files and not st.get("is_usb"):
            continue
        out.append({"mount": mount, "name": os.path.basename(mount.rstrip("/\\")) or mount,
                    "main": mount == main_mount, "usb": st.get("is_usb"), "fs": st.get("fs"),
                    "free_gb": round(free / 1e9, 1), "total_gb": round(total / 1e9, 1),
                    "files": files, "speed": _speed.get(mount), "warning": st.get("warning")})
    return out


def measure_speeds(drive_rows, force=False):
    """Measure each drive's cold read speed once (background thread), on its biggest model file."""
    todo = []
    with _lock:
        for d in drive_rows:
            s = _speed.get(d["mount"])
            if d["files"] and (force or s is None):
                _speed[d["mount"]] = {"measuring": True}
                todo.append(d)

    def work():
        for d in todo:
            f = max(d["files"], key=lambda x: x["gb"])["path"]
            try:
                r = profiler.measure_read_speed(f, mb=256)
                res = {"mbps": round(r["gb_per_s"] * 1000) if r.get("gb_per_s") else None,
                       "file": os.path.basename(f), "cold": r.get("cache_evicted"), "at": time.time()}
            except OSError as e:
                res = {"error": str(e)}
            with _lock:
                _speed[d["mount"]] = res
    if todo:
        threading.Thread(target=work, daemon=True).start()


def quant_profile(m: gguf.GGUFModel) -> dict:
    """Dominant tensor type of the routed experts and of everything else, by bytes."""
    by = {True: {}, False: {}}
    for t in m.tensors:
        if len(t.shape) < 2:
            continue
        k = TYPE_NAMES.get(t.ggml_type, str(t.ggml_type))
        by[t.is_expert][k] = by[t.is_expert].get(k, 0) + t.nbytes
    pick = lambda d: max(d, key=d.get) if d else None  # noqa: E731
    return {"experts": pick(by[True]), "rest": pick(by[False])}


# Measured on Qwen3-30B-A3B against its Q4_K_M file (docs/RESULTS.md R17, R19)
MEASURED = {"family": "Qwen3-30B-A3B", "ref": "Q4_K_M",
            "experts=Q3_K": ("near-lossless", 0.065), "experts=Q3_K,topk=6": ("acceptable", 0.119),
            "ud-q2": ("acceptable", 0.128), "ud-q2,topk=6": ("not recommended", 0.171),
            "experts=Q2_K": ("not recommended", 0.336), "topk=6": ("near-lossless", 0.061),
            "topk=4": ("not recommended", 0.299)}


def quality(m: gguf.GGUFModel, path: str, topk: int | None = None) -> dict:
    q = quant_profile(m)
    n_used = m.n_expert_used
    is_family = m.arch == "qwen3moe" and m.n_expert == 128
    base = os.path.basename(path).lower()
    # files made by `pocketllm requant` (or a published low-bit MoE quant) vs files left as published
    if "ud-q2" in base:
        key = "ud-q2"
    elif q["experts"] in ("Q3_K", "Q2_K") and q["rest"] == "Q8_0":
        key = f"experts={q['experts']}"
    else:
        key = "original"
    if topk and n_used and topk < n_used:
        key = f"topk={topk}" if key == "original" else f"{key},topk={topk}"
    out = {"quant": q, "tier": None, "kld": None, "measured_on": None}
    if key == "original":
        out.update(tier="original", kld=0.0)
    elif is_family and key in MEASURED:
        tier, kld = MEASURED[key]
        out.update(tier=tier, kld=kld, measured_on=f"{MEASURED['family']} vs {MEASURED['ref']}")
    return out


def fewer_topk(n_used: int) -> int | None:
    """The UI's "fewer experts" choice: 3/4 of the routed experts (6 of 8 measured near-lossless, R17)."""
    if not n_used or n_used < 2:
        return None
    return max(1, min(n_used - 1, -(-n_used * 3 // 4)))


def model_rows(profile: dict, drive_rows: list) -> list:
    """Local models with their parts per drive and estimated tok/s on 1 drive vs all drives,
    with the model's own experts per token and with "fewer experts"."""
    from . import core
    speeds = {d["mount"]: (d.get("speed") or {}).get("mbps") for d in drive_rows}
    rows = []
    for spec in core.local_specs():
        if spec.source != "gguf":
            continue
        r = drives.resolve(spec.path)
        parts = r.get("parts") or [spec.path]
        mounts = sorted({_mount(os.path.realpath(p)) for p in parts})
        m = gguf.load(spec.path)
        known = [speeds.get(x) for x in mounts if speeds.get(x)]
        one = min(known) if known else None
        allbw = sum(known) if len(known) == len(mounts) and known else None

        def estimate(topk):
            est = {}
            for label, mbps in (("one_drive", one), ("all_drives", allbw)):
                if mbps:
                    pl = planner.plan(spec, core.budget(profile, disk_gbps=mbps / 1000), topk=topk)
                    est[label] = {"mbps": mbps, "tok_s": pl.est_tok_s_upper, "strategy": pl.strategy,
                                  "verdict": pl.verdict}
            return est
        row = {"name": spec.name, "path": spec.path, "file": os.path.basename(spec.path),
               "size_gb": round(spec.total_bytes / 1e9, 2), "moe": spec.is_moe,
               "n_expert_used": spec.n_expert_used,
               "drives": [os.path.basename(x.rstrip("/\\")) or x for x in mounts], "n_drives": len(mounts),
               "estimate": estimate(None), "quality": quality(m, spec.path)}
        k = fewer_topk(spec.n_expert_used) if spec.is_moe else None
        if k:
            row["fewer"] = {"topk": k, "estimate": estimate(k), "quality": quality(m, spec.path, k)}
        rows.append(row)
    return rows

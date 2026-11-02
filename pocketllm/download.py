"""Download a published GGUF from Hugging Face onto the PocketLLM drives (provisioning only).

Standard library only, so it runs from the drive's portable Python. A model published as
split parts (``-00001-of-0000N.gguf``) has its parts dealt over the connected drives, which
PocketLLM reads in parallel; a single-file MoE model is striped over the drives afterwards
(`pocketllm stripe`). Files are written as ``.part`` and renamed when complete, so an
interrupted download never looks like a finished model.

Refused under an air-gap policy: the field side never downloads anything.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import urllib.request

HF_BASE = os.environ.get("POCKETLLM_HF_BASE", "https://huggingface.co")
SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")

state = {"running": False, "model": None, "done_bytes": 0, "total_bytes": 0, "file": None,
         "log": [], "error": None, "result": None, "started": None}
_lock = threading.Lock()


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "pocketllm"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def list_files(repo: str) -> list:
    """[{"path", "size"}] for every GGUF in the repo (sub-folders included)."""
    out = []
    for item in _get_json(f"{HF_BASE}/api/models/{repo}/tree/main?recursive=true"):
        if item.get("type") == "file" and item["path"].endswith(".gguf"):
            size = (item.get("lfs") or {}).get("size") or item.get("size") or 0
            out.append({"path": item["path"], "size": int(size)})
    return out


def pick(files: list, quant: str) -> list:
    """The file (or every part of the split set) for one quantization, e.g. Q4_K_M.
    "Q2_K" must not match "Q2_K_XL" or "UD-Q2_K_XL"."""
    pat = re.compile(rf"(?<![A-Za-z0-9])(?<!UD-){re.escape(quant)}(?=[-.])", re.I)
    hits = sorted((f for f in files if pat.search(os.path.basename(f["path"]))), key=lambda f: f["path"])
    if not hits:
        return []
    first = hits[0]
    m = SPLIT.search(first["path"])
    if not m:
        return [first]
    stem = first["path"][:m.start()]
    return [f for f in hits if f["path"].startswith(stem) and SPLIT.search(f["path"])]


def plan_placement(files: list, drive_rows: list) -> list:
    """Assign each file to a drive's models folder, spreading split parts over the drives.
    Returns [(file, dest_dir)] or raises ValueError when they do not fit."""
    targets = []
    for d in drive_rows:
        md = d.get("models_dir")
        if md:
            targets.append({"dir": md, "free": d["free_gb"] * 1e9 - 0.5e9})  # keep 0.5 GB spare
    if not targets:
        raise ValueError("no PocketLLM drive connected")
    out = []
    for i, f in enumerate(files):
        order = targets[i % len(targets):] + targets[:i % len(targets)]
        t = next((t for t in order if t["free"] >= f["size"]), None)
        if t is None:
            need = sum(x["size"] for x in files) / 1e9
            have = sum(max(0, x["free"]) for x in targets) / 1e9
            raise ValueError(f"not enough space on the drives: need {need:.1f} GB, free {have:.1f} GB")
        t["free"] -= f["size"]
        out.append((f, t["dir"]))
    return out


def _fetch(repo, f, dest_dir, log):
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(f["path"]))
    if os.path.isfile(dest) and os.path.getsize(dest) == f["size"]:
        with _lock:
            state["done_bytes"] += f["size"]
        log(f"already there: {dest}")
        return dest
    tmp = dest + ".part"
    url = f"{HF_BASE}/{repo}/resolve/main/{urllib.request.quote(f['path'])}"
    req = urllib.request.Request(url, headers={"User-Agent": "pocketllm"})
    with _lock:
        state["file"] = os.path.basename(dest)
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as out:
        while True:
            chunk = r.read(8 << 20)
            if not chunk:
                break
            out.write(chunk)
            with _lock:
                state["done_bytes"] += len(chunk)
    if f["size"] and os.path.getsize(tmp) != f["size"]:
        raise IOError(f"{os.path.basename(dest)}: got {os.path.getsize(tmp)} bytes, expected {f['size']}")
    os.replace(tmp, dest)
    log(f"saved {dest}")
    return dest


def start(entry: dict, quant: str, drive_rows: list, on_done=None) -> dict:
    """Begin downloading `quant` of a catalog entry in the background."""
    with _lock:
        if state["running"]:
            return {"error": "a download is already running"}
    files = pick(list_files(entry["hf"]), quant)
    if not files:
        return {"error": f"{quant} not found in {entry['hf']}"}
    placement = plan_placement(files, drive_rows)
    total = sum(f["size"] for f in files)
    with _lock:
        state.update(running=True, model=f"{entry['id']}:{quant}", done_bytes=0, total_bytes=total,
                     file=None, log=[], error=None, result=None, started=time.time())

    def log(msg):
        with _lock:
            state["log"].append(msg)

    def work():
        try:
            log(f"{entry['hf']}: {len(files)} file(s), {total / 1e9:.1f} GB onto "
                f"{len({d for _, d in placement})} drive(s)")
            paths = [_fetch(entry["hf"], f, d, log) for f, d in placement]
            result = {"paths": paths}
            if on_done:
                result.update(on_done(paths) or {})
            with _lock:
                state.update(running=False, result=result)
        except Exception as e:  # network, disk full, interrupted
            with _lock:
                state.update(running=False, error=f"{type(e).__name__}: {e}")
    threading.Thread(target=work, daemon=True).start()
    return {"started": True, "files": [os.path.basename(f["path"]) for f in files], "total_gb": round(total / 1e9, 2)}


def stripe_if_useful(paths: list, drive_dirs: list, log=print) -> dict:
    """A single-file MoE model with more than one drive connected: deal its experts over the drives."""
    from . import gguf, stripe
    if len(paths) != 1 or len(drive_dirs) < 2:
        return {}
    m = gguf.load(paths[0])
    if not m.is_moe:
        return {}
    src = paths[0]
    size = os.path.getsize(src)
    if any(shutil.disk_usage(d).free < size / len(drive_dirs) + 0.5e9 for d in drive_dirs):
        log("not striped: not enough free space to split it over the drives")
        return {}
    stem = os.path.basename(src)[:-5]
    from . import runtime
    link = stripe.stripe(src, drive_dirs, os.path.join(runtime.data_dir(), "links", stem), progress=log)
    os.remove(src)
    log(f"striped over {len(drive_dirs)} drives")
    return {"striped": link}

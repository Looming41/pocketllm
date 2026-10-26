"""Find the parts of a model striped over several USB drives, on whatever
computer they are plugged into.

Mount points differ between machines (/Volumes/X on macOS, /media/<user>/X or
/run/media/<user>/X on Linux), so a striped set cannot rely on fixed paths.
The main drive holds part 1 in PocketLLM/models/. At launch PocketLLM scans the
mounted volumes for ``*/PocketLLM*/models/<stem>-0000i-of-0000N.gguf`` and
links the parts it finds into data/links/<stem>/, which is what llama.cpp
loads. Missing drives are reported by name.
"""
from __future__ import annotations

import glob
import os
import platform

from . import gguf, runtime


def volume_roots() -> list:
    system = platform.system()
    pats = []
    if system == "Darwin":
        pats = ["/Volumes/*"]
    elif system == "Linux":
        pats = ["/media/*", "/media/*/*", "/run/media/*/*", "/mnt/*"]
    elif system == "Windows":
        return [f"{c}:\\" for c in "DEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{c}:\\")]
    extra = os.environ.get("POCKETLLM_EXTRA_DRIVES", "")
    roots = [p for pat in pats for p in glob.glob(pat) if os.path.isdir(p)]
    roots += [p for p in extra.split(os.pathsep) if p]
    return roots


def candidate_model_dirs() -> list:
    dirs = [runtime.models_dir()]
    for r in volume_roots():
        for d in glob.glob(os.path.join(r, "PocketLLM*", "models")) + glob.glob(os.path.join(r, "models")):
            if os.path.isdir(d) and os.path.abspath(d) not in map(os.path.abspath, dirs):
                dirs.append(d)
    return dirs


def resolve(first_part: str) -> dict:
    """Locate all parts of a split model. Returns {"path": loadable first part or None, "missing": [...], "parts": [...]}."""
    base = os.path.basename(first_part)
    m = gguf.SPLIT_RE.match(base)
    if not m:
        return {"path": first_part, "missing": [], "parts": [first_part]}
    stem, _, total = m.groups()
    names = [f"{stem}-{i:05d}-of-{total}.gguf" for i in range(1, int(total) + 1)]
    found = {}
    for d in [os.path.dirname(os.path.abspath(first_part))] + candidate_model_dirs():
        for n in names:
            p = os.path.join(d, n)
            if n not in found and os.path.isfile(p):
                found[n] = os.path.realpath(p)
    missing = [n for n in names if n not in found]
    if missing:
        return {"path": None, "missing": missing, "parts": list(found.values())}
    same_dir = len({os.path.dirname(p) for p in found.values()}) == 1
    if same_dir:
        return {"path": found[names[0]], "missing": [], "parts": [found[n] for n in names]}
    link_dir = os.path.join(runtime.data_dir(), "links", stem)
    os.makedirs(link_dir, exist_ok=True)
    for n in names:
        link = os.path.join(link_dir, n)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(found[n], link)
    return {"path": os.path.join(link_dir, names[0]), "missing": [], "parts": [found[n] for n in names],
            "drives": sorted({os.path.dirname(p) for p in found.values()})}

"""Glue shared by the CLI and the web UI."""
from __future__ import annotations

import os
import socket

from . import catalog, gguf, planner, profiler, runtime


def network_online(timeout=1.5) -> bool:
    """True if a public host is reachable. Used only to *display* status and to
    decide whether cloud features may be offered; inference never needs it."""
    for host, port in (("1.1.1.1", 443), ("8.8.8.8", 53)):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


INCOMPLETE = []   # split models whose other parts are on drives that are not plugged in


def local_specs(models_root=None) -> list:
    from . import drives
    specs = []
    INCOMPLETE.clear()
    for path in gguf.find_models(models_root or runtime.models_dir()):
        r = drives.resolve(path)   # parts may be striped over several USB drives
        if r["path"] is None:
            INCOMPLETE.append({"model": os.path.basename(path), "missing_parts": r["missing"]})
            continue
        try:
            specs.append(planner.ModelSpec.from_gguf(gguf.load(r["path"]), r["path"]))
        except (ValueError, OSError):
            continue
    return specs


def hardware_summary(p: dict) -> dict:
    gpus = [{"name": g.get("name"), "vram_gb": round((g.get("vram_total_bytes") or 0) / 1e9, 1),
             "unified": g.get("unified_memory")} for g in p.get("gpus", [])]
    st = p.get("storage") or {}
    return {
        "os": f'{p["os"]["system"]} {p["os"]["release"]}',
        "cpu": p["cpu"].get("model"), "cores": p["cpu"].get("physical_cores") or p["cpu"].get("logical_cores"),
        "cpu_flags": p["cpu"].get("flags"),
        "ram_gb": round((p["ram"].get("total_bytes") or 0) / 1e9, 1),
        "ram_free_gb": round((p["ram"].get("available_bytes") or 0) / 1e9, 1),
        "ram_bw_1thread_gbps": (p["ram"].get("bandwidth") or {}).get("gb_per_s_1thread"),
        "gpus": gpus,
        "drive": {"path": st.get("path"), "usb": st.get("is_usb"), "removable": st.get("is_removable"),
                  "fs": st.get("fs"), "free_gb": round((st.get("free_bytes") or 0) / 1e9, 1),
                  "read_gbps": (st.get("read_test") or {}).get("gb_per_s"), "warning": st.get("warning")},
    }


def full_profile(read_test=True) -> dict:
    p = profiler.profile(runtime.models_dir())
    st = p.get("storage")
    models = gguf.find_models(runtime.models_dir())
    if st is not None and read_test and models:
        biggest = max(models, key=os.path.getsize)
        try:
            st["read_test"] = profiler.measure_read_speed(biggest)
        except OSError as e:
            st["read_test"] = {"error": str(e)}
    return p


def budget(p: dict, ram_gbps=None, disk_gbps=None) -> planner.Budget:
    if disk_gbps is None:
        disk_gbps = ((p.get("storage") or {}).get("read_test") or {}).get("gb_per_s")
    return planner.budget_from_profile(p, disk_gbps=disk_gbps, ram_gbps=ram_gbps)


def recommendations(p: dict, ctx=4096, include_catalog=True, ram_gbps=None, disk_gbps=None) -> list:
    b = budget(p, ram_gbps, disk_gbps)
    specs = local_specs()
    if include_catalog:
        specs += catalog.specs()
    return planner.recommend(specs, b, ctx)

"""Measured auto-tuning: runs llama-bench over a small set of configurations
around the planner's choice and keeps the fastest one that does not fail.

Also measures the "naive default" (llama.cpp defaults) so the demo can show
default vs. tuned on the same model, prompt length and hardware.
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import subprocess
import threading
import time

from . import runtime


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + \
               [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]


def _win_process(pid):
    k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_VM_READ = 0x1000, 0x0010
    return k32, k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, int(pid))


def _rss_bytes(pid):
    if platform.system() == "Windows":
        try:
            k32, h = _win_process(pid)
            if not h:
                return None
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)  # type: ignore[attr-defined]
            k32.CloseHandle(h)
            return int(pmc.WorkingSetSize) if ok else None
        except (OSError, AttributeError):
            return None
    if platform.system() == "Linux":
        try:
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            return None
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=2).stdout
        return int(out.strip()) * 1024 if out.strip() else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def read_bytes(pid):
    """Bytes this process has caused to be read from storage (page-cache misses, read-ahead).
    Linux: /proc/<pid>/io; macOS: proc_pid_rusage(RUSAGE_INFO_V2).ri_diskio_bytesread."""
    system = platform.system()
    if system == "Linux":
        try:
            with open(f"/proc/{pid}/io") as f:
                for line in f:
                    if line.startswith("read_bytes:"):
                        return int(line.split()[1])
        except OSError:
            return None
    elif system == "Windows":
        # ReadTransferCount counts read I/O issued by the process; whether paging reads of
        # memory-mapped files are included is not verified, so treat it as indicative only
        try:
            k32, h = _win_process(pid)
            if not h:
                return None
            c = _IO_COUNTERS()
            ok = k32.GetProcessIoCounters(h, ctypes.byref(c))
            k32.CloseHandle(h)
            return int(c.ReadTransferCount) if ok else None
        except (OSError, AttributeError):
            return None
    elif system == "Darwin":
        try:
            lib = ctypes.CDLL("/usr/lib/libproc.dylib")
            buf = (ctypes.c_uint64 * 40)()
            if lib.proc_pid_rusage(int(pid), 2, ctypes.byref(buf)) == 0:
                # rusage_info_v2: 16-byte uuid, then uint64 fields; ri_diskio_bytesread is the 18th
                return int(buf[2 + 17])
        except (OSError, AttributeError):
            return None
    return None


def run_bench(model_path, cfg: dict, n_prompt=128, n_gen=64, reps=2, timeout=900, env=None):
    """Run one llama-bench configuration. Returns a result dict (never raises)."""
    exe = runtime.binary("llama-bench")
    args = [exe, "-m", model_path, "-p", str(n_prompt), "-n", str(n_gen), "-r", str(reps), "-o", "json"]
    for k, flag in (("threads", "-t"), ("ngl", "-ngl"), ("ncmoe", "-ncmoe"), ("fa", "-fa"), ("load_mode", "-lm"),
                    ("ubatch", "-ub"), ("repack", "--repack")):
        if cfg.get(k) is not None:
            args += [flag, str(cfg[k])]
    peak = [0]
    io = [None]
    t0 = time.perf_counter()
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                env=dict(os.environ, **(env or {})))
    except OSError as e:
        return {"config": cfg, "ok": False, "error": str(e)}

    def watch():
        while proc.poll() is None:
            r = _rss_bytes(proc.pid)
            if r:
                peak[0] = max(peak[0], r)
            b = read_bytes(proc.pid)
            if b is not None:
                io[0] = b
            time.sleep(0.2)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return {"config": cfg, "ok": False, "error": f"timeout after {timeout}s"}
    th.join(1)
    res = {"config": cfg, "ok": proc.returncode == 0, "wall_s": round(time.perf_counter() - t0, 1),
           "peak_rss_gb": round(peak[0] / 1e9, 2) if peak[0] else None,
           "storage_read_gb": round(io[0] / 1e9, 2) if io[0] is not None else None}
    if proc.returncode != 0:
        res["error"] = (err.strip().splitlines() or ["unknown error"])[-1][:300]
        return res
    try:
        rows = json.loads(out)
    except ValueError:
        res.update(ok=False, error="could not parse llama-bench output")
        return res
    for r in rows:
        if r.get("n_prompt") and not r.get("n_gen"):
            res["pp_tok_s"] = round(r["avg_ts"], 2)
        elif r.get("n_gen") and not r.get("n_prompt"):
            res["tg_tok_s"] = round(r["avg_ts"], 2)
            res["tg_stddev"] = round(r.get("stddev_ts", 0), 2)
        res.setdefault("backend", r.get("backends"))
        res.setdefault("build", r.get("build_commit"))
    return res


def candidates(plan: dict, profile: dict) -> list:
    """Small search space around the plan (kept small: each run loads the model)."""
    cpu = profile.get("cpu", {})
    phys = cpu.get("physical_cores") or cpu.get("logical_cores") or 4
    logical = cpu.get("logical_cores") or phys
    threads = sorted({phys, max(1, phys // 2), logical})
    has_gpu = bool(profile.get("gpus"))
    base = {"ngl": plan.get("n_gpu_layers", 0) if has_gpu else 0}
    if plan.get("n_cpu_moe"):
        base["ncmoe"] = plan["n_cpu_moe"]
    base["load_mode"] = "none" if plan.get("no_mmap") else "mmap"
    if not plan.get("no_mmap"):
        base["repack"] = 0
    out = []
    for t in threads:
        for fa in ("off", "on"):
            out.append({**base, "threads": t, "fa": fa})
    # neighbours of the MoE split: one more / one fewer layer of experts on CPU
    if plan.get("n_cpu_moe"):
        for d in (-2, 2):
            out.append({**base, "threads": phys, "fa": "on", "ncmoe": max(0, plan["n_cpu_moe"] + d)})
    if has_gpu and plan.get("n_gpu_layers") and not plan.get("n_cpu_moe"):
        for d in (-4, 4):
            out.append({**base, "threads": phys, "fa": "on", "ngl": max(0, plan["n_gpu_layers"] + d)})
    return out


def tune(model_path, plan: dict, profile: dict, n_prompt=128, n_gen=64, log=print, results_path=None):
    results = []
    default = run_bench(model_path, {"label": "llama.cpp defaults"}, n_prompt, n_gen)
    default["label"] = "llama.cpp defaults"
    results.append(default)
    log(_fmt(default))
    env = plan.get("env") or {}
    for cfg in candidates(plan, profile):
        r = run_bench(model_path, cfg, n_prompt, n_gen, env=env)
        if env:
            r["env"] = env
        results.append(r)
        log(_fmt(r))
    ok = [r for r in results if r.get("ok") and r.get("tg_tok_s")]
    best = max(ok, key=lambda r: (r["tg_tok_s"], r.get("pp_tok_s", 0)), default=None)
    # only call it an improvement if the gap exceeds the combined run-to-run noise
    significant = None
    if best and default.get("tg_tok_s") and best is not default:
        noise = 2 * ((best.get("tg_stddev") or 0) + (default.get("tg_stddev") or 0))
        significant = best["tg_tok_s"] - default["tg_tok_s"] > max(noise, 0.03 * default["tg_tok_s"])
    summary = {
        "model": os.path.basename(model_path), "n_prompt": n_prompt, "n_gen": n_gen,
        "default": default, "best": best, "all": results, "improvement_significant": significant,
        "speedup_tg": round(best["tg_tok_s"] / default["tg_tok_s"], 2) if best and default.get("tg_tok_s") else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": platform.node(),
    }
    if results_path:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
    return summary


def _fmt(r):
    cfg = {k: v for k, v in r["config"].items() if k != "label"} or "defaults"
    if not r.get("ok"):
        return f"  FAIL {cfg}: {r.get('error')}"
    return (f"  pp {r.get('pp_tok_s', '?'):>8} tok/s | tg {r.get('tg_tok_s', '?'):>7} tok/s | "
            f"peak RSS {r.get('peak_rss_gb')} GB | {cfg}")


STREAM_ENV = {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1"}


def evict_cache(model_path) -> bool:
    """Drop the model's pages from the OS cache (Linux, no root needed). Returns False where
    that is not possible (macOS needs `sudo purge`), so results there may start warm."""
    if not hasattr(os, "posix_fadvise"):
        return False
    from . import gguf
    ok = True
    for p in gguf.split_siblings(model_path):
        try:
            fd = os.open(os.path.realpath(p), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            ok = False
    return ok


def ab_compare(model_path, plan: dict, profile: dict, n_gen=16, log=print):
    """Quick demo comparison on the same model and settings: stock llama.cpp vs the PocketLLM engine."""
    cpu = profile.get("cpu", {})
    threads = cpu.get("physical_cores") or cpu.get("logical_cores") or 4
    cfg = {"threads": threads, "load_mode": "mmap", "repack": 0,
           "ngl": plan.get("n_gpu_layers", 0) if profile.get("gpus") else 0}
    out = {"cold_start": evict_cache(model_path)}
    for label, env in (("stock llama.cpp", {}), ("PocketLLM engine", plan.get("env") or STREAM_ENV)):
        evict_cache(model_path)   # both runs start from the drive, not from the other run's page cache
        r = run_bench(model_path, dict(cfg), n_prompt=0, n_gen=n_gen, reps=1, env=env)
        r["label"] = label
        out[label] = r
        log(f"  {label:18} {r.get('tg_tok_s', 'fail')} tok/s, read {r.get('storage_read_gb')} GB from storage")
    a, b = out["stock llama.cpp"].get("tg_tok_s"), out["PocketLLM engine"].get("tg_tok_s")
    out["speedup"] = round(b / a, 2) if a and b else None
    return out

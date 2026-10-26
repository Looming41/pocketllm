"""Measured auto-tuning: runs llama-bench over a small set of configurations
around the planner's choice and keeps the fastest one that does not fail.

Also measures the "naive default" (llama.cpp defaults) so the demo can show
default vs. tuned on the same model, prompt length and hardware.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time

from . import runtime


def _rss_bytes(pid):
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


def run_bench(model_path, cfg: dict, n_prompt=128, n_gen=64, reps=2, timeout=900):
    """Run one llama-bench configuration. Returns a result dict (never raises)."""
    exe = runtime.binary("llama-bench")
    args = [exe, "-m", model_path, "-p", str(n_prompt), "-n", str(n_gen), "-r", str(reps), "-o", "json"]
    for k, flag in (("threads", "-t"), ("ngl", "-ngl"), ("ncmoe", "-ncmoe"), ("fa", "-fa"), ("load_mode", "-lm"),
                    ("ubatch", "-ub")):
        if cfg.get(k) is not None:
            args += [flag, str(cfg[k])]
    peak = [0]
    t0 = time.perf_counter()
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as e:
        return {"config": cfg, "ok": False, "error": str(e)}

    def watch():
        while proc.poll() is None:
            r = _rss_bytes(proc.pid)
            if r:
                peak[0] = max(peak[0], r)
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
           "peak_rss_gb": round(peak[0] / 1e9, 2) if peak[0] else None}
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
    for cfg in candidates(plan, profile):
        r = run_bench(model_path, cfg, n_prompt, n_gen)
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

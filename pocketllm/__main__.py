"""PocketLLM command line.

    python -m pocketllm profile            # hardware report (+ USB read test)
    python -m pocketllm inspect MODEL.gguf # exact sizes from the GGUF header
    python -m pocketllm plan [--ctx N]     # feasibility for local + catalog models
    python -m pocketllm tune MODEL.gguf    # measured llama-bench sweep, default vs tuned
    python -m pocketllm serve              # web UI on http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import core, gguf, planner, runtime, tuner
from .profiler import human

VERDICT = {"fits": "OK", "slow": "SLOW", "impractical": "IMPRACTICAL", "does_not_fit": "NO"}


def cmd_profile(a):
    p = core.full_profile(read_test=not a.no_read_test)
    if a.json:
        print(json.dumps(p, indent=2))
        return
    s = core.hardware_summary(p)
    print(f"OS        {s['os']}")
    print(f"CPU       {s['cpu']}  ({s['cores']} cores)  {' '.join(s['cpu_flags'] or [])}")
    print(f"RAM       {s['ram_gb']} GB total, {s['ram_free_gb']} GB available, "
          f"~{s['ram_bw_1thread_gbps']} GB/s single-thread copy (lower bound)")
    for g in s["gpus"] or [{"name": "none detected", "vram_gb": 0, "unified": None}]:
        print(f"GPU       {g['name']}  {g['vram_gb']} GB{' (unified memory)' if g['unified'] else ''}")
    d = s["drive"]
    print(f"Drive     {d['path']}  usb={d['usb']} removable={d['removable']} fs={d['fs']} "
          f"free={d['free_gb']} GB read={d['read_gbps']} GB/s")
    if d.get("warning"):
        print(f"WARNING   {d['warning']}")
    runtime.save_json("profile.json", p)


def cmd_inspect(a):
    m = gguf.load(a.model)
    print(json.dumps(m.summary(), indent=2))


def cmd_plan(a):
    p = core.full_profile(read_test=not a.no_read_test)
    ranked = core.recommendations(p, ctx=a.ctx, include_catalog=not a.local_only, ram_gbps=a.ram_gbps,
                                  disk_gbps=a.disk_gbps)
    b = core.budget(p, a.ram_gbps, a.disk_gbps)
    print(f"budget: RAM {human(b.ram_budget)}, GPU {human(b.gpu_budget)}{' (unified)' if b.unified else ''}, "
          f"RAM bw {b.ram_bw and round(b.ram_bw / 1e9, 1)} GB/s{' (1-thread lower bound)' if b.bw_is_lower_bound else ''}, "
          f"storage {b.disk_bw and round(b.disk_bw / 1e9, 2)} GB/s, ctx {a.ctx}")
    print(f"{'model':48} {'src':8} {'size':>9} {'verdict':11} {'strategy':12} {'est tok/s<=':>11}  bottleneck")
    for pl, spec in ranked:
        est = f"{pl.est_tok_s_upper:.2f}" if pl.est_tok_s_upper else "-"
        print(f"{spec.name[:48]:48} {spec.source:8} {human(spec.total_bytes):>9} {VERDICT[pl.verdict]:11} "
              f"{pl.strategy:12} {est:>11}  {pl.bottleneck}")
        if a.verbose:
            for n in pl.notes:
                print(f"{'':50}- {n}")
            if pl.args:
                print(f"{'':50}llama-server {' '.join(pl.args)}")
    print("\nest tok/s is an upper bound from a bandwidth model, not a measurement. Run `tune` to measure.")


def cmd_tune(a):
    p = core.full_profile(read_test=False)
    spec = planner.ModelSpec.from_gguf(gguf.load(a.model), a.model)
    pl = planner.plan(spec, core.budget(p), ctx=a.ctx)
    print(f"plan: {pl.strategy} / {pl.verdict}  args: {' '.join(pl.args)}")
    out = os.path.join(runtime.data_dir(), "tuning", os.path.basename(a.model) + ".json")
    s = tuner.tune(a.model, pl.to_dict(), p, n_prompt=a.n_prompt, n_gen=a.n_gen, results_path=out)
    if s["best"]:
        print(f"\nbest: {s['best']['config']}  tg {s['best']['tg_tok_s']} tok/s "
              f"(defaults: {s['default'].get('tg_tok_s')} tok/s, x{s['speedup_tg']}, "
              f"{'beyond' if s['improvement_significant'] else 'within'} run-to-run noise)")
    print(f"saved {out}")


def cmd_serve(a):
    from . import server
    server.main(a.host, a.port, a.llama_port)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pocketllm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("profile")
    s.add_argument("--json", action="store_true")
    s.add_argument("--no-read-test", action="store_true")
    s.set_defaults(fn=cmd_profile)
    s = sub.add_parser("inspect")
    s.add_argument("model")
    s.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("plan")
    s.add_argument("--ctx", type=int, default=4096)
    s.add_argument("--local-only", action="store_true")
    s.add_argument("--ram-gbps", type=float, help="override RAM bandwidth (e.g. from a STREAM run)")
    s.add_argument("--disk-gbps", type=float, help="override storage read speed")
    s.add_argument("--no-read-test", action="store_true")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("tune")
    s.add_argument("model")
    s.add_argument("--ctx", type=int, default=4096)
    s.add_argument("--n-prompt", type=int, default=128)
    s.add_argument("--n-gen", type=int, default=64)
    s.set_defaults(fn=cmd_tune)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--llama-port", type=int, default=8081)
    s.set_defaults(fn=cmd_serve)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

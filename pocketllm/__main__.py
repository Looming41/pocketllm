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
    for x in core.INCOMPLETE:
        print(f"{x['model'][:48]:48} plug in the drive(s) holding: {', '.join(x['missing_parts'])}")
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


def cmd_stripe(a):
    from . import stripe
    first = stripe.stripe(a.model, a.drives, a.link_dir)
    print(f"\nload with: -m {first}")


def cmd_seal(a):
    from . import security
    path = security.seal(a.root, a.key, progress=print if a.verbose else (lambda *_: None))
    print(f"sealed: {path} (+ manifest.sig, pocketllm_pub.pem)")


def cmd_verify(a):
    from . import security
    r = security.verify(a.root, "full" if a.full else "quick", a.pubkey)
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["ok"] else 1)


def cmd_node(a):
    from . import cluster
    cluster.serve_node(a.host, a.port, a.threads, cache=not a.no_cache, allow_any=a.i_understand)


def cmd_cluster(a):
    from . import cluster
    sts = [cluster.node_status(n) for n in a.nodes.split(",")]
    m = gguf.load(a.model)
    plan = cluster.split(sts, m.weight_bytes)
    print(json.dumps(plan, indent=2))
    if not plan["fits"]:
        print("the pooled memory is not enough for this model", file=sys.stderr)
        sys.exit(1)
    args = cluster.llama_args(a.model, plan, a.ctx)
    print("llama-server " + " ".join(args))
    if a.run:
        srv = runtime.LlamaServer(port=a.llama_port)
        ok = srv.start(args)
        print(f"running at {srv.url}  (API key in {os.path.join(runtime.data_dir(), 'llama-server.key')})"
              if ok else "failed to start, see " + srv.log_path)
        if ok:
            try:
                srv.proc.wait()
            except KeyboardInterrupt:
                srv.stop()


def cmd_thermal(a):
    from . import thermal
    st = thermal.thermal_state()
    prof = thermal.auto_profile(st)
    cpu = core.full_profile(read_test=False)["cpu"]
    print(json.dumps(st | {"suggested_profile": prof,
                           "threads": {k: thermal.threads_for(k, cpu) for k in thermal.PROFILES}}, indent=2))


def _load_profiles(d):
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            with open(os.path.join(d, fn)) as f:
                out.append((fn[:-5], json.load(f)))
    return out


def cmd_fleet_plan(a):
    from . import provision
    plan = provision.fleet_plan(runtime.ROOT, _load_profiles(a.profiles), a.goal, a.ctx, use_meta=not a.no_meta)
    out = os.path.join(runtime.data_dir(), "fleet_plan.json")
    with open(out, "w") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"models that run on every device: {plan['runs_on_all_devices'][:6]}")
    if plan.get("brief"):
        print(f"\n[{plan['brief']['source']}]\n{plan['brief']['text']}")
    print(f"saved {out}")


def cmd_evalset(a):
    from . import provision
    items = provision.make_evalset(runtime.ROOT, a.domain, a.n)
    with open(a.out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} items to {a.out}")


def cmd_qualify(a):
    from . import provision
    items = [json.loads(l) for l in open(a.evalset) if l.strip()]
    cands = dict(c.split("=", 1) for c in a.candidates)
    rep = provision.qualify(runtime.ROOT, items, cands, a.pass_mean, a.local_judge)
    out = os.path.join(runtime.data_dir(), "qualification.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    for k, v in rep["candidates"].items():
        print(f"{k:30} mean {v['mean_score']}  {'PASS' if v['passed'] else 'fail'}")
    print(f"selected: {rep['selected']}  (saved {out})")


def cmd_ab(a):
    p = core.full_profile(read_test=False)
    spec = planner.ModelSpec.from_gguf(gguf.load(a.model), a.model)
    pl = planner.plan(spec, core.budget(p), ctx=a.ctx)
    print(f"plan: {pl.strategy} / {pl.verdict}")
    r = tuner.ab_compare(a.model, pl.to_dict(), p, n_gen=a.n_gen)
    print(f"PocketLLM engine speedup: x{r['speedup']}")


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
    s = sub.add_parser("ab", help="measure stock llama.cpp vs the PocketLLM engine on one model")
    s.add_argument("model")
    s.add_argument("--ctx", type=int, default=4096)
    s.add_argument("--n-gen", type=int, default=16)
    s.set_defaults(fn=cmd_ab)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--llama-port", type=int, default=8081)
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("stripe", help="split a GGUF across several drives for parallel expert reads")
    s.add_argument("model")
    s.add_argument("--drives", nargs="+", required=True, help="one directory per drive")
    s.add_argument("--link-dir", required=True, help="where to put the symlinked split set")
    s.set_defaults(fn=cmd_stripe)
    s = sub.add_parser("seal", help="(provisioning station) hash + sign everything on the drive")
    s.add_argument("--root", default=runtime.ROOT)
    s.add_argument("--key", required=True, help="private key PEM (keep it off the drive)")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_seal)
    s = sub.add_parser("verify", help="check the drive against its signed manifest")
    s.add_argument("--root", default=runtime.ROOT)
    s.add_argument("--full", action="store_true")
    s.add_argument("--pubkey", help="organisation public key (recommended)")
    s.set_defaults(fn=cmd_verify)
    s = sub.add_parser("node", help="lend this machine's memory to a cluster (llama.cpp rpc-server)")
    s.add_argument("--host", required=True, help="address on the private link")
    s.add_argument("--port", type=int, default=50052)
    s.add_argument("--threads", type=int)
    s.add_argument("--no-cache", action="store_true")
    s.add_argument("--i-understand", action="store_true", help="allow binding to all interfaces")
    s.set_defaults(fn=cmd_node)
    s = sub.add_parser("cluster", help="run a model across nodes started with `pocketllm node`")
    s.add_argument("model")
    s.add_argument("--nodes", required=True, help="host:port,host:port,...")
    s.add_argument("--ctx", type=int, default=4096)
    s.add_argument("--run", action="store_true")
    s.add_argument("--llama-port", type=int, default=8081)
    s.set_defaults(fn=cmd_cluster)
    s = sub.add_parser("thermal", help="thermal state and suggested power profile")
    s.set_defaults(fn=cmd_thermal)
    s = sub.add_parser("fleet-plan", help="(provisioning) plan a drive for a fleet of laptops")
    s.add_argument("--profiles", required=True, help="directory of profile JSON files from target laptops")
    s.add_argument("--goal", required=True)
    s.add_argument("--ctx", type=int, default=4096)
    s.add_argument("--no-meta", action="store_true")
    s.set_defaults(fn=cmd_fleet_plan)
    s = sub.add_parser("evalset", help="(provisioning) generate a synthetic evaluation set with Meta Model API")
    s.add_argument("--domain", required=True)
    s.add_argument("-n", type=int, default=20)
    s.add_argument("--out", default="evalset.jsonl")
    s.set_defaults(fn=cmd_evalset)
    s = sub.add_parser("qualify", help="grade candidate local models on an eval set; pick the smallest that passes")
    s.add_argument("--evalset", required=True)
    s.add_argument("--candidates", nargs="+", required=True, help="label=http://127.0.0.1:port (smallest first)")
    s.add_argument("--pass-mean", type=float, default=3.5)
    s.add_argument("--local-judge", help="llama-server URL of a local judge model (offline qualification)")
    s.set_defaults(fn=cmd_qualify)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

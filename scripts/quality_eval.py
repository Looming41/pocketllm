"""Measure what a speed recipe costs in quality, against a reference model.

For each variant (a GGUF file, optionally with environment such as POCKET_MOE_TOPK=4):
  * KL divergence of its next-token distribution from the reference, token by token, and how
    often both pick the same top token (llama-perplexity --kl-divergence). This is the most
    sensitive measure: it sees every token, not just the final answer.
  * perplexity on the same text
  * HellaSwag accuracy (optional; a common-sense multiple-choice task)

usage:
  python scripts/quality_eval.py --bin BIN --ref Q4.gguf --text wiki.test.raw \\
      --variant "top-k 4=Q4.gguf:POCKET_MOE_TOPK=4" --variant "Q2 experts=E2R8.gguf" \\
      [--hellaswag hellaswag_val_full.txt --hellaswag-tasks 100] --out results/quality.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

NUM = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
PATTERNS = {
    "ppl": rf"Mean PPL\(Q\)\s*:\s*{NUM}",
    "ppl_ref": rf"Mean PPL\(base\)\s*:\s*{NUM}",
    "ln_ppl_ratio": rf"Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*{NUM}",
    "kld_mean": rf"Mean\s+KLD:\s*{NUM}",
    "kld_99": rf"99\.0%\s+KLD:\s*{NUM}",
    "kld_max": rf"Maximum KLD:\s*{NUM}",
    "same_top_pct": rf"Same top p:\s*{NUM}",
    "rms_dp_pct": rf"RMS Δp\s*:\s*{NUM}",
}


def run(cmd, env_extra, log):
    env = dict(os.environ, **env_extra)
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = p.stdout + "\n" + p.stderr
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n" + json.dumps(env_extra) + "\n\n" + out)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({p.returncode}), see {log}")
    return out, round(time.time() - t0, 1)


def parse_kld(out):
    r = {}
    for k, pat in PATTERNS.items():
        m = re.search(pat, out)
        if m:
            r[k] = float(m.group(1))
    if "ppl" not in r:  # reference pass prints only the final estimate
        m = re.search(rf"Final estimate: PPL = {NUM}", out)
        if m:
            r["ppl"] = float(m.group(1))
    return r


def parse_hellaswag(out):
    rows = re.findall(r"^\s*(\d+)\s+([0-9.]+)", out, re.M)
    if not rows:
        return None
    n, acc = rows[-1]
    return {"tasks": int(n), "acc_pct": float(acc)}


def parse_variant(spec):
    label, rest = spec.split("=", 1)
    path, *envs = rest.split(":")
    env = dict(e.split("=", 1) for e in envs if e)
    return label, path, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--variant", action="append", default=[], help='"label=model.gguf[:ENV=VAL...]"')
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--hellaswag")
    ap.add_argument("--hellaswag-tasks", type=int, default=100)
    ap.add_argument("--workdir", default="quality_work")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    ppl = os.path.join(a.bin, "llama-perplexity")
    base = os.path.join(a.workdir, "reference.kld")
    common = ["-c", str(a.ctx), "--chunks", str(a.chunks), "-t", str(a.threads), "-f", a.text]
    res = {"reference": os.path.basename(a.ref), "ctx": a.ctx, "chunks": a.chunks,
           "hellaswag_tasks": a.hellaswag_tasks if a.hellaswag else 0, "variants": []}

    def hellaswag(model, env, tag):
        if not a.hellaswag:
            return None
        out, secs = run([ppl, "-m", model, "-f", a.hellaswag, "--hellaswag", "--hellaswag-tasks",
                         str(a.hellaswag_tasks), "-c", "512", "-t", str(a.threads)], env,
                        os.path.join(a.workdir, f"hellaswag_{tag}.log"))
        h = parse_hellaswag(out)
        if h:
            h["secs"] = secs
        return h

    out, secs = run([ppl, "-m", a.ref, *common, "--kl-divergence-base", base], {},
                    os.path.join(a.workdir, "kld_reference.log"))
    ref = {"label": "reference", "model": os.path.basename(a.ref), "env": {}, **parse_kld(out), "secs": secs}
    ref["hellaswag"] = hellaswag(a.ref, {}, "reference")
    res["variants"].append(ref)
    print(json.dumps(ref), flush=True)
    for i, spec in enumerate(a.variant):
        label, model, env = parse_variant(spec)
        r = {"label": label, "model": os.path.basename(model), "env": env}
        try:
            out, secs = run([ppl, "-m", model, *common, "--kl-divergence-base", base, "--kl-divergence"], env,
                            os.path.join(a.workdir, f"kld_{i}.log"))
            r.update(parse_kld(out), secs=secs)
            r["hellaswag"] = hellaswag(model, env, str(i))
        except RuntimeError as e:
            r["error"] = str(e)
        res["variants"].append(r)
        print(json.dumps(r), flush=True)
        with open(a.out, "w") as f:  # keep partial results if a later variant times out
            json.dump(res, f, indent=2)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(report(res))


def report(res):
    lines = [f"reference {res['reference']}, {res['chunks']} x {res['ctx']} tokens"
             + (f", HellaSwag {res['hellaswag_tasks']} tasks" if res["hellaswag_tasks"] else ""), "",
             "| variant | PPL | mean KLD | 99% KLD | same top token | HellaSwag |", "|---|---|---|---|---|---|"]
    for v in res["variants"]:
        if "error" in v:
            lines.append(f"| {v['label']} | error | | | | |")
            continue
        h = v.get("hellaswag") or {}
        f = lambda k, fmt: (fmt % v[k]) if k in v else "–"  # noqa: E731
        lines.append(f"| {v['label']} | {f('ppl', '%.3f')} | {f('kld_mean', '%.4f')} | {f('kld_99', '%.3f')} | "
                     f"{f('same_top_pct', '%.1f%%')} | "
                     + (f"{h['acc_pct']:.1f}% ({h['tasks']})" if h else "–") + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--report":
        print(report(json.load(open(sys.argv[2]))))
    else:
        main()

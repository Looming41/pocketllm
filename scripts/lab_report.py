"""Summarise stream_lab JSON files as a markdown table (mean ± sd over reps).

usage: python scripts/lab_report.py data/stream_lab_*.json
"""
import json
import statistics
import sys
from collections import OrderedDict


def main(paths):
    for path in paths:
        d = json.load(open(path))
        print(f"\n### {path}\nmodel {d['model']} ({d['model_gb']} GB), RAM limit {d['mem_mib']} MiB, "
              f"{d['drive_mbps']} MB/s per drive, {d['n_gen']} generated tokens, {d['threads']} threads\n")
        groups = OrderedDict()
        for r in d["runs"]:
            groups.setdefault((r["config"], r["drives"]), []).append(r)
        drives = sorted({k[1] for k in groups})
        configs = list(OrderedDict.fromkeys(k[0] for k in groups))
        base = {}
        print("| config | " + " | ".join(f"{n} drive{'s' if n > 1 else ''}" for n in drives) + " |")
        print("|---|" + "---|" * len(drives))
        for c in configs:
            cells = []
            for n in drives:
                rs = [r for r in groups.get((c, n), []) if r.get("ok")]
                if not rs:
                    cells.append("–")
                    continue
                v = [r["tg_tok_s"] for r in rs]
                m = statistics.mean(v)
                sd = statistics.stdev(v) if len(v) > 1 else 0.0
                gb = statistics.mean(r.get("drive_read_gb", 0) for r in rs)
                if c == configs[0]:
                    base[n] = m
                x = f" ×{m / base[n]:.1f}" if n in base and c != configs[0] else ""
                cells.append(f"{m:.2f}{f' ± {sd:.2f}' if len(v) > 1 else ''} tok/s{x} ({gb:.1f} GB read, n={len(v)})")
            print(f"| {c} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])

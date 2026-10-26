"""Print a feasibility matrix: catalog models x typical laptop memory tiers.

Memory verdicts come from the planner (estimates from parameter counts, not
measurements). Speed is shown as the bandwidth formula's upper bound at a
stated RAM bandwidth so the assumption is explicit; replace it with the
number measured on the real laptop.

usage: python scripts/feasibility_matrix.py [--ram-gbps 60] [--disk-gbps 1.0] [--ctx 4096]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pocketllm import catalog, planner  # noqa: E402

GiB = 2**30
TIERS = [
    ("MacBook Air 16GB", 16, 11, True),
    ("16GB laptop, no dGPU", 16, 0, False),
    ("32GB laptop, no dGPU", 32, 0, False),
    ("32GB + 8GB GPU", 32, 8, False),
    ("64GB + 16GB GPU", 64, 16, False),
    ("Mac 36GB unified", 36, 27, True),
    ("Mac 64GB unified", 64, 48, True),
    ("Mac 128GB unified", 128, 96, True),
    ("128GB mini-PC / workstation", 128, 0, False),
]
SHORT = {"fits": "OK", "slow": "slow", "impractical": "impract.", "does_not_fit": "no"}
STRAT = {"expert_stream": "stream", "disk_paging": "paging", "moe_hybrid": "moe split", "full_gpu": "gpu",
         "partial_gpu": "partial gpu", "cpu_ram": "ram", "impossible": "-"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ram-gbps", type=float, default=60.0)
    ap.add_argument("--disk-gbps", type=float, default=1.0)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--models", nargs="*", default=["qwen3-30b-a3b", "gpt-oss-20b", "llama-4-scout-17b-16e-instruct",
                                                    "gpt-oss-120b", "llama-3.3-70b-instruct", "llama-3.1-8b-instruct"])
    a = ap.parse_args()
    specs = catalog.specs(a.models)
    print(f"assumptions: RAM bandwidth {a.ram_gbps} GB/s, storage {a.disk_gbps} GB/s, ctx {a.ctx}, "
          f"unified-memory GPUs read at the RAM bandwidth; discrete-GPU speed is not estimated (blank)\n")
    head = "| model | size GB | " + " | ".join(t[0] for t in TIERS) + " |"
    print(head)
    print("|" + "---|" * (len(TIERS) + 2))
    for s in specs:
        cells = []
        for _, ram, gpu, unified in TIERS:
            prof = {"cpu": {"physical_cores": 8}, "ram": {"total_bytes": ram * GiB, "available_bytes": int(ram * 0.85 * GiB)},
                    "gpus": [{"vram_free_bytes": gpu * GiB, "unified_memory": unified}] if gpu else []}
            b = planner.budget_from_profile(prof, disk_gbps=a.disk_gbps, ram_gbps=a.ram_gbps)
            p = planner.plan(s, b, ctx=a.ctx)
            est = f" ≤{p.est_tok_s_upper:g}" if p.est_tok_s_upper else ""
            cells.append(f"{SHORT[p.verdict]} ({STRAT.get(p.strategy, p.strategy)}){est}")
        print(f"| {s.name} | {s.total_bytes / 1e9:.0f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

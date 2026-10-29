"""Expert-streaming lab (Linux, root): emulate "laptop with too little RAM +
N slow USB drives" and measure decode speed of a MoE model under different
PocketLLM streaming settings.

Emulation, not real hardware:
  * each "USB drive" is an ext4 loop device (direct I/O) whose reads are capped
    with the cgroup-v1 blkio throttle (--drive-mbps)
  * the RAM shortage is a cgroup-v1 memory limit (--mem-mib); page cache counts
    against it, so weights that do not fit are re-read from the "drives"
  * loop devices have almost no access latency, so the gains here come from
    bandwidth aggregation and I/O/compute overlap, not latency hiding

usage: sudo python3 scripts/stream_lab.py --model big.gguf --llama-bin build/bin \
            --drives 1,2,3 --drive-mbps 300 --mem-mib 1024 --out data/stream_lab.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pocketllm import stripe  # noqa: E402

CG_MEM = "/sys/fs/cgroup/memory/pocketlab"
CG_BLK = "/sys/fs/cgroup/blkio/pocketlab"
CG_V2 = "/sys/fs/cgroup/pocketlab"
# cgroup v1 (separate memory/blkio hierarchies) or v2 (unified, e.g. GitHub runners)
V2 = not os.path.isdir("/sys/fs/cgroup/memory") and os.path.exists("/sys/fs/cgroup/cgroup.controllers")


def sh(*cmd, check=True):
    return subprocess.run(cmd, check=check, capture_output=True, text=True).stdout.strip()


def make_drive(workdir, i, size_bytes):
    img = os.path.join(workdir, f"usb{i}.img")
    mnt = os.path.join(workdir, f"usb{i}")
    sh("truncate", "-s", str(size_bytes), img)
    dev = sh("losetup", "-f", "--show", "--direct-io=on", img)
    sh("mkfs.ext4", "-q", "-F", dev)
    os.makedirs(mnt, exist_ok=True)
    sh("mount", dev, mnt)
    majmin = open(f"/sys/block/{os.path.basename(dev)}/dev").read().strip()
    return {"img": img, "mnt": mnt, "dev": dev, "majmin": majmin}


def drop_drive(d):
    sh("umount", d["mnt"], check=False)
    sh("losetup", "-d", d["dev"], check=False)
    for p in (d["img"],):
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(d["mnt"], ignore_errors=True)


def setup_cgroups(mem_mib, drives, mbps):
    if V2:
        try:
            open("/sys/fs/cgroup/cgroup.subtree_control", "w").write("+memory +io")
        except OSError:
            pass
        os.makedirs(CG_V2, exist_ok=True)
        open(f"{CG_V2}/memory.max", "w").write(str(mem_mib << 20))
        try:
            open(f"{CG_V2}/memory.swap.max", "w").write("0")
        except OSError:
            pass
        for d in drives:
            open(f"{CG_V2}/io.max", "w").write(f"{d['majmin']} rbps={mbps * 1_000_000}")
        return
    os.makedirs(CG_MEM, exist_ok=True)
    os.makedirs(CG_BLK, exist_ok=True)
    open(f"{CG_MEM}/memory.limit_in_bytes", "w").write(str(mem_mib << 20))
    for d in drives:
        open(f"{CG_BLK}/blkio.throttle.read_bps_device", "w").write(f"{d['majmin']} {mbps * 1_000_000}")


def flush_cache(files=()):
    """Start every run cold: drop the model's pages from the page cache (wherever they are
    charged) and empty the lab cgroup."""
    for f in files:
        fd = os.open(os.path.realpath(f), os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    try:
        open("/proc/sys/vm/drop_caches", "w").write("1")
    except OSError:
        pass
    try:
        if not V2:
            open(f"{CG_MEM}/memory.force_empty", "w").write("0")
    except OSError:
        pass


def blk_read_bytes():
    if V2:
        try:
            total = 0
            for line in open(f"{CG_V2}/io.stat"):
                for kv in line.split()[1:]:
                    if kv.startswith("rbytes="):
                        total += int(kv.split("=")[1])
            return total
        except (OSError, ValueError):
            return 0
    try:
        return sum(int(l.split()[2]) for l in open(f"{CG_BLK}/blkio.throttle.io_service_bytes") if " Read " in l)
    except (OSError, ValueError, IndexError):
        return 0


def bench(bin_dir, model, env_extra, n_gen, threads, timeout):
    env_extra = dict(env_extra)
    extra = env_extra.pop("LAB_ARGS", "")  # extra llama-bench arguments (not an environment variable)
    env = dict(os.environ, **env_extra)
    read0 = blk_read_bytes()
    join = f"echo $$ > {CG_V2}/cgroup.procs; " if V2 else f"echo $$ > {CG_MEM}/cgroup.procs; echo $$ > {CG_BLK}/cgroup.procs; "
    cmd = (join +
           f"exec {bin_dir}/llama-bench -m {model} -p 0 -n {n_gen} -r 1 -t {threads} -lm mmap --repack 0 {extra} -o json")
    t0 = time.time()
    try:
        p = subprocess.run(["sh", "-c", cmd], env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout {timeout}s"}
    res = {"ok": p.returncode == 0, "wall_s": round(time.time() - t0, 1)}
    pocket = [l for l in p.stderr.splitlines() if l.startswith("pocket: tokens=") or l.startswith("pocket: v2")]
    if pocket:
        res["pocket"] = " | ".join(pocket)
    if p.returncode != 0:
        res["error"] = p.stderr.strip().splitlines()[-1:] if p.stderr else "?"
        return res
    rows = json.loads(p.stdout)
    r = rows[0]
    res["tg_tok_s"] = round(r["avg_ts"], 3)
    res["stddev"] = round(r.get("stddev_ts", 0), 3)
    res["drive_read_gb"] = round((blk_read_bytes() - read0) / 1e9, 2)
    return res


CONFIGS = [
    ("llama.cpp mmap (baseline)", {"POCKET_MOE": "0"}),                                                   # 0
    ("baseline, read-around off everywhere", {"POCKET_MOE": "0", "POCKET_MOE_NORA": "2"}),               # 1
    ("expert prefetch", {"POCKET_MOE": "1"}),                                                             # 2
    ("expert prefetch + no read-around on experts", {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1"}),         # 3
    ("expert prefetch + no read-around on experts + 8 I/O threads",
     {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1", "POCKET_MOE_IO_THREADS": "8"}),                          # 4
    ("expert prefetch + next-layer prediction", {"POCKET_MOE": "1", "POCKET_MOE_PREDICT": "1"}),         # 5
    ("v2 cache (no lookahead)", {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1", "POCKET_MOE_CACHE_MIB": "@cache",
                                 "POCKET_MOE_LOOKAHEAD": "0"}),                                          # 6
    ("v2 cache + router lookahead", {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1", "POCKET_MOE_CACHE_MIB": "@cache",
                                     "POCKET_MOE_LOOKAHEAD": "1"}),                                      # 7
    ("v1 + previous-token prefetch", {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1", "POCKET_MOE_PREVTOK": "1"}),  # 8
    ("v1 + top-k 6 (quality trade-off)", {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1", "POCKET_MOE_TOPK": "6"}),  # 9
    ("v1 + top-k 4 (quality trade-off)", {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1", "POCKET_MOE_TOPK": "4"}),  # 10
    ("stock llama.cpp + top-k 4", {"POCKET_MOE": "0", "POCKET_MOE_TOPK": "4"}),                          # 11
    ("v1 + warm start from routing census", {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1", "POCKET_MOE_STATS": "@stats",
                                             "POCKET_MOE_WARM_MIB": "@warm"}),                         # 12
    ("v1, 8K context, f16 KV", {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1", "LAB_ARGS": "-d 8192 -fa 1"}),  # 13
    ("v1, 8K context, q8_0 KV (planner default when streaming)",
     {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1", "LAB_ARGS": "-d 8192 -fa 1 -ctk q8_0 -ctv q8_0"}),       # 14
    ("v2 cache + top-k 4 (quality trade-off)", {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1", "POCKET_MOE_CACHE_MIB": "@cache",
                                                "POCKET_MOE_LOOKAHEAD": "0", "POCKET_MOE_TOPK": "4"}),  # 15
    ("v1 + lazy load + dense weights locked (slow-drive default)",
     {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1", "POCKET_MOE_LAZY_LOAD": "1", "POCKET_MOE_PIN_TRUNK": "1"}),  # 16
    ("v2 cache + lazy load + dense weights locked",
     {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1", "POCKET_MOE_CACHE_MIB": "@cache", "POCKET_MOE_LOOKAHEAD": "0",
      "POCKET_MOE_LAZY_LOAD": "1", "POCKET_MOE_PIN_TRUNK": "1"}),                                         # 17
    ("v1 + lazy load", {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1", "POCKET_MOE_LAZY_LOAD": "1"}),        # 18
    ("slow-drive default + top-k 4 (quality trade-off)",
     {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1", "POCKET_MOE_LAZY_LOAD": "1", "POCKET_MOE_PIN_TRUNK": "1",
      "POCKET_MOE_TOPK": "4"}),                                                                            # 19
    ("slow-drive default + top-k 6 (near-lossless, R17)",
     {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1", "POCKET_MOE_LAZY_LOAD": "1", "POCKET_MOE_PIN_TRUNK": "1",
      "POCKET_MOE_TOPK": "6"}),                                                                            # 20
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama-bin", required=True)
    ap.add_argument("--drives", default="1,2,3")
    ap.add_argument("--drive-mbps", type=int, default=300)
    ap.add_argument("--mem-mib", type=int, default=1024)
    ap.add_argument("--n-gen", type=int, default=24)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--pin-mib", type=int, default=0, help="also test hot-expert pinning with this budget")
    ap.add_argument("--workdir", default="/tmp/pocketlab")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="data/stream_lab.json")
    ap.add_argument("--reps", type=int, default=1, help="cold runs per configuration")
    ap.add_argument("--configs", help="comma-separated indexes into CONFIGS (default: all)")
    ap.add_argument("--cache-mib", type=int, default=0,
                    help="v2 cache size; default: a quarter of --mem-mib")
    ap.add_argument("--warm-mib", type=int, default=0, help="warm-start budget; default: a quarter of --mem-mib")
    a = ap.parse_args()
    if os.geteuid() != 0:
        sys.exit("needs root (loop devices, cgroups)")
    os.makedirs(a.workdir, exist_ok=True)
    configs = list(CONFIGS) if not a.configs else [CONFIGS[int(i)] for i in a.configs.split(",")]
    stats = os.path.join(a.workdir, "routing_stats.txt")
    if a.pin_mib:
        configs.append((f"expert prefetch + pin {a.pin_mib} MiB (calibrated)",
                        {"POCKET_MOE": "1",
                         "POCKET_MOE_PIN_MIB": str(a.pin_mib), "POCKET_MOE_STATS": stats}))
    size = os.path.getsize(a.model)
    from pocketllm import gguf, tuner
    gm = gguf.load(a.model)
    v2_cache = tuner.v2_cache_mib(a.mem_mib << 20, gm.weight_bytes - gm.expert_bytes())
    results = {"model": os.path.basename(a.model), "model_gb": round(size / 1e9, 2), "mem_mib": a.mem_mib,
               "drive_mbps": a.drive_mbps, "n_gen": a.n_gen, "threads": a.threads, "runs": []}
    for n in [int(x) for x in a.drives.split(",")]:
        drives = [make_drive(a.workdir, i, int(size / n * 1.3) + (256 << 20)) for i in range(n)]
        try:
            setup_cgroups(a.mem_mib, drives, a.drive_mbps)
            link_dir = os.path.join(a.workdir, f"model_{n}")
            if n == 1:
                dst = os.path.join(drives[0]["mnt"], os.path.basename(a.model))
                shutil.copyfile(a.model, dst)
                model = dst
            else:
                model = stripe.stripe(a.model, [d["mnt"] for d in drives], link_dir, progress=lambda *_: None)
            files = [os.path.join(os.path.dirname(model), x) for x in os.listdir(os.path.dirname(model))
                     if x.endswith(".gguf")] if n > 1 else [model]
            for label, env in configs:
                if ("POCKET_MOE_PIN_MIB" in env or "POCKET_MOE_STATS" in env) and not os.path.exists(stats):
                    # calibration pass to collect routing statistics (the census a previous session leaves)
                    flush_cache(files)
                    bench(a.llama_bin, model, {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1", "POCKET_MOE_STATS": stats},
                          32, a.threads, a.timeout)
                subst = {"@cache": str(a.cache_mib or v2_cache), "@stats": stats,
                         "@warm": str(a.warm_mib or a.mem_mib // 4)}
                env = {k: subst.get(v, v) for k, v in env.items()}
                for rep_i in range(a.reps):
                    flush_cache(files)
                    env = dict(env, POCKET_MOE_VERBOSE="1")
                    r = bench(a.llama_bin, model, env, a.n_gen, a.threads, a.timeout)
                    r.update(drives=n, config=label, rep=rep_i)
                    results["runs"].append(r)
                    print(json.dumps(r), flush=True)
        finally:
            for d in drives:
                drop_drive(d)
            shutil.rmtree(os.path.join(a.workdir, f"model_{n}"), ignore_errors=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=2)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()

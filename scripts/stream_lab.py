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
        open(f"{CG_MEM}/memory.force_empty", "w").write("0")
    except OSError:
        pass


def blk_read_bytes():
    try:
        return sum(int(l.split()[2]) for l in open(f"{CG_BLK}/blkio.throttle.io_service_bytes") if " Read " in l)
    except (OSError, ValueError, IndexError):
        return 0


def bench(bin_dir, model, env_extra, n_gen, threads, timeout):
    env = dict(os.environ, **env_extra)
    read0 = blk_read_bytes()
    cmd = (f"echo $$ > {CG_MEM}/cgroup.procs; echo $$ > {CG_BLK}/cgroup.procs; "
           f"exec {bin_dir}/llama-bench -m {model} -p 0 -n {n_gen} -r 1 -t {threads} -lm mmap --repack 0 -o json")
    t0 = time.time()
    try:
        p = subprocess.run(["sh", "-c", cmd], env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout {timeout}s"}
    res = {"ok": p.returncode == 0, "wall_s": round(time.time() - t0, 1)}
    pocket = [l for l in p.stderr.splitlines() if l.startswith("pocket:")]
    if pocket:
        res["pocket"] = pocket[-1]
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
                if "POCKET_MOE_PIN_MIB" in env and not os.path.exists(stats):
                    # calibration pass to collect routing statistics
                    flush_cache(files)
                    bench(a.llama_bin, model, {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1", "POCKET_MOE_STATS": stats},
                          8, a.threads, a.timeout)
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

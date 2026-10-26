"""Hardware profiler: OS, CPU, RAM, GPU/VRAM, disk, USB detection, and two
quick micro-benchmarks (RAM bandwidth, storage read speed).

Everything is best-effort and stdlib-only; any field that cannot be detected is
reported as None rather than guessed.
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import random
import shutil
import subprocess
import time


def _run(cmd, timeout=5):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


# ---------------------------------------------------------------- CPU / RAM
def cpu_info() -> dict:
    info = {
        "arch": platform.machine(),
        "logical_cores": os.cpu_count(),
        "physical_cores": None,
        "model": platform.processor() or None,
        "flags": [],
    }
    system = platform.system()
    if system == "Linux":
        try:
            text = open("/proc/cpuinfo").read()
            cores = set()
            phys = None
            for line in text.splitlines():
                k, _, v = line.partition(":")
                k, v = k.strip(), v.strip()
                if k == "physical id":
                    phys = v
                elif k == "core id":
                    cores.add((phys, v))
                elif k in ("flags", "Features") and not info["flags"]:
                    wanted = {"avx", "avx2", "avx512f", "avx512_vnni", "avx512_bf16", "amx_tile", "fma", "f16c", "asimd", "sve", "i8mm", "dotprod", "asimddp"}
                    info["flags"] = sorted(set(v.split()) & wanted)
            info["physical_cores"] = len(cores) or None
            for line in text.splitlines():
                if line.startswith("model name"):
                    info["model"] = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    elif system == "Darwin":
        info["model"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or info["model"]
        p = _run(["sysctl", "-n", "hw.physicalcpu"]).strip()
        info["physical_cores"] = int(p) if p.isdigit() else None
        perf = _run(["sysctl", "-n", "hw.perflevel0.physicalcpu"]).strip()
        info["performance_cores"] = int(perf) if perf.isdigit() else None
    elif system == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores | ConvertTo-Json"])
        try:
            d = json.loads(out)
            d = d[0] if isinstance(d, list) else d
            info["model"] = d.get("Name", info["model"])
            info["physical_cores"] = d.get("NumberOfCores")
        except (ValueError, AttributeError):
            pass
    return info


def ram_info() -> dict:
    total = avail = None
    system = platform.system()
    if system == "Linux":
        try:
            mem = {}
            for line in open("/proc/meminfo"):
                k, v = line.split(":", 1)
                mem[k] = int(v.split()[0]) * 1024
            total, avail = mem.get("MemTotal"), mem.get("MemAvailable")
        except OSError:
            pass
        # containers: honour cgroup limits
        for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            try:
                v = open(p).read().strip()
                if v.isdigit() and total and int(v) < total:
                    total = int(v)
                    avail = min(avail or total, total)
            except OSError:
                pass
    elif system == "Darwin":
        t = _run(["sysctl", "-n", "hw.memsize"]).strip()
        total = int(t) if t.isdigit() else None
        vm = _run(["vm_stat"])
        page = 16384 if platform.machine() == "arm64" else 4096
        free_pages = 0
        for line in vm.splitlines():
            if line.startswith(("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")):
                free_pages += int(line.split(":")[1].strip().rstrip("."))
        avail = free_pages * page if free_pages else None
    elif system == "Windows":
        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = MEMSTAT()
        s.dwLength = ctypes.sizeof(MEMSTAT)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):  # type: ignore[attr-defined]
            total, avail = s.ullTotalPhys, s.ullAvailPhys
    return {"total_bytes": total, "available_bytes": avail}


# ---------------------------------------------------------------- GPU
def gpu_info() -> list:
    gpus = []
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits"])
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({"vendor": "nvidia", "name": parts[0], "backend": "cuda",
                             "vram_total_bytes": int(float(parts[1])) * 2**20,
                             "vram_free_bytes": int(float(parts[2])) * 2**20,
                             "driver": parts[3] if len(parts) > 3 else None, "unified_memory": False})
    if shutil.which("rocm-smi"):
        out = _run(["rocm-smi", "--showmeminfo", "vram", "--json"])
        try:
            for card, d in json.loads(out).items():
                total = int(d.get("VRAM Total Memory (B)", 0))
                used = int(d.get("VRAM Total Used Memory (B)", 0))
                gpus.append({"vendor": "amd", "name": card, "backend": "rocm/vulkan",
                             "vram_total_bytes": total, "vram_free_bytes": total - used, "unified_memory": False})
        except (ValueError, AttributeError):
            pass
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # Apple Silicon: GPU shares system RAM. By default macOS lets Metal wire
        # roughly 2/3-3/4 of RAM; iogpu.wired_limit_mb overrides it when set.
        total = ram_info()["total_bytes"] or 0
        limit = _run(["sysctl", "-n", "iogpu.wired_limit_mb"]).strip()
        usable = int(limit) * 2**20 if limit.isdigit() and int(limit) > 0 else int(total * (0.75 if total > 36 * 2**30 else 0.66))
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        gpus.append({"vendor": "apple", "name": chip or "Apple GPU", "backend": "metal",
                     "vram_total_bytes": usable, "vram_free_bytes": usable, "unified_memory": True,
                     "note": "unified memory: GPU budget is part of system RAM"})
    if not gpus and platform.system() == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json"])
        try:
            d = json.loads(out)
            for g in d if isinstance(d, list) else [d]:
                # AdapterRAM is a uint32 and caps at 4 GB, so report it as a lower bound only
                gpus.append({"vendor": "unknown", "name": g.get("Name"), "backend": "vulkan?",
                             "vram_total_bytes": None, "vram_free_bytes": None, "unified_memory": None,
                             "note": "VRAM not reliably reported on Windows without vendor tools"})
        except (ValueError, AttributeError):
            pass
    return gpus


# ---------------------------------------------------------------- storage / USB
def _mount_point(path: str) -> str:
    path = os.path.realpath(path)
    while not os.path.ismount(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def storage_info(path: str) -> dict:
    path = os.path.abspath(path)
    du = shutil.disk_usage(path)
    info = {"path": path, "mount": _mount_point(path), "free_bytes": du.free, "total_bytes": du.total,
            "is_usb": None, "is_removable": None, "device": None, "fs": None}
    system = platform.system()
    if system == "Linux":
        mount = info["mount"]
        try:
            best = None
            for line in open("/proc/mounts"):
                dev, mnt, fs = line.split()[:3]
                if mnt == mount:
                    best = (dev, fs)
            if best:
                info["device"], info["fs"] = best
                dev = os.path.basename(os.path.realpath(best[0]))
                # find the parent block device (sdb1 -> sdb, nvme0n1p2 -> nvme0n1)
                blk = None
                for cand in os.listdir("/sys/block"):
                    if dev == cand or os.path.exists(f"/sys/block/{cand}/{dev}"):
                        blk = cand
                if blk:
                    info["is_removable"] = open(f"/sys/block/{blk}/removable").read().strip() == "1"
                    info["is_usb"] = "/usb" in os.path.realpath(f"/sys/block/{blk}")
        except OSError:
            pass
    elif system == "Darwin":
        out = _run(["diskutil", "info", info["mount"]])
        for line in out.splitlines():
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if k == "Protocol":
                info["is_usb"] = v.upper() == "USB"
            elif k in ("Removable Media", "Device Location"):
                info["is_removable"] = v in ("Removable", "External") or info["is_removable"]
            elif k == "Device Node":
                info["device"] = v
            elif k == "Type (Bundle)":
                info["fs"] = v
    elif system == "Windows":
        drive = os.path.splitdrive(path)[0].rstrip(":")
        if drive:
            ps = (f"$p=Get-Partition -DriveLetter {drive}; $d=$p|Get-Disk; "
                  "[pscustomobject]@{Bus=$d.BusType;Fs=(Get-Volume -DriveLetter " + drive + ").FileSystem}|ConvertTo-Json")
            try:
                d = json.loads(_run(["powershell", "-NoProfile", "-Command", ps]))
                info["is_usb"] = str(d.get("Bus")) in ("USB", "7")
                info["fs"] = d.get("Fs")
                info["device"] = drive + ":"
            except (ValueError, AttributeError):
                pass
    if info["fs"] and info["fs"].lower() in ("vfat", "fat32", "msdos"):
        info["warning"] = "FAT32 cannot hold files >4 GB; format the drive as exFAT (or use split GGUF parts)"
    return info


def measure_read_speed(file_path: str, mb: int = 256, cold: bool = True) -> dict:
    """Sequential read speed of a file (e.g. a model on the USB drive).

    cold=True tries to evict the range from the OS page cache first so the
    number reflects the device, not RAM. Eviction works on Linux
    (posix_fadvise) and macOS (F_NOCACHE); elsewhere the result may be warm.
    """
    size = os.path.getsize(file_path)
    n = min(size, mb * 2**20)
    start = random.randrange(0, max(1, size - n)) // 4096 * 4096
    evicted = False
    fd = os.open(file_path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        if cold and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, start, n, os.POSIX_FADV_DONTNEED)
            evicted = True
        elif cold and platform.system() == "Darwin":
            import fcntl
            fcntl.fcntl(fd, 48, 1)  # F_NOCACHE
            evicted = True
        os.lseek(fd, start, os.SEEK_SET)
        t0 = time.perf_counter()
        left = n
        while left > 0:
            chunk = os.read(fd, min(left, 8 * 2**20))
            if not chunk:
                break
            left -= len(chunk)
        dt = time.perf_counter() - t0
    finally:
        os.close(fd)
    read = n - left
    return {"bytes": read, "seconds": round(dt, 3), "gb_per_s": round(read / dt / 1e9, 3) if dt else None,
            "cache_evicted": evicted}


def measure_ram_bandwidth(mb: int = 512, rounds: int = 3) -> dict:
    """Single-thread memcpy bandwidth (copy counts read + write).

    This is a floor, not the platform peak: llama.cpp uses many threads and
    usually reaches noticeably more. It is still useful as a relative signal
    and to catch obviously slow memory.
    """
    n = mb * 2**20
    src = bytearray(n)
    dst = bytearray(n)
    best = None
    for _ in range(rounds):
        t0 = time.perf_counter()
        dst[:] = src
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    del src, dst
    return {"gb_per_s_1thread": round(2 * n / best / 1e9, 2), "method": "bytearray memcpy, 1 thread"}


def profile(model_dir: str | None = None, bench: bool = True) -> dict:
    p = {
        "os": {"system": platform.system(), "release": platform.release(), "version": platform.version(),
               "python": platform.python_version()},
        "cpu": cpu_info(),
        "ram": ram_info(),
        "gpus": gpu_info(),
    }
    if model_dir and os.path.isdir(model_dir):
        p["storage"] = storage_info(model_dir)
    if bench:
        try:
            p["ram"]["bandwidth"] = measure_ram_bandwidth()
        except MemoryError:
            p["ram"]["bandwidth"] = None
    return p


def human(nbytes) -> str:
    if nbytes is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1000 or unit == "TB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes /= 1000
    return str(nbytes)

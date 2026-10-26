"""Thermal / power profiles for fanless laptops (e.g. MacBook Air).

Heat per generated token is roughly proportional to the work per token, so
the biggest lever is model choice (a MoE with ~3B active parameters does less
work per token than a 7B dense model). Within a model the runtime can trade
speed for heat by using fewer threads: generation is memory-bound, so halving
threads usually costs much less than half the speed. The tuner measures this.
"""
from __future__ import annotations

import glob
import platform
import subprocess

PROFILES = {
    # fraction of performance cores used for generation
    "cool": 0.5,
    "balanced": 0.75,
    "max": 1.0,
}


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def thermal_state() -> dict:
    """Best-effort, no root needed. level: nominal | fair | serious | critical | unknown."""
    system = platform.system()
    st = {"level": "unknown", "detail": None, "on_battery": None, "fanless_hint": None}
    if system == "Darwin":
        out = _run(["pmset", "-g", "therm"])
        st["detail"] = " ".join(out.split())[:200] or None
        low = out.lower()
        if "cpu_speed_limit" in low:
            try:
                limit = int(low.split("cpu_speed_limit")[1].split("=")[1].split()[0])
                st["level"] = "nominal" if limit >= 100 else "serious" if limit < 70 else "fair"
            except (IndexError, ValueError):
                pass
        elif "no thermal warning level" in low or "thermal warning level" not in low:
            st["level"] = "nominal" if out else "unknown"
        else:
            st["level"] = "fair"
        batt = _run(["pmset", "-g", "batt"])
        st["on_battery"] = "battery power" in batt.lower() if batt else None
        model = _run(["sysctl", "-n", "hw.model"]).strip()
        st["fanless_hint"] = model.startswith("MacBookAir") or None
        st["model"] = model or None
    elif system == "Linux":
        temps = []
        for z in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
            try:
                temps.append(int(open(z).read().strip()) / 1000)
            except (OSError, ValueError):
                pass
        if temps:
            t = max(temps)
            st["detail"] = f"max zone {t:.0f} C"
            st["level"] = "nominal" if t < 75 else "fair" if t < 85 else "serious" if t < 95 else "critical"
        for b in glob.glob("/sys/class/power_supply/*/status"):
            try:
                if "Discharging" in open(b).read():
                    st["on_battery"] = True
            except OSError:
                pass
    return st


def threads_for(profile: str, cpu: dict) -> int:
    perf = cpu.get("performance_cores") or cpu.get("physical_cores") or cpu.get("logical_cores") or 4
    return max(1, round(perf * PROFILES.get(profile, 0.75)))


def auto_profile(state: dict) -> str:
    if state.get("level") in ("serious", "critical"):
        return "cool"
    if state.get("fanless_hint") or state.get("on_battery"):
        return "balanced"
    return "max"

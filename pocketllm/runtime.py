"""Locating the drive layout and binaries, and supervising llama-server.

Drive layout (the USB stick root):

    PocketLLM/
      launch.sh | launch.command | launch.bat
      pocketllm/            this Python package
      bin/<os>-<arch>/      llama.cpp builds (llama-server, llama-bench, ...)
      models/               *.gguf
      data/                 profiles, tuning results, logs (written at runtime)
"""
from __future__ import annotations

import json
import os
import platform
import secrets
import shutil
import socket
import subprocess
import time
import urllib.request

ROOT = os.environ.get("POCKETLLM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def platform_tag() -> str:
    system = {"Darwin": "macos", "Windows": "windows", "Linux": "linux"}.get(platform.system(), platform.system().lower())
    arch = platform.machine().lower()
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64"}.get(arch, arch)
    return f"{system}-{arch}"


def models_dir() -> str:
    return os.environ.get("POCKETLLM_MODELS") or os.path.join(ROOT, "models")


def data_dir() -> str:
    d = os.environ.get("POCKETLLM_DATA") or os.path.join(ROOT, "data")
    os.makedirs(d, exist_ok=True)
    return d


def binary(name: str) -> str:
    exe = name + (".exe" if platform.system() == "Windows" else "")
    candidates = []
    if os.environ.get("POCKETLLM_BIN"):
        candidates.append(os.path.join(os.environ["POCKETLLM_BIN"], exe))
    candidates.append(os.path.join(ROOT, "bin", platform_tag(), exe))
    for c in candidates:
        if os.path.isfile(c):
            return c
    found = shutil.which(exe)
    if found:
        return found
    raise FileNotFoundError(
        f"{exe} not found. Put llama.cpp binaries in {os.path.join(ROOT, 'bin', platform_tag())} "
        f"or set POCKETLLM_BIN.")


class LlamaServer:
    """Starts llama-server with a plan's arguments and falls back to safer
    settings if it fails to come up (e.g. GPU out of memory)."""

    def __init__(self, port=8081, log_path=None):
        self.port = port
        self.proc = None
        self.args = []
        self.log_path = log_path or os.path.join(data_dir(), "llama-server.log")
        self.history = []
        self.started_at = None
        self.env = {}
        # random per-session key: other web pages in the browser cannot use the local model
        self.api_key = secrets.token_urlsafe(24)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=1) as r:
                return r.status == 200
        except OSError:
            return False

    def start(self, args: list, timeout=600, env: dict | None = None) -> bool:
        self.stop()
        # a stale llama-server (e.g. from a crashed session) may still hold the port and would
        # answer our health checks; always start on a port nobody is listening on
        if not _port_free(self.port):
            self.port = _free_port()
        cmd = [binary("llama-server"), *args, "--host", "127.0.0.1", "--port", str(self.port), "--metrics"]
        log = open(self.log_path, "a")
        log.write(f"\n=== {time.strftime('%H:%M:%S')} {' '.join(cmd)}\n")
        log.flush()
        self.env = dict(env or {})
        # the key goes through the environment, not argv, so other local users cannot read it from `ps`
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, LLAMA_API_KEY=self.api_key, **self.env))
        self.args = args
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                self.history.append({"args": args, "ok": False, "error": self._tail()})
                return False
            if self.healthy() and self.proc.poll() is None:
                self.started_at = time.time()
                self.history.append({"args": args, "ok": True, "load_s": round(time.time() - t0, 1)})
                return True
            time.sleep(0.5)
        self.stop()
        self.history.append({"args": args, "ok": False, "error": f"not healthy after {timeout}s"})
        return False

    def start_with_fallback(self, args: list, env: dict | None = None) -> bool:
        """Try the planned args, then progressively safer variants."""
        attempts = [args]
        a = _set(args, "-c", str(min(int(_get(args, "-c", "4096")), 2048)))
        attempts.append(a)
        attempts.append(_set(_drop(a, "-ncmoe"), "-ngl", "0"))          # CPU only
        attempts.append(_set(_set(_drop(a, "-ncmoe"), "-ngl", "0"), "-lm", "mmap"))  # CPU + mmap (can page)
        seen = []
        for att in attempts:
            if att in seen:
                continue
            seen.append(att)
            if self.start(att, env=env):
                return True
        return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def rss_bytes(self):
        if not self.proc or self.proc.poll() is not None:
            return None
        from .tuner import _rss_bytes
        return _rss_bytes(self.proc.pid)

    def _tail(self, n=8):
        try:
            with open(self.log_path, errors="replace") as f:
                return "".join(f.readlines()[-n:])
        except OSError:
            return ""

    def status(self) -> dict:
        return {"running": bool(self.proc and self.proc.poll() is None), "healthy": self.healthy(),
                "args": self.args, "env": self.env, "rss_gb": round((self.rss_bytes() or 0) / 1e9, 2),
                "history": self.history[-5:]}


def _port_free(port) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(args, flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default


def _set(args, flag, value):
    args = list(args)
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args += [flag, value]
    return args


def _drop(args, flag):
    args = list(args)
    if flag in args:
        i = args.index(flag)
        del args[i:i + 2]
    return args


def save_json(name, obj):
    path = os.path.join(data_dir(), name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path

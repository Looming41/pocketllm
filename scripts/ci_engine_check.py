"""CI check of the PocketLLM engine on the current OS (Linux, macOS, Windows).

1. llama-bench with POCKET_MOE=1 on a MoE GGUF: the engine must report activity
   (`pocket: ... advise_calls=N` with N > 0) and the run must succeed.
2. llama-server greedy output must be identical with the engine off and on.

usage: python scripts/ci_engine_check.py <bin dir> <moe.gguf>
"""
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

EXE = ".exe" if os.name == "nt" else ""


def annotate(msg):
    """Surface details as a GitHub Actions annotation (readable through the API without the log)."""
    if os.environ.get("GITHUB_ACTIONS"):
        print("::error::" + msg.replace("%", "%25").replace("\r", "").replace("\n", "%0A")[:4000])


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def greedy(bin_dir, model, env_extra):
    port = free_port()
    env = dict(os.environ, **env_extra)
    proc = subprocess.Popen([os.path.join(bin_dir, "llama-server" + EXE), "-m", model, "-c", "512", "-t", "2",
                             "-ngl", "0", "-lm", "mmap", "--no-repack", "--host", "127.0.0.1", "--port", str(port)],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(240):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                    if r.status == 200:
                        break
            except OSError:
                time.sleep(0.5)
        body = json.dumps({"prompt": "The quick brown fox", "n_predict": 24, "temperature": 0}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)["content"]
    finally:
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()


def main(bin_dir, model):
    ok = True
    env = dict(os.environ, POCKET_MOE="1", POCKET_MOE_NORA="1", POCKET_MOE_VERBOSE="1", POCKET_MOE_PIN_MIB="32")
    p = subprocess.run([os.path.join(bin_dir, "llama-bench" + EXE), "-m", model, "-p", "0", "-n", "16", "-r", "1",
                        "-t", "2", "-ngl", "0", "-lm", "mmap", "--repack", "0", "-o", "json"],
                       env=env, capture_output=True, text=True, timeout=900)
    line = next((l for l in p.stderr.splitlines() if l.startswith("pocket:")), "")
    m = re.search(r"advise_calls=(\d+)", line)
    print(f"llama-bench exit={p.returncode}  {line or '(no pocket summary)'}")
    if p.returncode != 0 or not m or int(m.group(1)) == 0:
        print(p.stderr[-2000:])
        annotate(f"llama-bench exit={p.returncode} pocket={line!r}\n" + p.stderr[-1500:])
        ok = False
    else:
        print(f"tg {json.loads(p.stdout)[0]['avg_ts']:.1f} tok/s with the engine on")

    off = greedy(bin_dir, model, {"POCKET_MOE": "0"})
    on = greedy(bin_dir, model, {"POCKET_MOE": "1", "POCKET_MOE_NORA": "1"})
    same = off == on
    print(f"greedy output identical with engine off/on: {same}")
    if not same or not off:
        annotate(f"greedy mismatch or empty: off={off[:200]!r} on={on[:200]!r}")
    ok = ok and same and len(off) > 0
    print("ENGINE CHECK", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

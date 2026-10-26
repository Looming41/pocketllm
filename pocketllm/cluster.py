"""Multi-device mode: pool the memory of several machines ("compute sticks",
mini-PCs, other laptops) over a private link, using llama.cpp's RPC backend.

Every machine, the local one included, runs ``pocketllm node``: an
``rpc-server`` plus a tiny status endpoint (port + 1) that reports its memory.
``pocketllm cluster`` then asks each node for its free memory, splits the
model's layers in proportion (``-ts``), and starts llama-server with
``--rpc n1,n2,...``. Activations, not weights, cross the link per token, so
the link speed mostly affects latency.

Security: llama.cpp's RPC protocol has no authentication or encryption. Use it
only on a dedicated cable / isolated network (USB-C networking, a private switch),
never on a shared or open network. ``pocketllm node`` binds to the address you
give it and refuses 0.0.0.0 unless --i-understand is passed.
"""
from __future__ import annotations

import json
import subprocess
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import profiler, runtime


def serve_node(host: str, port: int, threads: int | None = None, cache: bool = True, allow_any=False):
    if host in ("0.0.0.0", "::") and not allow_any:
        raise SystemExit("refusing to expose the unauthenticated RPC server on all interfaces; "
                         "bind to the private link address or pass --i-understand")
    p = profiler.profile(bench=False)
    status = {"host": host, "rpc_port": port, "cpu": p["cpu"].get("model"),
              "cores": p["cpu"].get("physical_cores") or p["cpu"].get("logical_cores"),
              "ram_total": p["ram"]["total_bytes"], "ram_free": p["ram"]["available_bytes"],
              "gpus": [{"name": g.get("name"), "vram_free": g.get("vram_free_bytes")} for g in p["gpus"]]}
    cmd = [runtime.binary("ggml-rpc-server"), "-H", host, "-p", str(port)]
    if threads:
        cmd += ["-t", str(threads)]
    if cache:
        cmd += ["-c"]  # node keeps a local cache of the tensors it was sent
    proc = subprocess.Popen(cmd)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            s = dict(status, ram_free=profiler.ram_info()["available_bytes"], rpc_alive=proc.poll() is None)
            body = json.dumps(s).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer((host, port + 1), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"node: rpc-server on {host}:{port}, status on http://{host}:{port + 1}/")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
    finally:
        srv.shutdown()


def node_status(addr: str, timeout=3) -> dict:
    host, port = addr.rsplit(":", 1)
    try:
        with urllib.request.urlopen(f"http://{host}:{int(port) + 1}/", timeout=timeout) as r:
            return json.load(r) | {"addr": addr, "reachable": True}
    except (OSError, ValueError) as e:
        return {"addr": addr, "reachable": False, "error": str(e)}


def node_budget(st: dict, reserve_frac=0.15) -> int:
    """Memory a node can lend: its best GPU's free VRAM, else free RAM minus a reserve."""
    vram = max((g.get("vram_free") or 0 for g in st.get("gpus", [])), default=0)
    ram = int((st.get("ram_free") or 0) * (1 - reserve_frac))
    return max(vram, ram)


def split(statuses: list, model_bytes: int) -> dict:
    """Tensor-split proportions and a feasibility verdict."""
    live = [s for s in statuses if s.get("reachable")]
    budgets = [node_budget(s) for s in live]
    total = sum(budgets)
    plan = {"nodes": [s["addr"] for s in live], "budgets_gb": [round(b / 1e9, 2) for b in budgets],
            "total_gb": round(total / 1e9, 2), "model_gb": round(model_bytes / 1e9, 2),
            "unreachable": [s["addr"] for s in statuses if not s.get("reachable")]}
    if not live or total <= 0:
        plan.update(fits=False, ts=[])
        return plan
    plan["ts"] = [round(b / total, 3) for b in budgets]
    plan["fits"] = model_bytes * 1.1 <= total
    return plan


def llama_args(model_path: str, plan: dict, ctx=4096) -> list:
    return ["-m", model_path, "-c", str(ctx), "--rpc", ",".join(plan["nodes"]), "-ngl", "999",
            "-ts", ",".join(str(x) for x in plan["ts"])]

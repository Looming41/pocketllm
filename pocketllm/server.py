"""Local web UI + JSON API. Binds to 127.0.0.1 only.

Endpoints
  GET  /                 UI
  GET  /api/state        hardware summary, local models with plans, catalog recommendations
  GET  /api/status       runtime status: llama-server, RSS, network, drive attached
  POST /api/assist       {goal, use_meta} -> constraints (Meta API or offline rules) + ranked plans
  POST /api/load         {path, ctx} -> start llama-server with the planned args (with fallback)
  POST /api/unload
  POST /api/tune         {path} -> start a background llama-bench sweep
  GET  /api/tune         sweep progress / result
  POST /api/chat         OpenAI-style body; streamed through to llama-server
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import assistant, core, gguf, planner, runtime, security, thermal, tuner

UI = os.path.join(os.path.dirname(__file__), "ui", "index.html")


class State:
    def __init__(self, llama_port):
        self.lock = threading.Lock()
        self.profile = None
        self.llama = runtime.LlamaServer(port=llama_port)
        self.loaded = None
        self.loading = False
        self.tune = {"running": False, "log": [], "result": None}
        self.ab = {"running": False, "log": [], "result": None}
        self.io_last = None
        self.net = {"online": None, "checked": 0}
        self.policy = security.load_policy(runtime.ROOT)
        self.audit = security.AuditLog(os.path.join(runtime.data_dir(), "audit.jsonl"), self.policy)
        self.integrity = None
        if self.policy["verify_manifest"] != "off" and os.path.exists(os.path.join(runtime.ROOT, security.MANIFEST)):
            self.integrity = security.verify(runtime.ROOT, self.policy["verify_manifest"])
        self.audit.write("start", airgap=self.policy["airgap"],
                         integrity=None if self.integrity is None else self.integrity["ok"])

    def get_profile(self, refresh=False):
        with self.lock:
            if self.profile is None or refresh:
                self.profile = core.full_profile(read_test=True)
                runtime.save_json("profile.json", self.profile)
            return self.profile

    def online(self):
        if time.time() - self.net["checked"] > 10:
            if self.policy["airgap"]:
                # on a closed network even a probe packet is a violation: look at routes only
                self.net = {"online": security.network_routes()["online_possible"], "checked": time.time()}
            else:
                self.net = {"online": core.network_online(), "checked": time.time()}
        return self.net["online"]


def plan_rows(ranked):
    rows = []
    for pl, spec in ranked:
        d = pl.to_dict()
        d.update(source=spec.source, path=spec.path, size_gb=round(spec.total_bytes / 1e9, 2),
                 active_gb=round(spec.active_bytes / 1e9, 2), moe=spec.is_moe)
        rows.append(d)
    return rows


def make_handler(st: State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = open(UI, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/api/state"):
                p = st.get_profile(refresh="refresh" in self.path)
                ranked = core.recommendations(p)
                self._json({"hardware": core.hardware_summary(p), "plans": plan_rows(ranked),
                            "models_dir": runtime.models_dir(), "incomplete": core.INCOMPLETE})
            elif self.path == "/api/status":
                drive_ok = os.path.isdir(runtime.models_dir())
                if st.loaded and not all(os.path.exists(x) for x in st.loaded["files"]):
                    drive_ok = False
                th = thermal.thermal_state()
                io_rate = None
                if st.llama.proc and st.llama.proc.poll() is None:
                    b, now = tuner.read_bytes(st.llama.proc.pid), time.time()
                    if b is not None and st.io_last and st.io_last[0] == st.llama.proc.pid:
                        io_rate = round((b - st.io_last[1]) / max(now - st.io_last[2], 1e-3) / 1e6, 1)
                    if b is not None:
                        st.io_last = (st.llama.proc.pid, b, now)
                self._json({"llama": st.llama.status(), "loaded": st.loaded, "loading": st.loading,
                            "online": st.online(), "drive_attached": drive_ok, "airgap": st.policy["airgap"],
                            "integrity": None if st.integrity is None else
                            {"ok": st.integrity["ok"], "signature": st.integrity.get("signature"),
                             "problems": (st.integrity.get("modified", []) + st.integrity.get("missing", [])
                                          + st.integrity.get("unexpected", []))[:5],
                             "error": st.integrity.get("error")},
                            "thermal": {"level": th["level"], "suggested": thermal.auto_profile(th)},
                            "drive_read_mbps": io_rate})
            elif self.path == "/api/tune":
                self._json(st.tune)
            elif self.path == "/api/ab":
                self._json(st.ab)
            else:
                self.send_error(404)

        def _same_origin(self):
            # browsers attach Origin to cross-site requests; only our own page may drive the API
            origin = self.headers.get("Origin")
            if not origin:
                return True
            host = self.headers.get("Host", "")
            return origin in (f"http://{host}", f"https://{host}")

        def do_POST(self):
            if not self._same_origin():
                return self._json({"error": "cross-origin request refused"}, 403)
            if self.path == "/api/assist":
                b = self._body()
                p = st.get_profile()
                hw = core.hardware_summary(p)
                pre = core.recommendations(p)
                cands = [{"model": r["model"], "verdict": r["verdict"], "size_gb": r["size_gb"],
                          "est_tok_s_upper": r["est_tok_s_upper"]} for r in plan_rows(pre)]
                use_meta = b.get("use_meta", True) and not st.policy["airgap"] and st.online()
                c = assistant.interpret(b.get("goal", ""), hw, cands, use_meta=use_meta)
                if st.policy["airgap"]:
                    c["explanation"] += " (air-gap policy: cloud assistant disabled)"
                st.audit.write("assist", backend=c.get("backend"))
                ranked = assistant.choose(c, core.recommendations(p, ctx=c["min_ctx"]))
                self._json({"constraints": c, "plans": plan_rows(ranked)[:6]})
            elif self.path == "/api/load":
                b = self._body()
                path = b["path"]
                if st.policy["require_offline"] and st.online():
                    st.audit.write("load_refused", reason="network route present")
                    return self._json({"error": "policy requires an offline computer: disconnect network first"}, 403)
                if st.integrity is not None and not st.integrity["ok"]:
                    return self._json({"error": "drive failed its integrity check; refusing to run"}, 403)
                from . import mlx
                if os.path.isdir(path) and mlx.is_mlx_dir(path):
                    spec = mlx.load_spec(path)
                    files = [path]
                elif os.path.isfile(path):
                    m = gguf.load(path)
                    spec = planner.ModelSpec.from_gguf(m, path)
                    files = m.paths
                else:
                    return self._json({"error": "model file not found on drive; download it first"}, 400)
                prof = st.get_profile()
                topk = None
                if b.get("experts") == "fewer" and spec.n_expert_used > 1:
                    # top-k reduction to 3/4 of the routed experts: on Qwen3-30B-A3B, 6 of 8 measured
                    # mean KLD 0.060 and HellaSwag 75% vs 76% (4 of 8: 0.299 and 71%), RESULTS R17
                    topk = max(1, min(spec.n_expert_used - 1, -(-spec.n_expert_used * 3 // 4)))
                pl = planner.plan(spec, core.budget(prof), ctx=int(b.get("ctx", 4096)), topk=topk)
                if pl.verdict == "does_not_fit":
                    return self._json({"error": "does not fit", "plan": pl.to_dict()}, 400)
                power = b.get("power") or thermal.auto_profile(thermal.thermal_state())
                if "-t" in pl.args:
                    pl.args[pl.args.index("-t") + 1] = str(thermal.threads_for(power, prof["cpu"]))
                st.audit.write("load", model=spec.name, strategy=pl.strategy, power=power, topk=pl.topk)

                def load():
                    st.loading = True
                    t0 = time.time()
                    ok = st.llama.start_with_fallback(pl.args, env=pl.env, backend=pl.backend)
                    st.loading = False
                    st.loaded = {"name": spec.name, "path": path, "files": files, "plan": pl.to_dict(),
                                 "ok": ok, "load_s": round(time.time() - t0, 1), "args": st.llama.args,
                                 "env": st.llama.env, "power": power, "topk": pl.topk}
                threading.Thread(target=load, daemon=True).start()
                self._json({"started": True, "plan": pl.to_dict()})
            elif self.path == "/api/unload":
                st.llama.stop()
                st.loaded = None
                self._json({"ok": True})
            elif self.path == "/api/tune":
                b = self._body()
                if st.tune["running"]:
                    return self._json({"error": "already running"}, 409)
                path = b["path"]
                if st.loaded:  # free memory for the benchmark
                    st.llama.stop()
                    st.loaded = None
                p = st.get_profile()
                spec = planner.ModelSpec.from_gguf(gguf.load(path), path)
                pl = planner.plan(spec, core.budget(p))
                st.tune = {"running": True, "log": [f"plan: {pl.strategy} {' '.join(pl.args)}"], "result": None}

                def run():
                    out = os.path.join(runtime.data_dir(), "tuning", os.path.basename(path) + ".json")
                    res = tuner.tune(path, pl.to_dict(), p, log=st.tune["log"].append, results_path=out)
                    st.tune.update(running=False, result=res)
                threading.Thread(target=run, daemon=True).start()
                self._json({"started": True})
            elif self.path == "/api/ab":
                b = self._body()
                if st.ab["running"] or st.tune["running"]:
                    return self._json({"error": "a benchmark is already running"}, 409)
                path = b["path"]
                if st.loaded:  # free memory: the comparison must not compete with a loaded model
                    st.llama.stop()
                    st.loaded = None
                p = st.get_profile()
                spec = planner.ModelSpec.from_gguf(gguf.load(path), path)
                pl = planner.plan(spec, core.budget(p))
                st.ab = {"running": True, "log": [f"A/B on {spec.name} ({pl.strategy})"], "result": None}

                def run_ab():
                    res = tuner.ab_compare(path, pl.to_dict(), p, log=st.ab["log"].append)
                    st.ab.update(running=False, result=res)
                    st.audit.write("ab_compare", model=spec.name, speedup=res.get("speedup"))
                threading.Thread(target=run_ab, daemon=True).start()
                self._json({"started": True})
            elif self.path == "/api/chat":
                self._proxy_chat()
            else:
                self.send_error(404)

        def _proxy_chat(self):
            body = self._body()
            body["stream"] = True
            body.setdefault("stream_options", {"include_usage": True})
            msgs = body.get("messages") or []
            st.audit.write("chat", model=(st.loaded or {}).get("name"), turns=len(msgs),
                           prompt=str(msgs[-1].get("content", "")) if msgs else "")
            req = urllib.request.Request(st.llama.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {st.llama.api_key}"})
            try:
                upstream = urllib.request.urlopen(req, timeout=600)
            except OSError as e:
                return self._json({"error": f"local model not running: {e}"}, 503)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for line in upstream:
                    self.wfile.write(line)
                    if line.strip() == b"":
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                upstream.close()

    return H


def main(host="127.0.0.1", port=8080, llama_port=8081):
    st = State(llama_port)
    print(f"PocketLLM UI: http://{host}:{port}   (models: {runtime.models_dir()})")
    srv = ThreadingHTTPServer((host, port), make_handler(st))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        st.llama.stop()

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

from . import assistant, core, gguf, planner, runtime, tuner

UI = os.path.join(os.path.dirname(__file__), "ui", "index.html")


class State:
    def __init__(self, llama_port):
        self.lock = threading.Lock()
        self.profile = None
        self.llama = runtime.LlamaServer(port=llama_port)
        self.loaded = None
        self.loading = False
        self.tune = {"running": False, "log": [], "result": None}
        self.net = {"online": None, "checked": 0}

    def get_profile(self, refresh=False):
        with self.lock:
            if self.profile is None or refresh:
                self.profile = core.full_profile(read_test=True)
                runtime.save_json("profile.json", self.profile)
            return self.profile

    def online(self):
        if time.time() - self.net["checked"] > 10:
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
                            "models_dir": runtime.models_dir()})
            elif self.path == "/api/status":
                drive_ok = os.path.isdir(runtime.models_dir())
                if st.loaded and not all(os.path.exists(x) for x in st.loaded["files"]):
                    drive_ok = False
                self._json({"llama": st.llama.status(), "loaded": st.loaded, "loading": st.loading,
                            "online": st.online(), "drive_attached": drive_ok})
            elif self.path == "/api/tune":
                self._json(st.tune)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/api/assist":
                b = self._body()
                p = st.get_profile()
                hw = core.hardware_summary(p)
                pre = core.recommendations(p)
                cands = [{"model": r["model"], "verdict": r["verdict"], "size_gb": r["size_gb"],
                          "est_tok_s_upper": r["est_tok_s_upper"]} for r in plan_rows(pre)]
                c = assistant.interpret(b.get("goal", ""), hw, cands, use_meta=b.get("use_meta", True) and st.online())
                ranked = assistant.choose(c, core.recommendations(p, ctx=c["min_ctx"]))
                self._json({"constraints": c, "plans": plan_rows(ranked)[:6]})
            elif self.path == "/api/load":
                b = self._body()
                path = b["path"]
                if not os.path.isfile(path):
                    return self._json({"error": "model file not found on drive; download it first"}, 400)
                m = gguf.load(path)
                spec = planner.ModelSpec.from_gguf(m, path)
                pl = planner.plan(spec, core.budget(st.get_profile()), ctx=int(b.get("ctx", 4096)))
                if pl.verdict == "does_not_fit":
                    return self._json({"error": "does not fit", "plan": pl.to_dict()}, 400)

                def load():
                    st.loading = True
                    t0 = time.time()
                    ok = st.llama.start_with_fallback(pl.args)
                    st.loading = False
                    st.loaded = {"name": spec.name, "path": path, "files": m.paths, "plan": pl.to_dict(),
                                 "ok": ok, "load_s": round(time.time() - t0, 1), "args": st.llama.args}
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
            elif self.path == "/api/chat":
                self._proxy_chat()
            else:
                self.send_error(404)

        def _proxy_chat(self):
            body = self._body()
            body["stream"] = True
            body.setdefault("stream_options", {"include_usage": True})
            req = urllib.request.Request(st.llama.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
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

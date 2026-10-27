"""Tests for striping, security, cluster split, provisioning and thermal profiles."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pocketllm import assistant, cluster, gguf, provision, security, stripe, thermal  # noqa: E402

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
TEST_MOE = os.path.join(MODELS, "test-moe-Q4_K_M.gguf")


@unittest.skipUnless(os.path.exists(TEST_MOE), "test models not generated")
class TestStripe(unittest.TestCase):
    def test_roundtrip_same_tensors_and_bytes(self):
        d = tempfile.mkdtemp()
        try:
            drives = [os.path.join(d, f"usb{i}") for i in range(3)]
            first = stripe.stripe(TEST_MOE, drives, os.path.join(d, "set"), progress=lambda *_: None)
            a, b = gguf.load(TEST_MOE), gguf.load(first)
            self.assertEqual(len(b.paths), 3)
            self.assertEqual(sorted(t.name for t in a.tensors), sorted(t.name for t in b.tensors))
            self.assertEqual(a.n_params, b.n_params)
            self.assertEqual(a.metadata["general.architecture"], b.metadata["general.architecture"])
            # expert tensors are spread over all drives, small tensors stay on the first
            per_file = [gguf._read_one(p)[1] for p in b.paths]
            self.assertTrue(all(any(t.is_expert for t in ts) for ts in per_file))
            self.assertFalse(any(not t.is_expert for t in per_file[1]))
            # tensor bytes are identical
            with open(TEST_MOE, "rb") as fa:
                src = {t.name: t for t in a.tensors}
                meta, tensors = gguf._read_one(b.paths[1])
                t = tensors[0]
                start_a = self._data_start(TEST_MOE) + src[t.name].offset
                fa.seek(start_a)
                want = fa.read(min(t.nbytes, 4096))
            with open(b.paths[1], "rb") as fb:
                fb.seek(self._data_start(b.paths[1]) + t.offset)
                self.assertEqual(fb.read(len(want)), want)
        finally:
            shutil.rmtree(d)

    @staticmethod
    def _data_start(path):
        return stripe._scan(path)[4] if "-of-" not in path else _split_data_start(path)

    def test_assign_round_robin(self):
        ts = [{"name": n} for n in ("token_embd.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                                    "blk.0.ffn_down_exps.weight", "blk.0.attn_q.weight", "blk.1.ffn_gate_exps.weight")]
        self.assertEqual(stripe.assign(ts, 3), [0, 0, 1, 2, 0, 0])
        self.assertEqual(stripe.assign(ts, 1), [0] * 6)


def _split_data_start(path):
    import struct
    with open(path, "rb") as f:
        f.read(4)
        r = gguf._Reader(f)
        r.read("<I")
        n_t, n_kv = r.read("<Q"), r.read("<Q")
        for _ in range(n_kv):
            r.string()
            r.value(r.read("<I"), keep_arrays=False)
        for _ in range(n_t):
            r.string()
            nd = r.read("<I")
            for _ in range(nd):
                r.read("<Q")
            r.read("<I")
            r.read("<Q")
        return (f.tell() + 31) // 32 * 32


@unittest.skipUnless(shutil.which("openssl"), "openssl missing")
class TestSecurity(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.root = os.path.join(self.d, "drive")
        os.makedirs(os.path.join(self.root, "models"))
        with open(os.path.join(self.root, "models", "m.gguf"), "wb") as f:
            f.write(os.urandom(20 << 20))
        with open(os.path.join(self.root, "app.py"), "w") as f:
            f.write("print('hi')\n")
        self.key = os.path.join(self.d, "k.pem")
        subprocess.run(["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048", "-out", self.key],
                       check=True, capture_output=True)
        self.pub = os.path.join(self.d, "pub.pem")
        subprocess.run(["openssl", "pkey", "-in", self.key, "-pubout", "-out", self.pub], check=True, capture_output=True)
        security.seal(self.root, self.key, progress=lambda *_: None)

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_manifest_paths_are_portable(self):
        m = json.load(open(os.path.join(self.root, security.MANIFEST)))
        self.assertIn("models/m.gguf", m["files"])
        self.assertFalse(any("\\" in k for k in m["files"]))

    def test_clean_drive_verifies(self):
        r = security.verify(self.root, "full", self.pub)
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["signature"])

    def test_detects_modified_model_in_quick_mode_head(self):
        p = os.path.join(self.root, "models", "m.gguf")
        with open(p, "r+b") as f:
            f.write(b"X")
        r = security.verify(self.root, "quick", self.pub)
        self.assertFalse(r["ok"])
        self.assertIn("models/m.gguf", r["modified"])

    def test_detects_added_file_and_bad_signature(self):
        with open(os.path.join(self.root, "evil.py"), "w") as f:
            f.write("x")
        self.assertIn("evil.py", security.verify(self.root, "quick", self.pub)["unexpected"])
        with open(os.path.join(self.root, security.MANIFEST), "a") as f:
            f.write(" ")
        r = security.verify(self.root, "quick", self.pub)
        self.assertFalse(r["signature"])
        self.assertFalse(r["ok"])

    def test_signed_drive_fails_closed_without_openssl(self):
        real = security.shutil.which
        security.shutil.which = lambda name: None if name == "openssl" else real(name)
        try:
            r = security.verify(self.root, "quick", self.pub)
        finally:
            security.shutil.which = real
        self.assertFalse(r["ok"])
        self.assertIn("openssl", r["error"])

    def test_wrong_org_key_rejected(self):
        other = os.path.join(self.d, "other.pem")
        subprocess.run(["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048", "-out", other],
                       check=True, capture_output=True)
        otherpub = os.path.join(self.d, "otherpub.pem")
        subprocess.run(["openssl", "pkey", "-in", other, "-pubout", "-out", otherpub], check=True, capture_output=True)
        self.assertFalse(security.verify(self.root, "quick", otherpub)["ok"])

    def test_policy_and_audit(self):
        with open(os.path.join(self.root, "airgap.json"), "w") as f:
            json.dump({"airgap": True}, f)
        pol = security.load_policy(self.root)
        self.assertTrue(pol["airgap"])
        self.assertFalse(pol["allow_downloads"])
        log = os.path.join(self.d, "audit.jsonl")
        security.AuditLog(log, pol).write("chat", prompt="secret plan alpha", model="m")
        rec = json.loads(open(log).read())
        self.assertNotIn("prompt", rec)
        self.assertEqual(rec["prompt_chars"], len("secret plan alpha"))
        with self.assertRaises(provision.AirgapError):
            provision.fleet_plan(self.root, [], "anything", use_meta=False)

    def test_network_routes_sends_nothing(self):
        r = security.network_routes()
        self.assertIn("online_possible", r)


class TestCluster(unittest.TestCase):
    def test_split_proportional_and_feasibility(self):
        sts = [{"addr": "a:1", "reachable": True, "ram_free": 30e9, "gpus": []},
               {"addr": "b:1", "reachable": True, "ram_free": 10e9, "gpus": [{"vram_free": 12e9}]},
               {"addr": "c:1", "reachable": False}]
        p = cluster.split(sts, 30e9)
        self.assertEqual(p["nodes"], ["a:1", "b:1"])
        self.assertEqual(p["unreachable"], ["c:1"])
        self.assertAlmostEqual(sum(p["ts"]), 1.0, places=2)
        self.assertGreater(p["ts"][0], p["ts"][1])
        self.assertTrue(p["fits"])
        self.assertFalse(cluster.split(sts, 60e9)["fits"])
        args = cluster.llama_args("m.gguf", p)
        self.assertIn("--rpc", args)
        self.assertEqual(args[args.index("--rpc") + 1], "a:1,b:1")

    def test_node_refuses_open_bind(self):
        with self.assertRaises(SystemExit):
            cluster.serve_node("0.0.0.0", 50052)


class _Fake(BaseHTTPRequestHandler):
    reply = ""

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(n))
        text = self.server.responder(body)
        data = json.dumps({"choices": [{"message": {"content": text}}], "timings": {"predicted_per_second": 9.9}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _serve(responder):
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    srv.responder = responder
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class TestProvision(unittest.TestCase):
    def test_qualify_picks_smallest_passing(self):
        small = _serve(lambda b: "wrong")
        big = _serve(lambda b: "Paris")

        def judge(b):
            cand = json.loads(b["messages"][-1]["content"])["candidate"]
            return json.dumps({"score": 5 if cand == "Paris" else 1, "reason": "ok"})
        meta = _serve(judge)
        old = assistant.META_BASE_URL
        os.environ["MODEL_API_KEY"] = "k"
        try:
            assistant.META_BASE_URL = f"http://127.0.0.1:{meta.server_port}/v1"
            d = tempfile.mkdtemp()
            rep = provision.qualify(d, [{"prompt": "Capital of France?", "reference": "Paris"}],
                                    {"q2": f"http://127.0.0.1:{small.server_port}",
                                     "q4": f"http://127.0.0.1:{big.server_port}"}, log=lambda *_: None)
            self.assertEqual(rep["selected"], "q4")
            self.assertFalse(rep["candidates"]["q2"]["passed"])
            self.assertEqual(rep["candidates"]["q4"]["rows"][0]["judge"], f"meta:{assistant.META_MODEL}")
            shutil.rmtree(d)
        finally:
            assistant.META_BASE_URL = old
            os.environ.pop("MODEL_API_KEY", None)
            for s in (small, big, meta):
                s.shutdown()

    def test_fleet_plan_common_models(self):
        def prof(ram):
            return {"cpu": {"physical_cores": 8}, "ram": {"total_bytes": ram * 2**30, "available_bytes": ram * 2**29},
                    "gpus": []}
        d = tempfile.mkdtemp()
        plan = provision.fleet_plan(d, [("air16", prof(16)), ("pro64", prof(64))], "offline Q&A", use_meta=False)
        self.assertTrue(any("llama-3.1-8b" in m for m in plan["runs_on_all_devices"]))
        self.assertEqual(len(plan["devices"]), 2)
        shutil.rmtree(d)


class TestThermal(unittest.TestCase):
    def test_profiles(self):
        cpu = {"performance_cores": 4, "physical_cores": 8}
        self.assertEqual(thermal.threads_for("cool", cpu), 2)
        self.assertEqual(thermal.threads_for("max", cpu), 4)
        self.assertEqual(thermal.auto_profile({"level": "serious"}), "cool")
        self.assertEqual(thermal.auto_profile({"level": "nominal", "fanless_hint": True}), "balanced")


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(os.path.exists(TEST_MOE), "test models not generated")
class TestDrives(unittest.TestCase):
    def test_parts_found_across_volumes_and_missing_reported(self):
        from pocketllm import core, runtime
        d = tempfile.mkdtemp()
        vols = [os.path.join(d, f"vol{i}", "PocketLLM", "models") for i in range(3)]
        old = {k: os.environ.get(k) for k in ("POCKETLLM_MODELS", "POCKETLLM_EXTRA_DRIVES", "POCKETLLM_DATA")}
        try:
            stripe.stripe(TEST_MOE, vols, os.path.join(d, "ignored"), progress=lambda *_: None)
            os.environ["POCKETLLM_MODELS"] = vols[0]
            os.environ["POCKETLLM_EXTRA_DRIVES"] = os.pathsep.join(os.path.join(d, f"vol{i}") for i in (1, 2))
            os.environ["POCKETLLM_DATA"] = os.path.join(d, "data")
            specs = core.local_specs()
            self.assertEqual(len(specs), 1)
            self.assertIn(os.path.join(d, "data", "links"), specs[0].path)
            self.assertEqual(core.INCOMPLETE, [])
            # unplug the third drive
            os.environ["POCKETLLM_EXTRA_DRIVES"] = os.path.join(d, "vol1")
            self.assertEqual(core.local_specs(), [])
            self.assertEqual(len(core.INCOMPLETE), 1)
            self.assertTrue(core.INCOMPLETE[0]["missing_parts"][0].endswith("-00003-of-00003.gguf"))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            shutil.rmtree(d)


class TestMLX(unittest.TestCase):
    def _write(self, path, tensors):
        import struct
        header, off = {}, 0
        for name, nb in tensors:
            header[name] = {"dtype": "U8", "shape": [nb], "data_offsets": [off, off + nb]}
            off += nb
        h = json.dumps(header).encode()
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(h)) + h + b"\0" * off)

    def test_spec_and_swiftlm_args(self):
        from pocketllm import mlx, planner
        d = tempfile.mkdtemp()
        try:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"num_hidden_layers": 2, "num_experts": 8, "num_experts_per_tok": 2, "hidden_size": 64,
                           "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 4096}, f)
            ts = [("model.embed_tokens.weight", 1000)]
            for i in range(2):
                ts += [(f"model.layers.{i}.self_attn.q_proj.weight", 100),
                       (f"model.layers.{i}.mlp.switch_mlp.up_proj.weight", 8000)]
            self._write(os.path.join(d, "model.safetensors"), ts)
            self.assertTrue(mlx.is_mlx_dir(d))
            self.assertEqual(mlx.find_mlx_models(d), [d])
            s = mlx.load_spec(d)
            self.assertTrue(s.is_moe)
            self.assertEqual(s.total_bytes, 1000 + 2 * 100 + 2 * 8000)
            self.assertEqual(s.layer_expert_bytes, [8000, 8000])
            self.assertEqual(s.active_bytes, 1000 + 200 + 16000 // 4)
            b = planner.Budget(ram_budget=planner.COMPUTE_OVERHEAD + 10000, gpu_budget=0, unified=True, ram_bw=1e10, gpu_bw=None,
                               disk_bw=1e9, threads=4)
            p = planner.plan(s, b, ctx=16)
            self.assertEqual(p.backend, "swiftlm")
            self.assertEqual(p.args[:2], ["--model", d])
            self.assertIn("--stream-experts", p.args)
        finally:
            shutil.rmtree(d)


class TestRequantAndKV(unittest.TestCase):
    def test_estimate_shrinks_streamed_experts(self):
        from pocketllm import requant
        m = gguf.load(TEST_MOE)
        e = requant.estimate(m, "q2_k", "q8_0")
        self.assertLess(e["expert_bytes"], e["source_expert_bytes"])
        self.assertLess(e["expert_bytes_per_token"], e["source_expert_bytes_per_token"])
        self.assertEqual(e["total_bytes"], e["expert_bytes"] + e["rest_bytes"])
        with self.assertRaises(ValueError):
            requant.command(TEST_MOE, "out.gguf", "iq2_xxs", "q8_0")  # needs an imatrix

    def test_kv_quantized_only_when_ram_is_short(self):
        from pocketllm import planner
        spec = planner.ModelSpec.from_gguf(gguf.load(TEST_MOE), TEST_MOE)
        big = planner.Budget(ram_budget=64 << 30, gpu_budget=0, unified=False, ram_bw=2e10, gpu_bw=None,
                             disk_bw=3e8, threads=4)
        p = planner.plan(spec, big, ctx=4096)
        self.assertEqual(p.kv_type, "f16")
        self.assertNotIn("-ctk", p.args)
        small = planner.Budget(ram_budget=planner.COMPUTE_OVERHEAD + spec.total_bytes // 2, gpu_budget=0,
                               unified=False, ram_bw=2e10, gpu_bw=None, disk_bw=3e8, threads=4)
        q = planner.plan(spec, small, ctx=4096)
        self.assertEqual(q.kv_type, "q8_0")
        self.assertIn("-ctk", q.args)
        self.assertEqual(q.args[q.args.index("-fa") + 1], "on")
        self.assertLess(q.kv_bytes, planner.plan(spec, small, ctx=4096, kv_type="f16").kv_bytes)

"""Unit tests. Run: python -m unittest discover -s tests -v

Hardware scenarios are synthetic profiles (no real GPU needed); they test the
planner's logic, not real-world speed.
"""
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pocketllm import assistant, catalog, gguf, planner  # noqa: E402

GiB = 2**30
MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def profile(ram_gb, gpu_gb=0, unified=False, cores=8, bw_gbps=40):
    p = {"cpu": {"physical_cores": cores, "logical_cores": cores * 2},
         "ram": {"total_bytes": int(ram_gb * GiB), "available_bytes": int(ram_gb * 0.85 * GiB),
                 "bandwidth": {"gb_per_s_1thread": bw_gbps}}, "gpus": []}
    if gpu_gb:
        p["gpus"].append({"name": "test-gpu", "vram_total_bytes": int(gpu_gb * GiB),
                          "vram_free_bytes": int(gpu_gb * GiB), "unified_memory": unified})
    return p


def scout(q="Q4_K_M"):
    return next(s for s in catalog.specs(["llama-4-scout-17b-16e-instruct"]) if s.name.endswith(q))


class TestPlanner(unittest.TestCase):
    def test_scout_q4_on_16gb_laptop_pages_from_disk(self):
        b = planner.budget_from_profile(profile(16), disk_gbps=1.0)
        p = planner.plan(scout(), b)
        self.assertEqual(p.strategy, "expert_stream")
        self.assertEqual(p.env.get("POCKET_MOE"), "1")
        self.assertIn("--no-repack", p.args)
        self.assertIn(p.verdict, ("slow", "impractical"))
        self.assertGreater(p.paged_bytes, 40e9)
        self.assertIn("-lm", p.args)
        self.assertEqual(p.args[p.args.index("-lm") + 1], "mmap")

    def test_streaming_on_apple_uses_cpu_path(self):
        p = planner.plan(scout(), planner.budget_from_profile(profile(16, gpu_gb=11, unified=True), disk_gbps=2.0))
        self.assertEqual(p.strategy, "expert_stream")
        self.assertEqual(p.args[p.args.index("-ngl") + 1], "0")
        self.assertNotIn("-ncmoe", p.args)

    def test_dense_model_over_ram_pages_without_streaming_engine(self):
        s = next(s for s in catalog.specs(["llama-3.3-70b-instruct"]) if s.name.endswith("Q4_K_M"))
        p = planner.plan(s, planner.budget_from_profile(profile(16), disk_gbps=1.0))
        self.assertEqual(p.strategy, "disk_paging")
        self.assertEqual(p.env, {})

    def test_scout_q4_fits_in_128gb_ram(self):
        p = planner.plan(scout(), planner.budget_from_profile(profile(128)))
        self.assertEqual((p.strategy, p.verdict), ("cpu_ram", "fits"))
        self.assertTrue(p.no_mmap)

    def test_scout_moe_hybrid_with_discrete_gpu(self):
        # 64 GB RAM + 16 GB GPU: attention/shared weights on GPU, routed experts split
        p = planner.plan(scout(), planner.budget_from_profile(profile(96, gpu_gb=16)))
        self.assertEqual(p.strategy, "moe_hybrid")
        self.assertGreater(p.n_cpu_moe, 0)
        self.assertLess(p.n_cpu_moe, 48)
        self.assertIn("-ncmoe", p.args)
        self.assertEqual(p.verdict, "fits")

    def test_unified_memory_never_double_counts(self):
        b = planner.budget_from_profile(profile(64, gpu_gb=48, unified=True))
        self.assertLessEqual(b.gpu_budget, b.ram_budget)

    def test_unified_hybrid_shares_one_pool(self):
        # 54 GB of weights cannot "fit" a 36 GB Mac by splitting GPU and CPU shares of the same RAM
        p = planner.plan(scout("Q3_K_M"), planner.budget_from_profile(profile(36, gpu_gb=27, unified=True)))
        self.assertNotEqual(p.verdict, "fits")
        self.assertLess(p.gpu_bytes + p.ram_bytes, 36 * GiB)

    def test_small_model_full_gpu(self):
        s = next(s for s in catalog.specs(["llama-3.1-8b-instruct"]) if s.name.endswith("Q4_K_M"))
        p = planner.plan(s, planner.budget_from_profile(profile(16, gpu_gb=8)))
        self.assertEqual(p.strategy, "full_gpu")

    def test_dense_partial_offload(self):
        s = next(s for s in catalog.specs(["llama-3.3-70b-instruct"]) if s.name.endswith("Q4_K_M"))
        p = planner.plan(s, planner.budget_from_profile(profile(64, gpu_gb=12)))
        self.assertEqual(p.strategy, "partial_gpu")
        self.assertTrue(0 < p.n_gpu_layers < 80)

    def test_speed_bound_scales_with_bandwidth(self):
        s = scout()
        slow = planner.plan(s, planner.budget_from_profile(profile(128, bw_gbps=20)))
        fast = planner.plan(s, planner.budget_from_profile(profile(128, bw_gbps=80)))
        self.assertAlmostEqual(fast.est_tok_s_upper / slow.est_tok_s_upper, 4.0, places=1)

    def test_ctx_does_not_exceed_training_ctx(self):
        s = next(s for s in catalog.specs(["llama-3.2-3b-instruct"]))
        s.train_ctx = 2048
        self.assertEqual(planner.plan(s, planner.budget_from_profile(profile(16)), ctx=8192).ctx, 2048)


@unittest.skipUnless(os.path.exists(os.path.join(MODELS, "test-moe-Q4_K_M.gguf")), "test models not generated")
class TestGGUF(unittest.TestCase):
    def test_matches_llama_bench(self):
        # reference values reported by llama-bench (model_size, model_n_params) for the same file
        m = gguf.load(os.path.join(MODELS, "test-moe-Q4_K_M.gguf"))
        self.assertEqual(m.weight_bytes, 600613888)
        self.assertEqual(m.n_params, 947022336)
        self.assertTrue(m.is_moe)
        self.assertEqual((m.n_expert, m.n_expert_used, m.n_layer), (32, 4, 12))
        self.assertLess(m.active_bytes_per_token(), m.weight_bytes)

    def test_find_models_skips_split_parts(self):
        names = [os.path.basename(p) for p in gguf.find_models(MODELS)]
        self.assertFalse(any("-00002-of-" in n for n in names))


class TestSplitNames(unittest.TestCase):
    def test_split_regex(self):
        m = gguf.SPLIT_RE.match("Llama-4-Scout-Q4_K_M-00001-of-00002.gguf")
        self.assertEqual(m.groups(), ("Llama-4-Scout-Q4_K_M", "00001", "00002"))


class TestAssistant(unittest.TestCase):
    def test_offline_rules(self):
        c = assistant._local_parse("가장 큰 모델을 오프라인으로, 긴 문서 요약")
        self.assertEqual(c["priority"], "quality")
        self.assertTrue(c["must_be_offline"])
        self.assertEqual(c["min_ctx"], 16384)

    def test_validate_clamps_hallucinated_values(self):
        c = assistant.validate({"priority": "ludicrous", "min_ctx": 10**9, "max_model_b": "lots"})
        self.assertEqual(c["priority"], "balanced")
        self.assertEqual(c["min_ctx"], 131072)
        self.assertIsNone(c["max_model_b"])

    def test_meta_api_request_and_fallback(self):
        seen = {}

        class Fake(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                seen["path"] = self.path
                seen["auth"] = self.headers.get("Authorization")
                seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                reply = {"choices": [{"message": {"content": 'Sure: {"priority": "speed", "min_ctx": 8192, '
                                                             '"must_be_offline": true, "task": "coding"}'}}]}
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        srv = HTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        old = (assistant.META_BASE_URL, os.environ.get("MODEL_API_KEY"))
        try:
            assistant.META_BASE_URL = f"http://127.0.0.1:{srv.server_port}/v1"
            os.environ["MODEL_API_KEY"] = "test-key"
            c = assistant.interpret("fast coding helper", {"ram_gb": 16}, [])
            self.assertEqual(seen["path"], "/v1/chat/completions")
            self.assertEqual(seen["auth"], "Bearer test-key")
            self.assertEqual(seen["body"]["model"], assistant.META_MODEL)
            self.assertEqual((c["priority"], c["min_ctx"], c["task"]), ("speed", 8192, "coding"))
            self.assertTrue(c["backend"].startswith("meta:"))
            # unreachable API -> offline rules, never an exception
            assistant.META_BASE_URL = "http://127.0.0.1:9/v1"
            c = assistant.interpret("fast", {}, [], timeout=2)
            self.assertEqual(c["backend"], "local-rules")
        finally:
            srv.shutdown()
            assistant.META_BASE_URL = old[0]
            if old[1] is None:
                os.environ.pop("MODEL_API_KEY", None)
            else:
                os.environ["MODEL_API_KEY"] = old[1]


if __name__ == "__main__":
    unittest.main()

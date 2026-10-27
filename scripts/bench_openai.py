"""Measure decode speed of any OpenAI-compatible local server the same way
(llama-server with or without the PocketLLM engine, SwiftLM, ...).

Streams a chat completion, records time to first token and the decode rate
over the tokens after the first one. Each streamed content chunk is counted as
one token, which is how both llama-server and SwiftLM stream.

usage: python scripts/bench_openai.py URL [--tokens 64] [--runs 2] [--label X] [--out file.json]
"""
import argparse
import json
import statistics
import time
import urllib.request

PROMPT = "Explain in detail how a bicycle gear system works, step by step."


def one(url, n, model):
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": n,
            "temperature": 0, "stream": True}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first, last, count = None, None, 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            try:
                d = json.loads(line[5:])
            except ValueError:
                continue
            delta = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                now = time.perf_counter()
                first = first or now
                last = now
                count += 1
    if not first or count < 2:
        return {"ok": False, "tokens": count}
    return {"ok": True, "tokens": count, "ttft_s": round(first - t0, 3),
            "decode_tok_s": round((count - 1) / (last - first), 3)}


def wait_ready(url, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for path in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(url.rstrip("/") + path, timeout=5) as r:
                    body = r.read().decode("utf-8", "replace")
                    if r.status == 200 and "loading" not in body.lower():
                        return True
            except OSError:
                pass
        time.sleep(2)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--model", default="local")
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    a = ap.parse_args()
    if not wait_ready(a.url):
        raise SystemExit("server not ready")
    runs = [one(a.url, a.tokens, a.model) for _ in range(a.runs)]
    ok = [r for r in runs if r["ok"]]
    res = {"label": a.label, "runs": runs,
           "decode_tok_s": round(statistics.mean(r["decode_tok_s"] for r in ok), 3) if ok else None,
           "ttft_s": round(statistics.mean(r["ttft_s"] for r in ok), 3) if ok else None}
    print(json.dumps(res))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()

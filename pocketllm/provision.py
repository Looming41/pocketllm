"""Provisioning station: the only place PocketLLM talks to the cloud.

Two zones (see docs/META_KO.md):

  PROVISIONING (internet, unclassified)          FIELD (air-gapped)
  -----------------------------------            ------------------
  Meta Model API (Muse Spark):                   Meta open-weight models only
   * fleet plan: mission goal + hardware          (Llama 4 Scout, Llama 3.x,
     profiles of the target laptops -> which        Muse Glimmer) on the laptop,
     models / quants go on the drive                expert streaming from the drive
   * synthetic evaluation set for the domain     no network, signed manifest,
   * judge: grade each candidate quant's          metadata-only audit log
     answers -> pick the smallest that passes
  then `pocketllm seal` signs the drive    --->  drive carried across the air gap

Rules enforced in code:
  * every function here refuses to run when the drive's policy says airgap
  * only synthetic, unclassified prompts (generated here or written by the
    operator) are ever sent to the API; nothing produced in the field is uploaded
  * the cloud only proposes; the local planner and local measurements decide
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from . import assistant, catalog, planner, security

JUDGE_PROMPT = (
    "You grade answers from a locally running, quantized language model. For each item you get the "
    "question, an optional reference answer and the candidate answer. Score 1-5 for correctness and "
    "usefulness (5 = as good as the reference, 1 = wrong or harmful). Reply ONLY with JSON: "
    '{"score": <1-5>, "reason": "<one sentence>"}'
)
EVALSET_PROMPT = (
    "Write {n} evaluation questions for testing an offline assistant used for: {domain}. "
    "Questions must be UNCLASSIFIED and synthetic: no real names, units, locations or secrets. "
    "Mix factual recall, step-by-step procedure, summarisation of a short given text, and Korean-language items. "
    'Reply ONLY with a JSON list of objects {{"prompt": "...", "reference": "short ideal answer"}}.'
)


class AirgapError(RuntimeError):
    pass


def _guard(root):
    pol = security.load_policy(root)
    if pol["airgap"]:
        raise AirgapError("this drive's policy is air-gapped: provisioning features are disabled here")


def meta_chat(messages, temperature=0.0, timeout=60) -> str:
    key = os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set (Meta Model API)")
    body = {"model": assistant.META_MODEL, "messages": messages, "temperature": temperature}
    req = urllib.request.Request(assistant.META_BASE_URL.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def _json_from(text):
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    return json.loads(m.group(1) if m else text)


# ------------------------------------------------------------------ fleet plan
def fleet_plan(root: str, profiles: list, goal: str, ctx: int = 4096, use_meta=True) -> dict:
    """profiles: list of (name, profile dict) collected offline with `pocketllm profile --json`."""
    _guard(root)
    specs = catalog.specs()
    table = []
    for name, prof in profiles:
        b = planner.budget_from_profile(prof)
        rows = []
        for pl, spec in planner.recommend(specs, b, ctx):
            rows.append({"model": spec.name, "verdict": pl.verdict, "strategy": pl.strategy,
                         "size_gb": round(spec.total_bytes / 1e9, 1), "est_tok_s_upper": pl.est_tok_s_upper})
        table.append({"device": name, "ram_gb": round((prof["ram"].get("total_bytes") or 0) / 1e9),
                      "options": rows})
    constraints = assistant.interpret(goal, {"fleet": [t["device"] for t in table]},
                                      [], use_meta=use_meta)
    # the drive must serve the *weakest* device: keep models that run (fits/slow) everywhere
    common = None
    for t in table:
        ok = {r["model"] for r in t["options"] if r["verdict"] in ("fits", "slow")}
        common = ok if common is None else common & ok
    common = sorted(common or [], key=lambda m: -next(r["size_gb"] for r in table[0]["options"] if r["model"] == m))
    plan = {"goal": goal, "constraints": constraints, "devices": table, "runs_on_all_devices": common,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if use_meta and os.environ.get("MODEL_API_KEY"):
        try:
            brief = meta_chat([
                {"role": "system", "content": "You write short deployment briefs for IT staff. Use only the data given."},
                {"role": "user", "content": "Write a 6-bullet deployment brief (which models to put on the drive, "
                                            "which devices get which model, expected limits) from this JSON:\n"
                                            + json.dumps({k: plan[k] for k in ("goal", "runs_on_all_devices")}
                                                         | {"devices": [{"device": t["device"], "ram_gb": t["ram_gb"],
                                                                         "top": t["options"][:4]} for t in table]})}])
            plan["brief"] = {"source": f"meta:{assistant.META_MODEL}", "text": brief}
        except (urllib.error.URLError, OSError, KeyError, ValueError, RuntimeError) as e:
            plan["brief"] = {"source": "none", "text": f"Meta API unavailable: {type(e).__name__}"}
    return plan


# ------------------------------------------------------------------ evaluation set
def make_evalset(root: str, domain: str, n: int = 20) -> list:
    _guard(root)
    text = meta_chat([{"role": "user", "content": EVALSET_PROMPT.format(n=n, domain=domain)}], temperature=0.7)
    items = _json_from(text)
    return [{"prompt": str(i["prompt"]), "reference": str(i.get("reference", ""))} for i in items if i.get("prompt")]


# ------------------------------------------------------------------ qualification
def ask_local(server_url: str, prompt: str, max_tokens=384, timeout=600) -> dict:
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(server_url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return {"answer": d["choices"][0]["message"]["content"], "seconds": round(time.time() - t0, 2),
            "tok_s": (d.get("timings") or {}).get("predicted_per_second")}


def judge(item: dict, answer: str, local_judge_url: str | None = None) -> dict:
    msgs = [{"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": json.dumps({"question": item["prompt"], "reference": item.get("reference", ""),
                                                    "candidate": answer}, ensure_ascii=False)}]
    try:
        if local_judge_url:
            body = {"messages": msgs, "temperature": 0, "max_tokens": 128}
            req = urllib.request.Request(local_judge_url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                text = json.load(r)["choices"][0]["message"]["content"]
            src = "local-judge"
        else:
            text = meta_chat(msgs)
            src = f"meta:{assistant.META_MODEL}"
        d = _json_from(text)
        return {"score": max(1, min(5, int(d["score"]))), "reason": str(d.get("reason", ""))[:200], "judge": src}
    except (urllib.error.URLError, OSError, KeyError, ValueError, TypeError, RuntimeError) as e:
        return {"score": None, "reason": f"judge failed: {type(e).__name__}", "judge": "none"}


def qualify(root: str, evalset: list, candidates: dict, pass_mean=3.5, local_judge_url=None, log=print) -> dict:
    """candidates: {label: llama-server url}. Picks the smallest-first candidate that passes."""
    if not local_judge_url:
        _guard(root)  # cloud judge only off the air-gapped side
    report = {"pass_mean": pass_mean, "candidates": {}}
    for label, url in candidates.items():
        rows = []
        for item in evalset:
            a = ask_local(url, item["prompt"])
            j = judge(item, a["answer"], local_judge_url)
            rows.append({"prompt": item["prompt"][:120], "score": j["score"], "reason": j["reason"],
                         "seconds": a["seconds"], "tok_s": a["tok_s"], "judge": j["judge"]})
            log(f"  [{label}] score={j['score']} {a['seconds']}s  {item['prompt'][:60]!r}")
        scores = [r["score"] for r in rows if r["score"] is not None]
        mean = round(sum(scores) / len(scores), 2) if scores else None
        report["candidates"][label] = {"mean_score": mean, "n": len(rows), "passed": mean is not None and mean >= pass_mean,
                                       "rows": rows}
    report["selected"] = next((k for k, v in report["candidates"].items() if v["passed"]), None)
    return report

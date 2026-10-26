"""Setup assistant: turns a natural-language goal into run constraints.

Two backends, always labelled in the UI:

* ``meta``  - Meta Model API (cloud, needs internet + MODEL_API_KEY). The
  model only *proposes* constraints as JSON; the local planner validates every
  proposal against the measured hardware, so a hallucinated setting can never
  be applied.
* ``local`` - offline keyword rules. Used automatically when there is no key
  or no network, so the product keeps working offline.

API facts used here (dev.meta.ai docs, checked 2026-10-27 via search; the
page itself was not reachable from the build machine): base URL
https://api.meta.ai/v1, bearer auth, OpenAI-compatible /chat/completions,
model ids like ``muse-spark-1.3``. Override with POCKETLLM_META_BASE_URL /
POCKETLLM_META_MODEL if they differ.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

META_BASE_URL = os.environ.get("POCKETLLM_META_BASE_URL", "https://api.meta.ai/v1")
META_MODEL = os.environ.get("POCKETLLM_META_MODEL", "muse-spark-1.3")

SCHEMA = {
    "priority": "speed | quality | balanced",
    "min_ctx": "integer tokens of context the task needs",
    "must_be_offline": "boolean",
    "max_model_b": "integer or null - upper bound on parameters (billions) the user accepts",
    "task": "short label, e.g. 'summarise long PDFs', 'coding help', 'chat'",
    "explanation": "one sentence for the user",
}

SYSTEM_PROMPT = (
    "You configure a local LLM runtime. Given the user's goal, the measured hardware and the "
    "candidate models with their local feasibility verdicts, return ONLY a JSON object with keys: "
    + json.dumps(SCHEMA)
    + ". Never invent hardware or models; choose constraints, the local planner picks the model."
)


def _local_parse(goal: str) -> dict:
    g = goal.lower()
    c = {"priority": "balanced", "min_ctx": 4096, "must_be_offline": False, "max_model_b": None,
         "task": "chat", "explanation": "", "backend": "local-rules"}
    if re.search(r"fast|quick|speed|빠르|속도|실시간", g):
        c["priority"] = "speed"
    if re.search(r"best|smart|quality|accura|정확|품질|똑똑|큰 모델|largest|biggest|100b|120b|109b", g):
        c["priority"] = "quality"
    if re.search(r"long|pdf|document|book|문서|긴|요약", g):
        c["min_ctx"], c["task"] = 16384, "long documents"
    if re.search(r"code|coding|코드|코딩|프로그래밍", g):
        c["task"] = "coding"
    if re.search(r"offline|private|privacy|오프라인|개인정보|보안|인터넷 없이", g):
        c["must_be_offline"] = True
    m = re.search(r"(\d{1,3})\s*b\b", g)
    if m:
        c["max_model_b"] = int(m.group(1)) if c["priority"] != "quality" else None
    c["explanation"] = f"priority={c['priority']}, ctx>={c['min_ctx']}, task={c['task']} (offline rule parser)"
    return c


def interpret(goal: str, hardware_summary: dict, candidates: list, use_meta: bool = True, timeout=20) -> dict:
    key = os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY")
    if not (use_meta and key):
        return _local_parse(goal)
    body = {
        "model": META_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"goal": goal, "hardware": hardware_summary,
                                                     "candidates": candidates})},
        ],
        "temperature": 0,
    }
    req = urllib.request.Request(META_BASE_URL.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
        text = data["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", text, re.S)
        proposal = json.loads(m.group(0) if m else text)
    except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as e:
        c = _local_parse(goal)
        c["explanation"] += f" - Meta API unavailable ({type(e).__name__}); used offline rules"
        return c
    return validate(proposal) | {"backend": f"meta:{META_MODEL}"}


def validate(p: dict) -> dict:
    """Clamp a model-proposed constraint set to safe, known values."""
    out = _local_parse("")
    if p.get("priority") in ("speed", "quality", "balanced"):
        out["priority"] = p["priority"]
    try:
        out["min_ctx"] = max(512, min(int(p.get("min_ctx", 4096)), 131072))
    except (TypeError, ValueError):
        pass
    out["must_be_offline"] = bool(p.get("must_be_offline", False))
    try:
        out["max_model_b"] = int(p["max_model_b"]) if p.get("max_model_b") else None
    except (TypeError, ValueError):
        out["max_model_b"] = None
    out["task"] = str(p.get("task", "chat"))[:60]
    out["explanation"] = str(p.get("explanation", ""))[:300]
    return out


def choose(constraints: dict, ranked: list) -> list:
    """Re-rank planner output (list of (plan, spec)) using the constraints."""
    order = {"fits": 0, "slow": 1, "impractical": 2, "does_not_fit": 3}
    out = []
    for plan, spec in ranked:
        size_b = None
        m = re.search(r"(\d+(?:\.\d+)?)b", spec.name.lower())
        if m:
            size_b = float(m.group(1))
        if constraints.get("max_model_b") and size_b and size_b > constraints["max_model_b"]:
            continue
        if plan.ctx < constraints.get("min_ctx", 0) and plan.verdict != "does_not_fit":
            continue
        speed = plan.est_tok_s_upper or 0
        if constraints["priority"] == "speed":
            key = (order[plan.verdict], -speed)
        elif constraints["priority"] == "quality":
            key = (order[plan.verdict] if plan.verdict != "slow" else 0, -spec.total_bytes)
        else:
            key = (order[plan.verdict], -spec.total_bytes)
        out.append((key, plan, spec))
    out.sort(key=lambda x: x[0])
    return [(p, s) for _, p, s in out]

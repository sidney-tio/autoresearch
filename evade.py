"""evade.py — generation + detection + metric for the prompt-evasion loop.

Reads rules.md, injects it into Gemma's system prompt, generates the 60 varied
LinkedIn posts from seeds.py, scores each with the RoBERTa AI-text detector, and
prints the metric block. READ-ONLY harness file — the loop only edits rules.md.

Run:  uv run evade.py > run.log 2>&1
"""

import asyncio
import json
import re
import sys
from pathlib import Path

from langdetect import DetectorFactory, LangDetectException
from langdetect import detect as detect_lang
from openai import AsyncOpenAI
from transformers import pipeline

from seeds import SEEDS
from utils import clean_text

DetectorFactory.seed = 0  # make langdetect deterministic

# --- fixed configuration -----------------------------------------------------
GENERATOR = "google/gemma-4-31B-it"
DETECTOR = "fakespot-ai/roberta-base-ai-text-detection-v1"
DETECTOR_DEVICE = 0  # GPU 0; roberta-base co-exists with the vLLM generator on an H200
VLLM_BASE_URL = "http://localhost:8000/v1"

BASE_FRAMING = (
    "You are an experienced professional writing posts for LinkedIn.\n"
    "You will be given a brief. Write exactly one LinkedIn post that fulfills it.\n"
    "Output only the post text itself — no preamble, no quotation marks, no notes."
)

GEN_TEMPERATURE = 0.9
GEN_TOP_P = 0.95
GEN_MAX_TOKENS = 700
GEN_BASE_SEED = 7          # fixed -> runs differ only because rules.md differs

# quality-check bounds
MIN_WORDS = 40
MAX_WORDS = 400
MIN_ASCII_RATIO = 0.90
MIN_UNIQUE_RATIO = 0.35

# label-name heuristics for mapping detector output to P(AI)
AI_HINTS = ("ai", "machine", "generated", "fake", "gpt", "llm", "synthetic", "bot")
HUMAN_HINTS = ("human", "real", "genuine", "person", "authentic")


def build_system_prompt() -> str:
    raw = Path("rules.md").read_text(encoding="utf-8")
    rules = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL).strip()
    return f"{BASE_FRAMING}\n\n{rules}" if rules else BASE_FRAMING


async def _generate_all(system_prompt: str) -> list[str]:
    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")

    async def generate_one(idx: int, brief: str) -> str:
        resp = await client.chat.completions.create(
            model=GENERATOR,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": brief},
            ],
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            max_tokens=GEN_MAX_TOKENS,
            seed=GEN_BASE_SEED + idx,
        )
        return (resp.choices[0].message.content or "").strip()

    try:
        results = await asyncio.gather(
            *(generate_one(i, brief) for i, brief in enumerate(SEEDS))
        )
        return list(results)
    finally:
        await client.close()


def generate_all(system_prompt: str) -> list[str]:
    """Fire all 60 briefs concurrently at the vLLM server (continuous batching)."""
    return asyncio.run(_generate_all(system_prompt))


def quality_check(text: str) -> tuple[bool, str]:
    """Lightweight gate so the optimizer cannot win by producing junk."""
    if not text:
        return False, "empty"
    words = text.split()
    n = len(words)
    if n < MIN_WORDS:
        return False, f"too_short({n})"
    if n > MAX_WORDS:
        return False, f"too_long({n})"
    ascii_ratio = sum(c.isascii() for c in text) / len(text)
    if ascii_ratio < MIN_ASCII_RATIO:
        return False, f"low_ascii({ascii_ratio:.2f})"
    unique_ratio = len({w.lower() for w in words}) / n
    if unique_ratio < MIN_UNIQUE_RATIO:
        return False, f"repetitive({unique_ratio:.2f})"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 2 and len(set(lines)) < len(lines):
        return False, "duplicate_lines"
    try:
        if detect_lang(text) != "en":
            return False, "not_english"
    except LangDetectException:
        return False, "langdetect_failed"
    return True, "ok"


def p_ai_from_scores(scores: list[dict]) -> float:
    """Map a detector label distribution to P(text is AI-generated)."""
    by_label = {d["label"].lower(): float(d["score"]) for d in scores}
    for label, score in by_label.items():
        if any(hint in label for hint in AI_HINTS):
            return score
    for label, score in by_label.items():
        if any(hint in label for hint in HUMAN_HINTS):
            return 1.0 - score
    if "label_1" in by_label:  # common HF convention: LABEL_1 == positive == AI
        return by_label["label_1"]
    raise ValueError(f"cannot map detector labels to P(AI): {sorted(by_label)}")


def main() -> None:
    system_prompt = build_system_prompt()
    print(f"[system prompt — {len(system_prompt)} chars]", file=sys.stderr)
    print(system_prompt, file=sys.stderr)
    print(f"generating {len(SEEDS)} posts via {GENERATOR}...", file=sys.stderr)
    posts = generate_all(system_prompt)

    cleaned = [clean_text(p) for p in posts]
    checks = [quality_check(c) for c in cleaned]
    valid_idx = [i for i, (ok, _) in enumerate(checks) if ok]

    print(f"scoring with {DETECTOR}...", file=sys.stderr)
    classifier = pipeline("text-classification", model=DETECTOR, device=DETECTOR_DEVICE)
    p_ai = [1.0] * len(posts)  # invalid posts keep the max penalty
    labels_seen: set[str] = set()
    if valid_idx:
        raw = classifier(
            [cleaned[i] for i in valid_idx],
            top_k=None,
            batch_size=16,
            truncation=True,
        )
        for j, i in enumerate(valid_idx):
            labels_seen.update(d["label"] for d in raw[j])
            p_ai[i] = p_ai_from_scores(raw[j])

    records = []
    for i, brief in enumerate(SEEDS):
        ok, reason = checks[i]
        records.append({
            "brief": brief,
            "post": posts[i],
            "cleaned": cleaned[i],
            "valid": ok,
            "reason": reason,
            "p_ai": round(p_ai[i], 6),
        })
    Path("last_run.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )

    n = len(posts)
    mean_p_ai = sum(p_ai) / n
    detect_rate = sum(p > 0.5 for p in p_ai) / n
    n_invalid = sum(not ok for ok, _ in checks)

    print("---")
    print(f"mean_p_ai:     {mean_p_ai:.6f}")
    print(f"detect_rate:   {detect_rate:.3f}")
    print(f"n_invalid:     {n_invalid}")
    print(f"n_posts:       {n}")
    print(f"generator:     {GENERATOR}")
    print(f"detector:      {DETECTOR}")
    print(f"detector_labels: {sorted(labels_seen)}")


if __name__ == "__main__":
    main()

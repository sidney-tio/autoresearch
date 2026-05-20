"""loop.py — automated self-optimization loop for prompt-evasion.

Each iteration:
  1. Gemma (the "optimizer") reads the current rules.md and the worst-scoring
     posts from the best run so far, and proposes a new rules.md.
  2. evade.py runs the 60-post evaluation.
  3. The new rules are KEPT (git-committed) only if mean_p_ai improved,
     otherwise REVERTED (git checkout rules.md).

Gemma plays both roles — generator and optimizer — on the same vLLM server.

Requires the vLLM server running:
    vllm serve google/gemma-4-31B-it --port 8000

Run:
    uv run loop.py                 # loop until Ctrl-C
    uv run loop.py --iters 50      # bounded
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

GENERATOR = "google/gemma-4-31B-it"
VLLM_BASE_URL = "http://localhost:8000/v1"

RULES_PATH = Path("rules.md")
RESULTS_PATH = Path("results.tsv")
LAST_RUN_PATH = Path("last_run.jsonl")
BEST_RUN_PATH = Path("best_run.jsonl")
RUN_LOG_PATH = Path("run.log")

RESULTS_HEADER = "commit\tmean_p_ai\tdetect_rate\tn_invalid\tstatus\tdescription"
CRASH_METRICS = {"mean_p_ai": 1.0, "detect_rate": 1.0, "n_invalid": 0}

EVADE_TIMEOUT_S = 900          # 15 min — matches CLAUDE.md
N_WORST_EXAMPLES = 6
POST_SNIPPET_CHARS = 500
HISTORY_ROWS = 20

OPT_TEMPERATURE = 0.8
OPT_MAX_TOKENS = 1600

OPTIMIZER_SYSTEM = """You are the optimizer in an adversarial-robustness experiment on an AI-text detector.

A generator (Gemma) writes LinkedIn posts. Its system prompt contains a set of "evasion rules". Every generated post is scored by a RoBERTa detector that outputs P(AI) — the probability the text is AI-generated. The experiment's metric is mean_p_ai over 60 posts; lower is better.

Your job: rewrite the evasion rules to lower mean_p_ai while keeping the posts coherent, realistic LinkedIn posts.

Rules of the game:
- The rules are instructions to the writer. Write them in the second person ("Vary...", "Avoid...").
- Keep the rule set concise. Bloated rules dilute the writer's attention.
- Posts must stay 40-400 words, English, and coherent. Garbled or repetitive posts are marked invalid and scored P(AI)=1.0 — that hurts the metric badly.
- Encoding tricks (invisible characters, homoglyphs) do NOT work: text is normalized before detection. Evasion must be linguistic.
- Make ONE focused change per iteration so its effect is measurable. Build on what worked; drop what the history shows did not.

Respond in EXACTLY this format and nothing else:
<DESCRIPTION>one short line naming the change</DESCRIPTION>
<RULES>
the complete new rules text
</RULES>"""


# --- git helpers -------------------------------------------------------------
def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True)


def git_short_hash() -> str:
    return git("rev-parse", "--short", "HEAD").stdout.strip()


def rules_dirty() -> bool:
    return bool(git("status", "--porcelain", "rules.md").stdout.strip())


def commit_rules(description: str) -> str:
    git("add", "rules.md")
    result = git("commit", "-m", description, "--", "rules.md")
    if result.returncode != 0:
        print(f"  WARNING: git commit failed: {result.stderr.strip()}", file=sys.stderr)
    return git_short_hash()


def revert_rules() -> None:
    git("checkout", "--", "rules.md")


# --- evaluation --------------------------------------------------------------
def run_evade() -> dict | None:
    """Run one evade.py evaluation. Returns metrics dict, or None if it crashed."""
    with RUN_LOG_PATH.open("w") as log:
        try:
            subprocess.run(
                ["uv", "run", "evade.py"],
                stdout=log, stderr=subprocess.STDOUT,
                timeout=EVADE_TIMEOUT_S, check=False,
            )
        except subprocess.TimeoutExpired:
            return None
    text = RUN_LOG_PATH.read_text(errors="replace")
    metrics: dict = {}
    for key in ("mean_p_ai", "detect_rate", "n_invalid"):
        match = re.search(rf"^{key}:\s*([\d.]+)", text, re.MULTILINE)
        if not match:
            return None
        metrics[key] = float(match.group(1))
    metrics["n_invalid"] = int(metrics["n_invalid"])
    return metrics


def snapshot_best() -> None:
    if LAST_RUN_PATH.exists():
        shutil.copy(LAST_RUN_PATH, BEST_RUN_PATH)


# --- optimizer context -------------------------------------------------------
def worst_examples(path: Path, k: int) -> str:
    if not path.exists():
        return "(no run recorded yet)"
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    rows.sort(key=lambda r: r["p_ai"], reverse=True)
    blocks = []
    for r in rows[:k]:
        flag = "" if r["valid"] else f"  [INVALID: {r['reason']}]"
        blocks.append(
            f"--- p_ai={r['p_ai']:.3f}{flag}\n"
            f"brief: {r['brief']}\n"
            f"post: {r['post'][:POST_SNIPPET_CHARS]}"
        )
    return "\n\n".join(blocks)


def history(k: int) -> str:
    if not RESULTS_PATH.exists():
        return "(no history)"
    rows = RESULTS_PATH.read_text().splitlines()[1:]  # drop header
    return "\n".join(rows[-k:]) if rows else "(no history)"


def propose_rules(client: OpenAI, best: dict) -> tuple[str, str]:
    """Ask Gemma for the next rules.md. Returns (description, rules_text)."""
    user_msg = (
        f"CURRENT rules.md:\n```\n{RULES_PATH.read_text(encoding='utf-8')}\n```\n\n"
        f"CURRENT metric (the score to beat): mean_p_ai={best['mean_p_ai']:.6f}, "
        f"detect_rate={best['detect_rate']:.3f}, n_invalid={best['n_invalid']}\n\n"
        f"WORST-SCORING posts under the current best rules "
        f"(these are the failure modes to fix):\n"
        f"{worst_examples(BEST_RUN_PATH, N_WORST_EXAMPLES)}\n\n"
        f"HISTORY of attempts "
        f"(commit / mean_p_ai / detect_rate / n_invalid / status / description):\n"
        f"{history(HISTORY_ROWS)}\n\n"
        f"Propose the next rules.md."
    )
    for attempt in range(1, 4):
        resp = client.chat.completions.create(
            model=GENERATOR,
            messages=[
                {"role": "system", "content": OPTIMIZER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=OPT_TEMPERATURE,
            max_tokens=OPT_MAX_TOKENS,
        )
        out = resp.choices[0].message.content or ""
        desc = re.search(r"<DESCRIPTION>(.*?)</DESCRIPTION>", out, re.DOTALL)
        rules = re.search(r"<RULES>(.*?)</RULES>", out, re.DOTALL)
        if desc and rules and rules.group(1).strip():
            return desc.group(1).strip(), rules.group(1).strip()
        print(f"  optimizer output unparseable (attempt {attempt}/3)", file=sys.stderr)
    raise RuntimeError("optimizer failed to return parseable output 3x")


# --- results log -------------------------------------------------------------
def log_result(commit: str, metrics: dict, status: str, description: str) -> None:
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(RESULTS_HEADER + "\n")
    desc = description.replace("\t", " ").replace("\n", " ").strip()
    with RESULTS_PATH.open("a") as f:
        f.write(
            f"{commit}\t{metrics['mean_p_ai']:.6f}\t{metrics['detect_rate']:.3f}\t"
            f"{metrics['n_invalid']}\t{status}\t{desc}\n"
        )


def resume_best() -> dict:
    best = None
    for line in RESULTS_PATH.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 6 or parts[4] not in ("keep", "baseline"):
            continue
        m = {"mean_p_ai": float(parts[1]), "detect_rate": float(parts[2]),
             "n_invalid": int(parts[3])}
        if best is None or m["mean_p_ai"] < best["mean_p_ai"]:
            best = m
    if best is None:
        sys.exit("results.tsv has no keep/baseline row — delete it and start fresh.")
    return best


# --- loop --------------------------------------------------------------------
def baseline() -> dict:
    print("[baseline] running evade.py with the current rules.md ...")
    metrics = run_evade()
    if metrics is None:
        sys.exit("baseline run crashed — inspect run.log")
    commit = commit_rules("baseline rules") if rules_dirty() else git_short_hash()
    log_result(commit, metrics, "baseline", "baseline - initial rules")
    snapshot_best()
    print(f"[baseline] mean_p_ai={metrics['mean_p_ai']:.6f} "
          f"detect_rate={metrics['detect_rate']:.3f} n_invalid={metrics['n_invalid']}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=0,
                    help="number of iterations (0 = loop until Ctrl-C)")
    args = ap.parse_args()

    client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")

    has_results = RESULTS_PATH.exists() and len(RESULTS_PATH.read_text().splitlines()) > 1
    if has_results:
        if rules_dirty():
            print("WARNING: rules.md has uncommitted changes; resuming anyway.",
                  file=sys.stderr)
        best = resume_best()
        print(f"[resume] best mean_p_ai so far = {best['mean_p_ai']:.6f}")
    else:
        best = baseline()

    i = 0
    while args.iters == 0 or i < args.iters:
        i += 1
        print(f"\n=== iteration {i} ===")
        try:
            description, new_rules = propose_rules(client, best)
        except RuntimeError as e:
            print(f"  {e} — skipping iteration", file=sys.stderr)
            continue
        print(f"  proposal: {description}")
        RULES_PATH.write_text(new_rules + "\n", encoding="utf-8")

        metrics = run_evade()
        if metrics is None:
            print("  CRASH — reverting (see run.log)")
            revert_rules()
            log_result(git_short_hash(), CRASH_METRICS, "crash", description)
            continue

        improved = metrics["mean_p_ai"] < best["mean_p_ai"]
        print(f"  mean_p_ai={metrics['mean_p_ai']:.6f} "
              f"detect_rate={metrics['detect_rate']:.3f} "
              f"n_invalid={metrics['n_invalid']} -> "
              f"{'KEEP' if improved else 'discard'}")
        if improved:
            commit = commit_rules(description)
            log_result(commit, metrics, "keep", description)
            snapshot_best()
            best = metrics
        else:
            revert_rules()
            log_result(git_short_hash(), metrics, "discard", description)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped.", file=sys.stderr)

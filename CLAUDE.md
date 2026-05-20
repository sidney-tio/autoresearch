# prompt-evasion

An autoresearch-style experiment for **prompt optimization**. An autonomous agent
iteratively rewrites a set of **evasion rules** that get injected into a text
generator's system prompt, trying to make AI-generated LinkedIn posts slip past
an AI-text detector.

It mirrors the parent `autoresearch` project one-for-one:

| autoresearch        | prompt-evasion                          | role                          |
|---------------------|-----------------------------------------|-------------------------------|
| `program.md`        | **this `CLAUDE.md`**                    | instructions (human edits)    |
| `train.py`          | **`rules.md`**                          | the one artifact the agent edits |
| `prepare.py`        | **`evade.py` + `seeds.py` + `utils.py`** | fixed harness (read-only)     |
| `val_bpb` (lower better) | **`mean_p_ai`** (lower better)      | the metric being optimized    |

The agent's *only* job in the loop is to edit `rules.md`. Nothing else.

## The setup

Generator: `google/gemma-4-31B-it`, served via **vLLM** as a persistent
OpenAI-compatible server (loaded once, reused across every iteration).

Detector: `fakespot-ai/roberta-base-ai-text-detection-v1`, a RoBERTa
text-classification model, loaded in-process inside `evade.py`.

To set up a new run, work with the user to:

1. **Agree on a run tag**: propose one based on today's date (e.g. `may21`). The
   branch `evade/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b evade/<tag>` from current `master`.
3. **Install dependencies** (one-time): a `uv` project with `transformers`,
   `torch`, `openai` (the API client for vLLM), `vllm`, and `langdetect`.
4. **Build the harness** (one-time, if the files do not exist): create
   `evade.py`, `seeds.py`, and `utils.py` per the **Harness spec** below. Once
   built and confirmed working, these are **read-only** for the rest of the run.
5. **Start the vLLM server** and leave it running for the whole session:
   `vllm serve google/gemma-4-31B-it --port 8000`
   The 31B model loads once; every iteration just hits the API.
6. **Initialize `rules.md`**: an empty baseline (a header comment, no rules).
7. **Initialize `results.tsv`**: create it with only the header row.
8. **Baseline run**: `uv run evade.py > run.log 2>&1`. This measures how
   detectable Gemma's posts are with *no* evasion rules. Verify the detector's
   label→P(AI) mapping looks sane (baseline posts should mostly score as AI).
9. **Confirm and go**: confirm setup looks good, then start the loop.

## What you can and cannot edit

**You CAN edit:** `rules.md` — and only `rules.md`.

**You CANNOT edit:** `evade.py`, `seeds.py`, `utils.py`. They are the fixed
harness: generation pipeline, the varied seed prompts, the detector, the quality
checks, and the metric. The metric is the ground truth — do not touch it.

## rules.md — the artifact you optimize

`rules.md` is injected **verbatim** into Gemma's system prompt, appended after a
fixed task framing that `evade.py` controls. So:

- Write `rules.md` as **direct instructions to the generator** (second person:
  "Vary your sentence length…", "Avoid…"), NOT as meta-notes to yourself.
- The fixed framing ("write exactly one LinkedIn post, output only the post
  text") lives in `evade.py` and never changes. `rules.md` is purely the
  evasion guidance layered on top.
- **Simplicity criterion**: all else equal, shorter rules are better. A long,
  bloated rule set both dilutes the generator's attention and is harder to
  reason about. A rule that earns ~0 improvement should be deleted. Removing a
  rule and getting equal-or-better results is a win.

## How the dataset stays varied

Variety lives entirely in the **user prompt**, not the system prompt. `seeds.py`
holds a fixed grid of 60 briefs built by deterministically sampling across:

- **topic** — career milestone, layoff/job search, product launch, hiring,
  lesson-from-failure, industry hot take, conference recap, mentorship,
  work-life balance, team shoutout, fundraising, side project, …
- **industry** — SaaS, fintech, healthcare, e-commerce, manufacturing,
  education, marketing, biotech, logistics, real estate
- **persona** — junior engineer, eng manager, founder, recruiter, sales lead,
  PM, freelance consultant, C-suite exec, new grad, career switcher
- **format** — personal story, numbered listicle, contrarian take,
  announcement, question-to-audience, before/after, gratitude note
- **tone** — earnest, analytical, casual/witty, vulnerable, celebratory
- **target length** — short (~80 words), medium (~180), long (~320)

The 60 briefs and the generation sampling seed are **fixed** across iterations,
so the only thing that changes run-to-run is `rules.md` — making `mean_p_ai`
directly comparable between experiments (the same role `val_bpb` plays in
autoresearch).

## Quality checks (lightweight)

A post that evades the detector by being garbled is not a win. After generation,
`evade.py` runs cheap checks on each post:

- word count within `[40, 400]`
- detectable English (`langdetect`) and an ASCII ratio above ~0.9
- not degenerate repetition (unique-word ratio above a floor; no duplicated lines)
- non-empty after `clean_text`

Any post failing a check is marked **invalid** and assigned `P(AI) = 1.0` (max
penalty). This means the optimizer cannot win by producing junk — junk raises
`mean_p_ai`. `n_invalid` is reported every run; watch it.

Note: `clean_text` (in `utils.py`) applies Unicode NFKC normalization and strips
zero-width / control characters before the text reaches the detector. Encoding
tricks (invisible characters, homoglyphs) are therefore neutralized — evasion
must be **linguistic**.

## The metric

For each post, `P(AI)` is the detector's probability that the text is
AI-generated (`score` if the predicted label is the AI class, else `1 - score`).

- **`mean_p_ai`** — mean `P(AI)` across all 60 posts, including invalid-post
  penalties. **This is the primary metric. Lower is better.** The loop advances
  on this.
- **`detect_rate`** — fraction of posts with `P(AI) > 0.5`. Reported for
  interpretability; not the optimization target.

## Output format

When `evade.py` finishes it prints a summary like:

```
---
mean_p_ai:     0.612345
detect_rate:   0.583
n_invalid:     2
n_posts:       60
generator:     google/gemma-4-31B-it
detector:      fakespot-ai/roberta-base-ai-text-detection-v1
```

Extract the key lines from the log:

```
grep "^mean_p_ai:\|^detect_rate:\|^n_invalid:" run.log
```

`evade.py` also writes `last_run.jsonl` (one row per post: brief, generated
text, `p_ai`, `valid`). **Read this between iterations** — it is your research
feedback. Look at which posts still score high and why, and let that drive the
next rule edit.

## Logging results

When an experiment is done, append a row to `results.tsv` (tab-separated, NOT
comma-separated — commas break in descriptions). Do **not** commit `results.tsv`;
leave it untracked.

The TSV has a header row and 6 columns:

```
commit	mean_p_ai	detect_rate	n_invalid	status	description
```

1. git commit hash (short, 7 chars)
2. `mean_p_ai` achieved (e.g. `0.612345`) — use `1.000000` for crashes
3. `detect_rate` (e.g. `0.583`) — use `1.000` for crashes
4. `n_invalid` (integer count of posts failing quality checks)
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	mean_p_ai	detect_rate	n_invalid	status	description
a1b2c3d	0.871000	0.917	0	keep	baseline - no rules
b2c3d4e	0.794000	0.833	1	keep	vary sentence length and burstiness
c3d4e5f	0.880000	0.933	0	discard	add corporate buzzwords
d4e5f6e	1.000000	1.000	41	discard	force lowercase - too many invalid
```

## The experiment loop

The run lives on a dedicated branch (e.g. `evade/may21`).

LOOP FOREVER:

1. Look at the git state: the branch/commit you are on.
2. Read `last_run.jsonl` from the previous run — see which posts are still
   flagged and form a hypothesis.
3. Edit `rules.md` with one experimental change to the evasion rules.
4. `git commit`.
5. Run the experiment: `uv run evade.py > run.log 2>&1` (redirect everything —
   do NOT use `tee` or let output flood your context).
6. Read the result: `grep "^mean_p_ai:\|^detect_rate:\|^n_invalid:" run.log`.
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to
   read the stack trace and attempt a fix. If it is something dumb (typo,
   server not reachable) fix and re-run; if the idea is fundamentally broken,
   log `crash` and move on.
8. Record the result in `results.tsv` (keep it untracked).
9. If `mean_p_ai` improved (lower), **advance** the branch — keep the commit.
10. If `mean_p_ai` is equal or worse, `git reset --hard` back to where you
    started this iteration.

**Timeout**: a normal iteration is a couple of minutes. If a run exceeds 15
minutes, kill it and treat it as a crash (discard and revert).

**NEVER STOP**: once the loop has begun (after setup), do NOT pause to ask the
human whether to continue. Do NOT ask "should I keep going?" or "is this a good
stopping point?". The human may be asleep and expects you to run *indefinitely*
until manually stopped. If you run out of ideas, think harder: re-read
`last_run.jsonl` for patterns, reason about what makes text read as
human-written (burstiness, perplexity, idiosyncrasy, concrete personal detail,
imperfect punctuation), combine previous near-misses, try a more radical
rewrite. The loop runs until the human interrupts you, period.

## Harness spec (build once, then read-only)

### `utils.py`
- `clean_text(text: str) -> str`: Unicode NFKC normalization, strip zero-width
  and control characters, collapse whitespace, `.strip()`. This is the exact
  preprocessing applied before detection.

### `seeds.py`
- Defines the dimension lists above (topic, industry, persona, format, tone,
  length).
- With a **fixed** `random.seed`, deterministically samples 60 combinations and
  renders each into a natural-language brief string.
- Exports `SEEDS` (list of 60 brief strings) and `N_SEEDS = 60`.

### `evade.py`
- Connects to the vLLM OpenAI-compatible endpoint at `http://localhost:8000/v1`
  using the `openai` client.
- Reads `rules.md`; builds the system prompt as
  `BASE_FRAMING + "\n\n" + rules_text`, where `BASE_FRAMING` is a fixed string
  defined in `evade.py` (the task: write exactly one LinkedIn post, output only
  the post text).
- For each of the 60 briefs: calls Gemma with `system = SYSTEM_PROMPT`,
  `user = brief`, a **fixed** sampling `seed` and `temperature`, bounded
  `max_tokens`.
- Applies `clean_text` and the lightweight quality checks to each post.
- Scores valid posts with the `fakespot-ai/roberta-base-ai-text-detection-v1`
  `text-classification` pipeline (batched). Use `top_k=None` to read the full
  label distribution so `P(AI)` is unambiguous. Invalid posts get `P(AI) = 1.0`.
- Computes and prints the summary block above.
- Writes `last_run.jsonl` with one row per post:
  `{brief, post, p_ai, valid}`.

## Project structure

```
CLAUDE.md        — these instructions (human edits)
rules.md         — evasion rules injected into Gemma's system prompt (agent edits)
evade.py         — generation + detection + metric (read-only)
seeds.py         — the 60 varied briefs (read-only)
utils.py         — clean_text (read-only)
results.tsv      — experiment log (untracked, do not commit)
last_run.jsonl   — per-post output of the most recent run
run.log          — stdout/stderr of the most recent run
```

## Scope note

This is adversarial robustness research on an AI-text detector: probing where
`fakespot-ai/roberta-base-ai-text-detection-v1` is weak by optimizing a
generator prompt against it. Keep results and rule sets in this repo for
analysis; the point is to understand detector failure modes.

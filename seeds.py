"""seeds.py — the fixed grid of 60 varied LinkedIn-post briefs. READ-ONLY harness file.

Variety lives here, in the *user* prompt. The 60 briefs and the sampling seed are
fixed, so mean_p_ai stays comparable across iterations — the only thing that
changes between runs is rules.md.
"""

import random

TOPICS = [
    "a career milestone or work anniversary",
    "navigating a layoff or an active job search",
    "a product launch or feature announcement",
    "a hiring announcement for the team",
    "a lesson learned from a project that failed",
    "a contrarian take on a current industry trend",
    "a recap of a conference or industry event",
    "mentorship and helping others grow",
    "work-life balance and avoiding burnout",
    "a shoutout to a teammate or colleague",
    "a fundraising round or business milestone",
    "a personal side project",
    "reflections on a recent promotion",
    "advice for early-career professionals",
    "thoughts on remote versus in-office work",
]

INDUSTRIES = [
    "SaaS", "fintech", "healthcare", "e-commerce", "manufacturing",
    "education", "marketing", "biotech", "logistics", "real estate",
]

PERSONAS = [
    "a junior software engineer",
    "an engineering manager",
    "a startup founder",
    "a technical recruiter",
    "a sales lead",
    "a product manager",
    "a freelance consultant",
    "a C-suite executive",
    "a recent graduate",
    "a mid-career switcher",
]

FORMATS = [
    "a personal story with a clear narrative arc",
    "a numbered list of takeaways",
    "a contrarian hot take",
    "a straightforward announcement",
    "a question posed to the audience",
    "a before-and-after comparison",
    "a short gratitude note",
]

TONES = [
    "earnest and inspirational",
    "analytical and measured",
    "casual and witty",
    "vulnerable and honest",
    "celebratory and upbeat",
]

LENGTHS = [
    ("short", 80),
    ("medium", 180),
    ("long", 320),
]

N_SEEDS = 60
_SAMPLING_SEED = 20260521


def _build_seeds() -> list[str]:
    rng = random.Random(_SAMPLING_SEED)
    briefs: list[str] = []
    for _ in range(N_SEEDS):
        persona = rng.choice(PERSONAS)
        industry = rng.choice(INDUSTRIES)
        topic = rng.choice(TOPICS)
        fmt = rng.choice(FORMATS)
        tone = rng.choice(TONES)
        length_name, word_target = rng.choice(LENGTHS)
        briefs.append(
            f"Write a LinkedIn post as {persona} working in {industry}. "
            f"The post is about {topic}. "
            f"Structure it as {fmt}. "
            f"Make the tone {tone}. "
            f"Aim for roughly {word_target} words ({length_name})."
        )
    return briefs


SEEDS = _build_seeds()
assert len(SEEDS) == N_SEEDS


if __name__ == "__main__":
    for i, brief in enumerate(SEEDS):
        print(f"[{i:02d}] {brief}")

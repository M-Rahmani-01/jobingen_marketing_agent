"""
agent/nodes/trend_scorer.py

Layer 1 — "The Editor-in-Chief"

Scores each Theme using the formula from Spec Section 7.1:

    theme_score = w1 * audience_relevance
                + w2 * trend_velocity
                + w3 * brand_fit
                - penalty(risk)

audience_relevance and brand_fit need judgment, so we ask the LLM for
those two numbers per theme (one lightweight call). trend_velocity comes
from the real data (theme.heat, computed in signal_synthesizer from real
evidence). The final combination is pure math — deterministic, and easy
to unit test.

"No strong theme" is a valid, healthy outcome (spec Section 1, 5, 7.1) —
this function returns None when nothing clears the threshold.
"""

from agent.state import Theme
from agent.adapters.llm import generate_json


# Default weights + threshold — spec Section 7.1
WEIGHTS = {"relevance": 0.45, "velocity": 0.30, "brand_fit": 0.25}
TRIGGER_THRESHOLD = 0.62

SYSTEM_INSTRUCTION = """You score how well themes fit a specific audience
and brand. JobInGen is an AI career platform for students and early-career
job seekers in India (resume builder, AI mock interview, career copilot,
curated jobs, bootcamps). Audience is anxious about placements, distrusts
corporate HR-speak and "guaranteed job" scams.

For each numbered theme, return:
- audience_relevance (0.0-1.0): how much THIS audience cares about it
- brand_fit (0.0-1.0): how credibly JobInGen can speak to it
- risk_penalty (0.0-0.3): deduct points if the theme is off-brand,
  sensitive, or invites false claims (e.g. medical, legal, guaranteed
  outcomes)

Output strict JSON only, matching this exact shape:
{
  "scores": [
    {"index": 0, "audience_relevance": 0.0, "brand_fit": 0.0, "risk_penalty": 0.0}
  ]
}"""


def run_trend_scorer(themes: list[Theme]) -> tuple[Theme | None, list[dict]]:
    """
    Returns (chosen_theme, score_breakdown).
    chosen_theme is None if nothing clears TRIGGER_THRESHOLD — a valid,
    healthy "no post this run" outcome.
    """
    if not themes:
        return None, []

    numbered = "\n".join(
        f"[{i}] {t.label} (heat/velocity={t.heat})"
        for i, t in enumerate(themes)
    )

    result = generate_json(
        system_instruction=SYSTEM_INSTRUCTION,
        user_prompt=f"Score these {len(themes)} themes:\n\n{numbered}",
    )

    breakdown = []
    for raw_score in result.get("scores", []):
        idx = raw_score.get("index")
        if idx is None or not (0 <= idx < len(themes)):
            continue

        theme = themes[idx]
        relevance = raw_score.get("audience_relevance", 0.0)
        brand_fit = raw_score.get("brand_fit", 0.0)
        risk_penalty = raw_score.get("risk_penalty", 0.0)
        velocity = theme.heat  # real data, not LLM-judged

        final_score = (
            WEIGHTS["relevance"] * relevance
            + WEIGHTS["velocity"] * velocity
            + WEIGHTS["brand_fit"] * brand_fit
            - risk_penalty
        )

        breakdown.append({
            "label": theme.label,
            "audience_relevance": relevance,
            "trend_velocity": velocity,
            "brand_fit": brand_fit,
            "risk_penalty": risk_penalty,
            "final_score": round(final_score, 3),
        })

    if not breakdown:
        return None, []

    breakdown.sort(key=lambda b: b["final_score"], reverse=True)
    top = breakdown[0]

    print("[trend_scorer] Score breakdown:")
    for b in breakdown:
        print(f"  {b['final_score']:.3f}  {b['label']}")

    if top["final_score"] < TRIGGER_THRESHOLD:
        print(f"[trend_scorer] Nothing cleared threshold {TRIGGER_THRESHOLD} — no post this run.")
        return None, breakdown

    winner = next(t for t in themes if t.label == top["label"])
    print(f"[trend_scorer] Chosen theme: '{winner.label}' (score={top['final_score']})")
    return winner, breakdown


# Quick manual test — run this file directly
if __name__ == "__main__":
    from agent.nodes.trend_scout import run_trend_scout
    from agent.nodes.signal_synthesizer import run_signal_synthesizer

    signal = run_trend_scout()
    themes = run_signal_synthesizer(signal)
    chosen, scores = run_trend_scorer(themes)

    if chosen:
        print(f"\nWINNER: {chosen.label}")
    else:
        print("\nNo strong theme this run.")
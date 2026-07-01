"""
agent/nodes/audience_modeler.py

Layer 2 — "The Empath"

Takes the chosen Theme and names the SPECIFIC feeling underneath it —
one sharp tension, not a generic list. This tension is what the
Copywriter's hook and "emotional resonance" will be built on later.
"""

from agent.state import Theme
from agent.adapters.llm import generate_json


SYSTEM_INSTRUCTION = """You understand a stressed final-year student or
fresh job seeker in India deeply. Given a trending theme and real
evidence of what people are saying, name the ONE real feeling underneath
it — the specific anxiety, frustration, or quiet hope. Not a list of
feelings — one sharp, specific tension a human would immediately
recognize in themselves.

Output strict JSON only, matching this exact shape:
{"tension": "one or two sentences naming the specific feeling"}"""


def run_audience_modeler(theme: Theme) -> str:
    evidence_text = "\n".join(f"- {e.text}" for e in theme.evidence)

    user_prompt = (
        f"Theme: {theme.label}\n\n"
        f"Real evidence from the audience:\n{evidence_text}\n\n"
        "What is the one real tension here?"
    )

    result = generate_json(
        system_instruction=SYSTEM_INSTRUCTION,
        user_prompt=user_prompt,
    )

    tension = result.get("tension", "").strip()
    print(f"[audience_modeler] Tension: {tension}")
    return tension


# Quick manual test — run this file directly
if __name__ == "__main__":
    from agent.nodes.trend_scout import run_trend_scout
    from agent.nodes.signal_synthesizer import run_signal_synthesizer
    from agent.nodes.trend_scorer import run_trend_scorer

    signal = run_trend_scout()
    themes = run_signal_synthesizer(signal)
    chosen, _ = run_trend_scorer(themes)

    if chosen:
        tension = run_audience_modeler(chosen)
        print(f"\nTheme: {chosen.label}")
        print(f"Tension: {tension}")
    else:
        print("\nNo strong theme this run — nothing to model.")
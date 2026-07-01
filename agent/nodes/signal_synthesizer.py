"""
agent/nodes/signal_synthesizer.py

Layer 1 — "The Analyst"

Clusters raw SignalItems into a handful of coherent Themes.

IMPORTANT DESIGN CHOICE (traceability):
We do NOT ask the LLM to write out evidence text freely — that risks it
paraphrasing sloppily or inventing details. Instead, we number the input
items and ask the LLM to return which item INDICES belong to each theme.
Our own code then builds the real Theme.evidence list directly from the
original SignalItem objects. This guarantees every theme is traceable to
a real source, per the spec's "never invents a quote" guardrail.
"""

from agent.state import SignalItem, Theme
from agent.adapters.llm import generate_json


SYSTEM_INSTRUCTION = """You cluster raw online chatter into clear themes.
For each theme, identify which of the numbered input items belong to it.
Note how hot each theme is (0.0 to 1.0) based on how many strong items
support it and how emotionally intense the language is.
A theme needs at least 2 supporting items to be valid.
Output strict JSON only, matching this exact shape:
{
  "themes": [
    {"label": "short theme name", "heat": 0.0, "item_indices": [0, 3, 7]}
  ]
}
Do not include any text outside the JSON object."""


def run_signal_synthesizer(raw_signal: list[SignalItem]) -> list[Theme]:
    if not raw_signal:
        return []

    # Build a numbered list the LLM can reference by index
    numbered_items = "\n".join(
        f"[{i}] (source={item.source}, score={item.score}) {item.text}"
        for i, item in enumerate(raw_signal)
    )

    user_prompt = f"Cluster these {len(raw_signal)} items into themes:\n\n{numbered_items}"

    result = generate_json(
        system_instruction=SYSTEM_INSTRUCTION,
        user_prompt=user_prompt,
    )

    themes = []
    for raw_theme in result.get("themes", []):
        indices = raw_theme.get("item_indices", [])
        # Build evidence from REAL SignalItems only — never from LLM text
        evidence = [raw_signal[i] for i in indices if 0 <= i < len(raw_signal)]

        if len(evidence) < 2:
            continue  # skip themes with too little real support

        themes.append(Theme(
            label=raw_theme.get("label", "Untitled theme"),
            evidence=evidence,
            heat=raw_theme.get("heat", 0.0),
        ))

    print(f"[signal_synthesizer] Produced {len(themes)} themes from {len(raw_signal)} items.")
    return themes


# Quick manual test — run this file directly
if __name__ == "__main__":
    from agent.nodes.trend_scout import run_trend_scout

    signal = run_trend_scout()
    themes = run_signal_synthesizer(signal)

    for t in themes:
        print(f"\n--- {t.label} (heat={t.heat}) ---")
        for e in t.evidence:
            print(f"  - {e.text[:70]}...")
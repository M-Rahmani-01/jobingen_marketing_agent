"""
agent/nodes/platform_strategist.py

Layer 2 -- "The Native Speaker"

Takes the ONE angle from angle_selector.py and translates it into TWO
separate creative briefs -- one for LinkedIn, one for Instagram -- because
the two platforms reward completely different content shapes (Spec
Section 8). This node does NOT write the actual post; it writes the
instructions the Copywriter (next node) will follow.
"""

from agent.state import Angle, PlatformBrief
from agent.adapters.llm import generate_json


SYSTEM_INSTRUCTION = """You know LinkedIn and Instagram are different
worlds. Translate the given angle into a brief native to each platform --
the hook mechanism, ideal format, caption length, and CTA that platform
rewards. Do not produce one post and shrink it for the other platform;
produce the right post for each, based on these rules:

LINKEDIN:
- Reader mindset: "Am I doing my career right?" -- aspirational, professional, slightly anxious
- What stops the scroll: a contrarian insight, a number, a hard hiring truth (text-led)
- Hook lives in: the first 2 lines before "...more"
- Ideal format: text post or document/carousel, insight-dense
- Caption length: can be a long mini-essay if it earns it
- CTA that works: "comment / repost / follow for more" (drives reach)
- Hashtags: 3-5, professional

INSTAGRAM:
- Reader mindset: "Entertain or inspire me fast" -- casual scroll, low patience
- What stops the scroll: a bold visual or a relatable feeling (image-led)
- Hook lives in: the image itself; caption is secondary
- Ideal format: carousel or one bold graphic, emotionally led
- Caption length: short, punchy, front-loaded
- CTA that works: "save this / send to a friend job-hunting"
- Hashtags: 8-15, niche + discovery

Output strict JSON only, matching this exact shape:
{
  "linkedin": {
    "platform": "linkedin",
    "format": "...",
    "hook_style": "...",
    "caption_length": "...",
    "cta_type": "...",
    "tone": "..."
  },
  "instagram": {
    "platform": "instagram",
    "format": "...",
    "hook_style": "...",
    "caption_length": "...",
    "cta_type": "...",
    "tone": "..."
  }
}"""


def run_platform_strategist(angle: Angle) -> dict[str, PlatformBrief]:
    """
    Returns a dict with two PlatformBrief objects -- one per platform --
    built from the single angle passed in.
    """
    user_prompt = (
        f"Angle (the take): {angle.take}\n"
        f"Product tie-in: {angle.product_tie or 'none'}\n"
        f"Why only JobInGen can say this: {angle.why_only_jobingen}\n\n"
        "Produce the LinkedIn brief and the Instagram brief for this angle."
    )

    result = generate_json(
        system_instruction=SYSTEM_INSTRUCTION,
        user_prompt=user_prompt,
    )

    briefs = {
        "linkedin": PlatformBrief(**result["linkedin"]),
        "instagram": PlatformBrief(**result["instagram"]),
    }

    print(f"[platform_strategist] LinkedIn format: {briefs['linkedin'].format}")
    print(f"[platform_strategist] Instagram format: {briefs['instagram'].format}")
    return briefs


# Quick manual test -- run this file directly, chains the full pipeline
# so far (Week 1 nodes + audience_modeler + angle_selector) and feeds
# the resulting angle into this node.
if __name__ == "__main__":
    from agent.nodes.trend_scout import run_trend_scout
    from agent.nodes.signal_synthesizer import run_signal_synthesizer
    from agent.nodes.trend_scorer import run_trend_scorer
    from agent.nodes.audience_modeler import run_audience_modeler
    from agent.nodes.angle_selector import run_angle_selector

    signal = run_trend_scout()
    themes = run_signal_synthesizer(signal)
    chosen, _ = run_trend_scorer(themes)

    if chosen:
        tension = run_audience_modeler(chosen)
        angle = run_angle_selector(chosen, tension)

        if angle:
            briefs = run_platform_strategist(angle)

            print("\n--- LINKEDIN BRIEF ---")
            print(briefs["linkedin"].model_dump_json(indent=2))

            print("\n--- INSTAGRAM BRIEF ---")
            print(briefs["instagram"].model_dump_json(indent=2))
        else:
            print("\nNo differentiated angle found this run.")
    else:
        print("\nNo strong theme this run.")
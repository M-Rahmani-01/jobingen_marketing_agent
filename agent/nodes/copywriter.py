"""
agent/nodes/copywriter.py

Layer 3 -- "Gen, the Voice"

Takes the platform briefs from platform_strategist.py (the INSTRUCTIONS for
how to write each platform's post) and actually WRITES the post -- hooks,
body/slide copy, caption, hashtags, CTA, and alt-text. This is the
highest-value prompt in the whole system: it's where the JobInGen voice
comes alive.

Runs once PER PLATFORM (LinkedIn and Instagram get separately written --
never one post copy-pasted and reshaped, because their briefs are
genuinely different).
"""

from agent.state import Theme, Angle, PlatformBrief, PostDraft
from agent.adapters.llm import generate_json


SYSTEM_INSTRUCTION = """You are Gen, JobInGen's voice -- a sharp senior
who has survived placements and is on the reader's side. Write what a
stressed student would stop scrolling for: a real hook, genuine value,
zero cringe, zero corporate filler, zero guarantees.

Rules:
- Use the audience's own words and feelings where it fits naturally --
  don't sound like a brand, sound like a person who gets it.
- Never promise guaranteed jobs, interviews, or outcomes.
- Always include a clear save/share/comment CTA matching the brief's cta_type.
- Always write accessible alt_text describing what the visual should show
  (this is required for every image -- accessibility, not optional).
- Follow the brief's format, hook_style, caption_length, and tone exactly.

Output strict JSON only, matching this exact shape:
{
  "hook_options": ["hook option 1", "hook option 2", "hook option 3"],
  "body": ["slide or paragraph 1", "slide or paragraph 2", "..."],
  "caption": "the full caption text",
  "hashtags": ["tag1", "tag2"],
  "cta": "the specific call to action line",
  "alt_text": "a clear description of what the visual should show"
}"""


def run_copywriter(
    theme: Theme,
    tension: str,
    angle: Angle,
    briefs: dict[str, PlatformBrief],
    feedback: dict[str, str] | None = None,
) -> dict[str, PostDraft]:
    """
    Writes one PostDraft per platform, using that platform's brief plus
    the shared angle, tension, and real audience language from the
    theme's evidence pack.
    """
    # Pull a few real (paraphrased) audience quotes to hand the writer --
    # this is the "real language" ingredient the spec calls out in 6.7.
    evidence_snippets = [item.text for item in theme.evidence[:3]]

    drafts: dict[str, PostDraft] = {}

    for platform, brief in briefs.items():
        user_prompt = (
            f"Platform: {platform}\n"
            f"Brief -- format: {brief.format}\n"
            f"Brief -- hook_style: {brief.hook_style}\n"
            f"Brief -- caption_length: {brief.caption_length}\n"
            f"Brief -- cta_type: {brief.cta_type}\n"
            f"Brief -- tone: {brief.tone}\n\n"
            f"Angle (the take): {angle.take}\n"
            f"Product tie-in: {angle.product_tie or 'none'}\n"
            f"Audience tension: {tension}\n"
            f"Real audience language (paraphrased, for flavor only -- "
            f"never copy verbatim): {evidence_snippets}\n\n"
            "Write the full post for this platform, following the brief exactly."
        )

        # This ONLY adds retry feedback to the prompt text -- it does NOT
        # gate whether we call the LLM. The API call below always runs,
        # once per platform, whether or not feedback was given.
        if feedback and platform in feedback:
            user_prompt += (
                f"\n\nIMPORTANT: a previous draft for this platform failed "
                f"editorial review. Specific feedback to fix: {feedback[platform]}\n"
                f"Rewrite to directly address this feedback."
            )

        result = generate_json(
            system_instruction=SYSTEM_INSTRUCTION,
            user_prompt=user_prompt,
        )

        draft = PostDraft(
            platform=platform,
            hook_options=result.get("hook_options", []),
            body=result.get("body", []),
            caption=result.get("caption", ""),
            hashtags=result.get("hashtags", []),
            cta=result.get("cta", ""),
            alt_text=result.get("alt_text", ""),
        )
        drafts[platform] = draft

        print(f"[copywriter] {platform} hook: {draft.hook_options[0] if draft.hook_options else '(none)'}")

    return drafts


# Quick manual test -- run this file directly.
#
# FIRST run ever: no fixture exists yet, so it runs the full pipeline
# (trend_scout -> ... -> platform_strategist) ONE time, then saves the
# result to tests/fixtures/sample_run.json.
#
# EVERY run after that: it loads straight from that saved fixture --
# ZERO API calls spent on upstream nodes. Only copywriter.py itself
# uses your quota. This is how you avoid hitting the 429 quota error
# every time you're just testing THIS node.
#
# If you ever want to force a fresh live run (e.g. you changed an
# upstream node and want a new fixture), just delete
# tests/fixtures/sample_run.json and run this file again.
if __name__ == "__main__":
    from tests.fixtures import save_fixture, load_fixture

    try:
        chosen, tension, angle, briefs = load_fixture()
    except FileNotFoundError:
        print("[copywriter] No fixture found -- running the full live pipeline once...")
        from agent.nodes.trend_scout import run_trend_scout
        from agent.nodes.signal_synthesizer import run_signal_synthesizer
        from agent.nodes.trend_scorer import run_trend_scorer
        from agent.nodes.audience_modeler import run_audience_modeler
        from agent.nodes.angle_selector import run_angle_selector
        from agent.nodes.platform_strategist import run_platform_strategist

        signal = run_trend_scout()
        themes = run_signal_synthesizer(signal)
        chosen, _ = run_trend_scorer(themes)

        if not chosen:
            print("\nNo strong theme this run. Nothing to save as a fixture.")
            chosen = None

        if chosen:
            tension = run_audience_modeler(chosen)
            angle = run_angle_selector(chosen, tension)

            if angle:
                briefs = run_platform_strategist(angle)
                save_fixture(chosen, tension, angle, briefs)
            else:
                print("\nNo differentiated angle found this run.")
                angle = None

    if chosen and angle:
        drafts = run_copywriter(chosen, tension, angle, briefs)

        for platform, draft in drafts.items():
            print(f"\n--- {platform.upper()} DRAFT ---")
            print(draft.model_dump_json(indent=2))
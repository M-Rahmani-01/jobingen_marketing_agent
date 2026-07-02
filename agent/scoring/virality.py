"""
agent/scoring/virality.py

Layer 4 -- "The Predictor"

Predicts how well a finished draft is likely to perform, scored across
5 factors (Spec Section 11.2). This runs AFTER a draft has passed both
critics -- it's not a pass/fail gate like critics.py, it's a forecast
that goes into the Analysis Report so a human reviewer knows what to
expect and what the weakest lever is.

The 5 factors:
  1. hook_strength      -- does the first line actually stop the scroll?
  2. trend_alignment    -- how well does this ride the actual signal/theme?
  3. emotional_resonance -- does it genuinely connect with the audience tension?
  4. value_density       -- is there real substance, or just vibes?
  5. share_trigger       -- would someone actually save/send/repost this?
"""

from agent.state import Theme, Angle, PostDraft, ViralityScore
from agent.adapters.llm import generate_json


SYSTEM_INSTRUCTION = """You predict how well a social post is likely to
perform, BEFORE it's posted -- based on pattern knowledge of what
actually drives engagement on LinkedIn and Instagram for career content.

Score each factor 0-20 (5 factors, 100 total):

1. hook_strength -- Does the first line/hook actually stop a scroll?
   Generic openers score low. Specific, surprising, or number-led hooks score high.

2. trend_alignment -- Does this genuinely ride the theme/signal it's based
   on, or does it feel bolted-on to a trend that doesn't really fit?

3. emotional_resonance -- Does this actually connect with the stated
   audience tension, or does it stay surface-level/generic?

4. value_density -- Is there real, specific substance (a concrete insight,
   a specific mechanism, a real number) or just motivational filler?

5. share_trigger -- Would a real person actually save this, send it to a
   friend, or repost it -- or is it just "nice" with no reason to share?

Be honest and specific -- a score of 100 should be rare. Most solid posts
land in the 60-80 range. Identify the SINGLE weakest lever, even on a
good post -- there's always a most-improvable factor.

Output strict JSON only, matching this exact shape:
{
  "total": 74,
  "hook_strength": 16,
  "trend_alignment": 15,
  "emotional_resonance": 14,
  "value_density": 15,
  "share_trigger": 14,
  "weakest_lever": "share_trigger -- the CTA doesn't give anyone a concrete reason to send this to someone else"
}
"total" must equal the sum of the 5 factor scores."""


def run_virality_scorer(theme: Theme, angle: Angle, draft: PostDraft) -> ViralityScore:
    """
    Scores ONE platform's finished draft across the 5 virality factors.
    Call this once per platform (same pattern as critics.py).
    """
    user_prompt = (
        f"Theme: {theme.label}\n"
        f"Angle: {angle.take}\n\n"
        f"DRAFT ({draft.platform}):\n"
        f"hook_options: {draft.hook_options}\n"
        f"body: {draft.body}\n"
        f"caption: {draft.caption}\n"
        f"hashtags: {draft.hashtags}\n"
        f"cta: {draft.cta}\n\n"
        "Predict this draft's performance across the 5 factors."
    )

    result = generate_json(
        system_instruction=SYSTEM_INSTRUCTION,
        user_prompt=user_prompt,
        temperature=0.3,  # forecasting wants consistency, not creative variance
    )

    score = ViralityScore(**result)
    print(f"[virality] {draft.platform}: {score.total}/100 "
          f"(weakest lever: {score.weakest_lever})")
    return score


def run_virality_for_all(
    theme: Theme,
    angle: Angle,
    drafts: dict[str, PostDraft],
) -> dict[str, ViralityScore]:
    """Runs the scorer for every platform's draft. Returns a flat dict
    keyed by platform, e.g. {"linkedin": ViralityScore, "instagram": ViralityScore}."""
    return {
        platform: run_virality_scorer(theme, angle, draft)
        for platform, draft in drafts.items()
    }


# Quick manual test -- loads the saved fixture (zero upstream API calls),
# runs the full retry-loop pipeline live to get FINAL passing drafts,
# then scores virality on those.
if __name__ == "__main__":
    from tests.fixtures import load_fixture
    from agent.orchestrator import run_pipeline_with_retries

    chosen, tension, _, _ = load_fixture()
    result = run_pipeline_with_retries(chosen, tension)

    if not result["success"]:
        print(f"\nPipeline did not produce a passing draft: {result['reason']}")
    else:
        scores = run_virality_for_all(result["theme"], result["angle"], result["drafts"])

        print("\n--- VIRALITY REPORT ---")
        for platform, score in scores.items():
            print(f"\n{platform}:")
            print(score.model_dump_json(indent=2))
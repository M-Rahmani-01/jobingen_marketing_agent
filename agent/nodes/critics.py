"""
agent/nodes/critics.py

Layer 4 -- "The Editors"

Two independent critics review each platform's draft:

1. PLATFORM-FIT CRITIC -- did the Copywriter actually follow the brief?
   (right format, right hook style, right caption length, right CTA type)
   If this fails, the fix is almost always in the WRITING -- so it routes
   back to copywriter.py.

2. BRAND+SAFETY CRITIC -- is this something JobInGen can actually post?
   (no guarantees, no hype, no cringe, genuinely differentiated -- not a
   generic career-page post wearing a JobInGen logo)
   If this fails, the problem usually isn't the writing, it's the IDEA
   underneath it -- so it routes back to angle_selector.py.

This is the spec's "layer-routed retry" concept: a failure doesn't
trigger "regenerate everything," it triggers "regenerate the ONE node
that's actually responsible."
"""

from agent.state import Angle, PlatformBrief, PostDraft, Critique
from agent.adapters.llm import generate_json


PLATFORM_FIT_SYSTEM_INSTRUCTION = """You are a strict platform-fit editor.
You check ONLY whether a draft actually follows the creative brief it was
given -- not whether the idea is good, just whether the EXECUTION matches
the instructions.

Score against this rubric (each worth up to 20 points, 100 total):
1. Format match -- does the body/structure match the brief's format?
2. Hook strength -- does the first hook option match the brief's hook_style?
3. Caption length -- does the caption's length match the brief's caption_length?
4. CTA fit -- does the CTA match the brief's cta_type?
5. Tone match -- does the writing voice match the brief's tone?

A score of 70+ passes. Below 70 fails.

Output strict JSON only, matching this exact shape:
{
  "passed": true,
  "score": 85,
  "feedback": "specific, actionable feedback -- if failed, say EXACTLY what to fix",
  "failed_node": null
}
If passed is false, set "failed_node" to "copywriter" (this is always a
writing-execution problem when this critic fails, not an idea problem)."""


BRAND_SAFETY_SYSTEM_INSTRUCTION = """You are a strict brand-safety editor
for JobInGen, an AI career platform for Indian students and early-career
job seekers. You check whether this draft is something JobInGen can
actually post -- not the writing quality, the SUBSTANCE and RISK.

Score against this rubric (each worth up to 20 points, 100 total):
1. Zero guarantees -- no promise of jobs, interviews, or outcomes
2. Zero hype/cringe -- no fake urgency, no "guru" tone, no empty motivation
3. Genuinely differentiated -- could NOT appear on a generic career page
   or competitor feed as-is
4. Peer-to-peer voice -- speaks like a senior who gets it, not a brand
   talking down to candidates
5. Accessible -- alt_text is present and actually describes the visual

A score of 70+ passes. Below 70 fails.

Output strict JSON only, matching this exact shape:
{
  "passed": true,
  "score": 90,
  "feedback": "specific, actionable feedback -- if failed, say EXACTLY what is risky or generic",
  "failed_node": null
}
If passed is false, set "failed_node" to "angle_selector" (a brand/safety
failure almost always means the underlying ANGLE is the problem, not the
wording -- so the fix has to happen upstream, at the idea, not the copy)."""


def _run_platform_fit_critic(brief: PlatformBrief, draft: PostDraft) -> Critique:
    user_prompt = (
        f"BRIEF:\n"
        f"format: {brief.format}\n"
        f"hook_style: {brief.hook_style}\n"
        f"caption_length: {brief.caption_length}\n"
        f"cta_type: {brief.cta_type}\n"
        f"tone: {brief.tone}\n\n"
        f"DRAFT:\n"
        f"hook_options: {draft.hook_options}\n"
        f"body: {draft.body}\n"
        f"caption: {draft.caption}\n"
        f"cta: {draft.cta}\n\n"
        "Score this draft against the brief."
    )
    result = generate_json(
        system_instruction=PLATFORM_FIT_SYSTEM_INSTRUCTION,
        user_prompt=user_prompt,
        temperature=0.3,  # scoring wants consistency, not creativity
    )
    return Critique(**result)


def _run_brand_safety_critic(angle: Angle, draft: PostDraft) -> Critique:
    user_prompt = (
        f"ANGLE:\n"
        f"take: {angle.take}\n"
        f"product_tie: {angle.product_tie or 'none'}\n"
        f"why_only_jobingen: {angle.why_only_jobingen}\n\n"
        f"DRAFT:\n"
        f"hook_options: {draft.hook_options}\n"
        f"body: {draft.body}\n"
        f"caption: {draft.caption}\n"
        f"cta: {draft.cta}\n"
        f"alt_text: {draft.alt_text}\n\n"
        "Score this draft for brand and safety risk."
    )
    result = generate_json(
        system_instruction=BRAND_SAFETY_SYSTEM_INSTRUCTION,
        user_prompt=user_prompt,
        temperature=0.3,
    )
    return Critique(**result)


def run_critics(
    angle: Angle,
    briefs: dict[str, PlatformBrief],
    drafts: dict[str, PostDraft],
) -> dict[str, Critique]:
    """
    Runs BOTH critics against EACH platform's draft. Returns a flat dict
    keyed like "linkedin_platform_fit", "linkedin_brand_safety",
    "instagram_platform_fit", "instagram_brand_safety".
    """
    critiques: dict[str, Critique] = {}

    for platform, draft in drafts.items():
        brief = briefs[platform]

        platform_fit = _run_platform_fit_critic(brief, draft)
        critiques[f"{platform}_platform_fit"] = platform_fit
        print(f"[critics] {platform} platform-fit: {platform_fit.score} "
              f"({'PASS' if platform_fit.passed else 'FAIL -> ' + str(platform_fit.failed_node)})")

        brand_safety = _run_brand_safety_critic(angle, draft)
        critiques[f"{platform}_brand_safety"] = brand_safety
        print(f"[critics] {platform} brand+safety: {brand_safety.score} "
              f"({'PASS' if brand_safety.passed else 'FAIL -> ' + str(brand_safety.failed_node)})")

    return critiques


# Quick manual test -- loads the saved fixture (theme/tension/angle/briefs,
# zero API calls), runs copywriter.py live to get fresh drafts, then runs
# both critics against those drafts.
if __name__ == "__main__":
    from tests.fixtures import load_fixture
    from agent.nodes.copywriter import run_copywriter

    chosen, tension, angle, briefs = load_fixture()
    drafts = run_copywriter(chosen, tension, angle, briefs)
    critiques = run_critics(angle, briefs, drafts)

    print("\n--- FULL CRITIQUE REPORT ---")
    for key, critique in critiques.items():
        print(f"\n{key}:")
        print(critique.model_dump_json(indent=2))
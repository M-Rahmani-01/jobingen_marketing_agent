"""
agent/orchestrator.py

The layer-routed retry loop -- Week 3's core idea, tied together.

Runs: angle_selector -> platform_strategist -> copywriter -> critics

If a critic fails, this does NOT restart the whole pipeline. It re-runs
ONLY the node the critic pointed at:
  - brand+safety failure  -> re-run angle_selector (the IDEA is the problem)
  - platform-fit failure  -> re-run copywriter for that ONE platform only
                             (the WRITING is the problem)

Each node type is capped at MAX_RETRIES. If it still can't pass after
that many tries, the run fails CLEANLY with a logged reason -- it never
loops forever.
"""

from agent.state import Theme
from agent.nodes.angle_selector import run_angle_selector
from agent.nodes.platform_strategist import run_platform_strategist
from agent.nodes.copywriter import run_copywriter
from agent.nodes.critics import run_critics

MAX_RETRIES = 3


def run_pipeline_with_retries(theme: Theme, tension: str) -> dict:
    """
    Returns a dict:
      {"success": True, "theme":..., "angle":..., "briefs":..., "drafts":...,
       "critiques":..., "retry_counts":...}
    or, if it never gets a clean pass:
      {"success": False, "reason": "...", "last_critiques":...}
    """
    retry_counts = {"angle_selector": 0, "copywriter": 0}
    feedback_for_angle_selector = None

    while True:
        # ── STRATEGY: pick (or re-pick) the angle ──────────────────────
        angle = run_angle_selector(theme, tension, feedback=feedback_for_angle_selector)

        if angle is None:
            return {
                "success": False,
                "reason": "No differentiated angle found (angle_selector returned None).",
                "theme": theme,
            }

        briefs = run_platform_strategist(angle)
        drafts = run_copywriter(theme, tension, angle, briefs)
        critiques = run_critics(angle, briefs, drafts)

        # ── CHECK BRAND+SAFETY FIRST ────────────────────────────────────
        # If the angle itself is unsafe/generic, there's no point polishing
        # the copy yet -- the idea underneath has to change first.
        brand_safety_failures = [
            c for key, c in critiques.items()
            if key.endswith("_brand_safety") and not c.passed
        ]

        if brand_safety_failures:
            retry_counts["angle_selector"] += 1
            if retry_counts["angle_selector"] > MAX_RETRIES:
                return {
                    "success": False,
                    "reason": "angle_selector failed brand+safety after max retries.",
                    "last_critiques": critiques,
                    "retry_counts": retry_counts,
                }
            feedback_for_angle_selector = " | ".join(c.feedback for c in brand_safety_failures)
            print(f"[orchestrator] Brand+safety failed -- retrying angle_selector "
                  f"(attempt {retry_counts['angle_selector']}/{MAX_RETRIES})")
            continue  # angle changed -> briefs/drafts/critiques are stale, restart loop

        # ── ANGLE IS SAFE. NOW FIX ANY PLATFORM-FIT FAILURES ────────────
        # These only need copywriter to retry the ONE broken platform --
        # the angle stays exactly as-is.
        platform_fit_failures = {
            key.replace("_platform_fit", ""): c
            for key, c in critiques.items()
            if key.endswith("_platform_fit") and not c.passed
        }

        attempt = 0
        while platform_fit_failures and attempt < MAX_RETRIES:
            attempt += 1
            retry_counts["copywriter"] += 1
            print(f"[orchestrator] Platform-fit failed for {list(platform_fit_failures.keys())} "
                  f"-- retrying copywriter (attempt {attempt}/{MAX_RETRIES})")

            for platform, critique in platform_fit_failures.items():
                fixed = run_copywriter(
                    theme, tension, angle,
                    {platform: briefs[platform]},
                    feedback={platform: critique.feedback},
                )
                drafts[platform] = fixed[platform]

            recheck = run_critics(angle, briefs, {p: drafts[p] for p in platform_fit_failures})
            critiques.update(recheck)

            platform_fit_failures = {
                key.replace("_platform_fit", ""): c
                for key, c in recheck.items()
                if key.endswith("_platform_fit") and not c.passed
            }

        if platform_fit_failures:
            return {
                "success": False,
                "reason": f"copywriter failed platform-fit after max retries for: "
                          f"{list(platform_fit_failures.keys())}",
                "last_critiques": critiques,
                "retry_counts": retry_counts,
            }

        # ── EVERYTHING PASSED ────────────────────────────────────────────
        return {
            "success": True,
            "theme": theme,
            "tension": tension,
            "angle": angle,
            "briefs": briefs,
            "drafts": drafts,
            "critiques": critiques,
            "retry_counts": retry_counts,
        }


# Quick manual test -- loads the saved fixture (zero upstream API calls),
# runs the full retry loop, and prints a clean final verdict.
if __name__ == "__main__":
    from tests.fixtures import load_fixture

    chosen, tension, angle, briefs = load_fixture()
    result = run_pipeline_with_retries(chosen, tension)

    print("\n--- ORCHESTRATOR RESULT ---")
    print(f"Success: {result['success']}")

    if result["success"]:
        print(f"Retry counts: {result['retry_counts']}")
        for platform, draft in result["drafts"].items():
            print(f"\n--- FINAL {platform.upper()} DRAFT ---")
            print(draft.model_dump_json(indent=2))
    else:
        print(f"Reason: {result['reason']}")
        if "retry_counts" in result:
            print(f"Retry counts: {result['retry_counts']}")
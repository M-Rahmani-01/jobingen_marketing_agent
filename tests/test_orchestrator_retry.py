"""
tests/test_orchestrator_retry.py

Verifies the Week 3 milestone: "Tested with deliberately weak draft --
caught + routed correctly."

We do NOT call the real LLM here. Instead we fake (mock) angle_selector,
platform_strategist, copywriter, and critics so we can deliberately force
specific pass/fail outcomes and check the orchestrator reacts correctly:

  - a platform-fit failure retries ONLY copywriter, for ONLY that platform
  - a brand+safety failure retries angle_selector (a full re-pick)
  - repeated failure past MAX_RETRIES fails CLEANLY, never loops forever

This costs zero API quota and runs in under a second.
"""

from unittest.mock import patch

from agent.state import Angle, PlatformBrief, PostDraft, Critique, Theme
from agent.orchestrator import run_pipeline_with_retries


def _fake_briefs():
    return {
        "linkedin": PlatformBrief(
            platform="linkedin", format="document/carousel",
            hook_style="contrarian insight", caption_length="mini-essay",
            cta_type="comment + follow", tone="sharp senior peer",
        ),
        "instagram": PlatformBrief(
            platform="instagram", format="bold single graphic",
            hook_style="relatable POV", caption_length="short, punchy",
            cta_type="save/send to a friend", tone="casual, no corporate energy",
        ),
    }


def test_platform_fit_failure_routes_to_copywriter_and_recovers():
    """A weak Instagram draft fails platform-fit -> orchestrator should
    retry ONLY copywriter, ONLY for Instagram -- LinkedIn is untouched."""

    weak_ig_draft = PostDraft(
        platform="instagram", hook_options=["Job hunting is hard"],
        body=["generic filler"], caption="ok caption", hashtags=[],
        cta="click here", alt_text="an image",
    )
    fixed_ig_draft = PostDraft(
        platform="instagram", hook_options=["POV: your resume vanishes into the void"],
        body=["specific, on-brief content"], caption="better caption",
        hashtags=["#jobsearch"], cta="save this for later", alt_text="a detailed, specific alt text",
    )
    good_li_draft = PostDraft(
        platform="linkedin", hook_options=["The hard truth about ATS systems"],
        body=["insight-dense content"], caption="a real mini-essay",
        hashtags=["#careers"], cta="comment your experience", alt_text="a detailed alt text",
    )

    fail_critique = Critique(passed=False, score=40,
                              feedback="Hook is generic, doesn't match relatable POV style.",
                              failed_node="copywriter")
    pass_critique = Critique(passed=True, score=88, feedback="Matches brief well.", failed_node=None)
    pass_after_fix = Critique(passed=True, score=82, feedback="Fixed, now matches brief.", failed_node=None)

    with patch("agent.orchestrator.run_angle_selector") as mock_angle, \
         patch("agent.orchestrator.run_platform_strategist") as mock_strategist, \
         patch("agent.orchestrator.run_copywriter") as mock_copywriter, \
         patch("agent.orchestrator.run_critics") as mock_critics:

        mock_angle.return_value = Angle(take="test angle", product_tie=None, why_only_jobingen="test reason")
        mock_strategist.return_value = _fake_briefs()

        # 1st call: initial draft for BOTH platforms (Instagram is weak)
        # 2nd call: retry draft for Instagram ONLY (that's the point being tested)
        mock_copywriter.side_effect = [
            {"linkedin": good_li_draft, "instagram": weak_ig_draft},
            {"instagram": fixed_ig_draft},
        ]

        # 1st call: initial critique -- Instagram fails platform-fit
        # 2nd call: recheck -- ONLY Instagram gets re-checked, and now passes
        mock_critics.side_effect = [
            {
                "linkedin_platform_fit": pass_critique,
                "linkedin_brand_safety": pass_critique,
                "instagram_platform_fit": fail_critique,
                "instagram_brand_safety": pass_critique,
            },
            {
                "instagram_platform_fit": pass_after_fix,
                "instagram_brand_safety": pass_critique,
            },
        ]

        theme = Theme(label="test theme", evidence=[], heat=0.8)
        result = run_pipeline_with_retries(theme, "test tension")

        # ── Assertions ──
        assert result["success"] is True
        assert result["retry_counts"]["copywriter"] == 1
        assert result["retry_counts"]["angle_selector"] == 0  # angle was NEVER touched
        assert result["drafts"]["instagram"].hook_options[0] == "POV: your resume vanishes into the void"
        assert result["drafts"]["linkedin"].hook_options[0] == "The hard truth about ATS systems"  # untouched
        assert mock_copywriter.call_count == 2
        assert mock_critics.call_count == 2

        # Confirm the retry call only targeted Instagram, not LinkedIn
        second_call_briefs_arg = mock_copywriter.call_args_list[1].args[3]
        assert list(second_call_briefs_arg.keys()) == ["instagram"]

    print("PASS: platform-fit failure correctly routed to copywriter, Instagram-only, and recovered.")


def test_brand_safety_failure_routes_to_angle_selector_and_recovers():
    """A risky/generic angle fails brand+safety -> orchestrator should
    re-run angle_selector (a full re-pick), not just tweak the copy."""

    bad_angle = Angle(take="Just try harder and you'll get hired!",
                       product_tie=None, why_only_jobingen="generic, could be anyone")
    good_angle = Angle(take="Your resume isn't the problem, generic tailoring is.",
                        product_tie="AI resume builder", why_only_jobingen="ties to a real, specific tool")

    fail_brand = Critique(passed=False, score=35,
                           feedback="Sounds like generic motivational filler, implies guaranteed outcome.",
                           failed_node="angle_selector")
    pass_all = Critique(passed=True, score=85, feedback="Fine.", failed_node=None)

    draft_v1 = PostDraft(platform="linkedin", hook_options=["Try harder!"], body=["x"],
                          caption="x", hashtags=[], cta="follow", alt_text="alt")
    draft_v2 = PostDraft(platform="linkedin", hook_options=["Better hook"], body=["y"],
                          caption="y", hashtags=[], cta="comment", alt_text="alt")

    with patch("agent.orchestrator.run_angle_selector") as mock_angle, \
         patch("agent.orchestrator.run_platform_strategist") as mock_strategist, \
         patch("agent.orchestrator.run_copywriter") as mock_copywriter, \
         patch("agent.orchestrator.run_critics") as mock_critics:

        mock_angle.side_effect = [bad_angle, good_angle]
        mock_strategist.return_value = {"linkedin": _fake_briefs()["linkedin"]}
        mock_copywriter.side_effect = [
            {"linkedin": draft_v1},
            {"linkedin": draft_v2},
        ]
        mock_critics.side_effect = [
            {"linkedin_platform_fit": pass_all, "linkedin_brand_safety": fail_brand},
            {"linkedin_platform_fit": pass_all, "linkedin_brand_safety": pass_all},
        ]

        theme = Theme(label="test theme", evidence=[], heat=0.8)
        result = run_pipeline_with_retries(theme, "test tension")

        assert result["success"] is True
        assert result["retry_counts"]["angle_selector"] == 1
        assert result["angle"].take == good_angle.take
        assert mock_angle.call_count == 2

    print("PASS: brand+safety failure correctly routed to angle_selector and recovered.")


def test_persistent_failure_stops_cleanly_at_max_retries():
    """A draft that NEVER passes should stop after MAX_RETRIES, not loop
    forever -- and should report failure clearly, not crash."""

    always_bad_draft = PostDraft(platform="linkedin", hook_options=["bad"], body=["bad"],
                                  caption="bad", hashtags=[], cta="bad", alt_text="bad")
    always_fail = Critique(passed=False, score=20, feedback="Still bad.", failed_node="copywriter")
    always_pass_brand = Critique(passed=True, score=90, feedback="fine", failed_node=None)

    with patch("agent.orchestrator.run_angle_selector") as mock_angle, \
         patch("agent.orchestrator.run_platform_strategist") as mock_strategist, \
         patch("agent.orchestrator.run_copywriter") as mock_copywriter, \
         patch("agent.orchestrator.run_critics") as mock_critics:

        mock_angle.return_value = Angle(take="ok angle", product_tie=None, why_only_jobingen="fine")
        mock_strategist.return_value = {"linkedin": _fake_briefs()["linkedin"]}
        mock_copywriter.return_value = {"linkedin": always_bad_draft}
        mock_critics.return_value = {
            "linkedin_platform_fit": always_fail,
            "linkedin_brand_safety": always_pass_brand,
        }

        theme = Theme(label="test theme", evidence=[], heat=0.8)
        result = run_pipeline_with_retries(theme, "test tension")

        assert result["success"] is False
        assert "max retries" in result["reason"].lower()
        # Initial attempt + MAX_RETRIES(3) retries = 4 total copywriter calls
        assert mock_copywriter.call_count == 4

    print("PASS: persistent failure stopped cleanly at MAX_RETRIES, no infinite loop.")


if __name__ == "__main__":
    test_platform_fit_failure_routes_to_copywriter_and_recovers()
    test_brand_safety_failure_routes_to_angle_selector_and_recovers()
    test_persistent_failure_stops_cleanly_at_max_retries()
    print("\nAll orchestrator retry-routing tests passed.")
"""
tests/test_graph_retry.py

Verifies graph.py's retry loop (Version 2) the same way we verified
orchestrator.py's retry loop: mock every node function, force specific
pass/fail sequences, and confirm the graph routes and loops correctly.
Zero API calls, fully deterministic.
"""

from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver

from agent.state import MarketingState, SignalItem, Theme, Angle, PlatformBrief, PostDraft, Critique
from agent.graph import build_graph, MAX_RETRIES


def _fake_signal():
    return [SignalItem(source="reddit", text="test", score=0.5, url="http://example.com")]


def _fake_briefs():
    return {
        "linkedin": PlatformBrief(platform="linkedin", format="doc", hook_style="x",
                                    caption_length="y", cta_type="z", tone="w"),
        "instagram": PlatformBrief(platform="instagram", format="carousel", hook_style="x",
                                     caption_length="y", cta_type="z", tone="w"),
    }


def _common_mocks(mock_scout, mock_synth, mock_scorer, mock_modeler, mock_strategist):
    """Sets up the parts of the pipeline that stay the same across all retry tests."""
    fake_theme = Theme(label="test theme", evidence=[], heat=0.9)
    mock_scout.return_value = _fake_signal()
    mock_synth.return_value = [fake_theme]
    mock_scorer.return_value = (fake_theme, [])
    mock_modeler.return_value = "test tension"
    mock_strategist.return_value = _fake_briefs()
    return fake_theme


def test_platform_fit_failure_loops_through_retry_copywriter_and_succeeds():
    weak_draft = PostDraft(platform="instagram", hook_options=["weak"], body=["x"],
                            caption="x", hashtags=[], cta="x", alt_text="x")
    fixed_draft = PostDraft(platform="instagram", hook_options=["strong"], body=["y"],
                             caption="y", hashtags=["#x"], cta="save this", alt_text="detailed")
    good_li_draft = PostDraft(platform="linkedin", hook_options=["good"], body=["z"],
                               caption="z", hashtags=["#y"], cta="comment", alt_text="detailed")

    fail_critique = Critique(passed=False, score=40, feedback="doesn't match brief", failed_node="copywriter")
    pass_critique = Critique(passed=True, score=88, feedback="fine", failed_node=None)

    with patch("agent.graph.run_trend_scout") as mock_scout, \
         patch("agent.graph.run_signal_synthesizer") as mock_synth, \
         patch("agent.graph.run_trend_scorer") as mock_scorer, \
         patch("agent.graph.run_audience_modeler") as mock_modeler, \
         patch("agent.graph.run_angle_selector") as mock_angle, \
         patch("agent.graph.run_platform_strategist") as mock_strategist, \
         patch("agent.graph.run_copywriter") as mock_copywriter, \
         patch("agent.graph.run_critics") as mock_critics:

        _common_mocks(mock_scout, mock_synth, mock_scorer, mock_modeler, mock_strategist)
        mock_angle.return_value = Angle(take="test angle", product_tie=None, why_only_jobingen="test")

        # 1st copywriter call: initial (both platforms, instagram weak)
        # 2nd copywriter call: retry (instagram only, fixed)
        mock_copywriter.side_effect = [
            {"linkedin": good_li_draft, "instagram": weak_draft},
            {"instagram": fixed_draft},
        ]

        # 1st critics call: instagram platform-fit fails
        # 2nd critics call (full recheck after retry): everything passes
        mock_critics.side_effect = [
            {
                "linkedin_platform_fit": pass_critique, "linkedin_brand_safety": pass_critique,
                "instagram_platform_fit": fail_critique, "instagram_brand_safety": pass_critique,
            },
            {
                "linkedin_platform_fit": pass_critique, "linkedin_brand_safety": pass_critique,
                "instagram_platform_fit": pass_critique, "instagram_brand_safety": pass_critique,
            },
        ]

        graph = build_graph(checkpointer=InMemorySaver())
        final_state = graph.invoke(MarketingState(), config={"configurable": {"thread_id": "test-platform-fit-retry"}})

        assert final_state["retry_counts"]["copywriter"] == 1
        assert final_state["retry_counts"].get("angle_selector", 0) == 0
        assert final_state["drafts"]["instagram"].hook_options[0] == "strong"
        assert not final_state["errors"]  # no failure logged -- success
        assert "__interrupt__" in final_state  # paused for human approval, as it should
        assert mock_copywriter.call_count == 2
        assert mock_critics.call_count == 2

    print("PASS: platform-fit failure looped through retry_copywriter, succeeded, and paused for approval.")


def test_brand_safety_failure_loops_through_retry_angle_and_succeeds():
    bad_angle = Angle(take="generic take", product_tie=None, why_only_jobingen="generic")
    good_angle = Angle(take="sharp take", product_tie="resume builder", why_only_jobingen="specific")

    fail_brand = Critique(passed=False, score=35, feedback="too generic", failed_node="angle_selector")
    pass_all = Critique(passed=True, score=85, feedback="fine", failed_node=None)

    draft_v1 = PostDraft(platform="linkedin", hook_options=["a"], body=["a"], caption="a",
                          hashtags=[], cta="a", alt_text="a")
    draft_v2 = PostDraft(platform="linkedin", hook_options=["b"], body=["b"], caption="b",
                          hashtags=[], cta="b", alt_text="b")

    with patch("agent.graph.run_trend_scout") as mock_scout, \
         patch("agent.graph.run_signal_synthesizer") as mock_synth, \
         patch("agent.graph.run_trend_scorer") as mock_scorer, \
         patch("agent.graph.run_audience_modeler") as mock_modeler, \
         patch("agent.graph.run_angle_selector") as mock_angle, \
         patch("agent.graph.run_platform_strategist") as mock_strategist, \
         patch("agent.graph.run_copywriter") as mock_copywriter, \
         patch("agent.graph.run_critics") as mock_critics:

        fake_theme = _common_mocks(mock_scout, mock_synth, mock_scorer, mock_modeler, mock_strategist)
        mock_strategist.return_value = {"linkedin": _fake_briefs()["linkedin"]}

        mock_angle.side_effect = [bad_angle, good_angle]
        mock_copywriter.side_effect = [{"linkedin": draft_v1}, {"linkedin": draft_v2}]
        mock_critics.side_effect = [
            {"linkedin_platform_fit": pass_all, "linkedin_brand_safety": fail_brand},
            {"linkedin_platform_fit": pass_all, "linkedin_brand_safety": pass_all},
        ]

        graph = build_graph(checkpointer=InMemorySaver())
        final_state = graph.invoke(MarketingState(), config={"configurable": {"thread_id": "test-brand-safety-retry"}})

        assert final_state["retry_counts"]["angle_selector"] == 1
        assert final_state["angle"].take == "sharp take"
        assert not final_state["errors"]
        assert "__interrupt__" in final_state  # paused for human approval, as it should
        assert mock_angle.call_count == 2

    print("PASS: brand+safety failure looped through retry_angle_selector, succeeded, and paused for approval.")


def test_persistent_failure_stops_at_max_retries_and_logs_error():
    always_bad = PostDraft(platform="linkedin", hook_options=["bad"], body=["bad"],
                            caption="bad", hashtags=[], cta="bad", alt_text="bad")
    always_fail = Critique(passed=False, score=20, feedback="still bad", failed_node="copywriter")
    always_pass_brand = Critique(passed=True, score=90, feedback="fine", failed_node=None)

    with patch("agent.graph.run_trend_scout") as mock_scout, \
         patch("agent.graph.run_signal_synthesizer") as mock_synth, \
         patch("agent.graph.run_trend_scorer") as mock_scorer, \
         patch("agent.graph.run_audience_modeler") as mock_modeler, \
         patch("agent.graph.run_angle_selector") as mock_angle, \
         patch("agent.graph.run_platform_strategist") as mock_strategist, \
         patch("agent.graph.run_copywriter") as mock_copywriter, \
         patch("agent.graph.run_critics") as mock_critics:

        _common_mocks(mock_scout, mock_synth, mock_scorer, mock_modeler, mock_strategist)
        mock_strategist.return_value = {"linkedin": _fake_briefs()["linkedin"]}
        mock_angle.return_value = Angle(take="ok", product_tie=None, why_only_jobingen="fine")
        mock_copywriter.return_value = {"linkedin": always_bad}
        mock_critics.return_value = {
            "linkedin_platform_fit": always_fail,
            "linkedin_brand_safety": always_pass_brand,
        }

        graph = build_graph()  # no checkpointer needed for these mock tests
        final_state = graph.invoke(MarketingState())

        assert final_state["retry_counts"]["copywriter"] == MAX_RETRIES + 1
        assert len(final_state["errors"]) == 1
        assert "max retries" in final_state["errors"][0].lower()
        # initial + MAX_RETRIES retries = 4 total copywriter calls
        assert mock_copywriter.call_count == MAX_RETRIES + 1

    print("PASS: persistent failure stopped cleanly at MAX_RETRIES with a logged error.")


if __name__ == "__main__":
    test_platform_fit_failure_loops_through_retry_copywriter_and_succeeds()
    test_brand_safety_failure_loops_through_retry_angle_and_succeeds()
    test_persistent_failure_stops_at_max_retries_and_logs_error()
    print("\nAll graph retry tests passed.")
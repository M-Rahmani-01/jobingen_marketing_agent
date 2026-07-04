"""
tests/test_graph_routing.py

Verifies the ONE part of graph.py that today's live run never actually
exercised: the conditional early-exit routing.

- If trend_scorer finds no strong theme (chosen_theme is None), the
  graph must stop right there -- it must NOT call audience_modeler,
  angle_selector, platform_strategist, copywriter, or critics.
- If angle_selector finds no differentiated angle (angle is None), the
  graph must stop right there -- it must NOT call platform_strategist,
  copywriter, or critics.

We mock every node function so this costs ZERO API calls and is fully
deterministic -- we're testing OUR routing logic, not the LLM's judgment.
"""

from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver

from agent.state import MarketingState, SignalItem, Theme, Angle
from agent.graph import build_graph


def _fake_signal():
    return [SignalItem(source="reddit", text="test", score=0.5, url="http://example.com")]


def test_graph_stops_early_when_no_strong_theme():
    """chosen_theme=None after trend_scorer -> nothing downstream should run."""
    with patch("agent.graph.run_trend_scout") as mock_scout, \
         patch("agent.graph.run_signal_synthesizer") as mock_synth, \
         patch("agent.graph.run_trend_scorer") as mock_scorer, \
         patch("agent.graph.run_audience_modeler") as mock_modeler, \
         patch("agent.graph.run_angle_selector") as mock_angle, \
         patch("agent.graph.run_platform_strategist") as mock_strategist, \
         patch("agent.graph.run_copywriter") as mock_copywriter, \
         patch("agent.graph.run_critics") as mock_critics:

        mock_scout.return_value = _fake_signal()
        mock_synth.return_value = [Theme(label="weak theme", evidence=[], heat=0.1)]
        mock_scorer.return_value = (None, [])  # <-- the deliberate "no strong theme" outcome

        graph = build_graph()  # no checkpointer needed for these mock tests
        final_state = graph.invoke(MarketingState())

        # Should have stopped -- chosen_theme is None, nothing after it ran
        assert final_state["chosen_theme"] is None
        mock_modeler.assert_not_called()
        mock_angle.assert_not_called()
        mock_strategist.assert_not_called()
        mock_copywriter.assert_not_called()
        mock_critics.assert_not_called()

    print("PASS: graph correctly stopped early when no strong theme was found.")


def test_graph_stops_early_when_no_differentiated_angle():
    """A valid theme is found, but angle_selector returns None ->
    everything after angle_selector should NOT run."""
    fake_theme = Theme(label="valid theme", evidence=[], heat=0.9)

    with patch("agent.graph.run_trend_scout") as mock_scout, \
         patch("agent.graph.run_signal_synthesizer") as mock_synth, \
         patch("agent.graph.run_trend_scorer") as mock_scorer, \
         patch("agent.graph.run_audience_modeler") as mock_modeler, \
         patch("agent.graph.run_angle_selector") as mock_angle, \
         patch("agent.graph.run_platform_strategist") as mock_strategist, \
         patch("agent.graph.run_copywriter") as mock_copywriter, \
         patch("agent.graph.run_critics") as mock_critics:

        mock_scout.return_value = _fake_signal()
        mock_synth.return_value = [fake_theme]
        mock_scorer.return_value = (fake_theme, [])  # a real theme WAS chosen
        mock_modeler.return_value = "a real tension"
        mock_angle.return_value = None  # <-- the deliberate "no differentiated angle" outcome

        graph = build_graph()  # no checkpointer needed for these mock tests
        final_state = graph.invoke(MarketingState())

        # chosen_theme and tension should be set (they ran), but angle is None
        # and nothing after angle_selector should have run
        assert final_state["chosen_theme"] is not None
        assert final_state["tension"] == "a real tension"
        assert final_state["angle"] is None
        mock_strategist.assert_not_called()
        mock_copywriter.assert_not_called()
        mock_critics.assert_not_called()

    print("PASS: graph correctly stopped early when no differentiated angle was found.")


def test_graph_runs_full_happy_path_then_pauses_for_approval():
    """Sanity check: when a theme AND angle ARE found, the graph should
    run all the way through to critics (not stop early by accident),
    then PAUSE at human_approval -- it should NOT auto-finish, because
    the spec requires a human decision on every post, no exceptions."""
    fake_theme = Theme(label="valid theme", evidence=[], heat=0.9)
    fake_angle = Angle(take="a real angle", product_tie=None, why_only_jobingen="test")

    with patch("agent.graph.run_trend_scout") as mock_scout, \
         patch("agent.graph.run_signal_synthesizer") as mock_synth, \
         patch("agent.graph.run_trend_scorer") as mock_scorer, \
         patch("agent.graph.run_audience_modeler") as mock_modeler, \
         patch("agent.graph.run_angle_selector") as mock_angle, \
         patch("agent.graph.run_platform_strategist") as mock_strategist, \
         patch("agent.graph.run_copywriter") as mock_copywriter, \
         patch("agent.graph.run_critics") as mock_critics:

        mock_scout.return_value = _fake_signal()
        mock_synth.return_value = [fake_theme]
        mock_scorer.return_value = (fake_theme, [])
        mock_modeler.return_value = "a real tension"
        mock_angle.return_value = fake_angle
        mock_strategist.return_value = {}
        mock_copywriter.return_value = {}
        mock_critics.return_value = {}  # empty dict -> no failures -> "success" route

        checkpointer = InMemorySaver()  # required for interrupt() to work at all
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-happy-path"}}

        result = graph.invoke(MarketingState(), config=config)

        assert result["chosen_theme"] is not None
        assert result["angle"] is not None
        mock_strategist.assert_called_once()
        mock_copywriter.assert_called_once()
        mock_critics.assert_called_once()

        # The key new assertion: it PAUSED, it did not finish on its own
        assert "__interrupt__" in result
        pack = result["__interrupt__"][0].value
        assert pack["theme"] == "valid theme"
        assert pack["angle"] == "a real angle"
        assert result.get("approved") is None  # not yet decided

    print("PASS: graph runs the full happy path, then correctly PAUSES for human approval.")


if __name__ == "__main__":
    test_graph_stops_early_when_no_strong_theme()
    test_graph_stops_early_when_no_differentiated_angle()
    test_graph_runs_full_happy_path_then_pauses_for_approval()
    print("\nAll graph routing tests passed.")
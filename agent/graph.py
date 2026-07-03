"""
agent/graph.py

LangGraph wiring -- Version 2: HAPPY PATH + RETRY LOOP.

Builds on the proven Version 1 wiring (Scout -> ... -> Critics, with
early exits for "no strong theme" / "no differentiated angle"). This
version adds the layer-routed retry logic that used to live in
orchestrator.py's manual while-loop, now expressed as real graph edges:

  brand+safety failure  -> retry angle_selector (full re-pick, loops
                            back through platform_strategist + copywriter)
  platform-fit failure  -> retry copywriter (only the failing platforms),
                            then re-check with critics
  either failure > MAX_RETRIES -> stop cleanly, log the reason
  everything passes     -> success, END

SIMPLIFICATION vs orchestrator.py: after a copywriter retry, this
version re-critiques BOTH platforms, not just the fixed one. Slightly
more API calls per retry, but much simpler to reason about correctly
within LangGraph's cycle structure. Can optimize later (see
PROGRESS.md "Ideas for Later").
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.state import MarketingState
from agent.nodes.trend_scout import run_trend_scout
from agent.nodes.signal_synthesizer import run_signal_synthesizer
from agent.nodes.trend_scorer import run_trend_scorer
from agent.nodes.audience_modeler import run_audience_modeler
from agent.nodes.angle_selector import run_angle_selector
from agent.nodes.platform_strategist import run_platform_strategist
from agent.nodes.copywriter import run_copywriter
from agent.nodes.critics import run_critics

MAX_RETRIES = 3


# ─────────────────────────────────────────────
# NODE WRAPPERS — happy path (unchanged from v1, already proven working)
# ─────────────────────────────────────────────

def node_trend_scout(state: MarketingState) -> dict:
    signal = run_trend_scout()
    return {"raw_signal": signal}


def node_signal_synthesizer(state: MarketingState) -> dict:
    themes = run_signal_synthesizer(state.raw_signal)
    return {"themes": themes}


def node_trend_scorer(state: MarketingState) -> dict:
    chosen, _breakdown = run_trend_scorer(state.themes)
    return {"chosen_theme": chosen}


def node_audience_modeler(state: MarketingState) -> dict:
    tension = run_audience_modeler(state.chosen_theme)
    return {"tension": tension}


def node_angle_selector(state: MarketingState) -> dict:
    angle = run_angle_selector(state.chosen_theme, state.tension)
    return {"angle": angle}


def node_platform_strategist(state: MarketingState) -> dict:
    briefs = run_platform_strategist(state.angle)
    return {"briefs": briefs}


def node_copywriter(state: MarketingState) -> dict:
    drafts = run_copywriter(state.chosen_theme, state.tension, state.angle, state.briefs)
    return {"drafts": drafts}


def node_critics(state: MarketingState) -> dict:
    critiques = run_critics(state.angle, state.briefs, state.drafts)
    return {"critiques": critiques}


# ─────────────────────────────────────────────
# HELPERS — shared by the retry node and the router (kept in sync by
# living in ONE place instead of being duplicated)
# ─────────────────────────────────────────────

def _brand_safety_failures(critiques: dict) -> list:
    return [c for key, c in critiques.items() if key.endswith("_brand_safety") and not c.passed]


def _platform_fit_failures(critiques: dict) -> dict:
    return {
        key.replace("_platform_fit", ""): c
        for key, c in critiques.items()
        if key.endswith("_platform_fit") and not c.passed
    }


# ─────────────────────────────────────────────
# RETRY NODES — NEW in v2
# ─────────────────────────────────────────────

def node_record_critique_outcome(state: MarketingState) -> dict:
    """Updates retry_counts based on what just failed. Must be a real
    node (not just a router) because only nodes can update state."""
    retry_counts = dict(state.retry_counts)

    if _brand_safety_failures(state.critiques):
        retry_counts["angle_selector"] = retry_counts.get("angle_selector", 0) + 1
    elif _platform_fit_failures(state.critiques):
        retry_counts["copywriter"] = retry_counts.get("copywriter", 0) + 1

    return {"retry_counts": retry_counts}


def node_retry_angle_selector(state: MarketingState) -> dict:
    feedback = " | ".join(c.feedback for c in _brand_safety_failures(state.critiques))
    print(f"[graph] Retrying angle_selector (attempt {state.retry_counts.get('angle_selector', 0)}/{MAX_RETRIES})")
    angle = run_angle_selector(state.chosen_theme, state.tension, feedback=feedback)
    return {"angle": angle}


def node_retry_copywriter(state: MarketingState) -> dict:
    failures = _platform_fit_failures(state.critiques)
    feedback = {platform: c.feedback for platform, c in failures.items()}
    failing_briefs = {platform: state.briefs[platform] for platform in failures}

    print(f"[graph] Retrying copywriter for {list(failures.keys())} "
          f"(attempt {state.retry_counts.get('copywriter', 0)}/{MAX_RETRIES})")

    fixed_drafts = run_copywriter(state.chosen_theme, state.tension, state.angle, failing_briefs, feedback=feedback)

    merged_drafts = dict(state.drafts)
    merged_drafts.update(fixed_drafts)
    return {"drafts": merged_drafts}


def node_fail_max_retries(state: MarketingState) -> dict:
    reason = (
        f"Failed after max retries. retry_counts={state.retry_counts}, "
        f"last critiques={ {k: v.passed for k, v in state.critiques.items()} }"
    )
    print(f"[graph] {reason}")
    return {"errors": state.errors + [reason]}


# ─────────────────────────────────────────────
# CONDITIONAL ROUTING
# ─────────────────────────────────────────────

def route_after_scorer(state: MarketingState) -> str:
    if state.chosen_theme is None:
        print("[graph] No strong theme this run -- ending early.")
        return "end"
    return "continue"


def route_after_angle(state: MarketingState) -> str:
    if state.angle is None:
        print("[graph] No differentiated angle found -- ending early.")
        return "end"
    return "continue"


def route_after_critique_outcome(state: MarketingState) -> str:
    brand_fail = _brand_safety_failures(state.critiques)
    platform_fail = _platform_fit_failures(state.critiques)

    if brand_fail:
        if state.retry_counts.get("angle_selector", 0) > MAX_RETRIES:
            return "fail"
        return "retry_angle"

    if platform_fail:
        if state.retry_counts.get("copywriter", 0) > MAX_RETRIES:
            return "fail"
        return "retry_copy"

    return "success"


# ─────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    checkpointer=None -> no persistence, same as before (used by our
    mock tests, since they don't need a real DB file).
    checkpointer=<SqliteSaver instance> -> every step of the run gets
    saved to disk, so a run can be paused, resumed, or inspected later
    even after the program exits or crashes mid-run.
    """
    builder = StateGraph(MarketingState)

    # Happy path nodes
    builder.add_node("trend_scout", node_trend_scout)
    builder.add_node("signal_synthesizer", node_signal_synthesizer)
    builder.add_node("trend_scorer", node_trend_scorer)
    builder.add_node("audience_modeler", node_audience_modeler)
    builder.add_node("angle_selector", node_angle_selector)
    builder.add_node("platform_strategist", node_platform_strategist)
    builder.add_node("copywriter", node_copywriter)
    builder.add_node("critics", node_critics)

    # Retry nodes
    builder.add_node("record_critique_outcome", node_record_critique_outcome)
    builder.add_node("retry_angle_selector", node_retry_angle_selector)
    builder.add_node("retry_copywriter", node_retry_copywriter)
    builder.add_node("fail_max_retries", node_fail_max_retries)

    # Happy path edges
    builder.add_edge(START, "trend_scout")
    builder.add_edge("trend_scout", "signal_synthesizer")
    builder.add_edge("signal_synthesizer", "trend_scorer")

    builder.add_conditional_edges(
        "trend_scorer", route_after_scorer,
        {"end": END, "continue": "audience_modeler"},
    )

    builder.add_edge("audience_modeler", "angle_selector")

    builder.add_conditional_edges(
        "angle_selector", route_after_angle,
        {"end": END, "continue": "platform_strategist"},
    )

    builder.add_edge("platform_strategist", "copywriter")
    builder.add_edge("copywriter", "critics")
    builder.add_edge("critics", "record_critique_outcome")

    # Retry routing — the new part
    builder.add_conditional_edges(
        "record_critique_outcome", route_after_critique_outcome,
        {
            "retry_angle": "retry_angle_selector",
            "retry_copy": "retry_copywriter",
            "fail": "fail_max_retries",
            "success": END,
        },
    )

    # Loop-back edges — these create the cycles
    builder.add_edge("retry_angle_selector", "platform_strategist")  # full redo
    builder.add_edge("retry_copywriter", "critics")                  # recheck
    builder.add_edge("fail_max_retries", END)

    return builder.compile(checkpointer=checkpointer)


# Quick manual test — run this file directly
#
# Uses a REAL SQLite checkpointer this time, saved to
# content_store/checkpoints.sqlite (per FOLDER_STRUCTURE.md). Every step
# of the run gets persisted under a "thread_id" -- think of a thread_id
# as a save-slot name. Run this file twice with the SAME thread_id and
# LangGraph can resume/inspect that exact run instead of starting fresh.
if __name__ == "__main__":
    import os

    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")  # security: only allow known-safe types on checkpoint load

    db_path = "content_store/checkpoints.sqlite"
    thread_id = "manual-test-run-1"

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        final_state = graph.invoke(MarketingState(), config=config)

        print("\n--- GRAPH RUN COMPLETE ---")
        if final_state.get("chosen_theme"):
            print(f"Theme: {final_state['chosen_theme'].label}")
        if final_state.get("angle"):
            print(f"Angle: {final_state['angle'].take}")
        if final_state.get("critiques"):
            print("\nFinal critique summary:")
            for key, c in final_state["critiques"].items():
                print(f"  {key}: {'PASS' if c.passed else 'FAIL'} ({c.score})")
        print(f"\nRetry counts: {final_state.get('retry_counts')}")
        if final_state.get("errors"):
            print(f"Errors: {final_state['errors']}")

        print(f"\n[checkpointer] Run saved under thread_id='{thread_id}' at {db_path}")
        print("[checkpointer] Re-run this file to see the checkpoint history grow.")
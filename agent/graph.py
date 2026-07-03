"""
agent/graph.py

LangGraph wiring for the pipeline -- Version 1: HAPPY PATH ONLY.

This intentionally does NOT include retry logic yet. Goal right now is
just to confirm LangGraph can correctly call our existing node functions
in sequence, using MarketingState as the shared state object, and
correctly stop early when chosen_theme or angle come back None (the
"no post today" outcomes we built in Weeks 1-2).

Retry routing (the orchestrator.py logic) gets added in graph_v2 once
this base version is confirmed working.

IMPORTANT: none of our existing node functions take MarketingState
directly -- they take individual typed arguments (Theme, str, etc.) and
return typed objects, exactly as we built them in Weeks 1-3. So each
LangGraph node here is a small WRAPPER: it pulls what it needs off
state, calls our real function unchanged, and returns a dict of the
fields that changed. This is the standard LangGraph pattern -- your
actual thinking logic never has to know LangGraph exists.
"""

from langgraph.graph import StateGraph, START, END

from agent.state import MarketingState
from agent.nodes.trend_scout import run_trend_scout
from agent.nodes.signal_synthesizer import run_signal_synthesizer
from agent.nodes.trend_scorer import run_trend_scorer
from agent.nodes.audience_modeler import run_audience_modeler
from agent.nodes.angle_selector import run_angle_selector
from agent.nodes.platform_strategist import run_platform_strategist
from agent.nodes.copywriter import run_copywriter
from agent.nodes.critics import run_critics


# ─────────────────────────────────────────────
# NODE WRAPPERS — each one: read state -> call real function -> return updates
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
# CONDITIONAL ROUTING — the "no strong theme" / "no angle" early exits
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


# ─────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────

def build_graph():
    builder = StateGraph(MarketingState)

    builder.add_node("trend_scout", node_trend_scout)
    builder.add_node("signal_synthesizer", node_signal_synthesizer)
    builder.add_node("trend_scorer", node_trend_scorer)
    builder.add_node("audience_modeler", node_audience_modeler)
    builder.add_node("angle_selector", node_angle_selector)
    builder.add_node("platform_strategist", node_platform_strategist)
    builder.add_node("copywriter", node_copywriter)
    builder.add_node("critics", node_critics)

    builder.add_edge(START, "trend_scout")
    builder.add_edge("trend_scout", "signal_synthesizer")
    builder.add_edge("signal_synthesizer", "trend_scorer")

    builder.add_conditional_edges(
        "trend_scorer",
        route_after_scorer,
        {"end": END, "continue": "audience_modeler"},
    )

    builder.add_edge("audience_modeler", "angle_selector")

    builder.add_conditional_edges(
        "angle_selector",
        route_after_angle,
        {"end": END, "continue": "platform_strategist"},
    )

    builder.add_edge("platform_strategist", "copywriter")
    builder.add_edge("copywriter", "critics")
    builder.add_edge("critics", END)

    return builder.compile()


# Quick manual test — run this file directly
if __name__ == "__main__":
    graph = build_graph()
    final_state = graph.invoke(MarketingState())

    print("\n--- GRAPH RUN COMPLETE ---")
    if final_state.get("chosen_theme"):
        print(f"Theme: {final_state['chosen_theme'].label}")
    if final_state.get("angle"):
        print(f"Angle: {final_state['angle'].take}")
    if final_state.get("critiques"):
        print("\nCritique summary:")
        for key, c in final_state["critiques"].items():
            print(f"  {key}: {'PASS' if c.passed else 'FAIL'} ({c.score})")
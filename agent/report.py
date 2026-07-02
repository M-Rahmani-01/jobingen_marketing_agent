"""
agent/report.py

Assembles the Analysis Report (Spec Section 11.1) that ships alongside
every approved content pack -- so a human reviewer can see WHY this post
exists and what to expect from it, at a glance.

IMPORTANT: this node makes ZERO new LLM calls. Every piece of information
it needs (theme, tension, angle, critiques, virality scores) was already
produced by earlier nodes -- this is pure Python assembly, not generation.
That's intentional: a report should faithfully summarize what already
happened, not invent a new narrative on top of it.

Lives at agent/report.py (not agent/nodes/) because it isn't an agent
that thinks -- it's a formatter that reports.
"""

from agent.state import Theme, Angle, Critique, ViralityScore, AnalysisReport


def build_analysis_reports(
    theme: Theme,
    tension: str,
    angle: Angle,
    critiques: dict[str, Critique],
    virality_scores: dict[str, ViralityScore],
) -> dict[str, AnalysisReport]:
    """
    Builds one AnalysisReport PER PLATFORM (matches how MarketingState.report
    is keyed -- dict[str, AnalysisReport]).
    """
    reports: dict[str, AnalysisReport] = {}

    for platform, virality in virality_scores.items():
        platform_fit_critique = critiques.get(f"{platform}_platform_fit")
        brand_safety_critique = critiques.get(f"{platform}_brand_safety")

        platform_fit_summary = (
            f"Score {platform_fit_critique.score}/100. {platform_fit_critique.feedback}"
            if platform_fit_critique
            else "No platform-fit critique available."
        )

        # "What to watch" combines the weakest virality lever with any
        # lingering brand+safety note -- the two things a human reviewer
        # should actually pay attention to before approving.
        watch_points = [f"Weakest lever: {virality.weakest_lever}"]
        if brand_safety_critique and brand_safety_critique.score is not None and brand_safety_critique.score < 90:
            watch_points.append(f"Brand+safety note: {brand_safety_critique.feedback}")
        what_to_watch = " | ".join(watch_points)

        report = AnalysisReport(
            trend_basis=f"{theme.label} (heat: {theme.heat:.2f}, {len(theme.evidence)} evidence items)",
            audience_tension=tension,
            differentiation=angle.why_only_jobingen,
            platform_fit=platform_fit_summary,
            virality=virality,
            what_to_watch=what_to_watch,
        )
        reports[platform] = report

        print(f"[report] {platform} report built -- virality {virality.total}/100")

    return reports


# Quick manual test -- runs the full retry-loop pipeline + virality scorer
# on the saved fixture, then assembles and prints the final reports.
# No new API calls happen in THIS file -- all cost comes from upstream
# nodes it calls to gather the pieces.
if __name__ == "__main__":
    from tests.fixtures import load_fixture
    from agent.orchestrator import run_pipeline_with_retries
    from agent.scoring.virality import run_virality_for_all

    chosen, tension, _, _ = load_fixture()
    result = run_pipeline_with_retries(chosen, tension)

    if not result["success"]:
        print(f"\nPipeline did not produce a passing draft: {result['reason']}")
    else:
        virality_scores = run_virality_for_all(result["theme"], result["angle"], result["drafts"])
        reports = build_analysis_reports(
            theme=result["theme"],
            tension=result["tension"],
            angle=result["angle"],
            critiques=result["critiques"],
            virality_scores=virality_scores,
        )

        print("\n--- FINAL ANALYSIS REPORTS ---")
        for platform, report in reports.items():
            print(f"\n{platform}:")
            print(report.model_dump_json(indent=2))
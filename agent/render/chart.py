"""
agent/render/chart.py

Renders brand-styled DATA charts (comparisons, stats, before/after) using
matplotlib. This handles the spec's "data post" case -- when a post's
core value is a number or comparison, a clean chart communicates it far
better than trying to describe it in a code-drawn text card.

Pulls ALL colors/fonts from brand.py -- never redefines them here.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- we only ever save to file, never display a window
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

from agent.render.brand import COLORS, FONT_PATHS, CANVAS_SIZES


def _get_font_prop(style: str) -> fm.FontProperties:
    """Finds the first existing font path for a style (same fallback
    logic as brand.get_font) and wraps it for matplotlib's font system."""
    for path in FONT_PATHS.get(style, []):
        if path.exists():
            return fm.FontProperties(fname=str(path))
    print(f"[chart.py] WARNING: no font file found for '{style}', using matplotlib default.")
    return fm.FontProperties()  # matplotlib's own default


def render_bar_chart(
    title: str,
    subtitle: str,
    categories: list[str],
    values: list[float],
    output_path: str,
    value_suffix: str = "%",
    highlight_index: int | None = None,
    canvas: str = "square",
) -> str:
    """
    Renders a horizontal bar chart comparing a few categories, with one
    bar highlighted (typically the JobInGen-favorable outcome).

    highlight_index: which bar to draw in the primary brand color
                      (defaults to the LAST category, since comparisons
                      usually end on the best/JobInGen option)
    """
    headline_font = _get_font_prop("headline")
    body_font = _get_font_prop("body")

    size_px = CANVAS_SIZES[canvas]
    fig, ax = plt.subplots(figsize=(size_px[0] / 100, size_px[1] / 100), dpi=100)
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    if highlight_index is None:
        highlight_index = len(categories) - 1

    bar_colors = [COLORS["surface"] for _ in categories]
    bar_colors[highlight_index] = COLORS["primary"]

    y_pos = range(len(categories))
    bars = ax.barh(y_pos, values, color=bar_colors, height=0.55, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(val + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val}{value_suffix}", va="center", fontproperties=body_font,
                fontsize=22, color=COLORS["text_primary"])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontproperties=body_font, fontsize=20, color=COLORS["text_primary"])
    ax.invert_yaxis()  # first category on top, reading order

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.tick_params(left=False)

    fig.text(0.08, 0.93, title, fontproperties=headline_font, fontsize=28, color=COLORS["text_primary"])
    fig.text(0.08, 0.88, subtitle, fontproperties=body_font, fontsize=18, color=COLORS["text_muted"])

    plt.subplots_adjust(top=0.8, bottom=0.1, left=0.30, right=0.92)

    # Brand accent bars (top/bottom) + logo, matching brand_graphic.py's style
    fig.patches.append(plt.Rectangle((0, 0.985), 1, 0.015, transform=fig.transFigure,
                                       facecolor=COLORS["primary"], zorder=10))
    fig.patches.append(plt.Rectangle((0, 0), 1, 0.015, transform=fig.transFigure,
                                       facecolor=COLORS["dark"], zorder=10))
    fig.text(0.92, 0.04, "JobInGen", fontproperties=headline_font, fontsize=16,
              color=COLORS["primary"], ha="right")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=COLORS["background"])
    plt.close(fig)

    print(f"[chart.py] Saved bar chart to {output_path}")
    return output_path


# Quick manual test — run this file directly, zero API calls needed
if __name__ == "__main__":
    render_bar_chart(
        title="Why generic resumes get filtered",
        subtitle="ATS pass rate by resume type",
        categories=["Generic Template", "Keyword-Stuffed", "AI-Tailored (JobInGen)"],
        values=[34, 51, 89],
        output_path="content_store/test_renders/chart_test.png",
    )
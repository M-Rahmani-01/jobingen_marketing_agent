"""
agent/render/brand_graphic.py

Renders a branded PNG for TEXT-LED posts (insight cards, tips, quote-style
posts) using pure code -- no AI image generation involved. This is the
most common render path per the spec's Hybrid Renderer design (Section
10): when the content IS the hook, code-drawn graphics are more
reliable and more on-brand than an AI image model, which tends to drift
on exact colors and mangle any text baked into the image itself.

Pulls ALL colors/fonts from brand.py -- never redefines them here.
"""

from pathlib import Path
from PIL import Image, ImageDraw

from agent.render.brand import COLORS, get_font, CANVAS_SIZES, SAFE_MARGIN


def _wrap_text(draw: ImageDraw.Draw, text: str, font, max_width: int) -> list[str]:
    """Breaks text into lines that fit within max_width, measuring real
    pixel width rather than guessing by character count."""
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test_line = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test_line
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def render_insight_card(
    hook: str,
    subtext: str,
    pillar_label: str,
    output_path: str,
    canvas: str = "square",
) -> str:
    """
    Renders a single branded insight card PNG.

    hook: the headline text (from PostDraft.hook_options[0], usually)
    subtext: supporting line (e.g. from PostDraft.caption, shortened)
    pillar_label: e.g. "Educate", "Opportunity" -- shown as a small pill
    output_path: where to save the PNG
    canvas: "square" (1080x1080) or "portrait" (1080x1350) -- from brand.py

    Returns output_path, so callers can immediately use the result
    (e.g. to attach it to a PostDraft or hand it to the approval queue).
    """
    size = CANVAS_SIZES[canvas]
    img = Image.new("RGB", size, COLORS["background"])
    draw = ImageDraw.Draw(img)

    # Top accent bar — a small, consistent brand identity element
    draw.rectangle([0, 0, size[0], 16], fill=COLORS["primary"])

    # Pillar label pill (top left)
    label_font = get_font("body", 28)
    label_text = pillar_label.upper()
    label_bbox = draw.textbbox((0, 0), label_text, font=label_font)
    label_w = label_bbox[2] - label_bbox[0]
    pill_pad_x, pill_pad_y = 24, 14
    pill_x0, pill_y0 = SAFE_MARGIN, SAFE_MARGIN + 20
    pill_x1 = pill_x0 + label_w + pill_pad_x * 2
    pill_y1 = pill_y0 + 28 + pill_pad_y * 2
    draw.rounded_rectangle([pill_x0, pill_y0, pill_x1, pill_y1], radius=24, fill=COLORS["surface"])
    draw.text((pill_x0 + pill_pad_x, pill_y0 + pill_pad_y - 4), label_text,
               font=label_font, fill=COLORS["primary"])

    # Headline (the hook) — the main visual focus
    headline_font = get_font("headline", 64)
    max_text_width = size[0] - (SAFE_MARGIN * 2)
    hook_lines = _wrap_text(draw, hook, headline_font, max_text_width)

    y = pill_y1 + 60
    line_height = 78
    for line in hook_lines:
        draw.text((SAFE_MARGIN, y), line, font=headline_font, fill=COLORS["text_primary"])
        y += line_height

    # Subtext
    y += 20
    body_font = get_font("body", 32)
    sub_lines = _wrap_text(draw, subtext, body_font, max_text_width)
    for line in sub_lines:
        draw.text((SAFE_MARGIN, y), line, font=body_font, fill=COLORS["text_muted"])
        y += 44

    # Logo placeholder (bottom right) — swap for the real logo image
    # once you have one (see brand.py's LOGO_PATH)
    logo_text = "JobInGen"
    logo_font = get_font("headline", 36)
    logo_bbox = draw.textbbox((0, 0), logo_text, font=logo_font)
    logo_w = logo_bbox[2] - logo_bbox[0]
    draw.text((size[0] - SAFE_MARGIN - logo_w, size[1] - SAFE_MARGIN - 20),
               logo_text, font=logo_font, fill=COLORS["primary"])

    # Bottom accent bar
    draw.rectangle([0, size[1] - 16, size[0], size[1]], fill=COLORS["dark"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"[brand_graphic] Saved insight card to {output_path}")
    return output_path


# Quick manual test — run this file directly, zero API calls needed
if __name__ == "__main__":
    render_insight_card(
        hook="The hard truth about ATS systems: your resume isn't being ignored, it's being filtered",
        subtext="Generic tailoring gets you filtered out. See exactly what to fix.",
        pillar_label="Educate",
        output_path="content_store/test_renders/insight_card_test.png",
    )
"""
agent/render/brand.py

The SINGLE SOURCE OF TRUTH for JobInGen's visual identity. Every
rendering file (brand_graphic.py, chart.py, image_gen.py) imports from
here -- colors, fonts, canvas sizes never get redefined or drift out of
sync anywhere else.

DEFAULT PLACEHOLDER KIT -- swap these hex codes / font files any time
you get the real JobInGen brand guidelines. Nothing else in the
codebase needs to change when you do; this file is the only place
brand values live.
"""

from pathlib import Path
from PIL import ImageFont

# ─────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────
# A trustworthy blue/navy base (common for career/professional platforms)
# with one energetic accent reserved ONLY for CTAs and highlights --
# using the accent everywhere would dilute its job of drawing the eye.

COLORS = {
    "primary": "#1B4DFF",       # Engine Blue -- headlines, key accents
    "dark": "#0A1F44",          # Midnight Navy -- backgrounds, high-contrast text
    "background": "#FFFFFF",    # White -- default canvas background
    "surface": "#EAF1FF",       # Ice Blue -- soft background blocks, cards
    "accent": "#FF7A45",        # Warm coral -- CTAs and highlights ONLY, used sparingly
    "text_primary": "#0A1F44",  # body text on light backgrounds
    "text_on_dark": "#FFFFFF",  # body text on dark backgrounds
    "text_muted": "#5C6B8A",    # captions, secondary text
}


# ─────────────────────────────────────────────
# FONTS
# ─────────────────────────────────────────────

FONT_DIR = Path(__file__).parent / "fonts"

FONT_PATHS = {
    # Try the simple static file first (e.g. Poppins-Bold.ttf) -- if you
    # only have the variable font for a given family, that's tried next.
    "headline": [FONT_DIR / "Poppins-Bold.ttf", FONT_DIR / "Poppins-Variable.ttf"],
    "body": [FONT_DIR / "Inter-Regular.ttf", FONT_DIR / "Inter-Variable.ttf"],
}


def get_font(style: str, size: int) -> ImageFont.FreeTypeFont:
    """
    Loads a brand font at the given size. style is "headline" or "body".

    Tries each candidate path in order (static file first, variable
    font as a fallback). If a variable font is used for "headline", we
    select the Bold variation explicitly, since a variable font's
    default weight is usually Regular, not Bold.

    Falls back to Pillow's built-in default font (with a warning) if no
    real font file is found at all.
    """
    candidates = FONT_PATHS.get(style, [])

    for path in candidates:
        if not path.exists():
            continue

        font = ImageFont.truetype(str(path), size)

        if style == "headline" and "Variable" in path.name:
            try:
                font.set_variation_by_name("Bold")
            except Exception:
                pass  # font doesn't support named variations -- default weight is fine

        return font

    print(f"[brand.py] WARNING: no font file found for '{style}' "
          f"(checked: {[str(p) for p in candidates]}). Using Pillow's "
          f"default font as a placeholder -- download the real font "
          f"(see render/brand.py docstring) for actual brand-accurate output.")
    return ImageFont.load_default(size=size)


# ─────────────────────────────────────────────
# CANVAS SIZES — per spec Section 10 (Design Renderer)
# ─────────────────────────────────────────────

CANVAS_SIZES = {
    "square": (1080, 1080),      # standard feed post, works for both platforms
    "portrait": (1080, 1350),    # Instagram portrait / LinkedIn document pages
}


# ─────────────────────────────────────────────
# LAYOUT CONSTANTS
# ─────────────────────────────────────────────

SAFE_MARGIN = 80  # px kept clear on every edge so text/logo never touches the border
LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"  # placeholder -- add real logo later


# Quick manual check — run this file directly to confirm fonts load
if __name__ == "__main__":
    print("Brand colors:")
    for name, hex_code in COLORS.items():
        print(f"  {name}: {hex_code}")

    print("\nTesting font loading:")
    headline_font = get_font("headline", 48)
    body_font = get_font("body", 24)
    print(f"  headline font object: {headline_font}")
    print(f"  body font object: {body_font}")
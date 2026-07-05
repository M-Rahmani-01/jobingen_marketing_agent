"""
agent/render/image_gen.py

Renders a branded PNG for CONCEPT-LED posts where a plain text card feels
flat and an actual visual metaphor helps more -- e.g. "a resume vanishing
into the ATS void", a mood/scene that a solid-color card can't convey.

Design (per spec Section 10 -- Hybrid Renderer):
  1. Ask an AI image model for a MOOD/BACKGROUND image ONLY -- explicitly
     told to avoid any text, since image models are unreliable at
     rendering exact text (fonts drift, spelling breaks, brand colors
     shift when AI tries to draw letters).
  2. Composite the ACTUAL headline text on top using the exact same
     code-based text rendering as brand_graphic.py (same fonts, same
     wrapping helper, same brand colors) -- so text stays crisp and
     on-brand no matter what the AI draws underneath.

Uses Gemini's free-tier image model ("Nano Banana 2 Lite" --
gemini-3.1-flash-lite-image) via the google-genai SDK.
"""

import os
import io
from pathlib import Path
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from google import genai
from google.genai import types

from agent.render.brand import COLORS, get_font, CANVAS_SIZES, SAFE_MARGIN
from agent.render.brand_graphic import _wrap_text

load_dotenv()

IMAGE_MODEL = "gemini-3.1-flash-lite-image"

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _generate_background(mood_prompt: str, canvas: str) -> Image.Image:
    """
    Asks the image model for a MOOD/BACKGROUND ONLY. The prompt
    explicitly forbids text/letters/numbers -- code draws all real text
    afterward, so the AI's only job is atmosphere and color.
    """
    size = CANVAS_SIZES[canvas]
    aspect_hint = "square" if size[0] == size[1] else "portrait"

    full_prompt = (
        f"{mood_prompt}. Abstract, atmospheric background image only, "
        f"{aspect_hint} composition. "
        f"IMPORTANT: absolutely NO text, letters, words, numbers, or "
        f"typography anywhere in the image -- background/mood art only. "
        f"Real text will be added separately in code."
    )

    response = _client.models.generate_content(
        model=IMAGE_MODEL,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["Text", "Image"],
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            image_bytes = part.inline_data.data
            bg = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return bg.resize(size)

    raise RuntimeError(
        "Image model returned no image data. This can happen if the "
        "prompt tripped a safety filter or the model had an off run -- "
        "try rephrasing mood_prompt or running again."
    )


def render_concept_card(
    hook: str,
    subtext: str,
    pillar_label: str,
    mood_prompt: str,
    output_path: str,
    canvas: str = "square",
) -> str:
    """
    Renders a branded PNG with an AI-generated mood background and
    CODE-DRAWN text on top -- same font/wrapping rules as
    brand_graphic.py's render_insight_card, just with a generated image
    behind the text instead of a flat white canvas.

    mood_prompt: describe the VISUAL FEELING to generate (e.g. "a resume
                 dissolving into dark particles, navy and blue tones,
                 atmospheric") -- NOT the headline text itself. The
                 headline is drawn separately by this function.
    """
    size = CANVAS_SIZES[canvas]

    bg = _generate_background(mood_prompt, canvas)

    # Darken the background with a translucent navy overlay so white
    # text stays readable no matter what colors the AI happened to draw.
    overlay = Image.new("RGBA", size, (10, 20, 40, 130))  # dark navy, ~51% opacity
    bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(bg)

    # Top accent bar -- same identity element brand_graphic.py uses
    draw.rectangle([0, 0, size[0], 16], fill=COLORS["primary"])

    # Pillar label pill (top left) -- white pill since it now sits on a
    # dark/photo background instead of brand_graphic.py's plain white canvas
    label_font = get_font("body", 28)
    label_text = pillar_label.upper()
    label_bbox = draw.textbbox((0, 0), label_text, font=label_font)
    label_w = label_bbox[2] - label_bbox[0]
    pill_pad_x, pill_pad_y = 24, 14
    pill_x0, pill_y0 = SAFE_MARGIN, SAFE_MARGIN + 20
    pill_x1 = pill_x0 + label_w + pill_pad_x * 2
    pill_y1 = pill_y0 + 28 + pill_pad_y * 2
    draw.rounded_rectangle([pill_x0, pill_y0, pill_x1, pill_y1], radius=24,
                            fill=(255, 255, 255, 230))
    draw.text((pill_x0 + pill_pad_x, pill_y0 + pill_pad_y - 4), label_text,
               font=label_font, fill=COLORS["primary"])

    # Headline -- white, since it sits on a darkened background now
    headline_font = get_font("headline", 64)
    max_text_width = size[0] - (SAFE_MARGIN * 2)
    hook_lines = _wrap_text(draw, hook, headline_font, max_text_width)

    y = pill_y1 + 60
    line_height = 78
    for line in hook_lines:
        draw.text((SAFE_MARGIN, y), line, font=headline_font, fill=COLORS["text_on_dark"])
        y += line_height

    # Subtext -- soft light-blue-white for a touch of hierarchy vs. the headline
    y += 20
    body_font = get_font("body", 32)
    sub_lines = _wrap_text(draw, subtext, body_font, max_text_width)
    for line in sub_lines:
        draw.text((SAFE_MARGIN, y), line, font=body_font, fill=(220, 226, 240))
        y += 44

    # Logo (bottom right) -- white version, dark background here
    logo_text = "JobInGen"
    logo_font = get_font("headline", 36)
    logo_bbox = draw.textbbox((0, 0), logo_text, font=logo_font)
    logo_w = logo_bbox[2] - logo_bbox[0]
    draw.text((size[0] - SAFE_MARGIN - logo_w, size[1] - SAFE_MARGIN - 20),
               logo_text, font=logo_font, fill=COLORS["text_on_dark"])

    # Bottom accent bar
    draw.rectangle([0, size[1] - 16, size[0], size[1]], fill=COLORS["primary"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bg.save(output_path)
    print(f"[image_gen] Saved concept card to {output_path}")
    return output_path


# Quick manual test -- run this file directly. NOTE: this spends one
# real image-generation call every time you run it (free tier right now,
# but still a real API call) -- don't loop this in a test suite.
if __name__ == "__main__":
    render_concept_card(
        hook="Your resume isn't being read. It's disappearing into the ATS void.",
        subtext="89% of tailored applications get through. Generic ones don't.",
        pillar_label="Educate",
        mood_prompt=(
            "A resume paper dissolving into dark particles as it falls into "
            "an abstract void, dramatic lighting, navy and electric blue "
            "tones, professional and atmospheric, not literal or cartoonish"
        ),
        output_path="content_store/test_renders/concept_card_test.png",
    )
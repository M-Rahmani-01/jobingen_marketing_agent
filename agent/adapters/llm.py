"""
agent/adapters/llm.py

The ONE place that talks to the LLM provider(s). Every node calls
generate_json() instead of importing an SDK directly.

Primary: Google Gemini (free tier, daily quota).
Fallback: OpenRouter's free Nemotron model, used automatically if
Gemini's quota is exhausted or the call otherwise fails.

The fallback model doesn't have Gemini's native "force valid JSON" mode,
so on larger outputs (like Copywriter's) it can occasionally produce
almost-valid JSON (a stray comma, an unescaped quote). We run everything
through json_repair before parsing to fix small mistakes automatically
instead of crashing the whole node.
"""

import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from json_repair import repair_json

load_dotenv()

_gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)

GEMINI_MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


def _safe_parse(raw_text: str) -> dict:
    """Parses JSON, auto-repairing small mistakes (stray commas,
    unescaped quotes, minor truncation) before giving up."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        repaired = repair_json(raw_text)
        return json.loads(repaired)


def _call_gemini(system_instruction: str, user_prompt: str, temperature: float) -> dict:
    response = _gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )
    return _safe_parse(response.text)


def _call_openrouter_fallback(system_instruction: str, user_prompt: str, temperature: float) -> dict:
    response = _openrouter_client.chat.completions.create(
        model=FALLBACK_MODEL,
        temperature=temperature,
        max_tokens=4000,
        messages=[
            {"role": "system", "content": system_instruction + "\n\nOutput ONLY raw JSON, no markdown fences, no extra text."},
            {"role": "user", "content": user_prompt},
        ],
    )

    if not response.choices:
        raise RuntimeError(
            f"OpenRouter returned no choices (likely rate-limited or the "
            f"free model is temporarily overloaded). Raw response: {response}"
        )

    raw = response.choices[0].message.content
    if not raw:
        raise RuntimeError("OpenRouter returned an empty message content.")

    return _safe_parse(raw)


def generate_json(
    system_instruction: str,
    user_prompt: str,
    temperature: float = 0.7,
) -> dict:
    """
    Tries Gemini first. If that fails for ANY reason (quota exhausted,
    network error, bad JSON), automatically falls back to OpenRouter's
    free Nemotron model. Only raises if BOTH fail.
    """
    try:
        return _call_gemini(system_instruction, user_prompt, temperature)
    except Exception as gemini_error:
        print(f"[llm.py] Gemini failed ({gemini_error}) — falling back to OpenRouter...")
        try:
            return _call_openrouter_fallback(system_instruction, user_prompt, temperature)
        except Exception as fallback_error:
            raise RuntimeError(
                f"[llm.py] Both providers failed.\n"
                f"Gemini error: {gemini_error}\n"
                f"OpenRouter error: {fallback_error}"
            )


# Quick manual test — run this file directly
if __name__ == "__main__":
    result = generate_json(
        system_instruction="You output strict JSON only, no other text.",
        user_prompt='Return a JSON object like {"status": "ok", "message": "hello"}',
    )
    print(result)
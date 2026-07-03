"""
agent/adapters/llm.py

The ONE place that talks to the LLM provider(s). Every node calls
generate_json() instead of importing an SDK directly.

Primary: Google Gemini (free tier, daily quota).
Fallback: OpenRouter's "openrouter/free" auto-router -- instead of
hardcoding one specific free model (which can get overloaded by OTHER
users, as happened today with Nemotron), this automatically picks
whichever free model currently has capacity. More resilient than
pinning to a single model ID.

Also retries once with a short wait on TRANSIENT errors (503/502 "high
demand, try again") before giving up on a provider -- these often
resolve themselves within a few seconds.
"""

import os
import json
import time
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
FALLBACK_MODEL = "openrouter/free"  # auto-picks whichever free model has capacity right now

TRANSIENT_RETRY_WAIT_SECONDS = 5


def _safe_parse(raw_text: str) -> dict:
    """Parses JSON, auto-repairing small mistakes before giving up.
    Also unwraps a single-item list, since some fallback models wrap
    the object in [...] even when told not to."""
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        repaired = repair_json(raw_text)
        parsed = json.loads(repaired)

    if isinstance(parsed, list):
        if len(parsed) == 1 and isinstance(parsed[0], dict):
            return parsed[0]
        raise ValueError(f"Expected a JSON object but got a list: {parsed}")

    return parsed


def _is_transient_error(error: Exception) -> bool:
    """503/502/'UNAVAILABLE'/'high demand'/'ResourceExhausted' errors are
    usually temporary and worth one quick retry before giving up."""
    msg = str(error).lower()
    return any(term in msg for term in [
        "503", "502", "unavailable", "high demand", "resourceexhausted", "overloaded"
    ])


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
            {"role": "system", "content": system_instruction + "\n\nOutput ONLY a single raw JSON object, not an array, no markdown fences, no extra text."},
            {"role": "user", "content": user_prompt},
        ],
    )

    if not response.choices:
        raise RuntimeError(
            f"OpenRouter returned no choices (likely rate-limited or overloaded). "
            f"Raw response: {response}"
        )

    raw = response.choices[0].message.content
    if not raw:
        raise RuntimeError("OpenRouter returned an empty message content.")

    return _safe_parse(raw)


def _call_with_transient_retry(call_fn, *args) -> dict:
    """Tries call_fn(*args) once. If it fails with a TRANSIENT error,
    waits a few seconds and tries exactly once more before giving up."""
    try:
        return call_fn(*args)
    except Exception as first_error:
        if _is_transient_error(first_error):
            print(f"[llm.py] Transient error ({first_error}) -- "
                  f"waiting {TRANSIENT_RETRY_WAIT_SECONDS}s and retrying once...")
            time.sleep(TRANSIENT_RETRY_WAIT_SECONDS)
            return call_fn(*args)  # if this also fails, let it raise normally
        raise


def generate_json(
    system_instruction: str,
    user_prompt: str,
    temperature: float = 0.7,
    openrouter_max_attempts: int = 3,
) -> dict:
    """
    Tries Gemini first (with one transient-error retry). If that still
    fails for ANY reason, falls back to OpenRouter's free auto-router --
    retried up to openrouter_max_attempts times, since "openrouter/free"
    can land on a DIFFERENT underlying model each call, so a failure
    (bad JSON, empty response) is often worth just trying again rather
    than giving up immediately. Only raises if Gemini AND every
    OpenRouter attempt fail.
    """
    try:
        return _call_with_transient_retry(_call_gemini, system_instruction, user_prompt, temperature)
    except Exception as gemini_error:
        print(f"[llm.py] Gemini failed ({gemini_error}) — falling back to OpenRouter...")

        last_fallback_error = None
        for attempt in range(1, openrouter_max_attempts + 1):
            try:
                return _call_openrouter_fallback(system_instruction, user_prompt, temperature)
            except Exception as fallback_error:
                last_fallback_error = fallback_error
                print(f"[llm.py] OpenRouter attempt {attempt}/{openrouter_max_attempts} "
                      f"failed ({fallback_error}) — {'retrying...' if attempt < openrouter_max_attempts else 'giving up.'}")

        raise RuntimeError(
            f"[llm.py] Both providers failed.\n"
            f"Gemini error: {gemini_error}\n"
            f"OpenRouter error (after {openrouter_max_attempts} attempts): {last_fallback_error}"
        )


# Quick manual test — run this file directly
if __name__ == "__main__":
    result = generate_json(
        system_instruction="You output strict JSON only, no other text.",
        user_prompt='Return a JSON object like {"status": "ok", "message": "hello"}',
    )
    print(result)
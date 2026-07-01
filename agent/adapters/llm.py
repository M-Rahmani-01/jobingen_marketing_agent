"""
agent/adapters/llm.py

The ONE place that talks to the LLM provider. Every node (Synthesizer,
Scorer, Modeler, etc.) calls generate_json() instead of importing an
SDK directly. If we ever switch providers later, only this file changes.

Currently wired to Google Gemini's free tier.
"""

import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # reads GEMINI_API_KEY from .env

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

DEFAULT_MODEL = "gemini-2.5-flash"


def generate_json(
    system_instruction: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> dict:
    """
    Calls Gemini and forces the response to be valid JSON.
    Returns a parsed Python dict. Raises if the model returns
    something that isn't valid JSON (fail loud, not silent).
    """
    response = _client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )

    try:
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"[llm.py] Model did not return valid JSON.\n"
            f"Raw response: {response.text[:500]}\n"
            f"Error: {e}"
        )


# Quick manual test — run this file directly to sanity check the connection
if __name__ == "__main__":
    result = generate_json(
        system_instruction="You output strict JSON only, no other text.",
        user_prompt='Return a JSON object like {"status": "ok", "message": "hello from gemini"}',
    )
    print(result)
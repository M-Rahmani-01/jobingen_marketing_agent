"""
agent/adapters/trends.py

Real signal source #1 -- Google Trends (via the free `pytrends` library).

Why this instead of Reddit right now:
- Google Trends data is PUBLIC, AGGREGATE search-volume data -- it is not
  per-user content, so it doesn't carry the same "commercial use needs
  explicit approval" restriction that Reddit's Responsible Builder Policy
  places on business-adjacent Data API use.
- No OAuth, no app registration, no reCAPTCHA gate.

What this does:
For a small set of JobInGen-relevant seed keywords (resume, job interview,
campus placement, layoffs), it asks Google Trends "what related searches
are RISING right now in India" and turns each rising query into a
SignalItem -- the same shape trend_scout.py already expects from the fake
Reddit JSON, so nothing downstream needs to change.

Caveats (be aware, don't be surprised):
- pytrends scrapes Google Trends' public frontend rather than using an
  official API -- Google can rate-limit or temporarily block heavy use.
  This is handled defensively below: if a keyword's fetch fails, it's
  skipped and logged, not a crash.
- Google Trends' "rising" values are sometimes the string "Breakout"
  (meaning >5000% growth) instead of a number -- handled explicitly.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from pytrends.request import TrendReq
from agent.state import SignalItem

# JobInGen-relevant seed topics -- tune this list over time based on what
# actually surfaces useful rising queries.
SEED_KEYWORDS = [
    "resume",
    "mock interview",
    "campus placement",
    "layoffs india",
]

GEO = "IN"          # India
TIMEFRAME = "now 7-d"
REQUEST_DELAY_SECONDS = 2  # be polite between keyword requests

# Safety net: if a LIVE fetch fails entirely (Google's frontend changed,
# temporary block, etc.), fall back to the last successful fetch instead
# of returning an empty list -- keeps signal quality consistent even on
# a bad day. A cached result older than this is considered too stale to
# trust and won't be used.
CACHE_PATH = Path(__file__).parent.parent.parent / "content_store" / "trends_cache.json"
CACHE_MAX_AGE_HOURS = 48

# Rising queries containing these words are almost always noise -- word
# games, crossword clues, and trivia, not genuine career-anxiety signal.
# ("job interview" is a common crossword answer, which pollutes results.)
NOISE_TERMS = ["crossword", "wordle", "quiz", "trivia", "nyt "]


def _is_noise(query: str) -> bool:
    query_lower = query.lower()
    return any(term in query_lower for term in NOISE_TERMS)


def _save_cache(items: list[SignalItem]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "items": [item.model_dump() for item in items],
    }
    CACHE_PATH.write_text(json.dumps(data, indent=2))


def _load_cache() -> list[SignalItem] | None:
    """Returns cached items if a cache exists AND is fresh enough,
    otherwise None (meaning: don't use it, it's too old to trust)."""
    if not CACHE_PATH.exists():
        return None

    try:
        data = json.loads(CACHE_PATH.read_text())
        cached_at = datetime.fromisoformat(data["cached_at"])
        age_hours = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600

        if age_hours > CACHE_MAX_AGE_HOURS:
            print(f"[trends] Cache exists but is {age_hours:.1f}h old "
                  f"(max {CACHE_MAX_AGE_HOURS}h) -- too stale, not using it.")
            return None

        print(f"[trends] Using cached result from {age_hours:.1f}h ago.")
        return [SignalItem(**item) for item in data["items"]]

    except Exception as e:
        print(f"[trends] Cache file exists but couldn't be read ({e}) -- ignoring it.")
        return None


def _score_from_value(value) -> float:
    """
    Google Trends 'rising' value is usually an integer percent-growth
    number, but can be the string 'Breakout' for extreme growth (>5000%).
    Normalize both into a 0-1 score, capping at 1.0.
    """
    if isinstance(value, str) and value.strip().lower() == "breakout":
        return 1.0
    try:
        # Cap normal percent growth at 500% -> score 1.0, scale linearly below that.
        return min(float(value) / 500.0, 1.0)
    except (TypeError, ValueError):
        return 0.5  # unknown format -- neutral fallback, never crash on this


def fetch_trends_signal(keywords: list[str] | None = None) -> list[SignalItem]:
    """
    Returns a list of SignalItems built from RISING related-query data on
    Google Trends for each seed keyword, scoped to India.

    If the live fetch fails completely (Google's frontend changed, a
    temporary block, network issue) and returns nothing, falls back to
    the last successful cached result (if one exists and isn't too old)
    instead of returning an empty list.
    """
    keywords = keywords or SEED_KEYWORDS
    pytrends = TrendReq(hl="en-US", tz=330)  # tz=330 -> IST offset in minutes

    items: list[SignalItem] = []

    for keyword in keywords:
        try:
            pytrends.build_payload([keyword], timeframe=TIMEFRAME, geo=GEO)
            related = pytrends.related_queries()
            rising_df = related.get(keyword, {}).get("rising")

            if rising_df is None or rising_df.empty:
                print(f"[trends] No rising queries for '{keyword}' this run.")
                continue

            for _, row in rising_df.iterrows():
                query = str(row["query"])

                if _is_noise(query):
                    continue

                score = _score_from_value(row["value"])

                items.append(
                    SignalItem(
                        source="trends",
                        text=f"Rising search interest in India for '{query}' "
                             f"(related to '{keyword}')",
                        score=score,
                        url=f"https://trends.google.com/trends/explore?q={quote(query)}&geo={GEO}",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                )

            print(f"[trends] '{keyword}': found {len(rising_df)} rising queries.")

        except Exception as e:
            # Never let one bad keyword crash the whole signal fetch --
            # log it and move on, same philosophy as the llm.py fallback.
            print(f"[trends] Failed to fetch trends for '{keyword}': {e}")

        time.sleep(REQUEST_DELAY_SECONDS)

    if items:
        _save_cache(items)
        return items

    # Live fetch produced nothing at all (every keyword failed or was
    # empty) -- fall back to the last known-good cached result rather
    # than returning an empty list.
    print("[trends] Live fetch returned zero items -- checking cache fallback...")
    cached = _load_cache()
    if cached:
        return cached

    print("[trends] No usable cache either -- returning empty list.")
    return items


# Quick manual test -- run this file directly to see what it returns
# right now, live, with zero fake data involved.
if __name__ == "__main__":
    results = fetch_trends_signal()

    print(f"\n--- {len(results)} SIGNAL ITEMS FROM GOOGLE TRENDS ---")
    for item in results:
        print(f"\n{item.model_dump_json(indent=2)}")
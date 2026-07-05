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

import time
from datetime import datetime, timezone
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

# Rising queries containing these words are almost always noise -- word
# games, crossword clues, and trivia, not genuine career-anxiety signal.
# ("job interview" is a common crossword answer, which pollutes results.)
NOISE_TERMS = ["crossword", "wordle", "quiz", "trivia", "nyt "]


def _is_noise(query: str) -> bool:
    query_lower = query.lower()
    return any(term in query_lower for term in NOISE_TERMS)


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

    return items


# Quick manual test -- run this file directly to see what it returns
# right now, live, with zero fake data involved.
if __name__ == "__main__":
    results = fetch_trends_signal()

    print(f"\n--- {len(results)} SIGNAL ITEMS FROM GOOGLE TRENDS ---")
    for item in results:
        print(f"\n{item.model_dump_json(indent=2)}")
"""
agent/nodes/trend_scout.py

Layer 1 — "The Listener"

For now (Week 1), this reads FAKE sample data from a local JSON file
instead of hitting the real Reddit API. Same output shape either way —
that's the point. Week 5 swaps the inside of this function for a real
adapters/reddit.py call without changing anything downstream.
"""

import json
from pathlib import Path

from agent.state import SignalItem


FAKE_DATA_PATH = Path("content_store/fake_reddit_data.json")


def run_trend_scout(data_path: Path = FAKE_DATA_PATH) -> list[SignalItem]:
    """
    Reads raw signal items and returns them as a list of validated
    SignalItem objects. Never invents data — every item must come from
    the source file (later: a real API response).
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Fake signal data not found at {data_path}. "
            "Did you save fake_reddit_data.json into content_store/?"
        )

    with open(data_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    signal_items = [SignalItem(**item) for item in raw_items]

    print(f"[trend_scout] Loaded {len(signal_items)} signal items.")
    return signal_items


# Quick manual test — run this file directly to sanity check it
if __name__ == "__main__":
    items = run_trend_scout()
    for item in items[:3]:
        print(f"- ({item.source}, score={item.score}) {item.text[:60]}...")
"""
tests/fixtures.py

PURPOSE:
Every time you test a downstream node (copywriter.py, critics.py, etc.)
by re-running the FULL pipeline from trend_scout onward, you burn API
quota on nodes that already work and don't need re-testing.

This file lets you save ONE known-good pipeline state (theme, tension,
angle, briefs) to a JSON file once, then load it instantly for every
future test -- no API calls spent on the upstream nodes at all.

This is the same philosophy as your golden_set/ folder (Week 6 in the
roadmap) -- just an earlier, lighter version of "reuse known-good test
data instead of regenerating it every time."
"""

import json
from pathlib import Path
from agent.state import Theme, Angle, PlatformBrief

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_run.json"


def save_fixture(
    theme: Theme,
    tension: str,
    angle: Angle,
    briefs: dict[str, PlatformBrief],
) -> None:
    """Saves a known-good pipeline state to disk as JSON."""
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "theme": theme.model_dump(),
        "tension": tension,
        "angle": angle.model_dump(),
        "briefs": {platform: brief.model_dump() for platform, brief in briefs.items()},
    }

    FIXTURE_PATH.write_text(json.dumps(data, indent=2))
    print(f"[fixtures] Saved sample run to {FIXTURE_PATH}")


def load_fixture() -> tuple[Theme, str, Angle, dict[str, PlatformBrief]]:
    """Loads a previously saved pipeline state back into typed objects."""
    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(
            f"No fixture found at {FIXTURE_PATH}.\n"
            "Run copywriter.py once with a live pipeline first -- it will "
            "create this fixture automatically for future runs."
        )

    data = json.loads(FIXTURE_PATH.read_text())

    theme = Theme(**data["theme"])
    tension = data["tension"]
    angle = Angle(**data["angle"])
    briefs = {
        platform: PlatformBrief(**brief_data)
        for platform, brief_data in data["briefs"].items()
    }

    print(f"[fixtures] Loaded saved run from {FIXTURE_PATH} (no API calls used)")
    return theme, tension, angle, briefs
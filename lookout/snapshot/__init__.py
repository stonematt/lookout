"""
The dashboard snapshot: built in Python, published on Storj, read by the site.

- `model`: the `Snapshot` dataclass and its two JSON documents (the contract).
- `build`: `build_snapshot`, pure, with an injected `now`.
- `current`: live stitching, current conditions and the active rain event.
- `gaps`: the gap status of months and windows, and the guard that keeps
  invalid comparisons uncomputed.
- `rain_tab`: the Rain tab's section: window totals, daily and hourly rain,
  dry spells and the water-year curves.
- `publish`: the Storj keys and the write.
- `events`: the Rain Events tab's section and event curves (#84).
- `curves`: each completed event's curve file, written once.
- `almanac`: the Overview's almanac grid and rolling-window rain table (#82).
"""

from lookout.snapshot.build import build_snapshot
from lookout.snapshot.current import stitch_live
from lookout.snapshot.model import (
    CONTEXT_FIELDS,
    SCHEMA_VERSION,
    ActiveRainEvent,
    Gaps,
    GapStatus,
    SnapshotConditions,
    Snapshot,
    Station,
)

__all__ = [
    "CONTEXT_FIELDS",
    "SCHEMA_VERSION",
    "ActiveRainEvent",
    "Gaps",
    "GapStatus",
    "SnapshotConditions",
    "Snapshot",
    "Station",
    "build_snapshot",
    "stitch_live",
]

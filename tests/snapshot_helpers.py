"""
Shared builders for the snapshot tests: the fixture station, its two golden
scenarios, and `build_snapshot` called the way the refresh run calls it.

`REGENERATE` is set by `LOOKOUT_REGEN_GOLDEN=1`; a golden test that sees it
rewrites its file instead of comparing.
"""

import os
from typing import Dict, Optional

import pandas as pd

from lookout.core.energy_catalog import EnergyCatalog
from lookout.core.rain_events import RainEventCatalog
from lookout.snapshot import Snapshot, Station, build_snapshot
from tests.fixtures.ambient import load_devices, to_ambient_records
from tests.fixtures.archive import FAKE_MAC, SCENARIO, build_archive
from tests.fixtures.real import AMBIENT_END, load_real_slice

REGENERATE = os.environ.get("LOOKOUT_REGEN_GOLDEN") == "1"
STATION = Station(mac=FAKE_MAC, name="Fixture Station")


def ms(when: str) -> int:
    """Epoch ms for an ISO-8601 UTC time."""
    return int(pd.Timestamp(when).value // 10**6)


def live_row(df: pd.DataFrame, when: str) -> Dict:
    """The archive row at `when`, shaped as Ambient's `lastData`."""
    return to_ambient_records(df[df["dateutc"] == ms(when)])[0]


def scenario_until(end: str) -> pd.DataFrame:
    """The SCENARIO station, cut off before `end` (exclusive)."""
    return build_archive(SCENARIO.start, end, rain=SCENARIO.rain, gaps=SCENARIO.gaps)


def event_catalog(archive: pd.DataFrame) -> pd.DataFrame:
    """The Event Catalog detected from `archive`, with stable ids."""
    catalog = RainEventCatalog(FAKE_MAC).detect_and_catalog_events(
        archive, auto_save=False
    )
    catalog["event_id"] = [f"event-{i + 1}" for i in range(len(catalog))]
    return catalog


def energy_catalog(archive: pd.DataFrame) -> pd.DataFrame:
    """The energy catalog computed from `archive`, as the refresh run does."""
    return EnergyCatalog(FAKE_MAC).detect_and_calculate_periods(
        archive, auto_save=False
    )


def build(
    archive: pd.DataFrame,
    live: Optional[Dict],
    now: str,
    catalog: Optional[pd.DataFrame] = None,
    energy: Optional[pd.DataFrame] = None,
) -> Snapshot:
    """`build_snapshot` for the fixture station; no energy catalog unless given."""
    return build_snapshot(
        archive=archive,
        live=live,
        event_catalog=catalog,
        energy_catalog=energy,
        station=STATION,
        now=pd.Timestamp(now),
    )


# ---------- the golden scenarios ----------


def dry_station() -> Snapshot:
    """
    The real slice, with the recorded live reading three hours past its end.

    Dry: the slice's rain ended two days earlier.
    """
    archive = load_real_slice()
    live = load_devices()[0]["lastData"]
    return build(
        archive, live, AMBIENT_END, event_catalog(archive), energy_catalog(archive)
    )


# Mid event 3 (rain 22:00Z-02:00Z): archive to 00:00Z, live at 00:25Z.
RAIN_LIVE = "2026-01-09T00:25:00Z"


def raining_station() -> Snapshot:
    """
    The SCENARIO station mid event 3, the live reading ahead of the archive.

    Event 3 is ongoing in the Event Catalog; the live reading has more rain.
    """
    full = scenario_until("2026-01-09T00:30:00Z")
    archive = full[full["dateutc"] < ms("2026-01-09T00:00:00Z")]
    live = live_row(full, RAIN_LIVE)
    return build(
        archive,
        live,
        "2026-01-09T00:27:00Z",
        event_catalog(archive),
        energy_catalog(archive),
    )

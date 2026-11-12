"""
Shared builders for the snapshot tests: the fixture stations, the three golden
scenarios (`dry`, `raining` and `seasoned`), and `build_snapshot` called the
way the refresh run calls it.

`REGENERATE` is set by `LOOKOUT_REGEN_GOLDEN=1`; a golden test that sees it
rewrites its file instead of comparing.
"""

import os
from typing import Dict, Optional

import pandas as pd

from lookout.core.energy_catalog import EnergyCatalog
from lookout.core.rain_events import RainEventCatalog
from lookout.snapshot import Snapshot, Station, build_snapshot
from lookout.snapshot.model import StationLocation
from lookout.snapshot.status import CodeVersion
from tests.fixtures.ambient import load_devices, to_ambient_records
from tests.fixtures.archive import (
    FAKE_MAC,
    SCENARIO,
    SEASONED_MAC,
    SEASONED_NOW,
    STATION_TZ,
    build_archive,
    seasoned_archive,
)
from tests.fixtures.real import AMBIENT_END, load_real_slice

REGENERATE = os.environ.get("LOOKOUT_REGEN_GOLDEN") == "1"
# The recorded device's coordinates, as `Station.from_device` reads them.
STATION = Station(
    mac=FAKE_MAC,
    name="Fixture Station",
    tz=STATION_TZ,
    location=StationLocation(lat=45.0, lon=-123.0),
)
# The seasoned station (#461): its own fake MAC, the same zone and place.
SEASONED_STATION = Station(
    mac=SEASONED_MAC,
    name="Seasoned Station",
    tz=STATION_TZ,
    location=StationLocation(lat=45.0, lon=-123.0),
)
# The code a test run reports in its status document (#268): never git's.
FIXED_CODE = CodeVersion(version="2026.11.1", sha="0123456789abcdef" * 2 + "01234567")


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


# The seasoned station's archive stops 30 minutes before its live reading, as
# the refresh run's archive lags Ambient; it is built a minute after the live.
SEASONED_ARCHIVE_NEWEST = "2026-05-20T21:30:00Z"
SEASONED_BUILT = "2026-05-20T22:01:00Z"


def seasoned_station() -> Snapshot:
    """
    The seasoned station, its live reading at `SEASONED_NOW`.

    Two earlier years around 20 May and a recent run of 45 days. The archive
    ends at `SEASONED_ARCHIVE_NEWEST`; the live reading, half an hour ahead,
    carries the temperature pushed past the 90th percentile of its usual
    window and sensor 1's low battery.
    """
    full = seasoned_archive()
    archive = full[full["dateutc"] <= ms(SEASONED_ARCHIVE_NEWEST)].reset_index(
        drop=True
    )
    live = live_row(full, SEASONED_NOW)
    return build_snapshot(
        archive=archive,
        live=live,
        event_catalog=event_catalog(archive),
        energy_catalog=energy_catalog(archive),
        station=SEASONED_STATION,
        now=pd.Timestamp(SEASONED_BUILT),
    )

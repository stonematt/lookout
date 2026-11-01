"""
build.py: `build_snapshot`, the one pure function behind the dashboard.

It reads no storage, no clock and no session state: everything comes in as
arguments, and the `Snapshot` goes out. It makes the one `StationClock`, the
build time with the station's zone, and on it counts the archive's days and
extracts the daily rain series once (`lookout.snapshot.days.count_days`);
every section shares those, so no two judge a day two ways. The
refresh command (`lookout.cli.refresh`) loads the inputs and publishes the
result.

Later tickets add their per-tab sections here: compute the section from the
same `stitched` history (so live values are used once) and add its field to
`lookout.snapshot.model.Snapshot`.
"""

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from lookout.snapshot.almanac import almanac
from lookout.snapshot.clock import StationClock
from lookout.snapshot.current import (
    active_rain_event,
    current_conditions,
    stitch_live,
)
from lookout.snapshot.days import count_days
from lookout.snapshot.events import rain_events_section
from lookout.snapshot.gaps import snapshot_gaps
from lookout.snapshot.lead import hero_lead
from lookout.snapshot.model import Snapshot, Station
from lookout.snapshot.rain_tab import rain_tab
from lookout.snapshot.ribbon import condition_ribbon
from lookout.snapshot.solar import solar_tab
from lookout.snapshot.temperature import temperature_context


def build_snapshot(
    archive: pd.DataFrame,
    live: Optional[Dict[str, Any]],
    event_catalog: Optional[pd.DataFrame],
    energy_catalog: Optional[pd.DataFrame],
    station: Station,
    now: datetime,
) -> Snapshot:
    """
    Build the dashboard snapshot for one station.

    The archive is used as handed: rain corrections are not applied here.
    Storage holds the archive raw; the caller applies the corrections once
    before building (the refresh run's `correct_rain`, #95).

    :param archive: The station's archive, rain corrected (Ambient columns,
        `dateutc` in ms).
    :param live: The live `lastData` from Ambient, or None.
    :param event_catalog: The Event Catalog, or None when there is none.
    :param energy_catalog: The energy catalog, or None: the Solar tab's
        source.
    :param station: The station.
    :param now: The build time, timezone-aware.
    :return: The snapshot.
    :raises ValueError: If `now` has no timezone.
    """
    if now.tzinfo is None:
        raise ValueError("build_snapshot needs a timezone-aware `now`")
    # The zone is still `DEFAULT_TZ`; ADR-0005 moves it to the station's own.
    clock = StationClock(now)

    archive_newest_ms = int(archive["dateutc"].max()) if not archive.empty else None
    stitched = stitch_live(archive, live)
    observed_at_ms = int(stitched["dateutc"].iloc[0]) if not stitched.empty else None
    current = current_conditions(stitched, clock)
    active = active_rain_event(event_catalog, current, observed_at_ms)
    days = count_days(archive, current, clock)

    return Snapshot(
        station=station,
        built_at_ms=clock.now_ms,
        observed_at_ms=observed_at_ms,
        archive_newest_ms=archive_newest_ms,
        current=current,
        active_rain_event=active,
        gaps=snapshot_gaps(days.coverage),
        condition_ribbon=condition_ribbon(stitched, observed_at_ms, clock),
        rain_events=rain_events_section(
            event_catalog,
            stitched,
            active,
            observed_at_ms,
            clock,
            days.rain.water_year_status,
        ),
        hero_lead=hero_lead(
            archive, stitched, event_catalog, active, current, observed_at_ms, days
        ),
        rain_tab=rain_tab(stitched, days),
        temperature_context=temperature_context(
            stitched, current, observed_at_ms, days.coverage
        ),
        almanac=almanac(days),
        solar_tab=solar_tab(
            energy_catalog, archive, stitched, observed_at_ms, days.coverage
        ),
    )

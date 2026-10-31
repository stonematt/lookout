"""
build.py: `build_snapshot`, the one pure function behind the dashboard.

It reads no storage, no clock and no session state: everything comes in as
arguments, and the `Snapshot` goes out. The refresh command
(`lookout.cli.refresh`) loads the inputs and publishes the result.

Later tickets add their per-tab sections here: compute the section from the
same `stitched` history (so live values are used once) and add its field to
`lookout.snapshot.model.Snapshot`.
"""

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from lookout.snapshot.almanac import almanac
from lookout.snapshot.current import (
    active_rain_event,
    current_conditions,
    stitch_live,
    to_ms,
)
from lookout.snapshot.events import rain_events_section
from lookout.snapshot.gaps import snapshot_gaps
from lookout.snapshot.lead import hero_lead
from lookout.snapshot.model import Snapshot, Station
from lookout.snapshot.rain_tab import rain_tab
from lookout.snapshot.ribbon import condition_ribbon
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

    The archive is used as handed: rain corrections are not applied again
    (catchup has already applied them, see #95).

    :param archive: The station's archive (Ambient columns, `dateutc` in ms).
    :param live: The live `lastData` from Ambient, or None.
    :param event_catalog: The Event Catalog, or None when there is none.
    :param energy_catalog: The energy catalog, or None. Not read yet: the
        Solar tab ticket (#85) is its first reader.
    :param station: The station.
    :param now: The build time, timezone-aware.
    :return: The snapshot.
    :raises ValueError: If `now` has no timezone.
    """
    if now.tzinfo is None:
        raise ValueError("build_snapshot needs a timezone-aware `now`")

    archive_newest_ms = int(archive["dateutc"].max()) if not archive.empty else None
    stitched = stitch_live(archive, live)
    observed_at_ms = int(stitched["dateutc"].iloc[0]) if not stitched.empty else None
    current = current_conditions(stitched)
    active = active_rain_event(event_catalog, current, observed_at_ms)

    return Snapshot(
        station=station,
        built_at_ms=to_ms(pd.Timestamp(now)),
        observed_at_ms=observed_at_ms,
        archive_newest_ms=archive_newest_ms,
        current=current,
        active_rain_event=active,
        gaps=snapshot_gaps(archive, now),
        condition_ribbon=condition_ribbon(stitched, observed_at_ms),
        rain_events=rain_events_section(
            event_catalog, stitched, active, observed_at_ms
        ),
        hero_lead=hero_lead(
            archive, stitched, event_catalog, active, current, observed_at_ms, now
        ),
        rain_tab=rain_tab(archive, stitched, current, now),
        temperature_context=temperature_context(
            archive, stitched, current, observed_at_ms, now
        ),
        almanac=almanac(archive, current, now),
    )

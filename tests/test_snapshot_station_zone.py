"""
The station's zone, through the snapshot build (#197, ADR-0005).

The build's station clock takes its zone from the station, and the station
names it in both documents. The same rain, built for a station on the
Pacific coast and one in Tokyo (each archive's counters resetting at its own
midnight, as a real station's do), lands on different local days, water
years and ribbon resets.
"""

from typing import Optional

import pandas as pd
import pytest

from lookout.snapshot import Snapshot, Station, build_snapshot
from tests.fixtures.archive import FAKE_MAC, RainSpell, build_archive
from tests.snapshot_helpers import event_catalog

PACIFIC = "America/Los_Angeles"
TOKYO = "Asia/Tokyo"

# 14:00Z-17:00Z: 06:00-09:00 on the Pacific coast, 23:00-02:00 in Tokyo.
SPELL = RainSpell("2026-01-06T14:00:00Z", hours=3, inches_per_hour=0.12)
NOW = "2026-01-07T12:00:00Z"

# 03:00Z on 1 October: still 30 September (water year 2025) on the Pacific
# coast, already 1 October (water year 2026) in Tokyo.
OCTOBER_SPELL = RainSpell("2025-10-01T03:00:00Z", hours=1, inches_per_hour=0.24)
OCTOBER_NOW = "2025-10-03T00:00:00Z"


def build_in(
    tz: str,
    start: str,
    now: str,
    spell: RainSpell,
    catalog: bool = False,
) -> Snapshot:
    """
    Build one station's snapshot, its archive and its station both in `tz`.

    :param tz: The station's zone.
    :param start: The archive's first row.
    :param now: The build time, and the end of the archive.
    :param spell: The rain.
    :param catalog: Detect an Event Catalog from the archive.
    :return: The snapshot.
    """
    archive = build_archive(start, now, rain=[spell], tz=tz)
    events: Optional[pd.DataFrame] = event_catalog(archive) if catalog else None
    return build_snapshot(
        archive=archive,
        live=None,
        event_catalog=events,
        energy_catalog=None,
        station=Station(mac=FAKE_MAC, name="Fixture Station", tz=tz),
        now=pd.Timestamp(now),
    )


def daily_rain(snapshot: Snapshot) -> dict:
    """The Rain tab's recent days with rain, by local day."""
    recent = snapshot.rain_tab.recent
    first = pd.Timestamp(recent.first_day)
    return {
        (first + pd.Timedelta(days=i)).date().isoformat(): total
        for i, total in enumerate(recent.rain_in)
        if total
    }


@pytest.mark.parametrize("tz", [PACIFIC, TOKYO])
def test_both_documents_name_the_stations_zone(tz):
    snapshot = build_in(tz, "2026-01-05T00:00:00Z", NOW, SPELL)

    assert snapshot.dashboard()["station"]["tz"] == tz
    assert snapshot.context()["station"]["tz"] == tz


def test_the_days_break_at_the_stations_midnight():
    pacific = build_in(PACIFIC, "2025-12-30T00:00:00Z", NOW, SPELL)
    tokyo = build_in(TOKYO, "2025-12-30T00:00:00Z", NOW, SPELL)

    assert daily_rain(pacific) == {"2026-01-06": pytest.approx(0.36)}
    assert daily_rain(tokyo) == {
        "2026-01-06": pytest.approx(0.12),
        "2026-01-07": pytest.approx(0.24),
    }


@pytest.mark.parametrize("tz", [PACIFIC, TOKYO])
def test_the_ribbon_resets_the_counter_at_the_stations_midnight(tz):
    ribbon = build_in(tz, "2026-01-05T00:00:00Z", NOW, SPELL).condition_ribbon

    assert sum(r for r in ribbon.rain_in if r) == pytest.approx(0.36)
    assert ribbon.rain_hours == pytest.approx(3.0)


@pytest.mark.parametrize("tz, water_year", [(PACIFIC, 2025), (TOKYO, 2026)])
def test_an_event_falls_in_the_stations_water_year(tz, water_year):
    snapshot = build_in(
        tz, "2025-09-28T00:00:00Z", OCTOBER_NOW, OCTOBER_SPELL, catalog=True
    )

    assert [row.water_year for row in snapshot.rain_events.events] == [water_year]
    assert [season.water_year for season in snapshot.rain_events.seasons] == [
        water_year
    ]

"""
A counter drop is the daily rain counter's reset, whenever it falls (#654).

Every reader of the daily counter (hourly rain, daily totals, the ribbon,
event curves, event quality) takes the rain from the counter's increments,
where a drop is the reset: the reading after it holds only new rain. No
reader assumes the reset falls at local midnight. Two stations show why:

- Lookout Post's counter resets five minutes after local midnight, so its
  00:00 reading still holds yesterday's total (#328, #134).
- Salmon Run's counter resets at 21:00 Pacific (midnight Eastern), so every
  Pacific day spans two counter cycles (#328).

Through `build_snapshot` and the Event Catalog build; archives are synthetic
five-minute stations from `tests.fixtures.archive`; local days are Pacific.
"""

from datetime import date
from typing import Optional

import pandas as pd
import pytest

from lookout.snapshot import Snapshot
from lookout.snapshot.events import completed_curves
from tests.fixtures.archive import RainSpell, build_archive
from tests.snapshot_helpers import build, event_catalog, live_row, ms

LATE = pd.Timedelta(minutes=5)  # Lookout Post's reset, 00:05 local
EASTERN = pd.Timedelta(hours=-3)  # Salmon Run's reset, 21:00 local


def approx(value: float):
    """Rain totals to a thousandth of an inch."""
    return pytest.approx(value, abs=1e-3)


def recent_day(snapshot: Snapshot, iso: str):
    """The Rain tab's recent rain total and hours for one local day."""
    recent = snapshot.rain_tab.recent
    i = (date.fromisoformat(iso) - date.fromisoformat(recent.first_day)).days
    return recent.rain_in[i], recent.hourly_in[i]


def snapshot_of(
    archive: pd.DataFrame, now: str, catalog: Optional[pd.DataFrame] = None
) -> Snapshot:
    """`build_snapshot` at `now`, the live reading the archive's newest row."""
    newest = pd.Timestamp(int(archive["dateutc"].max()), unit="ms", tz="UTC")
    return build(archive, live_row(archive, newest.isoformat()), now, catalog)


class TestALateReset:
    """Lookout Post: the counter resets at 00:05, after the 00:00 reading."""

    def test_yesterdays_rain_stays_out_of_todays_midnight_hour(self):
        # Jan 5, 10:00-13:00 local: 0.3 in; Jan 6 is dry.
        archive = build_archive(
            "2026-01-04T08:00:00Z",
            "2026-01-07T08:00:00Z",
            rain=[RainSpell("2026-01-05T18:00:00Z", 3, 0.1)],
            daily_reset=LATE,
        )

        snapshot = snapshot_of(archive, "2026-01-07T08:00:00Z")

        jan_5, _ = recent_day(snapshot, "2026-01-05")
        jan_6, jan_6_hours = recent_day(snapshot, "2026-01-06")
        assert jan_5 == approx(0.3)
        assert jan_6 == 0.0
        assert jan_6_hours[0] == 0.0

    def test_rain_across_midnight_lands_in_the_hour_it_fell(self):
        # 0.12 in/h from 23:00 Jan 5 to 01:00 Jan 6 local.
        archive = build_archive(
            "2026-01-04T08:00:00Z",
            "2026-01-07T08:00:00Z",
            rain=[RainSpell("2026-01-06T07:00:00Z", 2, 0.12)],
            daily_reset=LATE,
        )

        snapshot = snapshot_of(archive, "2026-01-07T08:00:00Z")

        jan_5, jan_5_hours = recent_day(snapshot, "2026-01-05")
        jan_6, jan_6_hours = recent_day(snapshot, "2026-01-06")
        assert jan_5_hours[23] == approx(0.12)
        assert jan_6_hours[0] == approx(0.12)
        assert jan_5 == approx(0.12)
        assert jan_6 == approx(0.12)


class TestAResetAtNinePm:
    """Salmon Run: the counter resets at 21:00 Pacific, midnight Eastern."""

    def archive(self) -> pd.DataFrame:
        # Jan 5 local: 0.1 in at 10:00-11:00, 0.2 in at 21:00-23:00.
        # Jan 6 local: 0.1 in at 02:00-03:00.
        return build_archive(
            "2026-01-04T08:00:00Z",
            "2026-01-07T08:00:00Z",
            rain=[
                RainSpell("2026-01-05T18:00:00Z", 1, 0.1),
                RainSpell("2026-01-06T05:00:00Z", 2, 0.1),
                RainSpell("2026-01-06T10:00:00Z", 1, 0.1),
            ],
            daily_reset=EASTERN,
        )

    def test_hourly_rain_lands_in_the_hour_it_fell(self):
        snapshot = snapshot_of(self.archive(), "2026-01-07T08:00:00Z")

        _, jan_5 = recent_day(snapshot, "2026-01-05")
        _, jan_6 = recent_day(snapshot, "2026-01-06")
        assert jan_5[10] == approx(0.1)
        assert jan_5[21] == approx(0.1)
        assert jan_5[22] == approx(0.1)
        assert jan_6[0] == 0.0
        assert jan_6[2] == approx(0.1)

    def test_daily_totals_span_both_counter_cycles(self):
        snapshot = snapshot_of(self.archive(), "2026-01-07T08:00:00Z")

        jan_5, _ = recent_day(snapshot, "2026-01-05")
        jan_6, _ = recent_day(snapshot, "2026-01-06")
        assert jan_5 == approx(0.3)
        assert jan_6 == approx(0.1)


class TestAnEventAcrossAReset:
    """
    A shower at 0.36 in/h for 23:30-00:30 local, with a burst of 0.1 in in
    the five minutes before 00:00: the reading right after the reset is the
    event's peak (1.56 in/h) and makes it a storm.
    """

    def archive(self, daily_reset: pd.Timedelta = pd.Timedelta(0)) -> pd.DataFrame:
        return build_archive(
            "2026-01-05T08:00:00Z",
            "2026-01-07T08:00:00Z",
            rain=[
                RainSpell("2026-01-06T07:30:00Z", 1, 0.36),
                RainSpell("2026-01-06T08:00:00Z", 1 / 12, 1.2),
            ],
            daily_reset=daily_reset,
        )

    def test_the_catalog_keeps_the_first_reading_after_the_reset(self):
        (event,) = event_catalog(self.archive()).to_dict("records")

        assert event["max_hourly_rate"] == approx(1.56)
        assert event["quality_rating"] == "excellent"

    def test_the_listed_event_is_a_storm(self):
        archive = self.archive()
        catalog = event_catalog(archive)

        snapshot = snapshot_of(archive, "2026-01-07T08:00:00Z", catalog)

        (row,) = snapshot.rain_events.events
        assert row.peak_rate_in_hr == approx(1.56)
        assert row.category == "storm"
        assert row.quality == "excellent"

    def test_its_curve_keeps_the_rate_at_the_reset(self):
        archive = self.archive()

        (curve,) = completed_curves(event_catalog(archive), archive)

        at_reset = curve.t_ms.index(ms("2026-01-06T08:00:00Z"))
        assert curve.rate_in_hr[at_reset] == approx(1.56)


class TestTheRibbon:
    def test_a_reset_70_minutes_late_keeps_its_rain(self):
        # 0.12 in/h, 00:00-02:00 local Jan 6; the counter resets at 01:10.
        archive = build_archive(
            "2026-01-05T08:00:00Z",
            "2026-01-06T12:00:00Z",
            rain=[RainSpell("2026-01-06T08:00:00Z", 2, 0.12)],
            daily_reset=pd.Timedelta(minutes=70),
        )

        ribbon = snapshot_of(archive, "2026-01-06T12:00:00Z").condition_ribbon

        # 01:00-01:15 local holds three readings, the reset's among them.
        i = (ms("2026-01-06T09:00:00Z") - ribbon.start_ms) // (15 * 60 * 1000)
        assert ribbon.rain_in[i] == approx(0.03)

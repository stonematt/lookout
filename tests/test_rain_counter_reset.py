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

from lookout.core.rain_corrections import detect_carryovers
from lookout.core.rain_counter import rain_increments
from lookout.core.rainfall_analysis import extract_daily_rainfall, hourly_rain
from lookout.snapshot import Snapshot
from lookout.snapshot.events import completed_curves
from tests.fixtures.archive import Gap, RainSpell, build_archive
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


class TestAResetHiddenInAGap:
    """
    The station is offline from 16:00 to 12:00 the next day while 0.4 in
    falls after the reset, more than the 0.1 in the counter held before:
    the counter rises across the gap, and only the gap's span tells the
    reset (#654 review). With fewer than three drops to learn the reset
    time from, local midnight stands.
    """

    def archive(self) -> pd.DataFrame:
        return build_archive(
            "2026-01-04T08:00:00Z",
            "2026-01-08T08:00:00Z",
            rain=[
                RainSpell("2026-01-04T18:00:00Z", 1, 0.1),  # Jan 4, 10:00
                RainSpell("2026-01-05T18:00:00Z", 1, 0.1),  # Jan 5, 10:00
                RainSpell("2026-01-06T10:00:00Z", 4, 0.1),  # Jan 6, 02:00
            ],
            gaps=[Gap("2026-01-06T00:00:00Z", "2026-01-06T20:00:00Z")],
            daily_reset=LATE,
        )

    def test_the_day_after_the_gap_keeps_all_its_rain(self):
        totals = extract_daily_rainfall(self.archive()).set_index("date")
        hours = hourly_rain(self.archive()).set_index(["date", "hour"])["rain_in"]

        assert totals.loc[date(2026, 1, 5), "rainfall"] == approx(0.1)
        assert totals.loc[date(2026, 1, 6), "rainfall"] == approx(0.4)
        assert hours[(date(2026, 1, 6), 12)] == approx(0.4)

    def test_a_gap_over_midnight_but_not_the_reset_holds_none(self):
        # Salmon Run resets at 21:00, learned from three drops: offline
        # 22:00 to 06:00, it does not reset at midnight, so the rise across
        # the gap is the rain.
        archive = build_archive(
            "2026-01-03T08:00:00Z",
            "2026-01-07T08:00:00Z",
            rain=[
                RainSpell("2026-01-03T18:00:00Z", 1, 0.1),  # Jan 3, 10:00
                RainSpell("2026-01-04T18:00:00Z", 1, 0.1),  # Jan 4, 10:00
                RainSpell("2026-01-05T18:00:00Z", 1, 0.1),  # Jan 5, 10:00
                RainSpell("2026-01-06T05:00:00Z", 1, 0.1),  # Jan 5, 21:00
                RainSpell("2026-01-06T10:00:00Z", 1, 0.1),  # Jan 6, 02:00
            ],
            gaps=[Gap("2026-01-06T06:00:00Z", "2026-01-06T14:00:00Z")],
            daily_reset=EASTERN,
        )

        totals = extract_daily_rainfall(archive).set_index("date")["rainfall"]

        assert totals[date(2026, 1, 5)] == approx(0.2)
        assert totals[date(2026, 1, 6)] == approx(0.1)

    def test_a_reset_time_that_moves_is_followed(self):
        # Six resets at 21:00 in early December, then the console's zone is
        # set (#651) and three resets fall at 00:05 in January. Offline
        # 20:00 to 23:00 on Jan 15, the counter no longer resets at 21:00,
        # so the rise across the gap is the rain: 0.05 + 0.1 in that day.
        before = build_archive(
            "2025-12-01T08:00:00Z",
            "2025-12-20T08:00:00Z",
            rain=[RainSpell(f"2025-12-0{d}T18:00:00Z", 1, 0.05) for d in range(2, 8)],
            daily_reset=EASTERN,
        )
        after = build_archive(
            "2025-12-20T08:00:00Z",
            "2026-01-17T08:00:00Z",
            rain=[
                RainSpell("2026-01-10T18:00:00Z", 1, 0.05),  # Jan 10, 10:00
                RainSpell("2026-01-11T18:00:00Z", 1, 0.05),
                RainSpell("2026-01-12T18:00:00Z", 1, 0.05),
                RainSpell("2026-01-15T18:00:00Z", 1, 0.05),  # Jan 15, 10:00
                RainSpell("2026-01-16T05:30:00Z", 1, 0.1),  # Jan 15, 21:30
            ],
            gaps=[Gap("2026-01-16T04:00:00Z", "2026-01-16T07:00:00Z")],
            daily_reset=LATE,
        )
        archive = pd.concat([after, before], ignore_index=True)

        totals = extract_daily_rainfall(archive).set_index("date")["rainfall"]

        assert totals[date(2025, 12, 2)] == approx(0.05)
        assert totals[date(2026, 1, 15)] == approx(0.15)


class TestRainSoFar:
    def test_a_late_reset_never_counts_the_day_befores_rain(self):
        # 0.3 in on Jan 4 2025 at 10:00; Jan 5 is dry. At noon on Jan 5 the
        # rain so far is 0, though its 00:00 reading still held 0.3.
        archive = build_archive(
            "2024-12-01T08:00:00Z",
            "2026-01-06T20:00:00Z",
            rain=[RainSpell("2025-01-04T18:00:00Z", 1, 0.3)],
            daily_reset=LATE,
        )

        snapshot = snapshot_of(archive, "2026-01-06T20:02:00Z")
        (strip,) = [s for s in snapshot.temperature_context.strips if s.kind == "rain"]

        assert [v for v in strip.values if v > 0] == [approx(0.3)]


class TestCarryoverDetection:
    def archive(self) -> pd.DataFrame:
        # 0.3 in on Jan 5; offline over midnight, the counter keeps it.
        return build_archive(
            "2026-01-04T08:00:00Z",
            "2026-01-07T08:00:00Z",
            rain=[RainSpell("2026-01-05T18:00:00Z", 1, 0.3)],
            gaps=[Gap("2026-01-06T07:00:00Z", "2026-01-06T09:00:00Z", True)],
        )

    def test_a_kept_total_is_a_carryover(self):
        (carryover,) = detect_carryovers(self.archive())

        assert carryover["affected_date"] == "2026-01-06"
        assert carryover["carryover_amount"] == approx(0.3)

    def test_a_small_drop_is_a_reset_not_a_carryover(self):
        archive = self.archive()
        first = archive["dateutc"] == ms("2026-01-06T09:00:00Z")
        archive.loc[first, "dailyrainin"] = 0.28

        assert detect_carryovers(archive) == []


class TestIncrements:
    def test_an_increment_is_never_negative(self):
        # A negative counter (a bad reading) is a drop: its value, clamped.
        counter = pd.Series([0.2, -0.01])

        assert rain_increments(counter).tolist()[1:] == [0.0]

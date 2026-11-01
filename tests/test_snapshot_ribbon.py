"""
The 48-hour condition ribbon in the snapshot (#78), through `build_snapshot`.

The ribbon is 192 fifteen-minute slots ending with the slot that holds the
newest observation. Each slot carries the newest temperature in it, the rain
that fell in it (from the daily counter's increments), and whether the sun
was up. A slot with no reading, or whose rain cannot be told from the counter
(the span after the archive ends, a gap), has no rain value: the site hatches
it.
"""

import pandas as pd
import pytest

from lookout.snapshot import build_snapshot
from tests.snapshot_helpers import (
    RAIN_LIVE,
    STATION,
    build,
    dry_station,
    live_row,
    ms,
    raining_station,
    scenario_until,
)

SLOT_MS = 15 * 60 * 1000


def slot(ribbon, when: str) -> int:
    """The index of the slot that starts at `when`."""
    offset = ms(when) - ribbon.start_ms
    assert offset % SLOT_MS == 0
    return offset // SLOT_MS


class TestSlots:
    def test_the_ribbon_is_48_hours_of_15_minute_slots_ending_at_the_newest(self):
        ribbon = raining_station().condition_ribbon

        assert ribbon.slot_minutes == 15
        assert len(ribbon.temperature_f) == 192
        assert len(ribbon.rain_in) == 192
        assert len(ribbon.daylight) == 192
        # The live reading at 00:25Z sits in the last slot, 00:15Z-00:30Z.
        assert ribbon.start_ms == ms("2026-01-07T00:30:00Z")
        assert slot(ribbon, "2026-01-09T00:15:00Z") == 191

    def test_a_slot_holds_the_rain_that_fell_in_it(self):
        ribbon = raining_station().condition_ribbon

        # Event 3 starts at 22:00Z at 0.12 in/hr: 0.01 in every five minutes.
        assert ribbon.rain_in[slot(ribbon, "2026-01-08T21:45:00Z")] == 0.0
        assert ribbon.rain_in[slot(ribbon, "2026-01-08T22:00:00Z")] == 0.03
        # Event 2 at 0.06 in/hr, across local midnight (08:00Z) and its reset.
        assert ribbon.rain_in[slot(ribbon, "2026-01-07T07:45:00Z")] == 0.015
        assert ribbon.rain_in[slot(ribbon, "2026-01-07T08:00:00Z")] == 0.015

    def test_a_slot_says_whether_the_sun_was_up(self):
        ribbon = raining_station().condition_ribbon

        assert ribbon.daylight[slot(ribbon, "2026-01-08T22:00:00Z")] is True  # 2 pm
        assert ribbon.daylight[slot(ribbon, "2026-01-08T10:00:00Z")] is False  # 2 am

    def test_a_slot_holds_its_newest_temperature(self):
        full = scenario_until("2026-01-09T00:30:00Z")
        newest_in_slot = full[full["dateutc"] == ms("2026-01-08T22:10:00Z")]

        ribbon = raining_station().condition_ribbon

        assert ribbon.temperature_f[slot(ribbon, "2026-01-08T22:00:00Z")] == float(
            newest_in_slot["tempf"].iloc[0]
        )

    def test_the_last_slot_holds_the_live_temperature(self):
        snapshot = raining_station()
        ribbon = snapshot.condition_ribbon

        assert ribbon.temperature_f[-1] == snapshot.current.temperature_f == 59.3


class TestUnarchivedSpan:
    def test_the_span_after_the_archive_has_no_rain_value(self):
        """
        The archive ends at 23:55Z and the live reading is at 00:25Z: the rain
        between them is known only as the live total, so neither slot after
        the archive carries a rain value, and the empty one no temperature.
        """
        ribbon = raining_station().condition_ribbon

        empty = slot(ribbon, "2026-01-09T00:00:00Z")
        assert ribbon.temperature_f[empty] is None
        assert ribbon.rain_in[empty] is None
        assert ribbon.daylight[empty] is None
        assert ribbon.rain_in[-1] is None
        assert ribbon.daylight[-1] is True

    def test_a_gap_in_the_archive_has_no_rain_value(self):
        ribbon = dry_station().condition_ribbon

        # The real slice is silent from 13:00Z to 14:34Z on Nov 18.
        assert ribbon.rain_in[slot(ribbon, "2025-11-18T12:45:00Z")] == 0.0
        assert ribbon.rain_in[slot(ribbon, "2025-11-18T13:15:00Z")] is None
        assert ribbon.temperature_f[slot(ribbon, "2025-11-18T13:15:00Z")] is None
        # The archive ends at 04:55Z; the three hours to 07:55Z are live-only.
        assert ribbon.rain_in[slot(ribbon, "2025-11-19T04:45:00Z")] == 0.0
        assert ribbon.rain_in[slot(ribbon, "2025-11-19T05:00:00Z")] is None
        assert ribbon.temperature_f[-1] == 39.4

    def test_the_first_reading_adds_no_rain(self):
        """
        The real slice opens at local midnight with the day before's 0.185 in
        still on the counter; it reset on the next reading. That is not rain
        in the first slot.
        """
        ribbon = dry_station().condition_ribbon

        assert ribbon.start_ms == ms("2025-11-17T08:00:00Z")
        assert ribbon.rain_in[0] == pytest.approx(0.008)


class TestSummary:
    def test_the_ribbon_names_its_range_and_its_rain_hours(self):
        ribbon = raining_station().condition_ribbon

        assert ribbon.temperature_low_f == 40.0
        assert ribbon.temperature_high_f == 60.0
        # Event 2 (6 h + 1 h) and event 3's archived 2 h.
        assert ribbon.rain_hours == 9.0
        # Every slot but the two after the archive.
        assert ribbon.known_hours == 47.5

    def test_the_live_reading_counts_toward_the_range(self):
        full = scenario_until("2026-01-09T00:30:00Z")
        archive = full[full["dateutc"] < ms("2026-01-09T00:00:00Z")]
        live = {**live_row(full, RAIN_LIVE), "tempf": 71.0}

        ribbon = build(archive, live, "2026-01-09T00:27:00Z").condition_ribbon

        assert ribbon.temperature_high_f == 71.0


def test_no_reading_means_no_ribbon():
    empty = pd.DataFrame({"dateutc": pd.Series([], dtype="int64")})

    snapshot = build_snapshot(
        archive=empty,
        live=None,
        event_catalog=None,
        energy_catalog=None,
        station=STATION,
        now=pd.Timestamp("2026-01-09T00:00:00Z"),
    )

    assert snapshot.condition_ribbon is None
    assert snapshot.dashboard()["condition_ribbon"] is None


def test_the_ribbon_is_on_the_dashboard_not_the_context_block():
    snapshot = raining_station()

    dashboard = snapshot.dashboard()["condition_ribbon"]
    assert dashboard["start_ms"] == ms("2026-01-07T00:30:00Z")
    assert len(dashboard["temperature_f"]) == 192
    assert "condition_ribbon" not in snapshot.context()


@pytest.mark.parametrize("scenario", [dry_station, raining_station])
def test_slot_values_are_json_ready(scenario):
    ribbon = scenario().dashboard()["condition_ribbon"]

    assert all(v is None or isinstance(v, float) for v in ribbon["rain_in"])
    assert all(v is None or isinstance(v, bool) for v in ribbon["daylight"])


class TestCounter:
    def counter_archive(self, values):
        """Five-minute rows from 20:00Z (noon local) with these daily totals."""
        archive = scenario_until("2026-01-06T21:00:00Z").sort_values("dateutc")
        archive = archive.tail(len(values)).reset_index(drop=True)
        archive["dailyrainin"] = values
        return archive

    def test_a_fall_in_the_counter_is_new_rain_only_at_local_midnight(self):
        # A mid-day fall (a correction) is not a reset: no phantom rain.
        archive = self.counter_archive([0.50, 0.50, 0.50, 0.45, 0.45, 0.45])

        ribbon = build(archive, None, "2026-01-06T21:00:00Z").condition_ribbon

        assert ribbon.rain_in[-1] == 0.0
        assert ribbon.rain_hours == 0.0

    def test_one_missing_counter_reading_does_not_void_the_next(self):
        archive = self.counter_archive([0.10, 0.10, None, 0.12, 0.13, 0.13])

        ribbon = build(archive, None, "2026-01-06T21:00:00Z").condition_ribbon

        # 20:30Z-20:45Z: 0.10 -> (missing) -> 0.12 -> 0.13.
        assert ribbon.rain_in[-1] == 0.03

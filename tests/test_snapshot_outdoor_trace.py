"""
The 24-hour pressure and humidity series for the Overview's tiles (#554),
through `build_snapshot`.

Each series is the outdoor reading's hourly mean over the 24 hours ending in
the hour that holds the newest observation, on the same slots the indoor
sensors' traces use (#466). An hour with no reading is null: a gap stays a
gap (story 31). Pressure keeps two decimals and humidity whole percent.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd
import pytest

from lookout.snapshot.model import Snapshot
from tests.fixtures.archive import build_archive
from tests.snapshot_helpers import (
    GOLDEN_SCENARIOS,
    build,
    dry_station,
    ms,
    seasoned_station,
)

NOW = "2026-11-11T20:02:00Z"  # Wed noon PST, the build time
LAST_ROW = "2026-11-11T20:00:00Z"  # the newest reading
FIRST_ROW = "2026-11-09T08:00:00Z"  # Mon local midnight: 60 hours of record
HOUR_MS = 3_600_000
START = "2026-11-10T21:00:00Z"  # the first slot: 23 hours before LAST_ROW's


def station(
    change: Optional[Callable[[pd.DataFrame], None]] = None, now: str = NOW
) -> Snapshot:
    """
    A small station, oldest row first, no live reading.

    The builder's archive reads `baromrelin` 30.0 throughout and `humidity`
    on a daily swing. `change` edits the archive in place before the build.
    """
    df = build_archive(FIRST_ROW, pd.Timestamp(LAST_ROW) + pd.Timedelta(minutes=5))
    df = df.sort_values("dateutc").reset_index(drop=True)
    if change is not None:
        change(df)
    return build(df, None, now)


def slot(when: str) -> int:
    """The slot holding `when`."""
    return (ms(when) - ms(START)) // HOUR_MS


class TestSeasonedGolden:
    @pytest.fixture(scope="class")
    def seasoned(self) -> Snapshot:
        return seasoned_station()

    def test_both_series_share_the_indoor_slots(self, seasoned):
        trace = seasoned.outdoor_trace
        indoor = seasoned.indoor[0].trace

        assert (trace.start_ms, trace.slot_minutes) == (
            indoor.start_ms,
            indoor.slot_minutes,
        )
        assert len(trace.pressure_inhg) == 24
        assert len(trace.humidity_pct) == 24

    def test_every_hour_of_the_seasoned_day_has_both_readings(self, seasoned):
        trace = seasoned.outdoor_trace

        assert None not in trace.pressure_inhg
        assert None not in trace.humidity_pct

    def test_the_dashboard_carries_the_series_and_the_context_does_not(self, seasoned):
        doc = seasoned.dashboard()["outdoor_trace"]

        assert sorted(doc) == [
            "humidity_pct",
            "pressure_inhg",
            "slot_minutes",
            "start_ms",
        ]
        assert "outdoor_trace" not in seasoned.context()


@pytest.mark.parametrize("scenario", sorted(GOLDEN_SCENARIOS))
def test_every_golden_carries_both_series(scenario):
    trace = GOLDEN_SCENARIOS[scenario]().outdoor_trace

    assert len(trace.pressure_inhg) == len(trace.humidity_pct) == 24


def test_the_dry_golden_keeps_the_hours_before_its_live_reading_null():
    snapshot = dry_station()
    trace = snapshot.outdoor_trace
    # The real slice ends before 05:00Z; the live reading comes at 07:55Z.
    assert snapshot.observed_at_ms == ms("2025-11-19T07:55:00Z")
    first = (ms("2025-11-19T05:00:00Z") - trace.start_ms) // HOUR_MS
    gap = (first, first + 1)

    assert [trace.pressure_inhg[i] for i in gap] == [None, None]
    assert [trace.humidity_pct[i] for i in gap] == [None, None]
    assert trace.pressure_inhg[first - 1] is not None
    assert trace.pressure_inhg[-1] is not None


class TestSlots:
    def test_the_window_ends_in_the_hour_of_the_newest_reading(self):
        trace = station().outdoor_trace

        assert trace.start_ms == ms(START)
        assert trace.slot_minutes == 60

    def test_the_window_ends_at_the_newest_reading_not_the_clock(self):
        trace = station(now="2026-11-13T20:02:00Z").outdoor_trace

        assert trace.start_ms == ms(START)

    def test_each_slot_is_its_hours_mean(self):
        def ramp(df: pd.DataFrame) -> None:
            hour = (df["dateutc"] >= ms("2026-11-11T12:00:00Z")) & (
                df["dateutc"] < ms("2026-11-11T13:00:00Z")
            )
            # Twelve readings, a third at the first value and two at the second.
            df.loc[hour, "baromrelin"] = np.tile([29.90, 29.95, 29.95], 4)
            df.loc[hour, "humidity"] = np.tile([61.0, 62.0, 62.0], 4)

        trace = station(ramp).outdoor_trace
        i = slot("2026-11-11T12:00:00Z")

        assert trace.pressure_inhg[i] == 29.93  # 29.933, to two decimals
        assert trace.humidity_pct[i] == 62.0  # 61.667, to whole percent
        assert trace.pressure_inhg[i - 1] == 30.0


class TestGaps:
    def test_an_hour_with_no_rows_is_null_in_both(self):
        def hole(df: pd.DataFrame) -> None:
            rows = (df["dateutc"] >= ms("2026-11-11T10:00:00Z")) & (
                df["dateutc"] < ms("2026-11-11T11:00:00Z")
            )
            df.drop(index=df.index[rows], inplace=True)

        trace = station(hole).outdoor_trace
        i = slot("2026-11-11T10:00:00Z")

        assert (trace.pressure_inhg[i], trace.humidity_pct[i]) == (None, None)
        assert trace.pressure_inhg[i - 1] == 30.0
        assert trace.humidity_pct[i + 1] is not None

    def test_a_reading_missing_from_its_rows_is_null_in_its_series_alone(self):
        def no_humidity(df: pd.DataFrame) -> None:
            rows = (df["dateutc"] >= ms("2026-11-11T10:00:00Z")) & (
                df["dateutc"] < ms("2026-11-11T11:00:00Z")
            )
            df.loc[rows, "humidity"] = np.nan

        trace = station(no_humidity).outdoor_trace
        i = slot("2026-11-11T10:00:00Z")

        assert trace.humidity_pct[i] is None
        assert trace.pressure_inhg[i] == 30.0

    def test_a_station_with_no_barometer_has_an_all_null_pressure_series(self):
        trace = station(
            lambda df: df.drop(columns=["baromrelin"], inplace=True)
        ).outdoor_trace

        assert trace.pressure_inhg == (None,) * 24
        assert None not in trace.humidity_pct

    def test_int64_humidity_reads_as_floats(self):
        def salmon_run(df: pd.DataFrame) -> None:
            df["humidity"] = np.full(len(df), 71, dtype="int64")

        trace = station(salmon_run).outdoor_trace

        assert trace.humidity_pct == (71.0,) * 24
        assert isinstance(trace.humidity_pct[0], float)


class TestEmptyArchive:
    def test_no_observation_means_no_series(self):
        df = build_archive(FIRST_ROW, LAST_ROW).iloc[0:0]

        snapshot = build(df, None, NOW)

        assert snapshot.outdoor_trace is None
        assert snapshot.dashboard()["outdoor_trace"] is None

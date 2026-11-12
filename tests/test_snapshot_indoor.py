"""
Indoor readings and battery flags (#466), through `build_snapshot`.

A sensor counts only while it is still reporting: an indoor sensor needs a
temperature, and a battery flag a reading, in the last 24 hours of the
history, measured back from the newest reading. A flag reads 1 for OK and 0
for low; any other value, or one that does not parse as a number, is left
out and logged. Printable names come from one fixed map; an unmapped `batt*`
column is left out and logged. The Inside tile shows only while the station
has a current indoor sensor.

The live archives store their flags as several types (#466 column read):
Lookout Post's `battrain` is a text category ("1") and `batt_co2` int64,
Salmon Run's `humidityin` and `battout` are int64, the rest double. The
fixtures here store them the same way.
"""

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from lookout.snapshot.model import Snapshot
from tests.fixtures.archive import build_archive
from tests.snapshot_helpers import build, ms, seasoned_station

NOW = "2026-11-11T20:02:00Z"  # Wed noon PST, the build time
LAST_ROW = "2026-11-11T20:00:00Z"  # the newest reading
FIRST_ROW = "2026-11-09T08:00:00Z"  # Mon local midnight: 60 hours of record
DAY_MS = 24 * 3_600_000

INDOOR_LOG = "lookout.snapshot.indoor"


def station(change: Optional[Callable[[pd.DataFrame], None]] = None) -> Snapshot:
    """
    A small station built at `NOW`, oldest row first, no live reading.

    The builder's archive holds the console (`tempinf` 68.0, `humidityin`
    45.0) and two OK flags (`battin`, `battout`, double). `change` edits the
    archive in place before the build.
    """
    df = build_archive(FIRST_ROW, pd.Timestamp(LAST_ROW) + pd.Timedelta(minutes=5))
    df = df.sort_values("dateutc").reset_index(drop=True)
    if change is not None:
        change(df)
    return build(df, None, NOW)


def stale(df: pd.DataFrame) -> pd.Series:
    """The rows older than the 24 hours back from the newest reading."""
    return df["dateutc"] <= ms(LAST_ROW) - DAY_MS


def flags(snapshot: Snapshot) -> List[Tuple[str, str, bool]]:
    """Each battery flag as (column, name, low)."""
    return [(f.key, f.name, f.low) for f in snapshot.battery_flags]


def flag_keys(snapshot: Snapshot) -> List[str]:
    """The battery flags' columns."""
    return [f.key for f in snapshot.battery_flags]


def inside_tile(snapshot: Snapshot) -> Tuple[Optional[str], bool]:
    """The Inside tile's (needs, shown)."""
    tile = {t.key: t for t in snapshot.record_depth.tiles}["inside"]
    return tile.needs, tile.shown


def warnings(caplog: pytest.LogCaptureFixture) -> List[str]:
    """The indoor module's warning lines."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == INDOOR_LOG and r.levelno == logging.WARNING
    ]


class TestSeasonedGolden:
    @pytest.fixture(scope="class")
    def seasoned(self) -> Snapshot:
        return seasoned_station()

    def test_it_has_the_console_and_sensor_1(self, seasoned):
        found = [(s.key, s.name) for s in seasoned.indoor]

        assert found == [("in", "Console"), ("1", "Sensor 1")]

    def test_each_sensor_carries_its_current_reading_and_humidity(self, seasoned):
        found = {
            s.key: (s.temperature_f, s.humidity_pct, s.observed_at_ms)
            for s in seasoned.indoor
        }

        # The live reading at 22:00Z is 15:00 PDT: sensor 1's daily peak.
        live = ms("2026-05-20T22:00:00Z")
        assert found == {"in": (68.0, 45.0, live), "1": (66.0, 55.0, live)}

    def test_each_sensor_has_24_hourly_means_ending_in_the_newest_hour(self, seasoned):
        trace = seasoned.indoor[1].trace

        assert trace.start_ms == ms("2026-05-19T23:00:00Z")
        assert trace.slot_minutes == 60
        assert len(trace.temperature_f) == 24
        # The newest slot holds the live reading alone.
        assert trace.temperature_f[-1] == 66.0
        assert all(value is not None for value in trace.temperature_f)

    def test_it_has_four_flags_and_sensor_1_reads_low(self, seasoned):
        assert flags(seasoned) == [
            ("battout", "Outdoor sensor", False),
            ("battin", "Console", False),
            ("batt1", "Sensor 1", True),
            ("battrain", "Rain gauge", False),
        ]

    def test_the_inside_tile_is_last_and_shown(self, seasoned):
        assert seasoned.record_depth.tiles[-1].key == "inside"
        assert inside_tile(seasoned) == (None, True)

    def test_the_dashboard_carries_both_lists(self, seasoned):
        doc = seasoned.dashboard()

        assert doc["indoor"][0]["name"] == "Console"
        assert doc["battery_flags"][2] == {
            "key": "batt1",
            "name": "Sensor 1",
            "low": True,
        }


class TestNoIndoorSensor:
    @pytest.fixture(scope="class")
    def outdoor_only(self) -> Snapshot:
        return station(
            lambda df: df.drop(columns=["tempinf", "humidityin"], inplace=True)
        )

    def test_it_has_no_indoor_entry(self, outdoor_only):
        assert outdoor_only.indoor == ()
        assert outdoor_only.dashboard()["indoor"] == []

    def test_the_inside_tile_is_hidden_though_the_record_is_deep_enough(
        self, outdoor_only
    ):
        assert inside_tile(outdoor_only) == (None, False)
        others = [t.shown for t in outdoor_only.record_depth.tiles[:-1]]
        assert all(others)

    def test_its_flags_are_still_read(self, outdoor_only):
        assert flag_keys(outdoor_only) == ["battout", "battin"]


class TestIndoorCurrency:
    def test_a_sensor_silent_for_24_hours_is_absent(self):
        def silent(df: pd.DataFrame) -> None:
            df.loc[~stale(df), ["tempinf", "humidityin"]] = np.nan

        snapshot = station(silent)

        assert snapshot.indoor == ()
        assert inside_tile(snapshot) == (None, False)

    def test_a_sensor_with_one_reading_in_the_window_is_current(self):
        last = ms(LAST_ROW) - DAY_MS + 5 * 60_000

        def nearly_silent(df: pd.DataFrame) -> None:
            df.loc[df["dateutc"] > last, ["tempinf", "humidityin"]] = np.nan

        snapshot = station(nearly_silent)

        console = snapshot.indoor[0]
        assert (console.temperature_f, console.observed_at_ms) == (68.0, last)
        assert inside_tile(snapshot) == (None, True)

    def test_a_numbered_sensor_is_read_beside_the_console(self):
        def sensor_3(df: pd.DataFrame) -> None:
            df["temp3f"] = 61.0
            df["humidity3"] = 50.0

        snapshot = station(sensor_3)

        assert [(s.key, s.name) for s in snapshot.indoor] == [
            ("in", "Console"),
            ("3", "Sensor 3"),
        ]
        assert snapshot.indoor[1].temperature_f == 61.0

    def test_int64_humidity_reads_as_a_float(self):
        def salmon_run(df: pd.DataFrame) -> None:
            df["humidityin"] = np.full(len(df), 71, dtype="int64")

        console = station(salmon_run).indoor[0]

        assert console.humidity_pct == 71.0
        assert isinstance(console.humidity_pct, float)

    def test_a_sensor_without_humidity_has_none(self):
        console = station(
            lambda df: df.drop(columns=["humidityin"], inplace=True)
        ).indoor[0]

        assert console.temperature_f == 68.0
        assert console.humidity_pct is None

    def test_an_hour_without_readings_is_none_in_the_trace(self):
        hole = (ms("2026-11-11T10:00:00Z"), ms("2026-11-11T11:00:00Z"))

        def gap(df: pd.DataFrame) -> None:
            rows = (df["dateutc"] >= hole[0]) & (df["dateutc"] < hole[1])
            df.loc[rows, "tempinf"] = np.nan

        trace = station(gap).indoor[0].trace

        assert trace.start_ms == ms("2026-11-10T21:00:00Z")
        slot = (hole[0] - trace.start_ms) // 3_600_000
        assert trace.temperature_f[slot] is None
        assert trace.temperature_f[slot - 1] == 68.0


class TestFlagTypes:
    @pytest.fixture(scope="class")
    def live_types(self) -> Snapshot:
        def stored(df: pd.DataFrame) -> None:
            df["battrain"] = pd.Series(["1"] * len(df), dtype="category")
            df["batt_co2"] = np.full(len(df), 1, dtype="int64")
            df["battout"] = np.full(len(df), 1, dtype="int64")

        return station(stored)

    def test_a_text_an_int64_and_a_double_flag_each_read_ok(self, live_types):
        assert flags(live_types) == [
            ("battout", "Outdoor sensor", False),
            ("battin", "Console", False),
            ("battrain", "Rain gauge", False),
            ("batt_co2", "CO2 sensor", False),
        ]

    def test_an_int64_zero_reads_low(self):
        def low(df: pd.DataFrame) -> None:
            df["battout"] = np.full(len(df), 0, dtype="int64")

        assert flags(station(low))[0] == ("battout", "Outdoor sensor", True)

    def test_a_text_zero_reads_low(self):
        def low(df: pd.DataFrame) -> None:
            df["battrain"] = pd.Series(["0"] * len(df), dtype="category")

        assert ("battrain", "Rain gauge", True) in flags(station(low))


class TestFlagCurrency:
    def test_a_stale_flag_whose_last_value_is_0_is_absent_not_low(self):
        def gone(df: pd.DataFrame) -> None:
            df["batt_25"] = np.where(stale(df), 0.0, np.nan)
            df["battsm2"] = np.where(stale(df), 0.0, np.nan)

        assert flag_keys(station(gone)) == ["battout", "battin"]

    def test_a_flag_that_never_reported_is_absent(self):
        def never(df: pd.DataFrame) -> None:
            df["batt_lightning"] = np.nan

        assert flag_keys(station(never)) == ["battout", "battin"]

    def test_the_newest_reading_in_the_window_decides(self):
        recovered = ms(LAST_ROW) - 60 * 60_000

        def swapped(df: pd.DataFrame) -> None:
            df["batt1"] = np.where(df["dateutc"] < recovered, 0.0, 1.0)

        assert ("batt1", "Sensor 1", False) in flags(station(swapped))

    def test_a_low_reading_before_a_null_one_still_reads_low(self):
        def low_then_null(df: pd.DataFrame) -> None:
            df["batt1"] = 0.0
            df.loc[df.index[-3:], "batt1"] = np.nan

        assert ("batt1", "Sensor 1", True) in flags(station(low_then_null))

    def test_the_window_is_measured_from_the_newest_reading_not_the_clock(self):
        # Built two days after the newest reading: the flags are still read.
        df = build_archive(FIRST_ROW, pd.Timestamp(LAST_ROW) + pd.Timedelta(minutes=5))
        snapshot = build(df, None, "2026-11-13T20:02:00Z")

        assert flag_keys(snapshot) == ["battout", "battin"]
        assert [s.key for s in snapshot.indoor] == ["in"]


class TestFlagsLeftOut:
    def test_an_unparseable_value_is_absent_and_logged(self, caplog):
        def garbled(df: pd.DataFrame) -> None:
            df["battrain"] = pd.Series(["low"] * len(df), dtype="category")

        with caplog.at_level(logging.WARNING, logger=INDOOR_LOG):
            snapshot = station(garbled)

        assert flag_keys(snapshot) == ["battout", "battin"]
        assert len(warnings(caplog)) == 1
        assert "battrain" in warnings(caplog)[0]
        assert "'low'" in warnings(caplog)[0]

    @pytest.mark.parametrize("value", [2.0, 0.5, -1.0])
    def test_an_out_of_range_value_is_absent_and_logged(self, caplog, value):
        def odd(df: pd.DataFrame) -> None:
            df["batt1"] = value

        with caplog.at_level(logging.WARNING, logger=INDOOR_LOG):
            snapshot = station(odd)

        assert "batt1" not in flag_keys(snapshot)
        assert len(warnings(caplog)) == 1
        assert "batt1" in warnings(caplog)[0]

    def test_an_unmapped_flag_is_absent_and_logged(self, caplog):
        def unknown(df: pd.DataFrame) -> None:
            df["batt_leak"] = 1.0

        with caplog.at_level(logging.WARNING, logger=INDOOR_LOG):
            snapshot = station(unknown)

        assert flag_keys(snapshot) == ["battout", "battin"]
        assert len(warnings(caplog)) == 1
        assert "batt_leak" in warnings(caplog)[0]

    def test_ok_flags_log_nothing(self, caplog):
        with caplog.at_level(logging.WARNING, logger=INDOOR_LOG):
            station()

        assert warnings(caplog) == []


class TestEmptyArchive:
    def test_no_sensor_no_flag_and_no_inside_tile(self):
        df = build_archive(FIRST_ROW, LAST_ROW).iloc[0:0]
        snapshot = build(df, None, NOW)

        assert snapshot.indoor == ()
        assert snapshot.battery_flags == ()
        assert inside_tile(snapshot) == (None, False)

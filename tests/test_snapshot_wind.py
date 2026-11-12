"""
The wind rose (#465), through `build_snapshot`: for 24 h, 7 days and 30 days
back from the build time, 16 direction bins with average wind and highest
gust, the calm share (calm readings have no direction and are left out of
the bins) and the share of windy readings from each of 8 compass points.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.archive import build_archive
from tests.snapshot_helpers import (
    GOLDEN_SCENARIOS,
    build,
    live_row,
    ms,
    seasoned_station,
)

NOW = "2026-02-05T20:00:00Z"
"""Thursday 2026-02-05, noon PST: the build time and the newest reading."""

FIRST = "2026-01-01T08:00:00Z"
"""Local midnight 1 Jan: the record covers all three windows."""

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS

SIXTEEN = "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split()
EIGHT = "N NE E SE S SW W NW".split()


def station(first: str = FIRST, end: str = NOW) -> pd.DataFrame:
    """
    A station from `first` through `end` (its newest row), oldest first.

    Every reading is wind from the west (270°) at 3 mph, gusting 5.
    """
    df = build_archive(first, pd.Timestamp(end) + pd.Timedelta(minutes=5))
    df["winddir"], df["windspeedmph"], df["windgustmph"] = 270.0, 3.0, 5.0
    return df.sort_values("dateutc").reset_index(drop=True)


def last_day(df: pd.DataFrame) -> pd.Series:
    """The rows inside the 24 h window: after NOW - 24 h, through NOW."""
    return (df["dateutc"] > ms(NOW) - DAY_MS) & (df["dateutc"] <= ms(NOW))


def set_wind(
    df: pd.DataFrame,
    rows: pd.Series,
    deg: Optional[float],
    mph: float,
    gust: float = 9.0,
) -> None:
    """Set the wind on `rows`: direction (None for missing), speed, gust."""
    df.loc[rows, "winddir"] = np.nan if deg is None else deg
    df.loc[rows, "windspeedmph"] = mph
    df.loc[rows, "windgustmph"] = gust


def rose(df: pd.DataFrame, live: Optional[Dict] = None):
    """The wind rose built from `df` at NOW."""
    return build(df, live, NOW).wind_rose


def window(df: pd.DataFrame, key: str):
    """One window of the rose."""
    return next(w for w in rose(df).windows if w.key == key)


def bins(win) -> Dict[str, int]:
    """Each direction bin's readings, only the bins that hold any."""
    return {b.direction: b.readings for b in win.bins if b.readings}


def shares(win) -> Dict[str, Optional[int]]:
    """Each compass point's share of windy readings, the non-zero ones."""
    return {p.direction: p.share_pct for p in win.points if p.share_pct}


def test_three_windows_each_with_16_bins_and_8_points_in_order():
    windows = rose(station()).windows

    assert [w.key for w in windows] == ["24h", "7d", "30d"]
    assert [w.hours for w in windows] == [24, 168, 720]
    for win in windows:
        assert [b.direction for b in win.bins] == SIXTEEN
        assert [b.centre_deg for b in win.bins] == [i * 22.5 for i in range(16)]
        assert win.bins[0].from_deg == 348.75
        assert win.bins[0].to_deg == 11.25
        assert win.bins[1].from_deg == 11.25
        assert [p.direction for p in win.points] == EIGHT
        assert [p.centre_deg for p in win.points] == [i * 45.0 for i in range(8)]


def test_the_calm_threshold_is_a_decided_value():
    assert rose(station()).calm_below_mph == 1.0


def test_a_steady_west_wind_fills_one_bin():
    win = window(station(), "24h")

    assert bins(win) == {"W": 288}
    west = next(b for b in win.bins if b.direction == "W")
    assert (west.average_mph, west.gust_mph) == (3.0, 5.0)
    east = next(b for b in win.bins if b.direction == "E")
    assert (east.average_mph, east.gust_mph) == (None, None)
    assert (win.readings, win.calm_readings, win.calm_pct) == (288, 0, 0)
    assert shares(win) == {"W": 100}


def test_calm_readings_are_left_out_of_the_bins_and_given_as_a_share():
    df = station()
    inside = df.index[last_day(df)]
    # Ambient reports a direction at 0 mph: the vane's last resting point.
    set_wind(df, df.index.isin(inside[::2]), 180.0, 0.0, 0.0)

    win = window(df, "24h")

    assert bins(win) == {"W": 144}
    assert (win.readings, win.calm_readings, win.calm_pct) == (288, 144, 50)
    assert shares(win) == {"W": 100}


def test_a_reading_with_no_direction_is_calm():
    df = station()
    set_wind(df, last_day(df), None, 4.0)

    win = window(df, "24h")

    assert bins(win) == {}
    assert (win.calm_readings, win.calm_pct) == (288, 100)
    assert shares(win) == {}
    assert all(p.share_pct is None for p in win.points)


@pytest.mark.parametrize("mph, calm", [(0.9, True), (1.0, False)])
def test_under_one_mph_is_calm(mph, calm):
    df = station()
    set_wind(df, last_day(df), 90.0, mph)

    win = window(df, "24h")

    assert win.calm_pct == (100 if calm else 0)
    assert bins(win) == ({} if calm else {"E": 288})


@pytest.mark.parametrize(
    "deg, bin16, point8",
    [
        (0.0, "N", "N"),
        (360.0, "N", "N"),
        (348.75, "N", "N"),
        (348.74, "NNW", "N"),
        (11.24, "N", "N"),
        (11.25, "NNE", "N"),
        (22.49, "NNE", "N"),
        (22.5, "NNE", "NE"),
        (337.5, "NNW", "N"),
        (337.49, "NNW", "NW"),
        (191.25, "SSW", "S"),
    ],
)
def test_bin_edges(deg, bin16, point8):
    df = station()
    set_wind(df, last_day(df), deg, 5.0)

    win = window(df, "24h")

    assert bins(win) == {bin16: 288}
    assert shares(win) == {point8: 100}


def test_a_bin_averages_its_wind_and_keeps_its_highest_gust():
    df = station()
    inside = df.index[last_day(df)]
    set_wind(df, df.index.isin(inside[::2]), 180.0, 2.0, 6.0)
    set_wind(df, df.index.isin(inside[1::2]), 185.0, 4.25, 9.5)

    south = next(b for b in window(df, "24h").bins if b.direction == "S")

    assert south.readings == 288
    assert south.average_mph == 3.1  # 3.125, to one decimal
    assert south.gust_mph == 9.5


def test_the_eight_point_shares_are_of_windy_readings_only():
    df = station()
    inside = df.index[last_day(df)]
    # 288 readings: 96 calm, 144 from S or SSE (both "S"), 48 from W.
    set_wind(df, df.index.isin(inside[:96]), 180.0, 0.0)
    set_wind(df, df.index.isin(inside[96:168]), 180.0, 5.0)
    set_wind(df, df.index.isin(inside[168:240]), 160.0, 5.0)

    win = window(df, "24h")

    assert bins(win) == {"S": 72, "SSE": 72, "W": 48}
    assert shares(win) == {"S": 75, "W": 25}
    assert next(p for p in win.points if p.direction == "S").readings == 144
    assert win.calm_pct == 33


def test_each_window_reaches_back_its_span_from_the_build_time():
    df = station()
    # A reading exactly 24 h back is in 7 days, not 24 h; one 5 min later is in.
    set_wind(df, df["dateutc"] == ms(NOW) - DAY_MS, 90.0, 5.0)
    set_wind(df, df["dateutc"] == ms(NOW) - DAY_MS + 300_000, 0.0, 5.0)
    # Likewise exactly 7 days back: in 30 days only.
    set_wind(df, df["dateutc"] == ms(NOW) - 7 * DAY_MS, 135.0, 5.0)
    # And 30 days back: in none.
    set_wind(df, df["dateutc"] == ms(NOW) - 30 * DAY_MS, 225.0, 5.0)

    day, week, month = rose(df).windows

    assert bins(day) == {"N": 1, "W": 287}
    assert bins(week) == {"N": 1, "E": 1, "W": 2014}
    assert bins(month) == {"N": 1, "E": 1, "SE": 1, "W": 8637}
    assert (day.readings, week.readings, month.readings) == (288, 2016, 8640)


def test_the_live_reading_counts_once():
    full = station()
    set_wind(full, full["dateutc"] == ms(NOW), 90.0, 7.0)
    # The archive lags the live reading by half an hour, as the refresh run's does.
    archive = full[full["dateutc"] <= ms(NOW) - HOUR_MS // 2].reset_index(drop=True)
    live = live_row(full, NOW)

    win = next(w for w in rose(archive, live).windows if w.key == "24h")

    assert bins(win) == {"E": 1, "W": 288 - 6}


def test_a_window_the_record_does_not_cover_is_hidden_with_no_values():
    # The record starts ten days back: 24 h and 7 days are shown, 30 days not.
    day, week, month = rose(station(first="2026-01-26T08:00:00Z")).windows

    assert (day.status.shown, week.status.shown, month.status.shown) == (
        True,
        True,
        False,
    )
    assert (month.status.status, month.status.reason) == ("invalid", "before_archive")
    assert month.readings == 0
    assert month.calm_pct is None
    assert all(b.readings == 0 and b.average_mph is None for b in month.bins)
    assert all(p.share_pct is None for p in month.points)
    assert bins(week) == {"W": 2016}


def test_a_window_status_spans_the_station_local_days_it_touches():
    day, week, month = rose(station()).windows

    # NOW is noon local on 5 Feb: 24 h back is noon on the 4th.
    assert (day.status.first_day, day.status.last_day) == ("2026-02-04", "2026-02-05")
    assert week.status.first_day == "2026-01-29"
    assert month.status.first_day == "2026-01-06"
    assert [w.status.key for w in (day, week, month)] == ["24h", "7d", "30d"]
    assert all(w.status.status == "valid" for w in (day, week, month))


def test_a_gap_day_inside_a_week_makes_that_window_invalid_not_the_others():
    df = station()
    lost = (df["dateutc"] >= ms("2026-02-01T08:00:00Z")) & (
        df["dateutc"] < ms("2026-02-02T08:00:00Z")
    )
    day, week, month = rose(df[~lost].reset_index(drop=True)).windows

    assert (week.status.status, week.status.reason) == ("invalid", "gap")
    assert week.calm_pct is None and bins(week) == {}
    # One day of 31 is under a tenth: prorated, its values over what is held.
    assert month.status.status == "prorated"
    assert bins(month) == {"W": 8640 - 288}
    assert bins(day) == {"W": 288}


@pytest.mark.parametrize("name", sorted(GOLDEN_SCENARIOS))
def test_the_rose_is_in_the_dashboard_and_not_the_context(name):
    snapshot = GOLDEN_SCENARIOS[name]()

    published = snapshot.dashboard()["wind_rose"]
    assert [w["key"] for w in published["windows"]] == ["24h", "7d", "30d"]
    assert all(len(w["bins"]) == 16 for w in published["windows"])
    assert "wind_rose" not in snapshot.context()


def test_the_seasoned_station_has_a_varied_rose():
    week = next(w for w in seasoned_station().wind_rose.windows if w.key == "7d")

    assert week.status.shown and week.status.status == "valid"
    assert len(bins(week)) >= 6
    assert 0 < week.calm_pct < 50
    assert 98 <= sum(p.share_pct or 0 for p in week.points) <= 102
    averages = {b.average_mph for b in week.bins if b.readings}
    assert len(averages) > 1

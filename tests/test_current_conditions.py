"""
Characterizes current conditions and the active Rain Event decision.

Covers the temperature and barometer trend arrows, and `get_active_rain_event`
for an ongoing and an ended event (the Event Catalog passed in).
"""

import json

import pandas as pd
import pytest

from lookout.core.rain_events import RainEventCatalog
from lookout.core.weather_events import get_active_rain_event
from lookout.utils.trend_utils import (
    calculate_barometer_trend,
    calculate_temperature_trend,
    newest_row,
)
from tests.fixtures.archive import (
    FAKE_MAC,
    SCENARIO,
    build_archive,
    scenario_archive,
)

HOUR_MS = 3600 * 1000
UP, DOWN, STEADY = "↑", "↓", "→"


def _history(
    *, hours_back: float, value_then: float, field: str, base: float
) -> pd.DataFrame:
    """
    Two dry days of five-minute rows, newest first, with `field` constant at
    `base` except the row `hours_back` before the newest, which holds
    `value_then`. The other trend field is a constant too.
    """
    df = build_archive("2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z")
    df["tempf"] = 50.0
    df["baromrelin"] = 30.0
    df[field] = base
    target = df["dateutc"].iloc[0] - int(hours_back * HOUR_MS)
    df.loc[df["dateutc"] == target, field] = value_then
    return df


def _temp_history(then: float) -> pd.DataFrame:
    return _history(hours_back=24, value_then=then, field="tempf", base=50.0)


def _baro_history(then: float) -> pd.DataFrame:
    return _history(hours_back=3, value_then=then, field="baromrelin", base=30.0)


class TestTemperatureTrend:
    @pytest.mark.parametrize(
        "yesterday, arrow",
        [(45.0, UP), (48.9, UP), (49.0, STEADY), (50.0, STEADY), (51.0, STEADY)]
        + [(51.1, DOWN), (60.0, DOWN)],
    )
    def test_compares_now_with_the_same_time_yesterday(self, yesterday, arrow):
        # Now is 50.0F; steady is within one degree either way, edges included.
        assert calculate_temperature_trend(_temp_history(yesterday), 50.0) == arrow

    def test_the_current_value_is_the_argument_not_the_newest_row(self):
        df = _temp_history(40.0)

        assert calculate_temperature_trend(df, 40.0) == STEADY
        assert calculate_temperature_trend(df, 60.0) == UP

    def test_empty_history_is_steady(self):
        assert calculate_temperature_trend(pd.DataFrame(), 50.0) == STEADY

    def test_no_reading_within_thirty_minutes_of_yesterday_is_steady(self):
        df = _temp_history(40.0)
        target = df["dateutc"].iloc[0] - 24 * HOUR_MS
        far = (df["dateutc"] - target).abs() >= 35 * 60 * 1000

        assert calculate_temperature_trend(df[far].reset_index(drop=True), 50.0) == (
            STEADY
        )

    def test_a_reading_exactly_thirty_minutes_off_still_counts(self):
        df = _temp_history(40.0)
        target = df["dateutc"].iloc[0] - 24 * HOUR_MS
        df.loc[df["dateutc"] == target, "tempf"] = 50.0
        for side in (-30, 30):
            df.loc[df["dateutc"] == target + side * 60 * 1000, "tempf"] = 40.0
        edge = (df["dateutc"] - target).abs() >= 30 * 60 * 1000

        assert calculate_temperature_trend(df[edge].reset_index(drop=True), 50.0) == UP

    def test_history_shorter_than_a_day_is_steady(self):
        df = build_archive("2026-01-01T00:00:00Z", "2026-01-01T12:00:00Z")

        assert calculate_temperature_trend(df, 90.0) == STEADY

    def test_a_history_with_holes_in_its_index_reads_the_right_row(self):
        # Fixed in #72: the closest row was found by label but read by
        # position, so any index other than 0..n-1 read a neighbouring row.
        df = _temp_history(40.0)
        assert calculate_temperature_trend(df, 50.0) == UP

        holed = df.drop(index=[1, 2, 3])

        assert calculate_temperature_trend(holed, 50.0) == UP


class TestBarometerTrend:
    @pytest.mark.parametrize(
        "three_hours_ago, arrow",
        # Rate is (now - then) / 3 per hour; steady is within 0.05"/h.
        [(29.7, UP), (29.84, UP)]
        + [(29.9, STEADY), (29.88, STEADY), (30.0, STEADY), (30.1, STEADY)]
        + [(30.16, DOWN), (30.5, DOWN)],
    )
    def test_uses_the_three_hour_rate_of_change(self, three_hours_ago, arrow):
        # Now is 30.0".
        assert calculate_barometer_trend(_baro_history(three_hours_ago), 30.0) == (
            arrow
        )

    def test_empty_history_is_steady(self):
        assert calculate_barometer_trend(pd.DataFrame(), 30.0) == STEADY

    def test_no_reading_three_hours_back_is_steady(self):
        df = _baro_history(29.0)
        target = df["dateutc"].iloc[0] - 3 * HOUR_MS
        far = (df["dateutc"] - target).abs() >= 35 * 60 * 1000

        assert calculate_barometer_trend(df[far].reset_index(drop=True), 30.0) == (
            STEADY
        )

    def test_missing_column_is_steady(self):
        df = _baro_history(29.0).drop(columns="baromrelin")

        assert calculate_barometer_trend(df, 30.0) == STEADY


class TestTrendsReadEitherOrder:
    """#133: "now" is the newest `dateutc`, not the first row."""

    def test_newest_row_is_the_greatest_dateutc_and_skips_a_missing_one(self):
        df = pd.DataFrame(
            {"dateutc": [1000.0, float("nan"), 3000.0, 2000.0], "tempf": [1, 2, 3, 4]},
            index=[9, 8, 7, 6],
        )

        assert newest_row(df)["tempf"] == 3

    @pytest.mark.parametrize("yesterday, arrow", [(45.0, UP), (55.0, DOWN)])
    def test_temperature_trend_is_the_same_oldest_first(self, yesterday, arrow):
        df = _temp_history(yesterday)
        oldest_first = df.iloc[::-1].reset_index(drop=True)

        assert calculate_temperature_trend(df, 50.0) == arrow
        assert calculate_temperature_trend(oldest_first, 50.0) == arrow

    @pytest.mark.parametrize("three_hours_ago, arrow", [(29.7, UP), (30.3, DOWN)])
    def test_barometer_trend_is_the_same_oldest_first(self, three_hours_ago, arrow):
        df = _baro_history(three_hours_ago)
        oldest_first = df.iloc[::-1].reset_index(drop=True)

        assert calculate_barometer_trend(df, 30.0) == arrow
        assert calculate_barometer_trend(oldest_first, 30.0) == arrow


# ---- active rain event ------------------------------------------------------


def _catalog_df(archive: pd.DataFrame) -> pd.DataFrame:
    return RainEventCatalog(FAKE_MAC).detect_and_catalog_events(
        archive, auto_save=False
    )


def _until(end: str) -> pd.DataFrame:
    return build_archive(SCENARIO.start, end, rain=SCENARIO.rain, gaps=SCENARIO.gaps)


@pytest.fixture
def ongoing_catalog() -> pd.DataFrame:
    """Archive ends 2h into event 3, 0.24" fallen: the last event is ongoing."""
    return _catalog_df(_until("2026-01-09T00:00:00Z"))


def _fully_ended(catalog: pd.DataFrame) -> pd.DataFrame:
    """The same catalog with every event explicitly not ongoing."""
    return catalog.assign(ongoing=False, flags=[{"ongoing": False}] * len(catalog))


class TestGetActiveRainEvent:
    def test_ongoing_event_is_active_while_it_is_still_raining(self, ongoing_catalog):
        active = get_active_rain_event(ongoing_catalog, 0.24)

        assert active["total_rain"] == pytest.approx(0.24)
        assert active["duration_h"] == pytest.approx(115 / 60)
        assert active["rain_rate"] == pytest.approx(0.24 / (115 / 60))
        assert set(active) == {
            "duration",
            "total_rain",
            "rain_rate",
            "start_time",
            "duration_h",
        }

    def test_ongoing_event_is_dropped_once_the_live_event_rain_is_zero(
        self, ongoing_catalog
    ):
        assert get_active_rain_event(ongoing_catalog, 0) is None
        assert get_active_rain_event(ongoing_catalog) is None

    def test_the_live_amount_only_gates_it_and_is_not_reported(self, ongoing_catalog):
        active = get_active_rain_event(ongoing_catalog, 99.0)

        assert active["total_rain"] == pytest.approx(0.24)

    def test_ended_events_are_not_active_even_while_it_rains(self):
        catalog = _fully_ended(_catalog_df(scenario_archive()))

        assert get_active_rain_event(catalog, 0.1) is None
        assert get_active_rain_event(catalog, 0) is None

    def test_a_freshly_detected_completed_last_event_is_not_active(self):
        # Fixed in #72: a completed last event used to carry `ongoing` NaN,
        # which read as ongoing, so live rain made it the active event.
        assert get_active_rain_event(_catalog_df(scenario_archive()), 0.1) is None

    def test_a_nan_ongoing_from_an_older_catalog_is_not_active(self):
        catalog = _catalog_df(scenario_archive())
        catalog["ongoing"] = float("nan")
        catalog["flags"] = [{"ongoing": None}] * len(catalog)

        assert get_active_rain_event(catalog, 0.1) is None

    def test_ongoing_can_be_marked_only_in_the_flags_dict(self):
        catalog = _fully_ended(_catalog_df(scenario_archive()))
        catalog["flags"] = [{"ongoing": i == 1} for i in range(3)]

        active = get_active_rain_event(catalog, 0.1)

        assert active["total_rain"] == pytest.approx(0.48)
        assert active["duration_h"] == pytest.approx(950 / 60)

    def test_flags_stored_as_json_text_are_read_too(self):
        catalog = _fully_ended(_catalog_df(scenario_archive()))
        catalog["flags"] = [json.dumps({"ongoing": i == 0}) for i in range(3)]

        active = get_active_rain_event(catalog, 0.1)

        assert active["total_rain"] == pytest.approx(0.36)

    def test_the_latest_of_several_ongoing_events_wins(self):
        catalog = _fully_ended(_catalog_df(scenario_archive()))
        catalog["ongoing"] = True

        active = get_active_rain_event(catalog.iloc[::-1].reset_index(drop=True), 0.1)

        assert active["duration_h"] == pytest.approx(535 / 60)  # event 3

    def test_an_empty_or_missing_catalog_means_no_active_event(self):
        assert get_active_rain_event(pd.DataFrame(), 0.1) is None
        assert get_active_rain_event(None, 0.1) is None

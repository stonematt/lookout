"""
Wet mode, a pure replay over the daily wet and countable hours (#656).

Wet mode starts when the last 7 days' wet share reaches 15% and ends when the
last 14 days' share falls below 2%. When fewer than half the last 14 days'
hours can be counted, the calendar decides (Nov 1 to Apr 30 is wet). A day's
mode is decided by the days before it, replayed from a fixed start state, so
a rebuild always gives the same mode (spec #647, GLOSSARY: wet mode).

Through `lookout.core.wet_mode.wet_mode`, on synthetic series and on a
fixture of one station's last 365 days.
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from lookout.core.wet_mode import wet_mode


def series(start: str, days: List[Tuple[int, int]]) -> pd.DataFrame:
    """
    A daily series from `start`, one (wet hours, countable hours) per day.

    :param start: The first station-local day, ISO.
    :param days: Wet and countable hours for each day, in order.
    :return: Columns `wet_hours` and `countable_hours`, one row per day.
    """
    index = pd.date_range(start, periods=len(days), freq="D")
    return pd.DataFrame(days, index=index, columns=["wet_hours", "countable_hours"])


def dry(n: int) -> List[Tuple[int, int]]:
    """`n` fully counted days with no wet hour."""
    return [(0, 24)] * n


def wet(n: int, hours: int) -> List[Tuple[int, int]]:
    """`n` fully counted days with `hours` wet hours each."""
    return [(hours, 24)] * n


def modes(result: pd.DataFrame) -> List[str]:
    """The mode column as a list."""
    return result["mode"].tolist()


def test_a_week_at_fifteen_percent_starts_wet_mode_the_day_after() -> None:
    # Summer: the calendar would say dry. 4 of 24 hours is 16.7%.
    days = series("2026-07-01", dry(14) + wet(7, 4) + dry(1))
    result = wet_mode(days)
    assert modes(result)[:21] == ["dry"] * 21
    assert result["mode"].iloc[21] == "wet"
    assert result["decided_by"].iloc[21] == "weather"


def test_wet_mode_holds_between_the_two_thresholds() -> None:
    # After the wet week, 1 wet hour a day is 4.2%: below 15%, above 2%.
    days = series("2026-07-01", dry(14) + wet(7, 4) + wet(30, 1))
    assert modes(wet_mode(days))[21:] == ["wet"] * 30


def test_wet_mode_ends_when_the_fourteen_day_share_falls_below_two_percent() -> None:
    # 7 wet days at 4 h (days 14 to 20), then dry. With two wet days left in
    # the 14 days before, the share is 8 of 336 (2.4%): still wet. With one
    # left (day 34), it is 4 of 336 (1.2%): dry.
    days = series("2026-07-01", dry(14) + wet(7, 4) + dry(20))
    result = modes(wet_mode(days))
    first_dry = result.index("dry", 21)
    assert first_dry == 34
    assert result[first_dry:] == ["dry"] * (len(result) - first_dry)


def test_a_share_of_exactly_fifteen_percent_starts_wet_mode() -> None:
    # 3 wet of 20 countable hours a day is 21 of 140: exactly 15%.
    days = series("2026-07-01", dry(14) + [(3, 20)] * 7 + dry(1))
    assert wet_mode(days)["mode"].iloc[21] == "wet"


def test_a_share_of_exactly_two_percent_keeps_wet_mode() -> None:
    # Wet mode from day 21. On day 35 the 14 days before hold 300 countable
    # hours: 6 wet hours is exactly 2% and holds wet mode; 5 is 1.7%.
    def year(first_day_wet: int) -> pd.DataFrame:
        return series(
            "2026-07-01",
            dry(14)
            + wet(7, 4)
            + [(first_day_wet, 24)]
            + dry(11)
            + [(0, 6)] * 2
            + dry(1),
        )

    assert wet_mode(year(6))["mode"].iloc[35] == "wet"
    assert wet_mode(year(5))["mode"].iloc[35] == "dry"


def test_a_wet_share_between_the_thresholds_never_starts_wet_mode() -> None:
    days = series("2026-07-01", dry(14) + wet(30, 3))  # 12.5%
    assert modes(wet_mode(days)) == ["dry"] * 44


def missing(n: int) -> List[Tuple[int, int]]:
    """`n` days with no countable hour."""
    return [(0, 0)] * n


def test_the_calendar_decides_when_under_half_the_hours_are_countable() -> None:
    # December, a dead station: no countable hour, so the calendar says wet.
    days = series("2026-11-20", dry(14) + missing(10))
    result = wet_mode(days)
    assert result["mode"].iloc[-1] == "wet"
    assert result["decided_by"].iloc[-1] == "calendar"


def test_the_calendar_says_dry_from_may_to_october() -> None:
    days = series("2026-04-20", wet(14, 6) + missing(20))
    result = wet_mode(days)
    assert result.loc["2026-04-30", "mode"] == "wet"
    assert result.loc["2026-05-15", ["mode", "decided_by"]].tolist() == [
        "dry",
        "calendar",
    ]


def test_half_the_hours_counted_is_enough_for_the_weather() -> None:
    # The last day follows 7 full days and 7 dead ones: 168 of 336 hours,
    # exactly half. One hour fewer and the calendar decides.
    half = wet_mode(series("2026-12-01", dry(7) + missing(7) + dry(1)))
    assert half.iloc[-1].tolist() == ["dry", "weather"]
    short = series("2026-12-01", dry(6) + [(0, 23)] + missing(7) + dry(1))
    assert wet_mode(short).iloc[-1].tolist() == ["wet", "calendar"]


def test_the_replay_starts_on_the_calendar_and_hands_over_to_the_weather() -> None:
    # A first day in December has nothing before it: the calendar says wet.
    # Once 14 dry days are counted, the weather ends wet mode.
    days = series("2026-12-01", dry(20))
    result = wet_mode(days)
    assert result.iloc[0].tolist() == ["wet", "calendar"]
    assert result.iloc[-1].tolist() == ["dry", "weather"]


def test_a_day_missing_from_the_series_counts_no_hour() -> None:
    # Ten January rows dropped: the days before the last are under half
    # counted, so the calendar decides, not the dry days either side.
    days = series("2027-01-01", dry(14) + dry(10) + dry(4)).drop(
        pd.date_range("2027-01-15", periods=10, freq="D")
    )
    result = wet_mode(days)
    assert list(result.index) == list(days.index)
    assert result.iloc[-1].tolist() == ["wet", "calendar"]


FIXTURE = Path(__file__).parent / "fixtures" / "wet_mode_365_days.csv"
"""
One station's last 365 days to 2026-11-23: wet and countable hours per
station-local day, under the countable-hour rule and the counter helper
(#656 back-test). No station identifier.
"""


def test_a_year_of_real_weather_flips_six_times() -> None:
    days = pd.read_csv(FIXTURE, index_col="day", parse_dates=True)
    result = wet_mode(days)
    mode = result["mode"]
    flips = mode[mode != mode.shift()].iloc[1:]
    assert [(str(day.date()), m) for day, m in flips.items()] == [
        ("2026-01-22", "dry"),
        ("2026-03-13", "wet"),
        ("2026-05-06", "dry"),
        ("2026-09-05", "wet"),
        ("2026-09-21", "dry"),
        ("2026-10-29", "wet"),
    ]
    # The calendar decides only the fixture's first week, before half of 14
    # days can be counted from the fixture alone. Replayed over the whole
    # archive, the calendar decides no day of these 365.
    calendar = result.index[result["decided_by"] == "calendar"]
    assert list(calendar) == list(pd.date_range("2025-11-24", "2025-11-30"))

"""
The Rain tab's data prep in `lookout.core.rainfall_analysis` (#83).

Daily totals, the water year's day index, dry spells and rain by the hour,
each data in, value out. The archives are synthetic five-minute stations from
`tests.fixtures.archive`; local days are Pacific (ADR-0002).
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from lookout.core.rainfall_analysis import (
    dry_spells,
    extract_daily_rainfall,
    hourly_rain,
    water_year,
    water_year_day_index,
    water_year_start,
)
from tests.fixtures.archive import Gap, RainSpell, build_archive


def day(iso: str) -> date:
    """A station-local calendar day."""
    return date.fromisoformat(iso)


def daily(first: str, values) -> pd.Series:
    """Daily totals from `first` on, one per day; None is a day not known."""
    index = pd.date_range(first, periods=len(values), freq="D").date
    return pd.Series([np.nan if v is None else v for v in values], index=index)


def approx(value: float):
    """Rain totals to a ten-thousandth of an inch."""
    return pytest.approx(value, abs=1e-4)


class TestDailyTotals:
    def test_a_day_total_is_its_highest_daily_counter(self):
        # 0.1 in/h for 3 h, from 10:00 local on Jan 5.
        archive = build_archive(
            "2026-01-05T08:00:00Z",
            "2026-01-07T08:00:00Z",
            rain=[RainSpell("2026-01-05T18:00:00Z", 3, 0.1)],
        )

        totals = extract_daily_rainfall(archive).set_index("date")["rainfall"]

        assert totals[day("2026-01-05")] == 0.3
        assert totals[day("2026-01-06")] == 0.0


class TestWaterYear:
    def test_it_is_named_by_the_year_it_ends_in(self):
        assert water_year(day("2025-10-01")) == 2026
        assert water_year(day("2026-09-30")) == 2026
        assert water_year(day("2025-09-30")) == 2025
        assert water_year_start(2026) == day("2025-10-01")

    def test_the_day_index_counts_from_oct_1(self):
        assert water_year_day_index(day("2025-10-01")) == 0
        assert water_year_day_index(day("2025-10-31")) == 30
        assert water_year_day_index(day("2026-01-01")) == 92
        assert water_year_day_index(day("2026-09-30")) == 364

    def test_feb_29_shares_feb_28s_index(self):
        assert water_year_day_index(day("2024-02-29")) == water_year_day_index(
            day("2024-02-28")
        )
        assert water_year_day_index(day("2024-03-01")) == water_year_day_index(
            day("2025-03-01")
        )


class TestDrySpells:
    def test_a_dry_spell_runs_between_two_rain_days(self):
        totals = daily("2026-01-01", [0.2, 0, 0, 0, 0.05, 0, 0.3])

        spells = dry_spells(totals)

        assert list(spells.itertuples(index=False, name=None)) == [
            (day("2026-01-02"), day("2026-01-04"), 3),
            (day("2026-01-06"), day("2026-01-06"), 1),
        ]

    def test_a_trace_under_a_hundredth_is_a_dry_day(self):
        totals = daily("2026-01-01", [0.2, 0.004, 0, 0.3])

        assert dry_spells(totals)["days"].tolist() == [2]

    def test_a_spell_with_a_day_not_known_is_left_out(self):
        # A missing day is never read as a dry day.
        totals = daily("2026-01-01", [0.2, 0, None, 0, 0.1, 0, 0.1])

        assert dry_spells(totals)["days"].tolist() == [1]

    def test_spells_still_open_at_either_end_are_left_out(self):
        totals = daily("2026-01-01", [0, 0, 0.1, 0, 0.2, 0, 0, 0])

        assert dry_spells(totals)["days"].tolist() == [1]

    def test_no_rain_means_no_completed_spell(self):
        assert dry_spells(daily("2026-01-01", [0, 0, 0])).empty
        assert dry_spells(daily("2026-01-01", [])).empty


class TestHourlyRain:
    def test_rain_lands_in_the_local_hour_it_fell(self):
        # 0.12 in/h from 10:00 to 11:30 local on Jan 5 (18:00Z-19:30Z).
        archive = build_archive(
            "2026-01-05T08:00:00Z",
            "2026-01-06T08:00:00Z",
            rain=[RainSpell("2026-01-05T18:00:00Z", 1.5, 0.12)],
        )

        hours = hourly_rain(archive).set_index(["date", "hour"])["rain_in"]

        assert hours[(day("2026-01-05"), 10)] == approx(0.12)
        assert hours[(day("2026-01-05"), 11)] == approx(0.06)
        assert hours[(day("2026-01-05"), 9)] == 0

    def test_each_day_sums_to_its_daily_total(self):
        # Rain across local midnight: the counter resets, nothing is lost.
        archive = build_archive(
            "2026-01-05T08:00:00Z",
            "2026-01-07T08:00:00Z",
            rain=[RainSpell("2026-01-06T06:00:00Z", 4, 0.1)],
        )

        by_day = hourly_rain(archive).groupby("date")["rain_in"].sum()
        totals = extract_daily_rainfall(archive).set_index("date")["rainfall"]

        assert by_day[day("2026-01-05")] == approx(0.2)
        assert by_day[day("2026-01-06")] == approx(0.2)
        for d, total in totals.items():
            assert by_day[d] == approx(total)

    def test_rain_during_a_gap_lands_in_the_hour_after_it(self):
        archive = build_archive(
            "2026-01-05T08:00:00Z",
            "2026-01-06T08:00:00Z",
            rain=[RainSpell("2026-01-05T18:00:00Z", 1, 0.2)],
            gaps=[Gap("2026-01-05T18:00:00Z", "2026-01-05T19:00:00Z")],
        )

        hours = hourly_rain(archive).set_index(["date", "hour"])["rain_in"]

        assert (day("2026-01-05"), 10) not in hours.index
        assert hours[(day("2026-01-05"), 11)] == approx(0.2)

    def test_an_empty_archive_has_no_hours(self):
        empty = pd.DataFrame({"dateutc": [], "dailyrainin": []})

        assert hourly_rain(empty).empty

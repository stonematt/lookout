"""
The Overview's almanac grid and rolling-window rain table (#82; spec #66
seam 1).

`build_snapshot(...).almanac`: water-year rows of monthly rain, and rain over
rolling windows ending today against the same windows in earlier years. Every
total is an archive sum with today's live total in it (spec #66, decided
2026-10-30), every value follows its gap status, and seasons run from Oct 1.

The October station below crosses Oct 1 twice: the 30-day window ending
Oct 12, 2025 holds both Sep and Oct rain, and the season so far starts over
on Oct 1. Its almanac is also written as a golden document the TypeScript
render tests load (`tests/golden/almanac/october.json`); regenerate it with

    LOOKOUT_REGEN_GOLDEN=1 python -m pytest tests/test_snapshot_almanac.py \
        -p no:cacheprovider
"""

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pytest

from lookout.snapshot import Station, build_snapshot
from lookout.snapshot.model import Almanac, AlmanacYear, RollingWindow, Snapshot
from tests.fixtures.ambient import to_ambient_records
from tests.fixtures.archive import Gap, RainSpell, build_archive

STATION = Station(mac="00:11:22:33:44:55", name="Fixture Station")
GOLDEN = Path(__file__).parent / "golden" / "almanac" / "october.json"
REGENERATE = os.environ.get("LOOKOUT_REGEN_GOLDEN") == "1"

# Oct 2025 and Sep 2024 are PDT: local midnight is 07:00Z.
START = "2024-08-20T07:00:00Z"  # Aug 20, 2024: Aug 2024 began before it
END = "2025-10-12T19:00:00Z"  # noon on Oct 12, 2025
NOW = "2025-10-12T19:00:00Z"
RAIN = (
    RainSpell("2024-09-25T18:00:00Z", 1, 0.5),  # Sep 25, 2024: WY2024
    RainSpell("2024-10-05T18:00:00Z", 1, 0.25),  # Oct 5, 2024: WY2025
    RainSpell("2024-12-10T18:00:00Z", 2, 0.5),  # Dec 10, 2024: 1.0 in
    RainSpell("2025-09-28T18:00:00Z", 1, 0.3),  # Sep 28, 2025
    RainSpell("2025-09-30T18:00:00Z", 1, 0.4),  # Sep 30: WY2025's last day
    RainSpell("2025-10-01T18:00:00Z", 1, 0.2),  # Oct 1: WY2026's first day
    RainSpell("2025-10-12T16:00:00Z", 1, 0.1),  # today, 9 am
)


def build(archive: pd.DataFrame, now: str, live: Optional[Dict] = None) -> Snapshot:
    """`build_snapshot` for the fixture station, no catalogs."""
    return build_snapshot(
        archive=archive,
        live=live,
        event_catalog=None,
        energy_catalog=None,
        station=STATION,
        now=pd.Timestamp(now),
    )


def october(**changes) -> Snapshot:
    """The October station, built at noon on Oct 12, 2025."""
    archive = build_archive(START, END, rain=changes.pop("rain", RAIN), **changes)
    return build(archive, NOW)


def row(almanac: Almanac, water_year: int) -> AlmanacYear:
    """One water year's row."""
    return next(y for y in almanac.years if y.water_year == water_year)


def window(almanac: Almanac, key: str) -> RollingWindow:
    """One rolling window."""
    return next(w for w in almanac.rolling if w.key == key)


def by_year(almanac: Almanac, key: str) -> Dict[int, Optional[float]]:
    """A rolling window's rain by the year it ends in."""
    return {y.year: y.rain_in for y in window(almanac, key).years}


def approx(value: float):
    """Rain totals to a thousandth of an inch."""
    return pytest.approx(value, abs=1e-3)


@pytest.fixture(scope="module")
def almanac() -> Almanac:
    return october().almanac


class TestTheGrid:
    """Water-year rows of calendar months, Oct through Sep."""

    def test_rows_are_water_years_newest_first(self, almanac):
        assert [y.water_year for y in almanac.years] == [2026, 2025, 2024]

    def test_a_row_runs_oct_through_sep(self, almanac):
        keys = [m.status.key for m in row(almanac, 2025).months]

        assert keys[0] == "2024-10"
        assert keys[3] == "2025-01"
        assert keys[-1] == "2025-09"

    def test_a_month_is_the_sum_of_its_days(self, almanac):
        months = row(almanac, 2025).months

        assert months[0].rain_in == approx(0.25)  # Oct 2024
        assert months[2].rain_in == approx(1.0)  # Dec 2024
        assert months[1].rain_in == approx(0.0)  # Nov 2024: dry, not missing
        assert months[11].rain_in == approx(0.7)  # Sep 2025

    def test_sep_30_closes_one_season_and_oct_1_opens_the_next(self, almanac):
        assert row(almanac, 2025).months[11].rain_in == approx(0.7)
        assert row(almanac, 2025).season_in == approx(0.25 + 1.0 + 0.7)
        assert row(almanac, 2026).months[0].to_date_in == approx(0.3)

    def test_a_month_before_the_archive_began_is_not_compared(self, almanac):
        august = row(almanac, 2024).months[10]  # Aug 2024: began on Aug 20

        assert august.status.status == "invalid"
        assert august.status.reason == "before_archive"
        assert august.rain_in is None
        assert row(almanac, 2024).months[11].rain_in == approx(0.5)  # Sep 2024
        assert row(almanac, 2024).season_in is None

    def test_the_current_month_is_not_compared_but_has_its_rain_so_far(self, almanac):
        current = row(almanac, 2026).months[0]

        assert current.status.reason == "in_progress"
        assert current.rain_in is None
        assert (current.to_date.first_day, current.to_date.last_day) == (
            "2025-10-01",
            "2025-10-12",
        )
        assert current.to_date_in == approx(0.3)

    def test_only_the_current_month_carries_a_to_date_window(self, almanac):
        others = [m for y in almanac.years for m in y.months if m][1:]

        assert all(m.to_date is None and m.to_date_in is None for m in others)

    def test_months_not_begun_are_empty(self, almanac):
        assert row(almanac, 2026).months[1:] == (None,) * 11

    def test_the_running_season_is_not_compared(self, almanac):
        season = row(almanac, 2026).season

        assert (season.first_day, season.last_day) == ("2025-10-01", "2026-09-30")
        assert season.reason == "in_progress"
        assert row(almanac, 2026).season_in is None

    def test_a_month_missing_days_is_prorated_then_invalid(self):
        snapshot = october(
            gaps=(
                # Nov 2024: 2 of 30 days missing, so scaled up (PST: 08:00Z).
                Gap("2024-11-03T08:00:00Z", "2024-11-05T08:00:00Z"),
                # Jan 2025: 5 of 31 days missing, so not compared.
                Gap("2025-01-10T08:00:00Z", "2025-01-15T08:00:00Z"),
            ),
            rain=RAIN + (RainSpell("2024-11-20T18:00:00Z", 1, 0.28),),
        )
        months = row(snapshot.almanac, 2025).months

        assert months[1].status.status == "prorated"
        assert months[1].rain_in == approx(0.28 * 30 / 28)
        assert months[3].status.status == "invalid"
        assert months[3].rain_in is None
        # The season is judged as its own window: 7 of 365 days missing.
        season = row(snapshot.almanac, 2025)
        assert season.season.status == "prorated"
        assert season.season_in == approx((0.25 + 0.28 + 1.0 + 0.7) * 365 / 358)

    def test_month_statuses_are_the_snapshot_gap_statuses(self):
        snapshot = october()
        gaps = {m.key: m for m in snapshot.gaps.months}

        for year in snapshot.almanac.years:
            for month in year.months:
                if month is not None and month.status.key in gaps:
                    assert month.status == gaps[month.status.key]


class TestRollingWindows:
    """Rain over windows ending today, against the same days in earlier years."""

    def test_windows_are_the_week_month_season_and_year(self, almanac):
        keys = [w.key for w in almanac.rolling]

        assert keys == ["7d", "30d", "90d", "365d", "water_year"]

    def test_years_are_named_newest_first(self, almanac):
        assert [y.year for y in window(almanac, "30d").years] == [2025, 2024]

    def test_a_30_day_window_across_oct_1_holds_both_sides(self, almanac):
        assert by_year(almanac, "30d") == {
            2025: approx(0.3 + 0.4 + 0.2 + 0.1),  # Sep 13 to Oct 12, 2025
            2024: approx(0.5 + 0.25),  # Sep 13 to Oct 12, 2024
        }
        earlier = window(almanac, "30d").years[1].status

        assert (earlier.first_day, earlier.last_day) == ("2024-09-13", "2024-10-12")

    def test_the_season_so_far_starts_over_on_oct_1(self, almanac):
        assert by_year(almanac, "water_year") == {2025: approx(0.3), 2024: approx(0.25)}
        earlier = window(almanac, "water_year").years[1].status

        assert (earlier.first_day, earlier.last_day) == ("2024-10-01", "2024-10-12")

    def test_a_dry_week_is_zero_not_missing(self, almanac):
        assert by_year(almanac, "7d") == {2025: approx(0.1), 2024: approx(0.0)}

    def test_a_window_reaching_before_the_archive_is_not_compared(self, almanac):
        earlier = window(almanac, "365d").years[1]

        assert earlier.status.reason == "before_archive"
        assert earlier.rain_in is None

    def test_this_years_windows_are_the_rain_tabs_windows(self):
        snapshot = october()
        rain_tab = {w.key: w.total_in for w in snapshot.rain_tab.windows}
        statuses = {s.key: s for s in snapshot.gaps.windows}

        for rolling in snapshot.almanac.rolling:
            now = rolling.years[0]
            assert now.rain_in == rain_tab[rolling.key]
            assert now.status == statuses[rolling.key]

    def test_earlier_seasons_so_far_are_the_rain_tabs_curves(self):
        snapshot = october()
        curves = {c.water_year: c.to_date_in for c in snapshot.rain_tab.water_years}

        for year in window(snapshot.almanac, "water_year").years:
            assert year.rain_in == curves[year.water_year]

    def test_the_season_row_is_named_by_its_water_year(self, almanac):
        # Oct 12, 2025 is in water year 2026; the column stays the calendar
        # year the windows end in (#147).
        season = window(almanac, "water_year").years
        assert [y.water_year for y in season] == [2026, 2025]
        assert [y.year for y in season] == [2025, 2024]

    def test_a_december_season_is_the_next_water_year(self):
        archive = build_archive(
            "2024-10-01T07:00:00Z", "2025-12-15T20:00:00Z", rain=RAIN
        )
        snapshot = build(archive, "2025-12-15T20:00:00Z")
        season = window(snapshot.almanac, "water_year").years[0]

        assert season.water_year == 2026
        assert season.status.first_day == "2025-10-01"

    def test_feb_29_compares_with_feb_28(self):
        archive = build_archive(
            "2023-02-01T08:00:00Z",
            "2024-02-29T20:00:00Z",
            rain=(RainSpell("2023-02-28T18:00:00Z", 1, 0.2),),
        )
        snapshot = build(archive, "2024-02-29T20:00:00Z")
        earlier = window(snapshot.almanac, "7d").years[1]

        assert earlier.status.last_day == "2023-02-28"
        assert earlier.rain_in == approx(0.2)


class TestEveryTotalIsAnArchiveSum:
    """Totals sum each day's highest daily counter; today's is live."""

    def test_todays_live_total_is_in_every_window_ending_today(self):
        archive = build_archive(START, END, rain=RAIN)
        live = to_ambient_records(archive.head(1))[0]
        live["dateutc"] = int(pd.Timestamp(NOW).value // 10**6)
        live["dailyrainin"] = 0.35  # the archive has 0.1 so far today
        almanac = build(archive, NOW, live).almanac

        assert by_year(almanac, "7d")[2025] == approx(0.35)
        assert by_year(almanac, "water_year")[2025] == approx(0.2 + 0.35)
        assert row(almanac, 2026).months[0].to_date_in == approx(0.2 + 0.35)

    def test_ambients_year_month_and_week_counters_are_never_used(self):
        archive = build_archive(START, END, rain=RAIN)
        honest = build(archive, NOW).almanac
        archive[["yearlyrainin", "monthlyrainin", "weeklyrainin"]] = 99.0

        assert build(archive, NOW).almanac == honest

    def test_a_lagging_archive_leaves_todays_windows_uncompared(self):
        # The archive stops a day before now: today is not archived, so no
        # window ending today is computed, while complete months still are.
        archive = build_archive(START, "2025-10-11T19:00:00Z", rain=RAIN)
        almanac = build(archive, NOW).almanac
        now = window(almanac, "7d").years[0]

        assert (now.status.last_day, now.status.reason) == (
            "2025-10-12",
            "not_archived",
        )
        assert now.rain_in is None
        assert by_year(almanac, "7d")[2024] == approx(0.0)
        assert row(almanac, 2025).months[11].rain_in == approx(0.7)
        assert row(almanac, 2026).months[0].to_date.reason == "not_archived"

    def test_an_empty_archive_has_no_rows_and_nothing_compared(self):
        empty = build_archive(START, START)
        almanac = build(empty, NOW).almanac

        assert almanac.years == ()
        assert all(len(w.years) == 1 for w in almanac.rolling)
        assert all(w.years[0].rain_in is None for w in almanac.rolling)


def test_the_october_almanac_matches_its_golden_file(almanac):
    document = {"built_at": NOW, "almanac": asdict(almanac)}
    if REGENERATE:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(document, indent=2) + "\n")
    assert json.loads(GOLDEN.read_text()) == json.loads(json.dumps(document))

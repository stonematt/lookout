"""
The Overview's temperature context (#81), through `build_snapshot`.

Three views over the live-stitched history: the temperature across today,
yesterday, 7, 30, 90 and 365 days (low, middle half, high); the ten-day
high-low list against normal; and the "now against normal" strips.

Every window and every earlier year goes through the gap status: an invalid
one is never computed, so its values are null and its status says why.

The station is a synthetic two years and a bit, Dec 2023 to Tue 2026-01-06
noon local, whose temperature runs 4 degrees warmer in 2025 than in 2024 and
swings with the day, so each year's normal is told apart. The `temperature`
section also has a golden document, `tests/golden/temperature/<case>.json`,
that the TypeScript side renders. Regenerate with
`LOOKOUT_REGEN_GOLDEN=1 python -m pytest tests/test_snapshot_temperature.py
-p no:cacheprovider`.
"""

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.ambient import to_ambient_records
from tests.fixtures.archive import Gap, RainSpell, build_archive
from tests.snapshot_helpers import REGENERATE, build

GOLDEN_DIR = Path(__file__).parent / "golden" / "temperature"
NOW = "2026-01-06T20:02:00Z"  # Tue noon local; the newest row is 11:55
LAST_ROW = "2026-01-06T20:00:00Z"
YEAR_OFFSET = {2023: 0.0, 2024: 0.0, 2025: 4.0, 2026: 8.0}


def station(
    gaps: Sequence[Gap] = (), start: str = "2023-12-01T08:00:00Z"
) -> pd.DataFrame:
    """The synthetic station, with a per-year and per-day temperature."""
    rain = (
        RainSpell("2025-01-04T18:00:00Z", hours=1, inches_per_hour=0.10),
        RainSpell("2025-01-02T18:00:00Z", hours=1, inches_per_hour=0.30),
        RainSpell("2024-01-05T18:00:00Z", hours=1, inches_per_hour=0.20),
    )
    archive = build_archive(start, LAST_ROW, rain=rain, gaps=gaps)
    local = pd.to_datetime(archive["dateutc"], unit="ms", utc=True).dt.tz_convert(
        "America/Los_Angeles"
    )
    swing = 3.0 * np.sin(local.dt.dayofyear.to_numpy() * 0.9)
    year = local.dt.year.map(YEAR_OFFSET).to_numpy()
    archive["tempf"] = (archive["tempf"] + swing + year).round(1)
    return archive


def reading(archive: pd.DataFrame, **values) -> Dict:
    """A live `lastData` a minute past the archive, with `values` set."""
    row = to_ambient_records(archive[archive["dateutc"] == archive["dateutc"].max()])
    row[0]["dateutc"] = int(archive["dateutc"].max()) + 60_000
    row[0].update(values)
    return row[0]


@pytest.fixture(scope="module")
def whole() -> pd.DataFrame:
    return station()


@pytest.fixture(scope="module")
def snapshot(whole):
    live = reading(whole, tempf=55.0)
    return build(whole, live, NOW)


def section(snapshot):
    return snapshot.temperature_context


def local_days(archive: pd.DataFrame) -> pd.Series:
    """Each row's station-local date."""
    return (
        pd.to_datetime(archive["dateutc"], unit="ms", utc=True)
        .dt.tz_convert("America/Los_Angeles")
        .dt.date
    )


def assert_golden(snapshot, case: str) -> None:
    """Compare the section with `tests/golden/temperature/<case>.json`."""
    path = GOLDEN_DIR / f"{case}.json"
    text = json.dumps(snapshot.dashboard()["temperature_context"], indent=2) + "\n"
    if REGENERATE:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert json.loads(text) == json.loads(path.read_text())


class TestPeriods:
    def test_the_six_windows_in_order_with_now_live(self, snapshot):
        keys = [p.status.key for p in section(snapshot).periods]

        assert keys == ["today", "yesterday", "7d", "30d", "90d", "365d"]
        assert section(snapshot).now == 55.0

    def test_now_is_placed_against_the_middle_half_once(self, snapshot):
        ctx = section(snapshot)

        assert ctx.finding == "30d"
        by_key = {p.status.key: p for p in ctx.periods}
        assert by_key["30d"].position == "inside"
        assert by_key["today"].position == "inside"

    def test_a_window_is_its_low_middle_half_and_high(self, whole, snapshot):
        week = next(p for p in section(snapshot).periods if p.status.key == "7d")
        days = pd.date_range("2025-12-31", "2026-01-06").date
        values = whole[local_days(whole).isin(set(days))]["tempf"]

        assert week.status.status == "valid"
        assert week.low == values.min()
        assert week.high == values.max()
        assert week.q1 == pytest.approx(values.quantile(0.25), abs=0.06)
        assert week.q3 == pytest.approx(values.quantile(0.75), abs=0.06)
        assert week.low < week.q1 < week.q3 < week.high

    def test_wider_windows_reach_further(self, snapshot):
        by_key = {p.status.key: p for p in section(snapshot).periods}

        assert by_key["365d"].low <= by_key["30d"].low <= by_key["7d"].low
        assert by_key["365d"].high >= by_key["30d"].high >= by_key["7d"].high

    def test_a_window_over_a_gap_is_hatched_and_never_computed(self):
        # Five of the last seven days gone: invalid; 5 of 30 is 17%: invalid.
        gone = Gap("2026-01-01T08:00:00Z", "2026-01-06T08:00:00Z")
        archive = station(gaps=(gone,))

        periods = {
            p.status.key: p
            for p in build(archive, None, NOW).temperature_context.periods
        }

        for key in ("7d", "30d"):
            assert periods[key].status.status == "invalid"
            assert periods[key].status.reason == "gap"
            assert periods[key].low is None and periods[key].high is None
            assert periods[key].readings == 0
        assert periods["365d"].status.status == "prorated"
        assert periods["365d"].low is not None

    def test_the_counted_days_only(self):
        # One cold missing day inside 30 days: prorated, and the values come
        # from the days the archive holds.
        gone = Gap("2025-12-20T08:00:00Z", "2025-12-21T08:00:00Z")
        archive = station(gaps=(gone,))

        month = {
            p.status.key: p
            for p in build(archive, None, NOW).temperature_context.periods
        }["30d"]

        assert month.status.status == "prorated"
        assert month.status.missing_days == 1
        days = local_days(archive)
        held = archive[(days >= pd.Timestamp("2025-12-08").date())]
        assert month.low == held["tempf"].min()
        assert month.readings == len(held)

    def test_the_archive_lagging_hatches_today(self, whole):
        lagging = whole[whole["dateutc"] < whole["dateutc"].max() - 3 * 3_600_000]

        periods = {
            p.status.key: p
            for p in build(lagging, None, NOW).temperature_context.periods
        }

        assert periods["today"].status.reason == "not_archived"
        assert periods["today"].low is None


class TestTenDays:
    def test_ten_days_oldest_first_ending_today(self, snapshot):
        days = [d.day for d in section(snapshot).ten_days]

        assert days[0] == "2025-12-28"
        assert days[-1] == "2026-01-06"
        assert len(days) == 10

    def test_normal_is_the_median_daily_low_and_high_of_earlier_years(
        self, whole, snapshot
    ):
        today = section(snapshot).ten_days[-1]
        days = local_days(whole)
        lows, highs = [], []
        for year in (2025, 2024):
            centre = pd.Timestamp(f"{year}-01-06")
            span = set(
                pd.date_range(
                    centre - pd.Timedelta(days=7), centre + pd.Timedelta(days=7)
                ).date
            )
            for day in sorted(span):
                values = whole[days == day]["tempf"]
                lows.append(values.min())
                highs.append(values.max())

        assert today.usual_years == (2025, 2024)
        assert today.usual_days == len(lows)
        assert today.usual_low == pytest.approx(np.median(lows), abs=0.06)
        assert today.usual_high == pytest.approx(np.median(highs), abs=0.06)
        assert today.low < today.high

    def test_headline_years_are_every_year_a_day_used_newest_first(self, snapshot):
        # The Dec days reach back to Dec 2023; the Jan days do not.
        assert section(snapshot).ten_day_years == (2025, 2024, 2023)
        assert section(snapshot).ten_days[0].usual_years == (2024, 2023)
        assert section(snapshot).ten_days[-1].usual_years == (2025, 2024)

    def test_a_year_with_a_gap_in_its_window_is_left_out(self):
        gone = Gap("2025-01-03T08:00:00Z", "2025-01-07T08:00:00Z")
        archive = station(gaps=(gone,))

        today = build(archive, None, NOW).temperature_context.ten_days[-1]

        assert today.usual_years == (2024,)

    def test_a_day_the_archive_cannot_count_is_hatched_without_values(self):
        gone = Gap("2026-01-02T08:00:00Z", "2026-01-03T08:00:00Z")
        archive = station(gaps=(gone,))

        days = {
            d.day: d for d in build(archive, None, NOW).temperature_context.ten_days
        }

        assert days["2026-01-02"].status.status == "invalid"
        assert days["2026-01-02"].low is None and days["2026-01-02"].high is None
        assert days["2026-01-02"].usual_low is not None
        assert days["2026-01-03"].low is not None

    def test_a_young_archive_has_no_normal(self):
        archive = station(start="2025-12-01T08:00:00Z")

        today = build(archive, None, NOW).temperature_context.ten_days[-1]

        assert today.low is not None
        assert today.usual_low is None and today.usual_high is None
        assert today.usual_years == () and today.usual_days == 0


class TestStrips:
    def strips(self, snapshot) -> Dict:
        return {s.kind: s for s in section(snapshot).strips}

    def test_a_temperature_strip_is_now_among_past_days_at_this_hour(
        self, whole, snapshot
    ):
        strip = self.strips(snapshot)["temperature"]
        days = local_days(whole)
        hour = (
            pd.to_datetime(whole["dateutc"], unit="ms", utc=True)
            .dt.tz_convert("America/Los_Angeles")
            .dt.hour
        )
        expected = []
        for year in (2025, 2024):
            centre = pd.Timestamp(f"{year}-01-06")
            for day in pd.date_range(
                centre - pd.Timedelta(days=7), centre + pd.Timedelta(days=7)
            ).date:
                expected.append(whole[(days == day) & (hour == 11)]["tempf"].mean())

        assert strip.unit == "f" and strip.now == 55.0
        assert strip.years == (2025, 2024)
        assert len(strip.values) == len(expected) == 30
        assert list(strip.values) == pytest.approx(sorted(expected), abs=0.001)
        below = round(100 * sum(v < 55.0 for v in expected) / 30)
        assert strip.below_pct == below
        assert strip.above_pct == round(100 * sum(v > 55.0 for v in expected) / 30)

    def test_a_warm_now_is_above_every_past_day(self, whole):
        live = reading(whole, tempf=95.0)

        strip = build(whole, live, NOW).temperature_context.strips[0]

        assert strip.below_pct == 100 and strip.equal_pct == 0

    def test_a_rain_strip_compares_rain_so_far_by_this_time(self, whole):
        live = reading(whole, dailyrainin=0.05)

        strip = {s.kind: s for s in build(whole, live, NOW).temperature_context.strips}[
            "rain"
        ]

        assert strip.unit == "in" and strip.now == 0.05
        # The 2025 spell at 10:00 local has put 0.10 and 0.30 in by noon; the
        # 2024 one 0.20. Everything else is dry.
        assert [v for v in strip.values if v > 0] == pytest.approx([0.1, 0.2, 0.3])
        assert strip.below_pct == round(100 * 27 / 30)
        assert strip.equal_pct == 0

    def test_dry_now_ties_with_the_dry_days(self, whole):
        live = reading(whole, dailyrainin=0.0)

        strip = build(whole, live, NOW).temperature_context.strips[1]

        assert strip.below_pct == 0
        assert strip.equal_pct == round(100 * 27 / 30)

    def test_without_an_earlier_year_the_strip_says_so(self):
        archive = station(start="2025-12-01T08:00:00Z")

        strips = build(archive, None, NOW).temperature_context.strips

        for strip in strips:
            assert strip.missing == "no_comparison"
            assert strip.values == () and strip.below_pct is None

    def test_golden(self, snapshot):
        assert_golden(snapshot, "year")


class TestEmpty:
    def test_an_empty_archive_builds_nothing_to_compare(self):
        empty = station().iloc[0:0]

        context = build(empty, None, NOW).temperature_context

        assert context.now is None
        assert len(context.periods) == 6
        assert context.ten_days == () and context.strips == ()

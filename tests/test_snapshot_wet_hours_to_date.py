"""
Wet hours since Oct 1 against usual wet hours to date (#663, spec #647).

GLOSSARY: **usual wet hours to date** is the 25th to 75th percentile of
earlier water years' **wet share** from Oct 1 to this date, times this
year's countable hours. An earlier year covers the date only if at least
half its hours from Oct 1 to the date are countable. N, the years it rests
on: 3 or more give a band ("range" at 3, "percentile" from 4), 1 or 2 the
plain values, 0 none.

Through `build_snapshot`; archives are synthetic five-minute stations from
`tests.fixtures.archive`, local days Pacific. Each water year holds Sep 30,
23:00 local to Oct 9; the build is at 13:10 local on Wed Oct 8, 2025, so
this year's span (and each earlier year's) is Oct 1 00:00 to Oct 8 13:00:
181 hours.
"""

from typing import Optional, Sequence

import pandas as pd

from lookout.snapshot import Snapshot
from lookout.snapshot.model import WetHoursToDate
from tests.fixtures.archive import Gap, RainSpell, build_archive
from tests.snapshot_helpers import build, live_row

NOW = "2025-10-08T20:10:00Z"  # 13:10 local
SPAN_HOURS = 7 * 24 + 13


def year_archive(
    year: int, wet_hours: int = 0, gaps: Sequence[Gap] = (), end: str = ""
) -> pd.DataFrame:
    """
    One water year's first days: Sep 30, 23:00 local to Oct 9.

    :param year: The water year, named by the year it ends in.
    :param wet_hours: Whole wet hours of steady rain from Oct 2, 10:00 local.
    :param gaps: Stretches with no readings.
    :param end: The last reading's end, exclusive; Oct 9 by default.
    :return: The readings, oldest first.
    """
    first = year - 1
    rain = ()
    if wet_hours:
        rain = (RainSpell(f"{first}-10-02T17:00:00Z", wet_hours, 0.05),)
    return build_archive(
        f"{first}-10-01T06:00:00Z",
        end or f"{first}-10-09T07:00:00Z",
        rain=rain,
        gaps=gaps,
        newest_first=False,
    )


def snapshot_of(
    *earlier: pd.DataFrame, this_year: Optional[pd.DataFrame] = None, now: str = NOW
) -> Snapshot:
    """
    `build_snapshot` at `now` over this year and the earlier years given.

    :param earlier: Earlier water years' readings.
    :param this_year: This year's readings; a dry year to `now` by default.
    :param now: The build time, UTC.
    :return: The snapshot; the live reading is the archive's newest row.
    """
    current = this_year if this_year is not None else year_archive(2026, end=now)
    archive = pd.concat([*earlier, current], ignore_index=True)
    archive = archive.sort_values("dateutc", ascending=False, ignore_index=True)
    newest = pd.Timestamp(int(archive["dateutc"].max()), unit="ms", tz="UTC")
    return build(archive, live_row(archive, newest.isoformat()), now)


def to_date(
    *earlier: pd.DataFrame, this_year: Optional[pd.DataFrame] = None, now: str = NOW
) -> WetHoursToDate:
    """
    The Rain tab's wet hours to date, built as `snapshot_of` builds.

    :param earlier: Earlier water years' readings.
    :param this_year: This year's readings.
    :param now: The build time, UTC.
    :return: `RainTab.wet_hours_to_date`.
    """
    wet = snapshot_of(*earlier, this_year=this_year, now=now).rain_tab
    assert wet.wet_hours_to_date is not None
    return wet.wet_hours_to_date


class TestThisWaterYear:
    def test_counts_every_whole_hour_since_oct_1(self):
        wet = to_date(this_year=year_archive(2026, wet_hours=3, end=NOW))

        assert (wet.water_year, wet.first_day) == (2026, "2025-10-01")
        assert wet.hours == SPAN_HOURS
        assert wet.wet_hours == 3
        assert wet.countable_hours == SPAN_HOURS
        assert wet.missing_hours == 0

    def test_ends_where_the_overview_meter_ends(self):
        snapshot = snapshot_of(this_year=year_archive(2026, wet_hours=3, end=NOW))

        wet = snapshot.rain_tab.wet_hours_to_date
        assert wet.through_ms == snapshot.wet_hours.through_ms
        # Rain on Oct 2 is in both the 30-day window and the water year.
        assert wet.wet_hours == snapshot.wet_hours.last_30_days.wet_hours

    def test_the_first_hour_of_oct_1_has_no_hour_yet(self):
        # 00:10 local on Oct 1: the hour in progress is the year's first.
        oct_1 = "2025-10-01T07:10:00Z"
        last_year = year_archive(2025, wet_hours=6)

        wet = to_date(last_year, this_year=year_archive(2026, end=oct_1), now=oct_1)

        assert (wet.water_year, wet.first_day) == (2026, "2025-10-01")
        assert (wet.hours, wet.wet_hours, wet.countable_hours) == (0, 0, 0)
        assert wet.usual is None

    def test_a_gap_lowers_the_countable_hours_never_the_wet_ones(self):
        gap = Gap("2025-10-05T07:05:00Z", "2025-10-05T12:00:00Z")
        wet = to_date(this_year=year_archive(2026, wet_hours=3, gaps=[gap], end=NOW))

        assert wet.wet_hours == 3
        assert wet.missing_hours == 5
        assert wet.countable_hours == SPAN_HOURS - 5


class TestUsualWetHours:
    def test_no_earlier_year_has_no_usual(self):
        assert to_date().usual is None

    def test_one_earlier_year_gives_its_plain_value(self):
        usual = to_date(year_archive(2025, wet_hours=6)).usual

        assert usual.years == 1
        assert [(y.water_year, y.usual_hours) for y in usual.earlier] == [(2025, 6)]
        assert (usual.low_hours, usual.high_hours, usual.band) == (None, None, None)

    def test_two_earlier_years_give_both_plain_values(self):
        usual = to_date(
            year_archive(2024, wet_hours=4), year_archive(2025, wet_hours=8)
        ).usual

        assert usual.years == 2
        assert [y.usual_hours for y in usual.earlier] == [4, 8]
        assert usual.band is None
        assert usual.low_hours is None

    def test_three_earlier_years_give_a_range_not_a_percentile(self):
        usual = to_date(
            year_archive(2023, wet_hours=2),
            year_archive(2024, wet_hours=6),
            year_archive(2025, wet_hours=4),
        ).usual

        assert usual.years == 3
        assert usual.band == "range"
        # The 25th and 75th percentiles of 2, 4 and 6 wet hours of 181.
        assert (usual.low_hours, usual.high_hours) == (3, 5)

    def test_four_earlier_years_give_a_percentile_band(self):
        usual = to_date(
            year_archive(2022, wet_hours=16),
            year_archive(2023, wet_hours=4),
            year_archive(2024, wet_hours=12),
            year_archive(2025, wet_hours=8),
        ).usual

        assert usual.years == 4
        assert usual.band == "percentile"
        assert [y.water_year for y in usual.earlier] == [2022, 2023, 2024, 2025]
        assert (usual.low_hours, usual.high_hours) == (7, 13)

    def test_scales_each_share_to_this_years_countable_hours(self):
        # Last year: 10 wet of 181. This year loses 20 hours to a gap.
        gap = Gap("2025-10-05T07:05:00Z", "2025-10-06T03:00:00Z")
        wet = to_date(
            year_archive(2025, wet_hours=10),
            this_year=year_archive(2026, gaps=[gap], end=NOW),
        )

        assert wet.countable_hours == SPAN_HOURS - 20
        year = wet.usual.earlier[0]
        assert (year.wet_hours, year.countable_hours) == (10, SPAN_HOURS)
        assert year.usual_hours == round(10 / SPAN_HOURS * (SPAN_HOURS - 20))

    def test_a_year_covers_the_date_only_with_half_its_hours_countable(self):
        # Readings stop at Oct 4, 19:00 local: 91 of 181 hours, at least
        # half; stopping an hour sooner leaves 90, under half.
        half = year_archive(2025, wet_hours=6, end="2024-10-05T02:00:00Z")
        less = year_archive(2024, wet_hours=6, end="2023-10-05T01:00:00Z")

        usual = to_date(less, half).usual

        assert [y.water_year for y in usual.earlier] == [2025]
        assert usual.earlier[0].countable_hours == 91

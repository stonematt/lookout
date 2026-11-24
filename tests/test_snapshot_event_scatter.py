"""
The Rain Events page's headline counts and its scatter (#666), through
`build_snapshot`.

The headline counts the events that started in this water year and those
that started in the 7 days before the build. The scatter holds, for this
water year and the three before it, each completed event's start, total and
category, from the Event Catalog only: the live event is not a point, and a
point's total and category are the catalog's.
"""

import json

import pandas as pd
import pytest

from tests.snapshot_helpers import (
    build,
    event_catalog,
    ms,
    raining_station,
    scenario_until,
)

EVENT_1 = ms("2026-01-04T18:00:00Z")
EVENT_2 = ms("2026-01-07T04:00:00Z")
EVENT_3 = ms("2026-01-08T22:00:00Z")


def shifted(catalog: pd.DataFrame, row: int, years: int) -> pd.DataFrame:
    """One catalog row copied `years` years earlier (a past water year's event)."""
    event = catalog.iloc[[row]].copy()
    offset = pd.DateOffset(years=years)
    event["start_time"] = pd.to_datetime(event["start_time"], utc=True) - offset
    event["end_time"] = pd.to_datetime(event["end_time"], utc=True) - offset
    event["event_id"] = f"past-{years}-{row}"
    return event


def with_past_years(catalog: pd.DataFrame, years: range) -> pd.DataFrame:
    """The catalog with event 2 repeated `years` years back, one per year."""
    past = [shifted(catalog, 1, y) for y in years]
    return pd.concat([catalog, *past], ignore_index=True)


class TestHeadlineCounts:
    def test_counts_this_water_years_events_and_the_last_seven_days(self):
        counts = raining_station().rain_events.counts

        # Built Jan 9, 2026 (water year 2026): all three events, the live
        # one too, started in the last 7 days.
        assert counts.water_year == 2026
        assert counts.events == 3
        assert counts.last_7_days == 3

    def test_an_event_that_started_over_seven_days_ago_is_not_recent(self):
        archive = scenario_until("2026-01-12T20:00:00Z")

        # 7 days before Jan 12 20:00Z is Jan 5 20:00Z: event 1 started Jan 4.
        counts = build(
            archive, None, "2026-01-12T20:00:00Z", event_catalog(archive)
        ).rain_events.counts

        assert (counts.events, counts.last_7_days) == (3, 2)

    def test_events_of_earlier_water_years_are_not_counted(self):
        archive = scenario_until("2026-01-11T08:00:00Z")
        catalog = with_past_years(event_catalog(archive), range(1, 3))

        counts = build(archive, None, "2026-01-11T08:00:00Z", catalog).rain_events

        assert counts.counts.events == 3
        # The seasons still count each water year's own events.
        assert [(s.water_year, s.stats.count) for s in counts.seasons] == [
            (2026, 3),
            (2025, 1),
            (2024, 1),
        ]

    def test_no_catalog_counts_nothing_this_water_year(self):
        archive = scenario_until("2026-01-11T08:00:00Z")

        section = build(archive, None, "2026-01-11T08:00:00Z").rain_events

        assert section.counts.water_year == 2026
        assert (section.counts.events, section.counts.last_7_days) == (0, 0)
        assert section.scatter == ()


class TestScatter:
    def test_points_are_completed_catalog_events_newest_first(self):
        scatter = raining_station().rain_events.scatter

        # Event 3 is live, so it is no point: only the catalog's completed.
        assert [year.water_year for year in scatter] == [2026]
        (year,) = scatter
        assert year.start_ms == (EVENT_2, EVENT_1)
        assert year.total_in == pytest.approx((0.48, 0.36))
        assert year.category == ("shower", "shower")

    def test_a_points_category_is_pythons_from_the_catalog_total_and_peak(self):
        archive = scenario_until("2026-01-11T08:00:00Z")
        catalog = event_catalog(archive)
        catalog.loc[0, "total_rainfall"] = 2.4

        (year,) = build(
            archive, None, "2026-01-11T08:00:00Z", catalog
        ).rain_events.scatter

        by_start = dict(zip(year.start_ms, zip(year.total_in, year.category)))
        assert by_start[EVENT_1] == (2.4, "storm")

    def test_an_event_with_no_total_is_no_point(self):
        archive = scenario_until("2026-01-11T08:00:00Z")
        catalog = event_catalog(archive)
        catalog.loc[0, "total_rainfall"] = float("nan")

        snapshot = build(archive, None, "2026-01-11T08:00:00Z", catalog)
        (year,) = snapshot.rain_events.scatter

        assert EVENT_1 not in year.start_ms
        assert len(year.start_ms) == len(year.total_in) == len(year.category) == 2

    def test_it_holds_this_water_year_and_the_three_before(self):
        archive = scenario_until("2026-01-11T08:00:00Z")
        catalog = with_past_years(event_catalog(archive), range(1, 6))

        scatter = build(
            archive, None, "2026-01-11T08:00:00Z", catalog
        ).rain_events.scatter

        assert [year.water_year for year in scatter] == [2026, 2025, 2024, 2023]
        assert [len(year.start_ms) for year in scatter] == [3, 1, 1, 1]

    def test_a_water_year_with_no_events_inside_the_record_is_an_empty_row(self):
        archive = scenario_until("2026-01-11T08:00:00Z")
        catalog = with_past_years(event_catalog(archive), range(2, 3))

        scatter = build(
            archive, None, "2026-01-11T08:00:00Z", catalog
        ).rain_events.scatter

        # 2025 had no event, but the catalog reaches back to 2024.
        assert [(y.water_year, len(y.start_ms)) for y in scatter] == [
            (2026, 3),
            (2025, 0),
            (2024, 1),
        ]

    def test_this_water_year_is_a_row_before_its_first_event(self):
        # Built Oct 3: the catalog's events are all in water year 2026.
        archive = scenario_until("2026-01-11T08:00:00Z")
        catalog = event_catalog(archive)

        scatter = build(
            archive, None, "2026-10-03T08:00:00Z", catalog
        ).rain_events.scatter

        assert [(y.water_year, len(y.start_ms)) for y in scatter] == [
            (2027, 0),
            (2026, 3),
        ]

    def test_the_scatter_is_parallel_arrays_in_the_dashboard(self):
        dashboard = raining_station().dashboard()
        section = dashboard["rain_events"]

        assert section["counts"] == {"water_year": 2026, "events": 3, "last_7_days": 3}
        assert section["scatter"] == [
            {
                "water_year": 2026,
                "start_ms": [EVENT_2, EVENT_1],
                "total_in": [0.48, 0.36],
                "category": ["shower", "shower"],
            }
        ]
        json.dumps(dashboard, allow_nan=False)

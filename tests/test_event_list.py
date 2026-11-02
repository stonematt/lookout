"""
The Rain Events tab's filter and summary metrics, moved out of the Streamlit
tab (`lookout/ui/rain_events.py`) into `lookout.core.rain_events` (#84).

`filter_events` keeps the tab's three filters: a start-time range, the
quality ratings and a minimum rainfall. `summarize_events` gives its four
metrics: how many events, their total and median rain, and their mean
duration.
"""

import pandas as pd
import pytest

from lookout.core.rain_events import filter_events, summarize_events


def catalog_of(*rows) -> pd.DataFrame:
    """A small Event Catalog: (start UTC, total in, minutes, quality) per row."""
    return pd.DataFrame(
        [
            {
                "start_time": pd.Timestamp(start, tz="UTC"),
                "total_rainfall": total,
                "duration_minutes": minutes,
                "quality_rating": quality,
            }
            for start, total, minutes, quality in rows
        ]
    )


CATALOG = catalog_of(
    ("2025-09-20T12:00", 0.10, 60.0, "excellent"),
    ("2025-10-02T12:00", 0.50, 120.0, "good"),
    ("2025-11-15T12:00", 1.20, 600.0, "poor"),
    ("2026-02-01T12:00", 0.30, 180.0, "fair"),
)


class TestFilterEvents:
    def test_no_filter_keeps_every_event(self):
        assert len(filter_events(CATALOG)) == 4

    def test_the_range_keeps_events_starting_inside_it_end_exclusive(self):
        kept = filter_events(
            CATALOG,
            start=pd.Timestamp("2025-10-02T12:00", tz="UTC"),
            end=pd.Timestamp("2026-02-01T12:00", tz="UTC"),
        )

        assert list(kept["total_rainfall"]) == [0.50, 1.20]

    def test_quality_ratings_keep_only_those_named(self):
        kept = filter_events(CATALOG, qualities=["excellent", "fair"])

        assert list(kept["total_rainfall"]) == [0.10, 0.30]

    def test_minimum_rainfall_is_inclusive(self):
        kept = filter_events(CATALOG, min_rainfall=0.30)

        assert list(kept["total_rainfall"]) == [0.50, 1.20, 0.30]

    def test_the_filters_combine(self):
        kept = filter_events(
            CATALOG,
            start=pd.Timestamp("2025-10-01T07:00", tz="UTC"),
            qualities=["good", "poor", "fair"],
            min_rainfall=0.4,
        )

        assert list(kept["total_rainfall"]) == [0.50, 1.20]

    def test_an_empty_catalog_stays_empty(self):
        assert filter_events(pd.DataFrame()).empty


class TestSummarizeEvents:
    def test_the_four_metrics(self):
        summary = summarize_events(CATALOG)

        assert summary == {
            "count": 4,
            "total_in": pytest.approx(2.10),
            "median_in": pytest.approx(0.40),
            "mean_duration_minutes": pytest.approx(240.0),
        }

    def test_no_events_has_a_zero_count_and_unknown_averages(self):
        assert summarize_events(pd.DataFrame()) == {
            "count": 0,
            "total_in": 0.0,
            "median_in": None,
            "mean_duration_minutes": None,
        }

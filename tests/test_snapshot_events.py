"""
The Rain Events tab's section of the snapshot (#84), through `build_snapshot`.

The section lists every Event Catalog event newest first, keyed by its start
time, with quality, totals, its water year and its rank by total; it
summarises all events and each water year; and it carries the ongoing
event's curve, live stitched. Completed events' curves are files of their
own (`tests/test_refresh.py` covers their writes); `completed_curves` builds
them.
"""

import json

import pandas as pd
import pytest

from lookout.core.rain_events import RainEventCatalog
from lookout.snapshot.events import completed_curves
from tests.fixtures.archive import FAKE_MAC, Gap, RainSpell, build_archive
from tests.snapshot_helpers import (
    STATION,
    build,
    dry_station,
    event_catalog,
    ms,
    raining_station,
    scenario_until,
)

EVENT_1 = ms("2026-01-04T18:00:00Z")
EVENT_2 = ms("2026-01-07T04:00:00Z")
EVENT_3 = ms("2026-01-08T22:00:00Z")


def by_start(snapshot):
    """The section's rows keyed by start."""
    return {row.start_ms: row for row in snapshot.rain_events.events}


class TestTheEventList:
    def test_events_are_listed_newest_first_keyed_by_start(self):
        rows = raining_station().rain_events.events

        assert [row.start_ms for row in rows] == [EVENT_3, EVENT_2, EVENT_1]

    def test_a_completed_event_carries_the_catalog_totals_and_quality(self):
        row = by_start(raining_station())[EVENT_1]

        assert row.end_ms == ms("2026-01-05T02:50:00Z")
        assert row.total_in == pytest.approx(0.36)
        assert row.duration_minutes == 530.0
        assert row.peak_rate_in_hr == pytest.approx(0.12)
        assert row.quality == "excellent"
        assert row.completeness_pct == 100.0
        assert row.max_gap_minutes == 0
        assert (row.ongoing, row.live, row.has_curve) == (False, False, True)

    def test_the_live_event_shows_the_live_total_not_the_archived_one(self):
        row = by_start(raining_station())[EVENT_3]

        # The catalog says 0.24" at the archive's end (00:00Z); live is 0.3".
        assert (row.ongoing, row.live) == (True, True)
        assert row.total_in == 0.3
        assert row.duration_minutes == 145.0
        assert row.end_ms == ms("2026-01-09T00:25:00Z")

    def test_the_live_row_agrees_with_the_active_rain_event(self):
        snapshot = raining_station()
        row = by_start(snapshot)[EVENT_3]
        active = snapshot.active_rain_event

        assert row.start_ms == active.start_ms
        assert row.total_in == active.total_in
        assert row.duration_minutes == active.duration_minutes

    def test_rank_orders_events_by_total_wettest_first(self):
        rows = by_start(raining_station())

        # Event 2 holds 0.48", event 1 0.36", event 3 (live) 0.3".
        assert [rows[s].rank for s in (EVENT_2, EVENT_1, EVENT_3)] == [1, 2, 3]

    def test_an_event_with_no_archive_data_has_no_curve(self):
        archive = scenario_until("2026-01-11T08:00:00Z")
        catalog = event_catalog(archive)
        # Event 1 as the catalog holds it, but the archive lost its readings.
        lost = archive[
            (archive["dateutc"] < EVENT_1)
            | (archive["dateutc"] > ms("2026-01-05T03:00Z"))
        ]
        catalog.loc[0, "quality_rating"] = "invalid"

        row = by_start(build(lost, None, "2026-01-11T08:00:00Z", catalog))[EVENT_1]

        assert row.has_curve is False
        assert row.quality == "invalid"
        assert row.total_in == pytest.approx(0.36)

    def test_a_catalog_row_without_quality_metrics_reads_as_unknown(self):
        archive = scenario_until("2026-01-11T08:00:00Z")
        catalog = event_catalog(archive)
        catalog.loc[0, ["data_completeness", "max_gap_minutes", "max_hourly_rate"]] = (
            float("nan")
        )

        dashboard = build(archive, None, "2026-01-11T08:00:00Z", catalog).dashboard()
        oldest = dashboard["rain_events"]["events"][-1]

        assert oldest["completeness_pct"] is None
        assert oldest["max_gap_minutes"] is None
        assert oldest["peak_rate_in_hr"] is None
        json.dumps(dashboard, allow_nan=False)

    def test_no_catalog_lists_nothing(self):
        archive = scenario_until("2026-01-11T08:00:00Z")

        section = build(archive, None, "2026-01-11T08:00:00Z").rain_events

        assert section.events == ()
        assert section.all_time.count == 0
        assert section.seasons == ()
        assert section.ongoing_curve is None


class TestSummaries:
    def test_all_time_summarises_every_event(self):
        stats = raining_station().rain_events.all_time

        assert stats.count == 3
        # 0.36 + 0.48 + 0.3 (event 3 at its live total).
        assert stats.total_in == pytest.approx(1.14)
        assert stats.median_in == pytest.approx(0.36)

    def test_events_fall_in_the_water_year_they_start_in(self):
        # Sep 30 11:30 pm local (PDT) is water year 2025; Oct 2 is 2026.
        archive = build_archive(
            "2025-09-30T00:00:00Z",
            "2025-10-04T00:00:00Z",
            rain=(
                RainSpell("2025-10-01T06:30:00Z", hours=1, inches_per_hour=0.12),
                RainSpell("2025-10-02T18:00:00Z", hours=2, inches_per_hour=0.12),
            ),
        )
        catalog = event_catalog(archive)

        section = build(archive, None, "2025-10-04T00:00:00Z", catalog).rain_events

        assert [row.water_year for row in section.events] == [2026, 2025]
        assert [(s.water_year, s.stats.count) for s in section.seasons] == [
            (2026, 1),
            (2025, 1),
        ]
        assert section.seasons[0].stats.total_in == pytest.approx(0.24)


class TestCurves:
    def test_the_ongoing_curve_runs_from_the_start_to_the_live_reading(self):
        curve = raining_station().rain_events.ongoing_curve

        assert curve.start_ms == EVENT_3
        assert curve.t_ms[0] == EVENT_3
        assert curve.t_ms[-1] == curve.end_ms == ms("2026-01-09T00:25:00Z")
        assert curve.event_in[-1] == 0.3  # the live reading's event rain
        assert len(curve.t_ms) == len(curve.event_in) == len(curve.rate_in_hr)

    def test_steady_rain_reads_as_a_steady_rate(self):
        curve = raining_station().rain_events.ongoing_curve

        # 0.12 in/hr: 0.01" per 5-minute reading. The first reading has no
        # earlier one to compare with and reads 0.
        archived = curve.rate_in_hr[:-1]
        assert archived[0] == 0.0
        assert list(archived[1:]) == [0.12] * (len(archived) - 1)
        assert set(curve.gap_minutes[:-1]) == {None}

    def test_the_live_reading_past_the_archive_reads_as_a_gap(self):
        curve = raining_station().rain_events.ongoing_curve

        # The archive ends at 23:55Z; the live reading is at 00:25Z. Only the
        # live total is known across those 30 minutes, not a rate.
        assert curve.t_ms[-2] == ms("2026-01-08T23:55:00Z")
        assert curve.gap_minutes[-1] == 30.0
        assert curve.rate_in_hr[-1] is None

    def test_no_ongoing_event_means_no_ongoing_curve(self):
        assert dry_station().rain_events.ongoing_curve is None

    def test_a_reading_after_a_gap_has_no_rate(self):
        archive = build_archive(
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            rain=(RainSpell("2026-01-01T06:00:00Z", hours=3, inches_per_hour=0.12),),
            gaps=(Gap("2026-01-01T07:00:00Z", "2026-01-01T07:30:00Z"),),
        )
        catalog = RainEventCatalog(FAKE_MAC).detect_and_catalog_events(
            archive, auto_save=False
        )

        (curve,) = completed_curves(catalog, archive)
        after = curve.t_ms.index(ms("2026-01-01T07:30:00Z"))

        assert curve.gap_minutes[after] == 35.0
        assert curve.rate_in_hr[after] is None
        assert curve.rate_in_hr[after - 1] == pytest.approx(0.12)

    def test_completed_curves_skip_the_ongoing_and_the_already_stored(self):
        full = scenario_until("2026-01-09T00:30:00Z")
        archive = full[full["dateutc"] < ms("2026-01-09T00:00:00Z")]
        catalog = event_catalog(archive)

        curves = completed_curves(catalog, archive, skip={EVENT_2})

        assert [curve.start_ms for curve in curves] == [EVENT_1]

    def test_completed_curves_are_newest_first_up_to_the_limit(self):
        archive = scenario_until("2026-01-11T08:00:00Z")
        catalog = event_catalog(archive)

        curves = completed_curves(catalog, archive, limit=2)

        assert [curve.start_ms for curve in curves] == [EVENT_3, EVENT_2]

    def test_an_event_with_no_archive_data_gets_no_curve_file(self):
        archive = scenario_until("2026-01-11T08:00:00Z")
        catalog = event_catalog(archive)
        lost = archive[archive["dateutc"] > ms("2026-01-06T00:00Z")]

        curves = completed_curves(catalog, lost)

        assert [curve.start_ms for curve in curves] == [EVENT_3, EVENT_2]

    def test_a_curve_file_is_headed_like_the_snapshot_documents(self):
        archive = scenario_until("2026-01-11T08:00:00Z")
        (curve, *_) = completed_curves(event_catalog(archive), archive)

        document = curve.document(STATION)

        assert document["schema_version"] == 1
        assert document["kind"] == "event_curve"
        assert document["station"] == {"mac": FAKE_MAC, "name": "Fixture Station"}
        assert document["start_ms"] == EVENT_3
        json.dumps(document, allow_nan=False)


def test_the_completed_curve_matches_the_archive_slice():
    """A completed event's curve holds exactly its readings, start to end."""
    archive = scenario_until("2026-01-11T08:00:00Z")
    catalog = event_catalog(archive)

    curves = {c.start_ms: c for c in completed_curves(catalog, archive)}
    curve = curves[EVENT_1]
    readings = archive[
        (archive["dateutc"] >= EVENT_1) & (archive["dateutc"] <= curve.end_ms)
    ].sort_values("dateutc")

    assert list(curve.t_ms) == list(readings["dateutc"])
    assert curve.event_in[-1] == pytest.approx(0.36)
    assert pd.Series(curve.event_in).is_monotonic_increasing


def test_the_live_event_keeps_its_key_once_it_completes():
    """
    The link a viewer copies while an event is live names its start. The
    next run re-detects the event under a new catalog id; once it completes,
    its curve file is written under that same start.
    """
    full = scenario_until("2026-01-10T00:00:00Z")
    early = full[full["dateutc"] < ms("2026-01-09T00:00:00Z")]
    store = RainEventCatalog(FAKE_MAC)
    stored = store.detect_and_catalog_events(early, auto_save=False)
    live_link = by_start(build(early, None, "2026-01-09T00:00:00Z", stored))
    assert live_link[EVENT_3].ongoing

    updated = store.update_catalog_with_new_data(full, stored.copy())
    later = by_start(build(full, None, "2026-01-10T00:00:00Z", updated))

    old_id = stored.sort_values("start_time").iloc[-1]["event_id"]
    new_id = updated.sort_values("start_time").iloc[-1]["event_id"]
    assert old_id != new_id  # the catalog's id moved ...
    assert later[EVENT_3].ongoing is False  # ... the start did not
    assert EVENT_3 in {c.start_ms for c in completed_curves(updated, full)}

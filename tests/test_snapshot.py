"""
`build_snapshot`, spec #66 seam 1, and the snapshot JSON, seam 2.

`build_snapshot` is pure: archive, live `lastData`, catalogs, station and an
injected `now` in, a `Snapshot` out. This first cut (#72) covers current
conditions: live stitching, trends, the active rain event, the newest
observation time and the build time. Later tickets add per-tab fields.

The golden tests write the snapshot JSON the TypeScript side loads, under
`tests/golden/snapshots/<scenario>.<dashboard|context>.json`. After a change to
the snapshot, regenerate them and review the diff:

    LOOKOUT_REGEN_GOLDEN=1 python -m pytest tests/test_snapshot.py \
        -p no:cacheprovider

They are regenerated, never hand-merged: on a rebase conflict in one, take
either side, rerun that command, then rerun the suite.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pytest

from lookout.config import DEFAULT_MAC_ADDRESS
from lookout.snapshot import (
    CONTEXT_FIELDS,
    SCHEMA_VERSION,
    build_snapshot,
    stitch_live,
)
from lookout.snapshot.events import completed_curves
from lookout.snapshot.model import EventCurve
from tests.fixtures.ambient import load_devices, to_ambient_records
from tests.fixtures.archive import build_archive
from tests.fixtures.real import AMBIENT_END, load_real_slice
from tests.snapshot_helpers import (
    RAIN_LIVE,
    REGENERATE,
    STATION,
    build,
    dry_station,
    event_catalog,
    live_row,
    ms,
    raining_station,
    scenario_until,
)

GOLDEN_DIR = Path(__file__).parent / "golden" / "snapshots"


SCENARIOS = {"dry": dry_station, "raining": raining_station}


def assert_matches_golden(document: Dict, name: str) -> None:
    """
    Compare a document with `tests/golden/snapshots/<name>.json`.

    With `LOOKOUT_REGEN_GOLDEN=1` the file is rewritten instead.

    :param document: A snapshot document.
    :param name: The golden file's stem, `<scenario>.<kind>`.
    """
    path = GOLDEN_DIR / f"{name}.json"
    text = json.dumps(document, indent=2, allow_nan=False) + "\n"
    if REGENERATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert json.loads(text) == json.loads(path.read_text())


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_snapshot_documents_match_the_golden_files(scenario):
    snapshot = SCENARIOS[scenario]()

    assert_matches_golden(snapshot.dashboard(), f"{scenario}.dashboard")
    assert_matches_golden(snapshot.context(), f"{scenario}.context")


def test_golden_files_hold_no_real_station():
    texts = [p.read_text() for p in sorted(GOLDEN_DIR.glob("*.json"))]

    assert len(texts) == 2 * len(SCENARIOS)
    assert not any(DEFAULT_MAC_ADDRESS in text for text in texts)


# ---------- live stitching ----------


class TestLiveStitching:
    def test_live_values_replace_the_archived_ones(self):
        archive = load_real_slice()
        newest = archive.iloc[0]
        assert (newest["tempf"], newest["baromrelin"]) == (41.0, 29.705)

        current = dry_station().current

        # The recorded live reading's values, not the archive's newest row.
        assert current.temperature_f == 39.4
        assert current.feels_like_f == 39.4
        assert current.barometer_inhg == 29.673
        assert current.wind_speed_mph == 1.8
        assert current.wind_gust_mph == 2.5
        assert current.humidity_pct == 99.0
        # Built at local midnight from a 23:55 reading: yesterday's counter
        # is not today's rain (#122).
        assert current.rain_today_in is None
        assert current.last_rain_ms == ms("2025-11-17T17:03:00Z")

    def test_the_newest_observation_is_the_live_one(self):
        snapshot = dry_station()

        assert snapshot.observed_at_ms == ms("2025-11-19T07:55:00Z")
        assert snapshot.archive_newest_ms == ms("2025-11-19T04:55:00Z")
        assert snapshot.built_at_ms == ms(AMBIENT_END)

    def test_rain_today_is_the_live_total_while_the_archive_lags(self):
        full = scenario_until("2026-01-09T02:00:00Z")
        archive = full[full["dateutc"] < ms("2026-01-08T23:00:00Z")]
        live = live_row(full, "2026-01-09T01:00:00Z")
        assert live["dailyrainin"] > archive.iloc[0]["dailyrainin"]

        current = build(archive, live, "2026-01-09T01:01:00Z").current

        assert current.rain_today_in == live["dailyrainin"]
        assert current.rain_event_in == live["eventrainin"]

    def test_a_live_reading_already_archived_is_stitched_in_once(self):
        archive = load_real_slice()
        live = to_ambient_records(archive.head(1))[0]

        stitched = stitch_live(archive, live)

        assert len(stitched) == len(archive)
        assert stitched["dateutc"].is_unique

    def test_a_newer_live_reading_becomes_the_one_newest_row(self):
        archive = load_real_slice()
        live = load_devices()[0]["lastData"]

        stitched = stitch_live(archive, live)

        assert len(stitched) == len(archive) + 1
        assert stitched.iloc[0]["dateutc"] == live["dateutc"]
        assert stitched.iloc[0]["tempf"] == 39.4
        assert stitched["dateutc"].is_monotonic_decreasing

    def test_without_a_live_reading_the_archive_newest_row_is_current(self):
        archive = load_real_slice()

        snapshot = build(archive, None, AMBIENT_END)

        assert snapshot.current.temperature_f == 41.0
        assert snapshot.observed_at_ms == ms("2025-11-19T04:55:00Z")

    def test_the_temperature_trend_compares_live_with_a_day_before_it(self):
        archive = build_archive("2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z")
        archive["tempf"] = 50.0
        archive["baromrelin"] = 30.0
        # 24h before the live reading, not before the archive's newest row.
        archive.loc[archive["dateutc"] == ms("2026-01-02T03:00:00Z"), "tempf"] = 40.0
        live = {"dateutc": ms("2026-01-03T03:00:00Z"), "tempf": 50.0}

        current = build(archive, live, "2026-01-03T03:01:00Z").current

        assert current.temp_trend == "rising"
        assert current.barometer_trend is None  # live reported no barometer

    def test_the_barometer_trend_uses_three_hours_before_live(self):
        archive = build_archive("2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z")
        archive["baromrelin"] = 30.0
        archive.loc[archive["dateutc"] == ms("2026-01-02T22:00:00Z"), "baromrelin"] = (
            30.5
        )
        live = {"dateutc": ms("2026-01-03T01:00:00Z"), "baromrelin": 30.0}

        current = build(archive, live, "2026-01-03T01:01:00Z").current

        assert current.barometer_trend == "falling"

    def test_rain_corrections_are_not_applied_again(self):
        """
        `build_snapshot` reads the archive as handed: the refresh run applies
        the corrections once before building (#95), so the builder must not
        apply them again. Friday's daily counter here still holds Thursday's
        0.48" after the carryover gap, where a correction would make it 0;
        the snapshot keeps it.
        """
        archive = scenario_until("2026-01-09T12:00:00Z")
        assert archive.iloc[0]["dailyrainin"] == 0.48

        current = build(archive, None, "2026-01-09T12:00:00Z").current

        assert current.rain_today_in == 0.48


# ---------- the active rain event ----------


class TestActiveRainEvent:
    def test_an_ongoing_event_still_raining_live_is_active(self):
        event = raining_station().active_rain_event

        assert event.event_id == "event-3"
        assert event.start_ms == ms("2026-01-08T22:00:00Z")
        assert event.duration_minutes == 145.0
        # The live event total, not the catalog's 0.24" at the archive's end.
        assert event.total_in == 0.3
        assert event.avg_rate_in_hr == pytest.approx(0.3 / (145 / 60))

    def test_no_event_is_active_once_the_live_reading_is_dry(self):
        full = scenario_until("2026-01-09T00:30:00Z")
        archive = full[full["dateutc"] < ms("2026-01-09T00:00:00Z")]
        live = {**live_row(full, RAIN_LIVE), "eventrainin": 0.0}

        snapshot = build(archive, live, RAIN_LIVE, event_catalog(archive))

        assert snapshot.active_rain_event is None

    def test_a_completed_event_is_not_active_even_while_it_rains(self):
        archive = scenario_until("2026-01-11T08:00:00Z")
        live = {"dateutc": ms("2026-01-11T08:05:00Z"), "eventrainin": 0.05}

        snapshot = build(archive, live, "2026-01-11T08:06:00Z", event_catalog(archive))

        assert snapshot.active_rain_event is None

    def test_no_catalog_means_no_active_event(self):
        assert dry_station().active_rain_event is None
        full = scenario_until("2026-01-09T00:30:00Z")
        archive = full[full["dateutc"] < ms("2026-01-09T00:00:00Z")]

        assert build(
            archive, live_row(full, RAIN_LIVE), RAIN_LIVE
        ).active_rain_event is (None)


# ---------- the documents ----------


class TestDocuments:
    def test_both_documents_carry_the_schema_version_and_kind(self):
        snapshot = raining_station()

        assert snapshot.dashboard()["schema_version"] == SCHEMA_VERSION == 2
        assert snapshot.context()["schema_version"] == SCHEMA_VERSION
        assert snapshot.dashboard()["kind"] == "dashboard"
        assert snapshot.context()["kind"] == "context"

    def test_the_context_block_is_the_slice_of_the_dashboard_current_needs(self):
        dashboard = raining_station().dashboard()
        context = raining_station().context()

        assert set(context) == {"schema_version", "kind", *CONTEXT_FIELDS}
        assert set(CONTEXT_FIELDS) == {
            "station",
            "built_at_ms",
            "observed_at_ms",
            "current",
            "active_rain_event",
        }
        for name in CONTEXT_FIELDS:
            assert context[name] == dashboard[name]

    def test_documents_are_strict_json_with_null_for_unknown_values(self):
        archive = load_real_slice()
        live = {"dateutc": ms("2025-11-19T05:00:00Z"), "tempf": 40.0}

        dashboard = build(archive, live, AMBIENT_END).dashboard()

        json.dumps(dashboard, allow_nan=False)
        assert dashboard["current"]["solar_wm2"] is None
        assert dashboard["active_rain_event"] is None

    def test_now_must_carry_a_timezone(self):
        with pytest.raises(ValueError):
            build_snapshot(
                archive=load_real_slice(),
                live=None,
                event_catalog=None,
                energy_catalog=None,
                station=STATION,
                now=datetime(2025, 11, 19, 8, 0),
            )


# ---------- event curve files (#84) ----------

CURVE_DIR = GOLDEN_DIR / "events"


def scenario_curves(scenario: str) -> List[EventCurve]:
    """
    The completed events' curves of a golden scenario, as the refresh run
    writes them under `snapshot/<MAC>.events/`. The lane server serves these.
    """
    if scenario == "dry":
        archive = load_real_slice()
    else:
        full = scenario_until("2026-01-09T00:30:00Z")
        archive = full[full["dateutc"] < ms("2026-01-09T00:00:00Z")]
    return completed_curves(event_catalog(archive), archive)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_curve_files_match_the_golden_files(scenario):
    curves = scenario_curves(scenario)
    section = SCENARIOS[scenario]().rain_events
    still_open = section.open_curve.start_ms if section.open_curve else None
    listed = {
        row.start_ms
        for row in section.events
        if row.has_curve and row.start_ms != still_open
    }

    assert {curve.start_ms for curve in curves} == listed
    for curve in curves:
        assert_matches_golden(
            curve.document(STATION), f"events/{scenario}.{curve.start_ms}"
        )


def test_curve_golden_files_hold_no_real_station():
    texts = [p.read_text() for p in sorted(CURVE_DIR.glob("*.json"))]

    assert len(texts) == sum(len(scenario_curves(s)) for s in SCENARIOS)
    assert not any(DEFAULT_MAC_ADDRESS in text for text in texts)

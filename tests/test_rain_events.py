"""
Characterizes Rain Event detection and the Event Catalog merge/update.

Today's behaviour is pinned as it is, oddities included (see the NOTE comments).
Storage mechanics (paths, load/save/backup) are pinned in test_catalog_storage.
"""

import json

import pandas as pd
import pytest

from lookout.core.rain_events import (
    RainEventCatalog,
    detect_rain_events,
    newly_completed,
)
from tests.fixtures.archive import (
    FAKE_MAC,
    SCENARIO,
    RainSpell,
    build_archive,
    scenario_archive,
)
from lookout.storage import storj
from tests.fixtures.storage import bucket_keys
from tests.test_catchup import flaky_storj


def _scenario_until(end: str, newest_first: bool = True) -> pd.DataFrame:
    """The SCENARIO station, cut off before `end` (exclusive)."""
    return build_archive(
        SCENARIO.start,
        end,
        rain=SCENARIO.rain,
        gaps=SCENARIO.gaps,
        newest_first=newest_first,
    )


# Mid event 3 (rain 22:00Z-02:00Z): the station is still wet at the cut.
MID_EVENT_3 = "2026-01-09T00:00:00Z"
# Between events 2 and 3, long after event 2's counter reset.
BETWEEN_2_AND_3 = "2026-01-08T12:00:00Z"


def _summary(events) -> list:
    """(start, end, total_rainfall, duration_minutes) per event, as UTC strings."""
    return [
        (
            e["start_time"].strftime("%Y-%m-%dT%H:%M"),
            e["end_time"].strftime("%Y-%m-%dT%H:%M"),
            round(float(e["total_rainfall"]), 3),
            e["duration_minutes"],
        )
        for e in events
    ]


def _ms(iso: str) -> int:
    return int(pd.Timestamp(iso).timestamp() * 1000)


def _ids(catalog: pd.DataFrame) -> list:
    return list(catalog.sort_values("start_time")["event_id"])


@pytest.fixture
def catalog() -> RainEventCatalog:
    return RainEventCatalog(FAKE_MAC)


class TestDetectRainEvents:
    def test_scenario_yields_three_events(self):
        events = detect_rain_events(scenario_archive())

        # NOTE: an event ends at its last row with eventrainin > 0, which is the
        # row before the counter resets, so it includes the dry tail (about six
        # hours) after the last rain. Event 3's end is cut short by the gap that
        # starts at 07:00Z: the counter had not reset yet.
        assert _summary(events) == [
            ("2026-01-04T18:00", "2026-01-05T02:50", 0.36, 530.0),
            ("2026-01-07T04:00", "2026-01-07T19:50", 0.48, 950.0),
            ("2026-01-08T22:00", "2026-01-09T06:55", 0.48, 535.0),
        ]

    def test_two_spells_within_the_reset_window_are_one_event(self):
        events = detect_rain_events(scenario_archive())

        assert events[1]["total_rainfall"] == pytest.approx(0.48)  # 0.36 + 0.12

    def test_no_event_is_ongoing_when_the_archive_ends_dry(self):
        events = detect_rain_events(scenario_archive())

        # Spelled out as False on every event, so no reader sees a NaN.
        assert [e["ongoing"] for e in events] == [False, False, False]

    def test_result_does_not_depend_on_row_order(self):
        newest = _summary(detect_rain_events(scenario_archive(newest_first=True)))
        oldest = _summary(detect_rain_events(scenario_archive(newest_first=False)))

        assert newest == oldest

    def test_events_carry_ids_and_point_counts(self):
        events = detect_rain_events(scenario_archive())

        assert len({e["event_id"] for e in events}) == 3
        assert [e["data_point_count"] for e in events] == [107, 191, 108]

    def test_archive_ending_mid_event_marks_the_last_event_ongoing(self):
        events = detect_rain_events(_scenario_until(MID_EVENT_3))

        assert len(events) == 3
        assert bool(events[-1]["ongoing"]) is True
        assert events[-1]["end_time"] == pd.Timestamp("2026-01-08T23:55:00Z")
        assert events[-1]["total_rainfall"] == pytest.approx(0.24)
        assert [e.get("ongoing", False) for e in events[:-1]] == [False, False]

    def test_event_below_the_minimum_rainfall_is_dropped(self):
        # One row of 0.005" is shorter than 15 minutes and under 0.01".
        df = build_archive(
            "2026-01-04T00:00:00Z",
            "2026-01-04T12:00:00Z",
            rain=(RainSpell("2026-01-04T03:00:00Z", 1 / 12, 0.06),),
        )

        assert detect_rain_events(df) == []

    def test_one_row_event_is_kept_by_default_and_dropped_by_a_higher_floor(self):
        df = build_archive(
            "2026-01-04T00:00:00Z",
            "2026-01-04T12:00:00Z",
            rain=(RainSpell("2026-01-04T03:00:00Z", 1 / 12, 1.2),),
        )

        # NOTE: the event runs to the last wet counter row, so the dry tail
        # counts toward duration; the 15 minute floor is rarely what filters.
        events = detect_rain_events(df, min_event_duration_min=10_000)
        assert events == []
        assert len(detect_rain_events(df)) == 1

    def test_empty_archive_and_missing_column_give_no_events(self):
        assert detect_rain_events(pd.DataFrame()) == []
        assert detect_rain_events(scenario_archive().drop(columns="eventrainin")) == []

    def test_a_dry_archive_gives_no_events(self):
        df = build_archive("2026-01-04T00:00:00Z", "2026-01-05T00:00:00Z")

        assert detect_rain_events(df) == []


class TestDetectAndCatalogEvents:
    def test_builds_the_catalog_with_quality_and_metadata(self, catalog):
        result = catalog.detect_and_catalog_events(scenario_archive(), auto_save=False)

        assert len(result) == 3
        assert set(result["catalog_version"]) == {"1.0"}
        assert {"quality_rating", "data_completeness", "flags", "updated_at"} <= set(
            result.columns
        )
        assert list(result["total_rainfall"].round(3)) == [0.36, 0.48, 0.48]

    def test_no_events_gives_an_empty_frame(self, catalog):
        dry = build_archive("2026-01-04T00:00:00Z", "2026-01-05T00:00:00Z")

        assert catalog.detect_and_catalog_events(dry, auto_save=False).empty

    def test_the_ongoing_flag_reaches_the_catalog_flags(self, catalog):
        result = catalog.detect_and_catalog_events(
            _scenario_until(MID_EVENT_3), auto_save=False
        )

        last = result.sort_values("start_time").iloc[-1]
        assert bool(last["flags"]["ongoing"]) is True

    def test_auto_save_writes_a_catalog_that_loads_back(self, catalog, fake_s3):
        result = catalog.detect_and_catalog_events(
            scenario_archive(), backup_existing=False
        )

        assert f"{FAKE_MAC}.event_catalog.parquet" in bucket_keys(fake_s3)
        assert _ids(catalog.load_catalog()) == _ids(result)


class TestUpdateCatalogWithNewData:
    def test_empty_catalog_triggers_full_detection(self, catalog):
        result = catalog.update_catalog_with_new_data(
            scenario_archive(), existing_catalog=pd.DataFrame()
        )

        assert len(result) == 3

    def test_adds_the_events_after_the_last_one_and_keeps_the_earlier_ids(
        self, catalog
    ):
        stored = catalog.detect_and_catalog_events(
            _scenario_until(BETWEEN_2_AND_3), auto_save=False
        )
        assert len(stored) == 2

        updated = catalog.update_catalog_with_new_data(scenario_archive(), stored)

        assert len(updated) == 3
        assert _ids(updated)[0] == _ids(stored)[0]
        assert list(updated.sort_values("start_time")["total_rainfall"].round(3)) == [
            0.36,
            0.48,
            0.48,
        ]

    def test_a_completed_last_event_keeps_its_id(self, catalog):
        # Fixed in #72: detection used to leave `ongoing` NaN on completed
        # events, NaN read as ongoing, and the last event was re-detected with
        # a new id on every update.
        stored = catalog.detect_and_catalog_events(
            _scenario_until(BETWEEN_2_AND_3), auto_save=False
        )

        updated = catalog.update_catalog_with_new_data(scenario_archive(), stored)

        assert _ids(updated)[:2] == _ids(stored)
        assert len(updated) == 3

    def test_a_stored_catalog_with_nan_ongoing_reads_as_completed(self, catalog):
        # Catalogs saved before #72 carry NaN; it must not mean ongoing.
        stored = catalog.detect_and_catalog_events(
            _scenario_until(BETWEEN_2_AND_3), auto_save=False
        )
        stored["ongoing"] = float("nan")
        stored["flags"] = [{"ongoing": None}] * len(stored)

        updated = catalog.update_catalog_with_new_data(scenario_archive(), stored)

        assert _ids(updated)[:2] == _ids(stored)

    def test_ongoing_last_event_is_replaced_by_the_completed_version(self, catalog):
        stored = catalog.detect_and_catalog_events(
            _scenario_until(MID_EVENT_3), auto_save=False
        )
        old_last = stored.sort_values("start_time").iloc[-1]
        assert bool(old_last["flags"]["ongoing"]) is True

        updated = catalog.update_catalog_with_new_data(scenario_archive(), stored)

        assert len(updated) == 3
        new_last = updated.sort_values("start_time").iloc[-1]
        assert new_last["start_time"] == old_last["start_time"]
        assert new_last["event_id"] != old_last["event_id"]  # re-detected
        assert new_last["total_rainfall"] == pytest.approx(0.48)
        assert new_last["end_time"] == pd.Timestamp("2026-01-09T06:55:00Z")
        assert bool(new_last["flags"]["ongoing"]) is False
        assert _ids(updated)[:2] == _ids(stored)[:2]

    def test_ongoing_event_that_is_still_wet_stays_ongoing_and_grows(self, catalog):
        stored = catalog.detect_and_catalog_events(
            _scenario_until("2026-01-08T23:00:00Z"), auto_save=False
        )

        updated = catalog.update_catalog_with_new_data(
            _scenario_until("2026-01-09T01:00:00Z"), stored
        )

        assert len(updated) == 3
        last = updated.sort_values("start_time").iloc[-1]
        assert bool(last["flags"]["ongoing"]) is True
        assert (
            last["total_rainfall"]
            > stored.sort_values("start_time").iloc[-1]["total_rainfall"]
        )

    def test_ongoing_flag_is_also_read_from_json_string_flags(self, catalog):
        stored = catalog.detect_and_catalog_events(
            _scenario_until(MID_EVENT_3), auto_save=False
        )
        stored["flags"] = stored["flags"].map(json.dumps)

        updated = catalog.update_catalog_with_new_data(scenario_archive(), stored)

        assert len(updated) == 3
        assert not updated.sort_values("start_time").iloc[-1]["flags"]["ongoing"]

    def test_no_new_rain_leaves_the_events_in_place(self, catalog):
        stored = catalog.detect_and_catalog_events(scenario_archive(), auto_save=False)

        updated = catalog.update_catalog_with_new_data(scenario_archive(), stored)

        assert len(updated) == 3
        assert list(updated["start_time"]) == list(stored["start_time"])
        assert list(updated["total_rainfall"].round(3)) == [0.36, 0.48, 0.48]
        # NOTE: the last event is re-detected (see the NaN note above).
        assert _ids(updated)[:2] == _ids(stored)[:2]

    def test_archive_with_nothing_after_the_last_event_changes_nothing(self, catalog):
        stored = catalog.detect_and_catalog_events(scenario_archive(), auto_save=False)
        cut = scenario_archive()
        cut = cut[cut["dateutc"] < _ms("2026-01-09T06:00:00Z")]

        updated = catalog.update_catalog_with_new_data(cut, stored)

        assert len(updated) == 3
        assert _ids(updated)[:2] == _ids(stored)[:2]

    def test_loads_the_stored_catalog_when_none_is_passed(self, catalog, fake_s3):
        catalog.detect_and_catalog_events(
            _scenario_until(BETWEEN_2_AND_3), backup_existing=False
        )

        updated = catalog.update_catalog_with_new_data(scenario_archive())

        assert len(updated) == 3


CATALOG_KEY = f"{FAKE_MAC}.event_catalog.parquet"


class TestSaveIfEventsCompleted:
    """
    The refresh run's Event Catalog save rule: save when the updated catalog
    holds a completed event the stored one lacks, matched by start.
    """

    @pytest.fixture
    def ongoing(self, catalog):
        """Events 1 and 2 completed, event 3 ongoing."""
        return catalog.detect_and_catalog_events(
            _scenario_until(MID_EVENT_3), auto_save=False
        )

    @pytest.fixture
    def completed(self, catalog):
        """All three events completed."""
        return catalog.detect_and_catalog_events(scenario_archive(), auto_save=False)

    def test_saves_when_the_ongoing_event_completes(
        self, catalog, ongoing, completed, fake_s3
    ):
        assert catalog.save_if_events_completed(ongoing, completed) is True

        assert CATALOG_KEY in bucket_keys(fake_s3)

    def test_the_updated_catalog_from_a_real_update_triggers_the_save(
        self, catalog, ongoing, fake_s3
    ):
        # Fixed in #72: the re-detected completed event used to carry `ongoing`
        # NaN, which read as ongoing, so this hand-off never fired.
        updated = catalog.update_catalog_with_new_data(scenario_archive(), ongoing)

        assert catalog.save_if_events_completed(ongoing, updated) is True
        assert CATALOG_KEY in bucket_keys(fake_s3)

    def test_saves_a_new_completed_event_after_a_completed_last_event(
        self, catalog, completed, fake_s3
    ):
        # The steady state: the stored last event completed at the last save,
        # and event 3 started and completed since.
        stored = catalog.detect_and_catalog_events(
            _scenario_until(BETWEEN_2_AND_3), auto_save=False
        )
        assert len(stored) == 2

        assert catalog.save_if_events_completed(stored, completed) is True
        assert CATALOG_KEY in bucket_keys(fake_s3)

    def test_does_not_save_while_the_event_is_still_ongoing(
        self, catalog, ongoing, fake_s3
    ):
        assert catalog.save_if_events_completed(ongoing, ongoing) is False

        assert bucket_keys(fake_s3) == []

    def test_does_not_save_when_nothing_new_completed(
        self, catalog, completed, fake_s3
    ):
        assert catalog.save_if_events_completed(completed, completed) is False

        assert bucket_keys(fake_s3) == []

    def test_matches_events_by_start_not_by_event_id(self, catalog, completed, fake_s3):
        redetected = completed.copy()
        redetected["event_id"] = [f"new-{i}" for i in range(len(redetected))]

        assert catalog.save_if_events_completed(completed, redetected) is False
        assert bucket_keys(fake_s3) == []

    def test_with_none_stored_the_first_completed_event_seeds_it(
        self, catalog, completed, fake_s3
    ):
        assert catalog.save_if_events_completed(None, completed) is True

        assert CATALOG_KEY in bucket_keys(fake_s3)

    def test_with_none_stored_an_ongoing_event_alone_seeds_nothing(
        self, catalog, fake_s3
    ):
        wet = catalog.detect_and_catalog_events(
            _scenario_until("2026-01-04T19:00:00Z"), auto_save=False
        )
        assert len(wet) == 1

        assert catalog.save_if_events_completed(pd.DataFrame(), wet) is False
        assert catalog.save_if_events_completed(None, wet) is False
        assert bucket_keys(fake_s3) == []

    def test_does_not_save_an_empty_update(self, catalog, completed, fake_s3):
        assert catalog.save_if_events_completed(completed, pd.DataFrame()) is False

        assert bucket_keys(fake_s3) == []

    def test_backs_up_the_stored_catalog_before_saving(
        self, catalog, ongoing, completed, fake_s3
    ):
        catalog.save_catalog(ongoing)

        catalog.save_if_events_completed(ongoing, completed)

        backups = [
            key
            for key in bucket_keys(fake_s3)
            if key.startswith(f"backups/{FAKE_MAC}.event_catalog")
        ]
        assert len(backups) == 1

    def test_a_failed_backup_leaves_the_stored_catalog_alone(
        self, catalog, ongoing, completed, fake_s3, monkeypatch
    ):
        catalog.save_catalog(ongoing)
        before = fake_s3.get_object(Bucket=catalog.bucket, Key=CATALOG_KEY)[
            "Body"
        ].read()
        flaky_storj(monkeypatch, fake_s3, copy=True)

        with pytest.raises(storj.BackupError):
            catalog.save_if_events_completed(ongoing, completed)

        after = fake_s3.get_object(Bucket=catalog.bucket, Key=CATALOG_KEY)["Body"]
        assert after.read() == before


class TestNewlyCompleted:
    def test_names_the_completed_events_the_stored_catalog_lacks(self, catalog):
        stored = catalog.detect_and_catalog_events(
            _scenario_until(MID_EVENT_3), auto_save=False
        )
        updated = catalog.update_catalog_with_new_data(scenario_archive(), stored)

        new = newly_completed(stored, updated)

        assert list(new["start_time"]) == [pd.Timestamp("2026-01-08T22:00:00Z")]

    def test_with_nothing_stored_every_completed_event_is_new(self, catalog):
        updated = catalog.detect_and_catalog_events(
            _scenario_until(MID_EVENT_3), auto_save=False
        )

        new = newly_completed(None, updated)

        assert list(new["start_time"]) == [
            pd.Timestamp("2026-01-04T18:00:00Z"),
            pd.Timestamp("2026-01-07T04:00:00Z"),
        ]

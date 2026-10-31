"""
Characterize how the AWN controller merges history into the archive.

Pins what `lookout.api.awn_controller` does today with data in and data out:
combine, dedup and sort on `dateutc`; paging forward from the newest archive
row; gap filling and its outcome (filled, no new data, refused). Where the
behaviour looks wrong
the test pins it as it is and says so, so the phase-1 refactor changes it on
purpose or not at all.

Archives come from `tests/fixtures/archive.py`, Ambient pages from
`AmbientReplay`, and the rain corrections
catalog lives in the moto bucket.
"""

from typing import Sequence

import pandas as pd
import pytest

from lookout.api import awn_controller as awn
from lookout.core import rain_corrections
from tests.fixtures.ambient import devices_body, to_ambient_records
from tests.fixtures.archive import (
    FAKE_MAC,
    SCENARIO,
    Gap,
    build_archive,
    scenario_archive,
)

COUNTERS = ["dailyrainin", "weeklyrainin", "monthlyrainin", "yearlyrainin"]
CARRY_GAP = SCENARIO.gaps[0]
PLAIN_GAP = SCENARIO.gaps[1]
# The carryover gap's rows are missing but the counters reset properly.
NO_CARRY_GAP = Gap(CARRY_GAP.start, CARRY_GAP.end)


def ms(when: str) -> int:
    """Epoch ms for an ISO-8601 UTC time."""
    return int(pd.Timestamp(when).value // 10**6)


def week(gaps: Sequence[Gap] = SCENARIO.gaps) -> pd.DataFrame:
    """The SCENARIO week with the given gaps, newest first."""
    return build_archive(SCENARIO.start, SCENARIO.end, rain=SCENARIO.rain, gaps=gaps)


@pytest.fixture
def device() -> dict:
    """The fixture station as the controller receives it, `lastData.tz` set."""
    return devices_body({"dateutc": 1})[0]


def serve(ambient_replay, df: pd.DataFrame) -> None:
    """Make the fake Ambient API serve `df` as its history."""
    ambient_replay.history = sorted(
        to_ambient_records(df), key=lambda r: r["dateutc"], reverse=True
    )


class TestCombine:
    """`combine_df` / `combine_full_history`: dedup and sort on `dateutc`."""

    def test_overlapping_slices_combine_to_the_full_archive(self):
        full = week()
        older = full[full.dateutc <= ms("2026-01-08T00:00:00Z")]
        newer = full[full.dateutc >= ms("2026-01-07T00:00:00Z")]

        combined = awn.combine_df(newer, older)

        pd.testing.assert_frame_equal(combined, full)

    def test_sorts_newest_first_whatever_the_input_order(self):
        full = week()
        shuffled = full.sample(frac=1, random_state=3)
        ascending = full.sort_values("dateutc")

        for frame in (shuffled, ascending):
            combined = awn.combine_df(frame, full.iloc[0:0])
            assert combined["dateutc"].is_monotonic_decreasing
            assert len(combined) == len(full)

    def test_a_duplicate_timestamp_leaves_one_row(self):
        """
        Held finding: which duplicate survives is not defined.

        `normalize` sorts with an unstable sort before `keep="last"`, so the
        winner between equal timestamps can differ by platform. Only the
        dedup is pinned.
        """
        old = week().head(3).copy()
        new = old.copy()
        new["tempf"] = 99.0

        combined = awn.combine_df(old, new)

        assert len(combined) == 3
        assert combined["dateutc"].is_unique
        assert set(combined["tempf"]) <= set(old["tempf"]) | {99.0}

    def test_drops_rows_with_invalid_dateutc(self):
        good = week().head(2)
        junk = good.copy()
        junk["dateutc"] = [12345, -1]  # below the post-2020 floor
        missing = good.head(1).copy()
        missing["dateutc"] = float("nan")

        combined = awn.combine_df(good, pd.concat([junk, missing]))

        assert len(combined) == 2
        assert set(combined["dateutc"]) == set(good["dateutc"])

    def test_full_history_without_interim_returns_the_archive_untouched(self):
        archive = week().head(10).sample(frac=1, random_state=1)

        for interim in (None, archive.iloc[0:0]):
            result = awn.combine_full_history(archive, interim)
            # Not re-sorted or deduped: the archive comes back as it went in.
            pd.testing.assert_frame_equal(result, archive)

    def test_full_history_adds_only_the_interim_rows_it_lacks(self):
        full = week()
        archive = full[full.dateutc < ms("2026-01-09T00:00:00Z")]
        interim = full[full.dateutc >= ms("2026-01-08T20:00:00Z")]

        combined = awn.combine_full_history(archive, interim)

        pd.testing.assert_frame_equal(combined, full)
        assert combined["dateutc"].is_unique


class TestSeekOverTimeGap:
    """`seek_over_time_gap`: how a page that does not advance moves the cursor."""

    def test_each_attempt_jumps_five_minutes_per_record_per_attempt(self):
        start = pd.Timestamp("2026-01-08T00:00:00Z").to_pydatetime()

        first, attempts, stop = awn.seek_over_time_gap(start, 0, limit=250)
        second, attempts2, stop2 = awn.seek_over_time_gap(start, 1, limit=250)

        assert (attempts, stop) == (1, False)
        assert first - start == pd.Timedelta(minutes=5 * 250 * 1)
        assert (attempts2, stop2) == (2, False)
        assert second - start == pd.Timedelta(minutes=5 * 250 * 2)

    def test_gives_up_on_the_tenth_attempt_without_moving(self):
        start = pd.Timestamp("2026-01-08T00:00:00Z").to_pydatetime()

        cursor, attempts, stop = awn.seek_over_time_gap(start, 9, limit=250)

        assert (cursor, attempts, stop) == (start, 10, True)


class TestHistorySinceLastArchive:
    """`get_history_since_last_archive`: page forward and merge into the archive."""

    def test_invalid_archive_comes_back_unchanged_with_no_requests(
        self, fake_s3, ambient_replay, device
    ):
        empty = pd.DataFrame()
        no_dateutc = pd.DataFrame({"tempf": [1.0]})

        assert awn.get_history_since_last_archive(device, empty) is empty
        assert awn.get_history_since_last_archive(device, no_dateutc) is no_dateutc
        assert ambient_replay.calls == []

    def test_new_pages_extend_the_archive_without_duplicates(
        self, fake_s3, ambient_replay, device
    ):
        full = week(gaps=())
        archive = full[full.dateutc < ms("2026-01-08T00:00:00Z")]
        serve(ambient_replay, full)

        result = awn.get_history_since_last_archive(
            device, archive, end_date=pd.Timestamp("2026-01-08T03:00:00Z")
        )

        assert result["dateutc"].is_unique
        assert result["dateutc"].is_monotonic_decreasing
        # One page of 250 records, four of which the archive already had.
        assert len(result) == len(archive) + 246
        # Every row the archive had is still there, unchanged.
        kept = result[result.dateutc < ms("2026-01-08T00:00:00Z")]
        pd.testing.assert_frame_equal(
            kept.reset_index(drop=True)[archive.columns],
            archive.reset_index(drop=True),
            check_dtype=False,
        )
        # The rows it added carry the same values the station reported.
        new = result[result.dateutc >= ms("2026-01-08T00:00:00Z")]
        expected = full.set_index("dateutc").loc[new.dateutc]
        assert (new.set_index("dateutc")[COUNTERS] == expected[COUNTERS]).all().all()

    def test_end_date_caps_the_cursor_not_the_rows_returned(
        self, fake_s3, ambient_replay, device
    ):
        """
        Held finding: rows past `end_date` are kept.

        The page request ends at the cursor plus `(limit - 3) * 5` minutes, not
        at `end_date`, so a short `end_date` still brings in later rows.
        """
        full = week(gaps=())
        archive = full[full.dateutc < ms("2026-01-08T00:00:00Z")]
        serve(ambient_replay, full)
        end_date = pd.Timestamp("2026-01-08T03:00:00Z")

        result = awn.get_history_since_last_archive(device, archive, end_date=end_date)

        # The page ends at cursor + 1235 minutes, 2026-01-08T20:25Z at the last row.
        assert result["dateutc"].max() == ms("2026-01-08T20:25:00Z")

    def test_archive_already_at_end_date_merges_one_page_ending_now(
        self, fake_s3, ambient_replay, device
    ):
        full = week(gaps=())
        archive = full[full.dateutc < ms("2026-01-11T07:00:00Z")]
        serve(ambient_replay, full)

        result = awn.get_history_since_last_archive(
            device, archive, end_date=pd.Timestamp("2026-01-10T00:00:00Z")
        )

        pd.testing.assert_frame_equal(
            result[full.columns].reset_index(drop=True), full, check_dtype=False
        )

    def test_nothing_newer_returns_the_archive_rows(
        self, fake_s3, ambient_replay, device
    ):
        archive = week(gaps=())
        serve(ambient_replay, archive)

        result = awn.get_history_since_last_archive(
            device, archive, end_date=pd.Timestamp("2026-01-12T00:00:00Z")
        )

        pd.testing.assert_frame_equal(
            result[archive.columns].reset_index(drop=True), archive, check_dtype=False
        )

    def test_known_corrections_are_not_applied_to_what_it_returns(
        self, fake_s3, ambient_replay, device
    ):
        """
        The combined archive comes back raw, catalog or not (#95).

        Rows the archive had come back as handed; rows fetched come back as
        the station reported them. The carryover day keeps its raw counters.
        """
        carried = scenario_archive()
        rain_corrections.save_catalog(
            {"corrections": rain_corrections.detect_carryovers(carried)}, FAKE_MAC
        )
        archive = carried[carried.dateutc < ms("2026-01-09T12:00:00Z")]
        serve(ambient_replay, carried)

        result = awn.get_history_since_last_archive(
            device, archive, end_date=pd.Timestamp("2026-01-10T12:00:00Z")
        )

        merged = result.merge(carried, on="dateutc", suffixes=("", "_raw"))
        assert len(merged) == len(result)
        assert merged["dateutc"].min() < ms("2026-01-09T12:00:00Z")
        assert merged["dateutc"].max() >= ms("2026-01-10T00:00:00Z")
        for col in COUNTERS:
            assert (merged[col] - merged[col + "_raw"]).abs().max() < 1e-9


class TestFillArchiveGap:
    """`fill_archive_gap`: fetch a missing stretch and report how it went."""

    def test_fills_a_gap_from_the_station_history(
        self, fake_s3, ambient_replay, device
    ):
        archive = week()
        truth = week(gaps=(CARRY_GAP,))
        serve(ambient_replay, truth)

        result = awn.fill_archive_gap(device, archive, PLAIN_GAP.start, PLAIN_GAP.end)

        assert result.status == "filled"
        assert not result.skipped
        assert result.rows_added == len(truth) - len(archive)
        filled = result.history_df
        assert len(filled) == len(truth)
        assert filled["dateutc"].is_unique
        assert filled["dateutc"].is_monotonic_decreasing
        assert set(filled["dateutc"]) == set(truth["dateutc"])

    def test_fill_leaves_a_raw_archive_raw(self, fake_s3, ambient_replay, device):
        """
        A fill applies no rain corrections (#95), so a raw archive saves raw.

        Neither the archive rows nor the fetched page (Saturday 03:15 local
        to the end of the week) lose the carryover in the catalog.
        """
        carried = scenario_archive()
        rain_corrections.save_catalog(
            {"corrections": rain_corrections.detect_carryovers(carried)}, FAKE_MAC
        )
        truth = week(gaps=(CARRY_GAP,))
        serve(ambient_replay, truth)

        filled = awn.fill_archive_gap(
            device, carried, PLAIN_GAP.start, PLAIN_GAP.end
        ).history_df

        merged = filled.merge(truth, on="dateutc", suffixes=("", "_raw"))
        assert len(merged) == len(filled) == len(truth)
        for col in COUNTERS:
            assert (merged[col] - merged[col + "_raw"]).abs().max() < 1e-9
    def test_a_gap_with_no_new_data_comes_back_skipped(
        self, fake_s3, ambient_replay, device
    ):
        archive = week()
        ambient_replay.history = []

        result = awn.fill_archive_gap(device, archive, PLAIN_GAP.start, PLAIN_GAP.end)

        assert result.history_df is archive
        assert result.status == "no_new_data"
        assert result.skipped
        assert result.rows_added == 0
        assert result.start == pd.Timestamp(PLAIN_GAP.start, tz="UTC")
        assert result.end == pd.Timestamp(PLAIN_GAP.end, tz="UTC")

    def test_an_inverted_gap_is_refused_without_a_request(
        self, fake_s3, ambient_replay, device
    ):
        archive = week()

        result = awn.fill_archive_gap(device, archive, PLAIN_GAP.end, PLAIN_GAP.start)

        assert result.history_df is archive
        assert result.status == "refused"
        assert not result.skipped
        assert ambient_replay.calls == []

    def test_a_gap_before_all_archive_data_is_refused(
        self, fake_s3, ambient_replay, device
    ):
        archive = week()

        result = awn.fill_archive_gap(
            device, archive, "2025-12-01T00:00:00Z", "2025-12-01T04:00:00Z"
        )

        assert result.history_df is archive
        assert result.status == "refused"
        assert ambient_replay.calls == []

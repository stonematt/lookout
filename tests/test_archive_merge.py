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

    def test_every_empty_probe_moves_the_cursor_one_page(self):
        """
        Probes 1, 2 and 5 move the cursor by the same 1250 minutes (#132).

        It used to move 5 minutes x records x attempt number from the last
        cursor, so the offsets accumulated and skipped time.
        """
        start = pd.Timestamp("2026-01-08T00:00:00Z").to_pydatetime()

        for done, attempt in ((0, 1), (1, 2), (4, 5)):
            cursor, attempts, stop = awn.seek_over_time_gap(start, done, limit=250)

            assert (attempts, stop) == (attempt, False)
            assert cursor - start == pd.Timedelta(minutes=1250)

    def test_the_step_is_one_page_at_any_limit(self):
        start = pd.Timestamp("2026-01-08T00:00:00Z").to_pydatetime()

        cursor, _, _ = awn.seek_over_time_gap(start, 3, limit=288)

        assert cursor - start == pd.Timedelta(hours=24)

    def test_gives_up_on_the_tenth_attempt_without_moving(self):
        start = pd.Timestamp("2026-01-08T00:00:00Z").to_pydatetime()

        cursor, attempts, stop = awn.seek_over_time_gap(start, 9, limit=250)

        assert (cursor, attempts, stop) == (start, 10, True)


PAGE = pd.Timedelta(minutes=5 * 250)
OUTAGE_START = pd.Timestamp("2026-01-05T00:00:00Z")
ARCHIVE_START = "2026-01-04T00:00:00Z"


def outage(hours: float, after_days: float = 4) -> tuple:
    """
    An archive that ends where an outage of `hours` begins, and the truth.

    :param hours: How long the station sent nothing.
    :param after_days: Days of readings Ambient holds after the outage.
    :return: `(archive, truth, end)`: the archive (rows before the outage),
        everything Ambient holds, and the time after its last reading.
    """
    resume = OUTAGE_START + pd.Timedelta(hours=hours)
    end = resume + pd.Timedelta(days=after_days)
    truth = build_archive(ARCHIVE_START, end, gaps=(Gap(OUTAGE_START, resume),))
    archive = build_archive(ARCHIVE_START, OUTAGE_START)
    return archive, truth, end


def windows(ambient_replay, limit: int = 250) -> list:
    """
    The span of readings each history request asked for, in request order.

    A request ending at `endDate` for `limit` five-minute readings covers
    `[endDate - limit * 5 min, endDate)`.

    :return: `(start, end)` Timestamps, one pair per history request.
    """
    spans = []
    for path, params in ambient_replay.calls:
        if path == "/devices" or "endDate" not in params:
            continue
        end = pd.Timestamp(int(params["endDate"]), unit="ms", tz="UTC")
        spans.append((end - pd.Timedelta(minutes=5 * limit), end))
    return spans


class TestGapSeekAcrossAnOutage:
    """The forward walk across an outage requests every stretch of time (#132)."""

    def test_windows_of_consecutive_empty_probes_touch(
        self, fake_s3, ambient_replay, device
    ):
        archive, truth, end = outage(hours=150)
        serve(ambient_replay, truth)

        awn.get_history_since_last_archive(device, archive, pages=20, end_date=end)

        spans = windows(ambient_replay)
        assert len(spans) >= 6
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
            assert next_start <= prev_end, "unrequested time between windows"
            assert prev_end - next_start <= PAGE, "more than a page of overlap"

    @pytest.mark.parametrize("hours", [42, 72, 180])
    def test_every_reading_after_the_outage_is_fetched(
        self, fake_s3, ambient_replay, device, hours
    ):
        """
        No reading Ambient holds after the outage is skipped.

        At 42 hours and beyond the old offset left a stretch unrequested
        (244 rows lost at 42 h). 180 h is near the walk's reach.
        """
        archive, truth, end = outage(hours)
        serve(ambient_replay, truth)
        resume = OUTAGE_START + pd.Timedelta(hours=hours)

        result = awn.get_history_since_last_archive(
            device, archive, pages=20, end_date=end
        )

        newest = result["dateutc"].max()
        assert newest >= ms(str(resume + pd.Timedelta(days=1)))
        held = truth[truth.dateutc <= newest]
        assert set(result["dateutc"]) == set(held["dateutc"])

    def test_an_outage_of_41_hours_makes_the_requests_it_always_made(
        self, fake_s3, ambient_replay, device
    ):
        """
        One seek step crosses 41 hours, and the first step is unchanged.

        The cursor starts at the last archive row, 2026-01-04T23:55Z; each
        request ends 1235 minutes after its cursor. The first page brings
        back the archive's own newest rows (counted as progress against an
        empty interim), the second the same rows (not fresh), then the seek.
        """
        archive, truth, end = outage(hours=41)
        serve(ambient_replay, truth)

        result = awn.get_history_since_last_archive(
            device, archive, pages=4, end_date=end
        )

        ends = [str(e) for _, e in windows(ambient_replay)]
        assert ends == [
            "2026-01-05 20:30:00+00:00",  # archive rows only
            "2026-01-05 20:35:00+00:00",  # from 23:55 + 5 min: nothing new
            "2026-01-06 17:25:00+00:00",  # one page on; back at 17:00
            "2026-01-07 14:00:00+00:00",  # paging forward from 17:20 + 5 min
        ]
        newest = result["dateutc"].max()
        assert newest == ms("2026-01-07T13:55:00Z")
        held = truth[truth.dateutc <= newest]
        assert set(result["dateutc"]) == set(held["dateutc"])

    def test_an_outage_past_the_walks_reach_gives_up_on_the_tenth_probe(
        self, fake_s3, ambient_replay, device
    ):
        """
        Held finding: ten one-page probes reach about 208 hours (#132).

        A 240-hour outage is past that, so the walk ends on the cap with
        nothing fetched, and the next run starts from the same archive row.
        The windows it did request still touch, so nothing is skipped.
        The first request returns the archive's own newest rows; the ten
        after it are the empty probes.
        """
        archive, truth, end = outage(hours=240)
        serve(ambient_replay, truth)

        result = awn.get_history_since_last_archive(
            device, archive, pages=20, end_date=end
        )

        spans = windows(ambient_replay)
        assert len(spans) == 1 + 10
        assert spans[-1][1] - OUTAGE_START < pd.Timedelta(hours=240)
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
            assert next_start <= prev_end
        assert len(result) == len(archive)

    def test_the_seek_never_probes_from_past_the_end_bound(
        self, fake_s3, ambient_replay, device
    ):
        """
        Every probe starts before `end_date`; the walk stops at the bound.

        A window can still run past the bound by up to a page: that is the
        page width (see `test_end_date_caps_the_cursor_not_the_rows_returned`).
        """
        resume_never = OUTAGE_START + pd.Timedelta(hours=6)
        truth = build_archive(ARCHIVE_START, resume_never)
        archive = build_archive(ARCHIVE_START, OUTAGE_START)
        serve(ambient_replay, truth)
        bound = OUTAGE_START + pd.Timedelta(hours=100)

        awn.get_history_since_last_archive(device, archive, pages=20, end_date=bound)

        cursors = [e - pd.Timedelta(minutes=1235) for _, e in windows(ambient_replay)]
        assert len(cursors) >= 5
        assert all(c < bound for c in cursors)

    def test_the_seek_never_requests_past_the_present(
        self, fake_s3, ambient_replay, device
    ):
        now = pd.Timestamp.now(tz="UTC").floor("5min")
        archive = build_archive(now - pd.Timedelta(days=2), now - pd.Timedelta(hours=30))
        serve(ambient_replay, archive)

        awn.get_history_since_last_archive(device, archive, pages=20)

        latest = pd.Timestamp.now(tz="UTC")
        spans = windows(ambient_replay)
        assert len(spans) >= 2
        assert all(e <= latest for _, e in spans)


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

    def test_an_interior_fill_drops_no_archive_row_and_warns_nothing(
        self, fake_s3, ambient_replay, device, caplog
    ):
        """
        The slice before the gap keeps `dateutc` as int ms.

        It used to be sliced from a copy converted to Timestamps, so the
        merge inside `get_history_since_last_archive` saw a mixed column
        and `normalize` dropped every archive row ("dropped N of M").
        """
        archive = week()
        serve(ambient_replay, week(gaps=(CARRY_GAP,)))
        seen = []
        real = awn.get_history_since_last_archive

        def spy(device, archive_df, **kwargs):
            seen.append(archive_df["dateutc"].dtype)
            result = real(device, archive_df, **kwargs)
            seen.append(len(result) >= len(archive_df))
            return result

        with caplog.at_level("WARNING", logger="lookout.utils.dateutc"):
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(awn, "get_history_since_last_archive", spy)
                result = awn.fill_archive_gap(
                    device, archive, PLAIN_GAP.start, PLAIN_GAP.end
                )

        assert result.status == "filled"
        assert seen == ["int64", True]
        assert not [r for r in caplog.records if "dateutc.normalize" in r.message]

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

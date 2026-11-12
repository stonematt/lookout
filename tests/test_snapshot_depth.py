"""
Full history and the record depth (#463), through `build_snapshot`.

A comparison is shown once the record covers its first day: the archive's
first station-local day is on or before it. One that is shown still reports
its gaps by its gap status; one that is not is hidden, and the snapshot names
the comparison that unlocks next and its date.

The young station is Salmon Run's shape: a record from Mon 2026-11-03,
built Wed 2026-11-11 at noon local, days only and no earlier year. The
seasoned golden's record begins 2024-05-04, so every window lies inside it.
"""

from datetime import date
from typing import Dict, Sequence

import pytest

from lookout.snapshot.model import DepthComparison, GapStatus, Snapshot
from tests.fixtures.archive import Gap, build_archive
from tests.snapshot_helpers import (
    build,
    dry_station,
    live_row,
    ms,
    seasoned_station,
)

NOW = "2026-11-11T20:02:00Z"  # Wed noon PST
LAST_ROW = "2026-11-11T20:00:00Z"
FIRST_ROW = "2026-11-03T08:00:00Z"  # Mon 2026-11-03, local midnight


def young(start: str = FIRST_ROW, gaps: Sequence[Gap] = ()) -> Snapshot:
    """A station whose record starts at `start`, built at `NOW`."""
    archive = build_archive(start, LAST_ROW, gaps=gaps)
    return build(archive, None, NOW)


def periods(snapshot: Snapshot) -> Dict[str, GapStatus]:
    """The temperature periods' gap statuses by key."""
    return {p.status.key: p.status for p in snapshot.temperature_context.periods}


def comparisons(snapshot: Snapshot) -> Dict[str, DepthComparison]:
    """The record depth's comparisons by key."""
    return {c.key: c for c in snapshot.record_depth.comparisons}


def tiles(snapshot: Snapshot) -> Dict[str, bool]:
    """Each tile's shown flag by key."""
    return {t.key: t.shown for t in snapshot.record_depth.tiles}


class TestYoungStation:
    @pytest.fixture(scope="class")
    def salmon_run(self) -> Snapshot:
        return young()

    def test_the_windows_the_record_covers_are_shown(self, salmon_run):
        shown = {key: s.shown for key, s in periods(salmon_run).items()}

        assert shown == {
            "today": True,
            "yesterday": True,
            "7d": True,
            "30d": False,
            "90d": False,
            "365d": False,
        }

    def test_a_window_the_record_does_not_cover_is_not_shown(self, salmon_run):
        month = periods(salmon_run)["30d"]

        assert month.status == "invalid"
        assert month.reason == "before_archive"
        assert month.shown is False

    def test_the_rain_windows_follow_the_same_depth(self, salmon_run):
        shown = {w.key: w.status.shown for w in salmon_run.rain_tab.windows}

        assert shown["7d"] is True
        assert shown["30d"] is False
        assert shown["water_year"] is False

    def test_the_gaps_section_carries_the_depth(self, salmon_run):
        shown = {w.key: w.shown for w in salmon_run.gaps.windows}

        assert shown["7d"] is True
        assert shown["365d"] is False

    def test_full_history_is_the_record_s_first_day_to_today(self, salmon_run):
        full = salmon_run.temperature_context.full_history
        archive = build_archive(FIRST_ROW, LAST_ROW)

        assert full.status.key == "full_history"
        assert full.status.first_day == "2026-11-03"
        assert full.status.last_day == "2026-11-11"
        assert full.status.days == 9
        assert full.status.status == "valid"
        assert full.status.shown is True
        assert full.low == pytest.approx(archive["tempf"].min(), abs=0.05)
        assert full.high == pytest.approx(archive["tempf"].max(), abs=0.05)
        assert full.readings == len(archive)

    def test_the_depth_names_the_record_s_first_day(self, salmon_run):
        depth = salmon_run.record_depth

        assert depth.first_day == "2026-11-03"
        assert depth.days == 9

    def test_each_comparison_not_shown_names_the_day_it_unlocks(self, salmon_run):
        found = {
            key: (c.shown, c.unlocks_on) for key, c in comparisons(salmon_run).items()
        }

        assert found == {
            "today": (True, None),
            "yesterday": (True, None),
            "7d": (True, None),
            "30d": (False, "2026-12-02"),
            "90d": (False, "2027-01-31"),
            "365d": (False, "2027-11-02"),
            "water_year": (False, "2027-10-01"),
            "earlier_year": (False, "2027-11-10"),
            "full_history": (True, None),
        }

    def test_the_next_unlock_is_the_earliest(self, salmon_run):
        unlock = salmon_run.record_depth.next_unlock

        assert (unlock.key, unlock.date) == ("30d", "2026-12-02")

    def test_every_tile_has_the_record_it_needs(self, salmon_run):
        assert tiles(salmon_run) == {
            "temperature": True,
            "rain": True,
            "last_10_days": True,
            "dry_spell": True,
            "wind": True,
            "pressure": True,
            "humidity": True,
            "sun": True,
            "daylight": True,
            "inside": True,
        }

    def test_ten_day_rows_before_the_record_are_not_shown(self, salmon_run):
        rows = {r.day: r.status.shown for r in salmon_run.temperature_context.ten_days}

        assert rows["2026-11-02"] is False
        assert rows["2026-11-03"] is True
        assert rows["2026-11-11"] is True


class TestGapInsideTheRecord:
    @pytest.fixture(scope="class")
    def gapped(self) -> Snapshot:
        # Sat 2026-11-08 has no rows at all.
        return young(gaps=[Gap("2026-11-08T08:00:00Z", "2026-11-09T08:00:00Z")])

    def test_a_gap_inside_a_shown_window_still_reports_as_a_gap(self, gapped):
        week = periods(gapped)["7d"]

        assert week.shown is True
        assert week.status == "invalid"
        assert week.reason == "gap"

    def test_full_history_reports_the_gap_too(self, gapped):
        full = gapped.temperature_context.full_history.status

        assert full.shown is True
        assert full.missing_days == 1
        assert full.reason == "gap"

    def test_the_gap_does_not_move_the_next_unlock(self, gapped):
        unlock = gapped.record_depth.next_unlock

        assert (unlock.key, unlock.date) == ("30d", "2026-12-02")


class TestFirstDay:
    @pytest.fixture(scope="class")
    def first_day(self) -> Snapshot:
        return young(start="2026-11-11T08:00:00Z")

    def test_only_today_and_full_history_are_shown(self, first_day):
        shown = [key for key, c in comparisons(first_day).items() if c.shown]

        assert shown == ["today", "full_history"]

    def test_yesterday_unlocks_tomorrow(self, first_day):
        unlock = first_day.record_depth.next_unlock

        assert (unlock.key, unlock.date) == ("yesterday", "2026-11-12")

    def test_the_tiles_that_compare_with_earlier_days_hide(self, first_day):
        hidden = [key for key, shown in tiles(first_day).items() if not shown]

        assert hidden == ["temperature", "last_10_days"]


class TestEmptyArchive:
    @pytest.fixture(scope="class")
    def empty(self) -> Snapshot:
        archive = build_archive(FIRST_ROW, LAST_ROW).iloc[0:0]
        return build(archive, None, NOW)

    def test_nothing_is_shown_and_nothing_unlocks(self, empty):
        depth = empty.record_depth

        assert depth.first_day is None
        assert depth.days == 0
        assert all(not c.shown and c.unlocks_on is None for c in depth.comparisons)
        assert depth.next_unlock is None
        assert empty.temperature_context.full_history is None

    def test_only_daylight_needs_no_record(self, empty):
        # Inside needs no record either, but an empty archive has no sensor.
        shown = [key for key, shown in tiles(empty).items() if shown]

        assert shown == ["daylight"]


class TestSeasonedGolden:
    @pytest.fixture(scope="class")
    def seasoned(self) -> Snapshot:
        return seasoned_station()

    def test_every_comparison_is_shown_and_none_unlocks(self, seasoned):
        depth = seasoned.record_depth

        assert depth.first_day == "2024-05-04"
        assert all(c.shown for c in depth.comparisons)
        assert all(shown for shown in tiles(seasoned).values())
        assert depth.next_unlock is None

    def test_the_long_windows_are_shown_with_their_gap(self, seasoned):
        found = periods(seasoned)

        for key in ("90d", "365d"):
            assert found[key].shown is True
            assert (found[key].status, found[key].reason) == ("invalid", "gap")

    def test_full_history_spans_the_record_and_reports_its_gap(self, seasoned):
        full = seasoned.temperature_context.full_history

        assert full.status.first_day == "2024-05-04"
        assert full.status.last_day == "2026-05-20"
        assert full.status.shown is True
        assert (full.status.status, full.status.reason) == ("invalid", "gap")
        assert full.low is None

    def test_an_earlier_year_s_window_before_the_record_is_not_shown(self, seasoned):
        rolling = {w.key: w for w in seasoned.almanac.rolling}
        by_year = {
            key: {y.year: y.status.shown for y in rolling[key].years}
            for key in ("7d", "30d")
        }

        # 2024's 7 days end May 20 from May 14, inside the record; its 30
        # days start Apr 21, before the record's May 4.
        assert by_year["7d"] == {2026: True, 2025: True, 2024: True}
        assert by_year["30d"] == {2026: True, 2025: True, 2024: False}


class TestDryGolden:
    def test_the_young_golden_names_a_next_unlock(self):
        snapshot = dry_station()
        depth = snapshot.record_depth
        first = date.fromisoformat(depth.first_day)
        unlock = depth.next_unlock

        assert (
            depth.first_day
            == snapshot.temperature_context.full_history.status.first_day
        )
        assert unlock is not None
        assert date.fromisoformat(unlock.date) > first
        assert not comparisons(snapshot)[unlock.key].shown


def test_full_history_carries_the_live_reading():
    full = build_archive(FIRST_ROW, "2026-11-11T20:05:00Z")
    archive = full[full["dateutc"] < ms(LAST_ROW)]
    live = {**live_row(full, LAST_ROW), "tempf": 99.0}
    period = build(archive, live, NOW).temperature_context.full_history

    assert period.high == 99.0

"""
Gap status for months and windows (#80; spec #66 "Gap status", seam 1).

Each month and each window gets one status: valid, prorated or invalid. Days
are station-local (ADR-0002). The archives are synthetic five-minute stations
from `tests.fixtures.archive`, and every `now` is fixed.
"""

from datetime import date

import pandas as pd
import pytest

from lookout.snapshot import Station, build_snapshot
from lookout.snapshot.clock import StationClock
from lookout.snapshot.gaps import (
    ArchiveCoverage,
    comparable,
    month_status,
    prorate,
    window_status,
)
from tests.fixtures.archive import SCENARIO, Gap, build_archive, scenario_archive

STATION = Station(mac="00:11:22:33:44:55", name="Fixture Station")


def day(iso: str) -> date:
    """A station-local calendar day."""
    return date.fromisoformat(iso)


def at(when: str) -> StationClock:
    """The station clock at a fixed, timezone-aware build time."""
    return StationClock(pd.Timestamp(when))


def status_of(gap_status) -> tuple:
    """The status, reason and count of days that cannot be counted."""
    return gap_status.status, gap_status.reason, gap_status.missing_days


# Jan 1 to Jan 11 local (PST, UTC-8), archived up to 23:00 local on Jan 11.
JAN_START = "2026-01-01T08:00:00Z"
JAN_NOW = "2026-01-12T07:00:00Z"  # Jan 11, 23:00 local


def january(*gaps: Gap) -> pd.DataFrame:
    """Five-minute rows from Jan 1 local to `JAN_NOW`, less `gaps`."""
    return build_archive(JAN_START, JAN_NOW, gaps=gaps)


class TestDays:
    def test_a_fully_archived_week_is_valid(self):
        status = window_status(
            january(), day("2026-01-05"), day("2026-01-11"), at(JAN_NOW)
        )

        assert status_of(status) == ("valid", None, 0)
        assert (status.first_day, status.last_day, status.days) == (
            "2026-01-05",
            "2026-01-11",
            7,
        )

    def test_the_scenario_outages_leave_every_day_countable(self):
        """Its 2 h and 4 h gaps each leave the day above 80% coverage."""
        status = window_status(
            scenario_archive(), day("2026-01-04"), day("2026-01-10"), at(SCENARIO.end)
        )

        assert status_of(status) == ("valid", None, 0)

    def test_a_day_under_80_percent_coverage_is_low_coverage(self):
        # Jan 8, 10:00-16:00 local: 18 of 24 hours, 75%.
        archive = january(Gap("2026-01-08T18:00:00Z", "2026-01-09T00:00:00Z"))

        one_day = window_status(
            archive, day("2026-01-08"), day("2026-01-08"), at(JAN_NOW)
        )
        week = window_status(archive, day("2026-01-05"), day("2026-01-11"), at(JAN_NOW))

        assert status_of(one_day) == ("invalid", "low_coverage", 1)
        assert status_of(week) == ("invalid", "low_coverage", 1)

    def test_a_four_hour_outage_keeps_the_day(self):
        # 20 of 24 hours, 83%.
        archive = january(Gap("2026-01-08T18:00:00Z", "2026-01-08T22:00:00Z"))

        status = window_status(
            archive, day("2026-01-08"), day("2026-01-08"), at(JAN_NOW)
        )

        assert status_of(status) == ("valid", None, 0)

    def test_a_week_with_a_day_missing_from_the_record_is_invalid(self):
        archive = january(Gap("2026-01-07T08:00:00Z", "2026-01-08T08:00:00Z"))

        status = window_status(
            archive, day("2026-01-05"), day("2026-01-11"), at(JAN_NOW)
        )

        assert status_of(status) == ("invalid", "gap", 1)

    def test_days_follow_the_station_clock_across_daylight_saving(self):
        """
        Nov 2, 2025 is 25 hours long locally. A 5.5 h outage leaves 19.5 h,
        78% of the day: low coverage, though it is 81% of a 24-hour day.
        """
        archive = build_archive(
            "2025-10-31T07:00:00Z",
            "2025-11-04T08:00:00Z",
            gaps=[Gap("2025-11-02T15:00:00Z", "2025-11-02T20:30:00Z")],
        )
        now = at("2025-11-04T08:00:00Z")

        fall_back = window_status(archive, day("2025-11-02"), day("2025-11-02"), now)
        next_day = window_status(archive, day("2025-11-03"), day("2025-11-03"), now)

        assert status_of(fall_back) == ("invalid", "low_coverage", 1)
        assert status_of(next_day) == ("valid", None, 0)


class TestProrating:
    # Dec 13 to Jan 11 local: 30 days.
    START, END = day("2025-12-13"), day("2026-01-11")

    def thirty_days(self, *gaps: Gap) -> pd.DataFrame:
        return build_archive("2025-12-01T08:00:00Z", JAN_NOW, gaps=gaps)

    def test_up_to_ten_percent_of_days_missing_is_prorated(self):
        # Jan 1, 2 and 3 local are absent: 3 of 30 days.
        archive = self.thirty_days(Gap("2026-01-01T08:00:00Z", "2026-01-04T08:00:00Z"))

        status = window_status(archive, self.START, self.END, at(JAN_NOW))

        assert status_of(status) == ("prorated", "gap", 3)

    def test_more_than_ten_percent_of_days_missing_is_invalid(self):
        # Jan 1 to 4 local: 4 of 30 days.
        archive = self.thirty_days(Gap("2026-01-01T08:00:00Z", "2026-01-05T08:00:00Z"))

        status = window_status(archive, self.START, self.END, at(JAN_NOW))

        assert status_of(status) == ("invalid", "gap", 4)

    def test_low_coverage_days_count_as_missing(self):
        # Two days at 50% coverage.
        archive = self.thirty_days(
            Gap("2026-01-02T08:00:00Z", "2026-01-02T20:00:00Z"),
            Gap("2026-01-05T08:00:00Z", "2026-01-05T20:00:00Z"),
        )

        status = window_status(archive, self.START, self.END, at(JAN_NOW))

        assert status_of(status) == ("prorated", "low_coverage", 2)

    def test_a_prorated_total_is_scaled_to_the_whole_window(self):
        prorated = window_status(
            self.thirty_days(Gap("2026-01-01T08:00:00Z", "2026-01-04T08:00:00Z")),
            self.START,
            self.END,
            at(JAN_NOW),
        )
        valid = window_status(self.thirty_days(), self.START, self.END, at(JAN_NOW))

        # 2.7 in over the 27 counted days is 0.1 in a day, 3.0 in over 30.
        assert prorate(prorated, 2.7) == pytest.approx(3.0)
        assert prorate(valid, 2.7) == 2.7

    def test_the_counted_days_leave_out_the_missing_ones(self):
        archive = self.thirty_days(Gap("2026-01-01T08:00:00Z", "2026-01-04T08:00:00Z"))

        counted = ArchiveCoverage(archive, at(JAN_NOW)).counted_days(
            self.START, self.END
        )

        assert len(counted) == 27
        assert day("2026-01-02") not in counted
        assert day("2025-12-31") in counted and day("2026-01-04") in counted


class TestArchiveEdges:
    def test_a_window_reaching_past_the_archive_end_is_not_archived_yet(self):
        # The archive stops at 20:00 local on Jan 11; it is now 23:00.
        archive = build_archive(JAN_START, "2026-01-12T04:00:00Z")

        today = window_status(
            archive, day("2026-01-11"), day("2026-01-11"), at(JAN_NOW)
        )
        yesterday = window_status(
            archive, day("2026-01-10"), day("2026-01-10"), at(JAN_NOW)
        )

        assert status_of(today) == ("invalid", "not_archived", 1)
        assert status_of(yesterday) == ("valid", None, 0)

    def test_an_archive_under_an_hour_behind_still_counts_today(self):
        # The archive stops at 22:30 local; the half hour since is not missing.
        archive = build_archive(JAN_START, "2026-01-12T06:30:00Z")

        today = window_status(
            archive, day("2026-01-11"), day("2026-01-11"), at(JAN_NOW)
        )

        assert status_of(today) == ("valid", None, 0)

    def test_today_counts_only_the_hours_so_far(self):
        # 09:00 local on Jan 11, archived to the minute.
        now = "2026-01-11T17:00:00Z"
        archive = build_archive(JAN_START, now)

        today = window_status(archive, day("2026-01-11"), day("2026-01-11"), at(now))

        assert status_of(today) == ("valid", None, 0)

    def test_days_before_the_archive_began_are_invalid(self):
        archive = scenario_archive()  # starts Jan 4 local

        status = window_status(
            archive, day("2026-01-01"), day("2026-01-07"), at(SCENARIO.end)
        )

        assert status_of(status) == ("invalid", "before_archive", 3)

    def test_days_still_to_come_leave_a_window_in_progress(self):
        status = window_status(
            january(), day("2026-01-10"), day("2026-01-12"), at(JAN_NOW)
        )

        assert status_of(status) == ("invalid", "in_progress", 1)


class TestMonths:
    # Dec 15, 2025 to Feb 10, 2026 local, with Jan 10 and 11 missing.
    NOW = "2026-02-11T07:00:00Z"  # Feb 10, 23:00 local

    def archive(self) -> pd.DataFrame:
        return build_archive(
            "2025-12-15T08:00:00Z",
            self.NOW,
            gaps=[Gap("2026-01-10T08:00:00Z", "2026-01-12T08:00:00Z")],
        )

    def test_a_month_with_two_days_missing_is_prorated(self):
        status = month_status(self.archive(), 2026, 1, at(self.NOW))

        assert status.key == "2026-01"
        assert (status.first_day, status.last_day, status.days) == (
            "2026-01-01",
            "2026-01-31",
            31,
        )
        assert status_of(status) == ("prorated", "gap", 2)

    def test_the_month_the_archive_began_in_is_invalid(self):
        status = month_status(self.archive(), 2025, 12, at(self.NOW))

        assert status_of(status) == ("invalid", "before_archive", 14)

    def test_the_current_month_is_in_progress(self):
        status = month_status(self.archive(), 2026, 2, at(self.NOW))

        assert status_of(status) == ("invalid", "in_progress", 18)

    def test_a_month_that_ended_today_is_complete(self):
        # 23:00 local on Jan 31: the whole month is archived but not over.
        now = "2026-02-01T07:00:00Z"
        archive = build_archive("2025-12-31T08:00:00Z", now)

        status = month_status(archive, 2026, 1, at(now))

        assert status_of(status) == ("invalid", "in_progress", 0)


class TestComparable:
    def test_an_invalid_window_is_never_computed(self):
        archive = january(Gap("2026-01-07T08:00:00Z", "2026-01-08T08:00:00Z"))
        status = window_status(
            archive, day("2026-01-05"), day("2026-01-11"), at(JAN_NOW)
        )
        calls = []

        value = comparable(status, lambda: calls.append(1) or 1.5)

        assert value is None
        assert calls == []

    def test_a_valid_or_prorated_window_is_computed(self):
        status = window_status(
            january(), day("2026-01-05"), day("2026-01-11"), at(JAN_NOW)
        )

        assert comparable(status, lambda: 1.5) == 1.5

    def test_an_invalid_window_has_no_prorated_total(self):
        archive = january(Gap("2026-01-07T08:00:00Z", "2026-01-08T08:00:00Z"))
        status = window_status(
            archive, day("2026-01-05"), day("2026-01-11"), at(JAN_NOW)
        )

        assert prorate(status, 2.0) is None


# ---------- through build_snapshot ----------


def snapshot(archive: pd.DataFrame, now: str):
    return build_snapshot(
        archive=archive,
        live=None,
        event_catalog=None,
        energy_catalog=None,
        station=STATION,
        now=pd.Timestamp(now),
    )


class TestSnapshotGaps:
    NOW = TestMonths.NOW

    def test_every_month_of_the_archive_gets_a_status(self):
        months = snapshot(TestMonths().archive(), self.NOW).gaps.months

        assert [(m.key, m.status, m.reason) for m in months] == [
            ("2025-12", "invalid", "before_archive"),
            ("2026-01", "prorated", "gap"),
            ("2026-02", "invalid", "in_progress"),
        ]

    def test_the_standard_windows_end_on_the_station_local_today(self):
        """At 23:00 local on Feb 10 it is already Feb 11 in UTC."""
        windows = snapshot(TestMonths().archive(), self.NOW).gaps.windows

        assert [(w.key, w.first_day, w.last_day) for w in windows] == [
            ("today", "2026-02-10", "2026-02-10"),
            ("yesterday", "2026-02-09", "2026-02-09"),
            ("7d", "2026-02-04", "2026-02-10"),
            ("30d", "2026-01-12", "2026-02-10"),
            ("90d", "2025-11-13", "2026-02-10"),
            ("365d", "2025-02-11", "2026-02-10"),
            ("water_year", "2025-10-01", "2026-02-10"),
        ]

    def test_the_standard_windows_get_their_status(self):
        windows = snapshot(TestMonths().archive(), self.NOW).gaps.windows

        assert {w.key: (w.status, w.reason) for w in windows} == {
            "today": ("valid", None),
            "yesterday": ("valid", None),
            "7d": ("valid", None),
            "30d": ("valid", None),
            "90d": ("invalid", "before_archive"),
            "365d": ("invalid", "before_archive"),
            "water_year": ("invalid", "before_archive"),
        }

    def test_a_window_with_a_gap_is_invalid_in_the_snapshot(self):
        # A day missing three days ago: the week is invalid, the month is not.
        archive = build_archive(
            "2025-12-15T08:00:00Z",
            self.NOW,
            gaps=[Gap("2026-02-07T08:00:00Z", "2026-02-08T08:00:00Z")],
        )

        windows = {w.key: w for w in snapshot(archive, self.NOW).gaps.windows}

        assert status_of(windows["7d"]) == ("invalid", "gap", 1)
        assert status_of(windows["30d"]) == ("prorated", "gap", 1)

    def test_the_water_year_starts_on_october_first(self):
        now = "2025-10-01T17:00:00Z"  # Oct 1, 10:00 local
        archive = build_archive("2025-09-01T07:00:00Z", now)

        windows = {w.key: w for w in snapshot(archive, now).gaps.windows}

        assert (windows["water_year"].first_day, windows["water_year"].days) == (
            "2025-10-01",
            1,
        )
        assert windows["water_year"].status == "valid"

    def test_no_comparison_is_computed_for_an_invalid_window(self):
        windows = snapshot(TestMonths().archive(), self.NOW).gaps.windows
        computed = []

        values = {
            w.key: comparable(w, lambda w=w: computed.append(w.key)) for w in windows
        }

        assert computed == ["today", "yesterday", "7d", "30d"]
        assert values["90d"] is values["365d"] is values["water_year"] is None

    def test_an_empty_archive_has_no_months_and_nothing_archived(self):
        empty = build_archive(JAN_START, JAN_NOW).iloc[0:0]

        gaps = snapshot(empty, JAN_NOW).gaps

        assert gaps.months == ()
        assert {(w.status, w.reason) for w in gaps.windows} == {
            ("invalid", "not_archived")
        }

    def test_gaps_are_in_the_dashboard_document_not_the_context(self):
        built = snapshot(TestMonths().archive(), self.NOW)

        dashboard, context = built.dashboard(), built.context()

        assert dashboard["gaps"]["months"][1] == {
            "key": "2026-01",
            "first_day": "2026-01-01",
            "last_day": "2026-01-31",
            "days": 31,
            "missing_days": 2,
            "status": "prorated",
            "reason": "gap",
        }
        assert "gaps" not in context

"""
The Solar tab's section of the snapshot (#85; spec #66 seam 1).

`build_snapshot(...).solar_tab`: daily energy and window totals from the
energy catalog, monthly totals, each recent day's 15-minute energy, and the
24-hour view with its usual range. Every value over a day, a month or a
window follows the gap rules; the live solar reading is the 24-hour trace's
newest point.

The catalogs here are written by hand, so every expected kWh comes from the
numbers put in, not from the catalog's own integration. Archives are
synthetic five-minute stations from `tests.fixtures.archive` (only their
coverage and their solar trace are read), and every `now` is fixed.
"""

from typing import Dict, Optional, Sequence

import pandas as pd
import pytest

from lookout.snapshot import Station, build_snapshot
from lookout.snapshot.model import Snapshot
from tests.fixtures.ambient import to_ambient_records
from tests.fixtures.archive import Gap, build_archive

STATION = Station(mac="00:11:22:33:44:55", name="Fixture Station")
TZ = "America/Los_Angeles"
FIVE_MIN_MS = 5 * 60 * 1000

# June 2026 is PDT: local midnight is 07:00Z. Now is Jun 20, 12:07 pm local.
START = "2026-05-01T07:00:00Z"
NOW = "2026-06-20T19:07:00Z"
ARCHIVE_END = "2026-06-20T19:05:00Z"  # newest archived row 19:00Z (noon)
SUNNY = ("10:00", "14:00")  # each 15 minutes in it holds 0.01 kWh/m²


def ms(when: str) -> int:
    """Epoch ms for an ISO-8601 UTC time."""
    return int(pd.Timestamp(when).value // 10**6)


def catalog(
    first_day: str,
    last_ms: int,
    kwh: float = 0.01,
    sunny: Sequence[str] = SUNNY,
    skip_days: Sequence[str] = (),
) -> pd.DataFrame:
    """
    An energy catalog of 15-minute periods, as `EnergyCatalog` stores it.

    :param first_day: The first local day.
    :param last_ms: Periods start before this time.
    :param kwh: Each sunny period's energy; the rest hold 0.
    :param sunny: The local clock span, [start, end), that has sun.
    :param skip_days: Local days with no periods at all.
    :return: The catalog.
    """
    starts = pd.date_range(
        pd.Timestamp(first_day).tz_localize(TZ),
        pd.Timestamp(last_ms, unit="ms", tz="UTC").tz_convert(TZ),
        freq="15min",
        inclusive="left",
    )
    starts = starts[[d.strftime("%Y-%m-%d") not in skip_days for d in starts]]
    clock = starts.strftime("%H:%M")
    energy = [kwh if sunny[0] <= c < sunny[1] else 0.0 for c in clock]
    return pd.DataFrame(
        {
            "period_start": starts,
            "period_end": starts + pd.Timedelta(minutes=15),
            "energy_kwh": energy,
        }
    )


def build(
    archive: pd.DataFrame,
    energy: Optional[pd.DataFrame],
    now: str = NOW,
    live: Optional[Dict] = None,
) -> Snapshot:
    """`build_snapshot` for the fixture station."""
    return build_snapshot(
        archive=archive,
        live=live,
        event_catalog=None,
        energy_catalog=energy,
        station=STATION,
        now=pd.Timestamp(now),
    )


@pytest.fixture(scope="module")
def june() -> pd.DataFrame:
    """May 1 to Jun 20 noon local, no gaps."""
    return build_archive(START, ARCHIVE_END)


@pytest.fixture(scope="module")
def june_snapshot(june) -> Snapshot:
    """The June station with a full catalog."""
    return build(june, catalog("2026-05-01", ms(ARCHIVE_END) - FIVE_MIN_MS))


def day_value(snapshot: Snapshot, day: str) -> Optional[float]:
    """One day's energy from the tab's daily totals."""
    days = snapshot.solar_tab.days
    index = (pd.Timestamp(day) - pd.Timestamp(days.first_day)).days
    return days.energy_kwh_m2[index] if index >= 0 else None


def windows(snapshot: Snapshot) -> Dict[str, Optional[float]]:
    """The tab's window totals by key."""
    return {w.key: w.energy_kwh_m2 for w in snapshot.solar_tab.windows}


class TestDailyEnergy:
    def test_a_day_is_the_sum_of_its_catalog_periods(self, june_snapshot):
        # Sixteen sunny periods of 0.01 kWh/m², 10 am to 2 pm.
        assert day_value(june_snapshot, "2026-06-19") == pytest.approx(0.16)
        assert day_value(june_snapshot, "2026-05-01") == pytest.approx(0.16)

    def test_the_days_run_from_the_archive_s_first_day_through_today(
        self, june_snapshot
    ):
        days = june_snapshot.solar_tab.days
        assert days.first_day == "2026-05-01"
        assert len(days.energy_kwh_m2) == 51  # May 1 .. Jun 20
        assert len(days.reason) == 51

    def test_today_is_the_energy_so_far(self, june_snapshot):
        # Periods 10:00 .. 11:45 have started by noon: eight of them.
        assert day_value(june_snapshot, "2026-06-20") == pytest.approx(0.08)
        assert windows(june_snapshot)["today"] == pytest.approx(0.08)

    def test_windows_sum_their_days(self, june_snapshot):
        totals = windows(june_snapshot)
        assert totals["yesterday"] == pytest.approx(0.16)
        assert totals["7d"] == pytest.approx(6 * 0.16 + 0.08)
        assert totals["30d"] == pytest.approx(29 * 0.16 + 0.08)

    def test_the_windows_are_the_snapshot_s_standard_windows(self, june_snapshot):
        keys = [w.key for w in june_snapshot.solar_tab.windows]
        assert keys == ["today", "yesterday", "7d", "30d", "90d", "365d"]
        # 90 and 365 days reach back before the archive began.
        assert windows(june_snapshot)["90d"] is None
        assert windows(june_snapshot)["365d"] is None


class TestGaps:
    @pytest.fixture(scope="class")
    def gappy(self) -> Snapshot:
        """A whole day missing on Jun 10, in both the archive and the catalog."""
        archive = build_archive(
            START,
            ARCHIVE_END,
            gaps=[Gap("2026-06-10T07:00:00Z", "2026-06-11T07:00:00Z")],
        )
        energy = catalog(
            "2026-05-01", ms(ARCHIVE_END) - FIVE_MIN_MS, skip_days=["2026-06-10"]
        )
        return build(archive, energy)

    def test_a_day_missing_from_the_record_is_not_read_as_a_dark_day(self, gappy):
        assert day_value(gappy, "2026-06-10") is None
        index = (pd.Timestamp("2026-06-10") - pd.Timestamp("2026-05-01")).days
        assert gappy.solar_tab.days.reason[index] == "gap"
        assert gappy.solar_tab.days.reason[index - 1] is None

    def test_a_window_missing_a_tenth_of_its_days_or_less_is_prorated(self, gappy):
        # 30 days, Jun 10 missing: 28 full days and today, scaled by 30/29.
        # Published to a thousandth of a kWh.
        assert windows(gappy)["30d"] == pytest.approx(
            (28 * 0.16 + 0.08) * 30 / 29, abs=1e-3
        )

    def test_a_short_window_with_a_missing_day_is_not_computed(self):
        archive = build_archive(
            START,
            ARCHIVE_END,
            gaps=[Gap("2026-06-17T07:00:00Z", "2026-06-18T07:00:00Z")],
        )
        snapshot = build(archive, catalog("2026-05-01", ms(ARCHIVE_END)))
        assert windows(snapshot)["7d"] is None

    def test_today_is_not_computed_while_the_archive_lags_behind(self, june):
        # Two hours after the newest archived row: today is not archived yet.
        energy = catalog("2026-05-01", ms(ARCHIVE_END))
        snapshot = build(june, energy, now="2026-06-20T21:05:00Z")
        assert windows(snapshot)["today"] is None
        assert day_value(snapshot, "2026-06-20") is None
        assert snapshot.solar_tab.days.reason[-1] == "not_archived"


class TestMonths:
    def test_a_complete_month_is_the_sum_of_its_days(self, june_snapshot):
        months = {m.key: m.energy_kwh_m2 for m in june_snapshot.solar_tab.months}
        assert months["2026-05"] == pytest.approx(31 * 0.16)

    def test_the_month_under_way_is_not_compared(self, june_snapshot):
        months = {m.key: m.energy_kwh_m2 for m in june_snapshot.solar_tab.months}
        assert months["2026-06"] is None

    def test_the_months_are_the_gap_section_s_months(self, june_snapshot):
        assert [m.key for m in june_snapshot.solar_tab.months] == [
            s.key for s in june_snapshot.gaps.months
        ]


class TestFifteenMinuteEnergy:
    def test_each_recent_day_has_its_96_slots_in_wh(self, june_snapshot):
        recent = june_snapshot.solar_tab.recent
        index = (pd.Timestamp("2026-06-19") - pd.Timestamp(recent.first_day)).days
        day = recent.energy_wh_m2[index]
        assert recent.slot_minutes == 15
        assert len(day) == 96
        assert day[39] == 0.0  # 9:45 am
        assert day[40] == pytest.approx(10.0)  # 10:00 am: 0.01 kWh/m²
        assert day[55] == pytest.approx(10.0)  # 1:45 pm
        assert day[56] == 0.0  # 2:00 pm

    def test_a_day_s_slots_add_up_to_its_daily_total(self, june_snapshot):
        recent = june_snapshot.solar_tab.recent
        for offset, slots in enumerate(recent.energy_wh_m2):
            day = (pd.Timestamp(recent.first_day) + pd.Timedelta(days=offset)).date()
            total = day_value(june_snapshot, day.isoformat())
            if total is None:
                continue
            assert sum(s for s in slots if s is not None) / 1000 == pytest.approx(total)

    def test_they_cover_the_90_day_window_ending_today(self, june_snapshot):
        recent = june_snapshot.solar_tab.recent
        assert recent.window == "90d"
        assert recent.first_day == "2026-03-23"
        assert len(recent.energy_wh_m2) == 90

    def test_a_day_before_the_record_has_no_slots(self, june_snapshot):
        recent = june_snapshot.solar_tab.recent
        assert recent.energy_wh_m2[0] == (None,) * 96

    def test_slots_still_to_come_today_are_empty(self, june_snapshot):
        today = june_snapshot.solar_tab.recent.energy_wh_m2[-1]
        assert today[47] == pytest.approx(10.0)  # 11:45 am, started by noon
        assert today[48:] == (None,) * 48  # noon on: not in the catalog yet


class TestTheLast24Hours:
    @pytest.fixture(scope="class")
    def live(self, june) -> Snapshot:
        """The live reading 25 minutes ahead of the archive, brighter."""
        full = build_archive(START, "2026-06-20T19:30:00Z")
        archive = full[full["dateutc"] < ms("2026-06-20T19:00:00Z")]
        reading = to_ambient_records(
            full[full["dateutc"] == ms("2026-06-20T19:25:00Z")]
        )
        reading[0]["solarradiation"] = 912.0
        energy = catalog("2026-05-01", ms("2026-06-20T19:00:00Z"))
        return build(archive, energy, now="2026-06-20T19:27:00Z", live=reading[0])

    def test_the_trace_ends_at_the_live_reading(self, live):
        trace = live.solar_tab.last_24h
        assert live.current.solar_wm2 == 912.0
        assert trace.solar_wm2[-1] == 912.0
        assert trace.start_ms + (len(trace.solar_wm2) - 1) * FIVE_MIN_MS == ms(
            "2026-06-20T19:25:00Z"
        )

    def test_it_covers_24_hours_in_five_minute_steps(self, live):
        trace = live.solar_tab.last_24h
        assert trace.step_minutes == 5
        assert len(trace.solar_wm2) == 24 * 12 + 1
        assert trace.start_ms == ms("2026-06-19T19:25:00Z")

    def test_the_span_only_the_live_reading_covers_is_empty(self, live):
        trace = live.solar_tab.last_24h
        # 19:00Z .. 19:20Z are not archived; 18:55Z is.
        assert trace.solar_wm2[-6:-1] == (None,) * 5
        assert trace.solar_wm2[-7] is not None

    def test_the_usual_range_is_the_middle_half_of_the_last_30_days(self, live):
        trace = live.solar_tab.last_24h
        # Every synthetic day is the same: at local noon (19:00Z) each holds
        # 800 W/m², so the band's quartiles and median are all 800.
        noon = (ms("2026-06-20T19:00:00Z") - trace.start_ms) // FIVE_MIN_MS
        assert trace.usual.status == "valid"
        assert trace.usual.first_day == "2026-05-21"
        assert trace.usual.last_day == "2026-06-19"
        assert trace.usual_low_wm2[noon] == pytest.approx(800.0)
        assert trace.usual_median_wm2[noon] == pytest.approx(800.0)
        assert trace.usual_high_wm2[noon] == pytest.approx(800.0)

    def test_the_energy_is_the_catalog_s_over_the_24_hours(self, live):
        trace = live.solar_tab.last_24h
        # From yesterday 12:25 pm: periods 12:30 .. 1:45 pm (6), then today
        # 10:00 .. 11:45 am (8), the 12:00 period not catalogued yet.
        assert trace.energy_kwh_m2 == pytest.approx(0.14)
        assert trace.previous_kwh_m2 == pytest.approx(0.16)

    def test_no_usual_range_over_too_few_recorded_days(self):
        archive = build_archive("2026-06-15T07:00:00Z", ARCHIVE_END)
        snapshot = build(archive, catalog("2026-06-15", ms(ARCHIVE_END)))
        trace = snapshot.solar_tab.last_24h
        assert trace.usual.status == "invalid"
        assert trace.usual_median_wm2 is None


class TestNoCatalog:
    def test_there_is_no_solar_tab_without_an_energy_catalog(self, june):
        assert build(june, None).solar_tab is None


class TestDaylightSaving:
    def test_the_repeated_hour_is_summed_into_its_clock_slots(self):
        # Nov 1, 2026: clocks fall back at 2 am, so 1:00-1:59 am runs twice.
        archive = build_archive("2026-10-25T07:00:00Z", "2026-11-02T20:05:00Z")
        energy = catalog(
            "2026-10-25", ms("2026-11-02T20:00:00Z"), sunny=("01:00", "02:00")
        )
        snapshot = build(archive, energy, now="2026-11-02T20:07:00Z")
        recent = snapshot.solar_tab.recent
        index = (pd.Timestamp("2026-11-01") - pd.Timestamp(recent.first_day)).days
        day = recent.energy_wh_m2[index]
        assert day[4] == pytest.approx(20.0)  # 1:00 am, both passes
        assert day[7] == pytest.approx(20.0)  # 1:45 am, both passes
        assert day_value(snapshot, "2026-11-01") == pytest.approx(0.08)
        assert day_value(snapshot, "2026-10-31") == pytest.approx(0.04)


def solar_status(snapshot: Snapshot, key: str):
    """A solar window's or month's own gap status."""
    for item in (*snapshot.solar_tab.windows, *snapshot.solar_tab.months):
        if item.key == key:
            return item.status
    raise KeyError(key)


def day_reason(snapshot: Snapshot, day: str) -> Optional[str]:
    """One day's reason from the tab's daily totals."""
    days = snapshot.solar_tab.days
    return days.reason[(pd.Timestamp(day) - pd.Timestamp(days.first_day)).days]


class TestCatalogCoverage:
    """
    A day counts for solar only when the energy catalog has it (#123).

    The archive is whole in every case here; only the catalog lacks days, as
    after a gap backfilled into the archive the catalog never went back for.
    """

    @pytest.fixture(scope="class")
    def backfilled(self, june) -> Snapshot:
        """Jun 10 is in the archive but not the catalog."""
        energy = catalog(
            "2026-05-01", ms(ARCHIVE_END) - FIVE_MIN_MS, skip_days=["2026-06-10"]
        )
        return build(june, energy)

    def test_an_archived_day_the_catalog_lacks_is_not_a_dark_day(self, backfilled):
        assert day_value(backfilled, "2026-06-10") is None
        assert day_reason(backfilled, "2026-06-10") == "gap"

    def test_a_window_over_it_is_prorated_not_summed_as_zero(self, backfilled):
        # Read as 0 kWh it would be 28 * 0.16 + 0.08, valid.
        assert windows(backfilled)["30d"] == pytest.approx(
            (28 * 0.16 + 0.08) * 30 / 29, abs=1e-3
        )
        status = solar_status(backfilled, "30d")
        assert (status.status, status.reason, status.missing_days) == (
            "prorated",
            "gap",
            1,
        )

    def test_the_archive_s_own_status_is_unchanged(self, backfilled):
        archive_30d = next(w for w in backfilled.gaps.windows if w.key == "30d")
        assert archive_30d.status == "valid"

    def test_a_short_window_over_it_is_not_computed(self, june):
        energy = catalog(
            "2026-05-01", ms(ARCHIVE_END) - FIVE_MIN_MS, skip_days=["2026-06-18"]
        )
        snapshot = build(june, energy)
        assert windows(snapshot)["7d"] is None
        assert solar_status(snapshot, "7d").reason == "gap"
        assert snapshot.solar_tab.last_24h.previous_kwh_m2 is None

    def test_a_month_missing_too_many_catalog_days_is_not_computed(self, june):
        skip = [f"2026-05-{d:02d}" for d in range(10, 15)]  # 5 of 31 days
        energy = catalog("2026-05-01", ms(ARCHIVE_END) - FIVE_MIN_MS, skip_days=skip)
        snapshot = build(june, energy)
        months = {m.key: m.energy_kwh_m2 for m in snapshot.solar_tab.months}
        assert months["2026-05"] is None
        assert solar_status(snapshot, "2026-05").reason == "gap"

    def test_a_month_missing_one_catalog_day_is_prorated(self, june):
        energy = catalog(
            "2026-05-01", ms(ARCHIVE_END) - FIVE_MIN_MS, skip_days=["2026-05-10"]
        )
        snapshot = build(june, energy)
        months = {m.key: m.energy_kwh_m2 for m in snapshot.solar_tab.months}
        assert months["2026-05"] == pytest.approx(30 * 0.16 * 31 / 30)
        assert solar_status(snapshot, "2026-05").status == "prorated"

    def test_a_day_the_catalog_holds_too_little_of_is_low_coverage(self, june):
        energy = catalog("2026-05-01", ms(ARCHIVE_END) - FIVE_MIN_MS)
        start = pd.to_datetime(energy["period_start"])
        morning = (start.dt.strftime("%Y-%m-%d") == "2026-06-10") & (start.dt.hour < 12)
        snapshot = build(june, energy[~morning])
        assert day_value(snapshot, "2026-06-10") is None
        assert day_reason(snapshot, "2026-06-10") == "low_coverage"

    def test_days_before_the_catalog_begins_are_before_the_record(self, june):
        energy = catalog("2026-05-05", ms(ARCHIVE_END) - FIVE_MIN_MS)
        snapshot = build(june, energy)
        assert day_reason(snapshot, "2026-05-04") == "before_archive"
        assert day_reason(snapshot, "2026-05-05") is None
        assert solar_status(snapshot, "2026-05").reason == "before_archive"

    def test_days_after_the_catalog_ends_are_not_archived_yet(self, june):
        energy = catalog("2026-05-01", ms("2026-06-18T07:00:00Z"))
        snapshot = build(june, energy)
        assert day_reason(snapshot, "2026-06-17") is None
        assert day_reason(snapshot, "2026-06-18") == "not_archived"
        assert windows(snapshot)["today"] is None
        assert solar_status(snapshot, "today").reason == "not_archived"
        assert snapshot.solar_tab.last_24h.energy_kwh_m2 is None

    def test_a_whole_catalog_gives_the_archive_s_statuses(self, june_snapshot):
        archive = {s.key: s for s in june_snapshot.gaps.windows}
        for window in june_snapshot.solar_tab.windows:
            assert window.status == archive[window.key]
        months = {s.key: s for s in june_snapshot.gaps.months}
        for month in june_snapshot.solar_tab.months:
            assert month.status == months[month.key]

    def test_a_day_with_nothing_due_yet_needs_no_periods(self):
        # Ten minutes past midnight, the newest row 11:55 pm: today is counted
        # with nothing due, so the catalog owes it nothing.
        archive = build_archive(START, "2026-06-20T07:00:00Z")
        energy = catalog("2026-05-01", ms("2026-06-20T07:00:00Z"))
        snapshot = build(archive, energy, now="2026-06-20T07:10:00Z")
        assert day_value(snapshot, "2026-06-20") == 0.0
        assert day_reason(snapshot, "2026-06-20") is None
        assert windows(snapshot)["7d"] == pytest.approx(6 * 0.16)

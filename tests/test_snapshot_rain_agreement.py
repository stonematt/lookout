"""
Rain numbers that must agree across the snapshot (#122).

- Rain today belongs to today: just after local midnight, with no live
  reading newer than yesterday's last row, yesterday's total is not today's.
- One "live event" decision (`Snapshot.active_rain_event`) feeds the hero
  and the Rain Events tab.
- The Rain Events tab's water-year totals are events-only sums, not the
  water year's rain (`RainTab.water_years` holds that).
- A day whose rain counter has no reading is unknown for every rain window,
  as it is for dry spells.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from lookout.snapshot import days, gaps
from lookout.utils.dateutc import DEFAULT_TZ, from_local_date
from tests.fixtures.archive import Gap, RainSpell, build_archive
from tests.snapshot_helpers import (
    RAIN_LIVE,
    build,
    energy_catalog,
    event_catalog,
    live_row,
    ms,
    raining_station,
    scenario_until,
)

# Jan 2026, local midnight is 08:00Z.
JAN_RAIN = [
    RainSpell("2026-01-02T18:00:00Z", 2, 0.1),  # Jan 2: 0.2 in
    RainSpell("2026-01-05T18:00:00Z", 3, 0.1),  # Jan 5: 0.3 in
    RainSpell("2026-01-09T18:00:00Z", 1, 0.25),  # Jan 9: 0.25 in
]


def approx(value: float):
    """Rain totals to a thousandth of an inch."""
    return pytest.approx(value, abs=1e-3)


def windows(snapshot):
    """The Rain tab's window totals by key."""
    return {w.key: w.total_in for w in snapshot.rain_tab.windows}


def rain_window(snapshot, key):
    """One of the Rain tab's windows."""
    return next(w for w in snapshot.rain_tab.windows if w.key == key)


def rolling(snapshot, key):
    """The almanac's rolling-window row for this year."""
    row = next(r for r in snapshot.almanac.rolling if r.key == key)
    return row.years[0]


class TestRainTodayAtMidnight:
    """00:02 local on Jan 10; the newest row is 23:55 on Jan 9 (0.25 in)."""

    ARCHIVE_END = "2026-01-10T08:00:00Z"
    NOW = "2026-01-10T08:02:00Z"

    def snapshot(self, live=None):
        archive = build_archive("2025-12-01T08:00:00Z", self.ARCHIVE_END, rain=JAN_RAIN)
        return build(archive, live, self.NOW)

    def test_yesterdays_counter_is_not_todays_rain(self):
        snapshot = self.snapshot()

        assert snapshot.current.rain_today_in is None

    def test_yesterdays_rain_is_counted_once_in_every_window(self):
        snapshot = self.snapshot()
        totals = windows(snapshot)

        assert totals["today"] == 0.0
        assert totals["yesterday"] == approx(0.25)
        assert totals["7d"] == approx(0.55)  # not 0.80: Jan 9 once
        assert snapshot.rain_tab.recent.rain_in[-1] == 0.0
        assert rolling(snapshot, "7d").rain_in == approx(0.55)

    def test_a_live_reading_after_midnight_is_todays_rain(self):
        live = {"dateutc": ms("2026-01-10T08:01:00Z"), "dailyrainin": 0.0}

        snapshot = self.snapshot(live)

        assert snapshot.current.rain_today_in == 0.0
        assert windows(snapshot)["7d"] == approx(0.55)


def dry_live_snapshot():
    """Mid event 3, the catalog still open, but the live counter reads 0."""
    full = scenario_until("2026-01-09T00:30:00Z")
    archive = full[full["dateutc"] < ms("2026-01-09T00:00:00Z")]
    live = dict(live_row(full, RAIN_LIVE), eventrainin=0.0)
    return build(archive, live, "2026-01-09T00:27:00Z", event_catalog(archive))


class TestOneLiveDecision:
    def test_catalog_open_and_live_dry_is_not_live_anywhere(self):
        snapshot = dry_live_snapshot()
        rain_rule = next(r for r in snapshot.hero_lead.rules if r.kind == "rain")

        assert snapshot.active_rain_event is None
        assert rain_rule.value > 0  # the hero: the event has ended
        assert not any(row.live for row in snapshot.rain_events.events)

    def test_an_event_the_live_reading_has_ended_is_not_open(self):
        # The Events tab reads the same decision as the hero (#147): no
        # open curve rides in the snapshot for an event that is not live.
        # Its curve file follows once the catalog closes it.
        snapshot = dry_live_snapshot()

        assert snapshot.rain_events.open_curve is None

    def test_the_live_event_is_the_open_one(self):
        snapshot = raining_station()

        assert snapshot.rain_events.open_curve.start_ms == (
            snapshot.active_rain_event.start_ms
        )

    def test_the_live_row_is_the_active_event(self):
        snapshot = raining_station()
        live = [row for row in snapshot.rain_events.events if row.live]

        assert [row.start_ms for row in live] == [snapshot.active_rain_event.start_ms]

    def test_a_row_has_no_second_ongoing_flag(self):
        row = raining_station().rain_events.events[0]

        assert not hasattr(row, "ongoing")


class TestEventSeasonsAreEventSums:
    def test_a_season_total_is_its_events_added_not_the_water_years_rain(self):
        # A shower under the catalog's 0.01 in minimum is rain, not an event.
        archive = build_archive(
            "2025-12-01T08:00:00Z",
            "2026-01-10T08:00:00Z",
            rain=[*JAN_RAIN, RainSpell("2026-01-07T18:00:00Z", 1 / 6, 0.048)],
        )
        snapshot = build(archive, None, "2026-01-10T07:59:00Z", event_catalog(archive))
        events = snapshot.rain_events
        season = next(s for s in events.seasons if s.water_year == 2026)

        listed = sum(r.total_in for r in events.events if r.water_year == 2026)
        assert season.stats.total_in == approx(listed)
        assert windows(snapshot)["30d"] == approx(0.758)
        assert season.stats.total_in < windows(snapshot)["30d"]

    def test_a_season_carries_the_water_years_gap_status(self):
        # A two-week hole in December: the water year's rain cannot be
        # compared, and the events-only sum says so beside it (#147).
        archive = build_archive(
            "2025-10-01T07:00:00Z",
            "2026-01-10T08:00:00Z",
            rain=JAN_RAIN,
            gaps=[Gap("2025-12-01T08:00:00Z", "2025-12-15T08:00:00Z")],
        )
        snapshot = build(archive, None, "2026-01-10T07:59:00Z", event_catalog(archive))
        season = next(s for s in snapshot.rain_events.seasons if s.water_year == 2026)
        curve = next(c for c in snapshot.rain_tab.water_years if c.water_year == 2026)

        assert season.status == curve.status
        assert season.status.status == "invalid"
        assert season.stats.count == 3  # the listed events, still summed
        assert season.stats.total_in == approx(0.75)

    def test_a_whole_seasons_status_is_valid(self):
        archive = build_archive(
            "2025-10-01T07:00:00Z", "2026-01-10T08:00:00Z", rain=JAN_RAIN
        )
        snapshot = build(archive, None, "2026-01-10T07:59:00Z", event_catalog(archive))
        season = snapshot.rain_events.seasons[0]

        assert season.status.status == "valid"
        assert season.status == snapshot.rain_tab.water_years[-1].status

    def test_a_season_before_the_archive_still_has_a_status(self):
        # The catalog reaches back to a water year the archive does not: the
        # snapshot says so itself, so the browser never makes a status up
        # (#156).
        full = build_archive(
            "2025-09-20T07:00:00Z",
            "2026-01-10T08:00:00Z",
            rain=[RainSpell("2025-09-25T18:00:00Z", 2, 0.1), *JAN_RAIN],
        )
        archive = full[full["dateutc"] >= ms("2025-10-01T07:00:00Z")]
        snapshot = build(archive, None, "2026-01-10T07:59:00Z", event_catalog(full))
        season = next(s for s in snapshot.rain_events.seasons if s.water_year == 2025)

        assert season.status.key == "water_year_2025"
        assert (season.status.first_day, season.status.last_day) == (
            "2024-10-01",
            "2025-09-30",
        )
        assert season.status.days == season.status.missing_days == 365
        assert season.status.status == "invalid"
        assert season.status.reason == "before_archive"

    def test_every_seasons_status_is_keyed_like_its_curve(self):
        archive = build_archive(
            "2025-10-01T07:00:00Z", "2026-01-10T08:00:00Z", rain=JAN_RAIN
        )
        snapshot = build(archive, None, "2026-01-10T07:59:00Z", event_catalog(archive))

        curve = snapshot.rain_tab.water_years[-1]

        assert [s.status.key for s in snapshot.rain_events.seasons] == [
            "water_year_2026"
        ]
        assert snapshot.rain_events.seasons[0].status.key == curve.status.key


def nan_counter_archive():
    """Jan 5 has rows but no rain counter reading; it had 0.3 in."""
    archive = build_archive(
        "2025-12-01T08:00:00Z", "2026-01-10T08:00:00Z", rain=JAN_RAIN
    )
    day = (archive["dateutc"] >= ms("2026-01-05T08:00:00Z")) & (
        archive["dateutc"] < ms("2026-01-06T08:00:00Z")
    )
    archive.loc[day, "dailyrainin"] = np.nan
    return archive


class TestARainCounterWithNoReading:
    NOW = "2026-01-10T07:59:00Z"

    def test_the_day_is_missing_from_rain_windows(self):
        snapshot = build(nan_counter_archive(), None, self.NOW)

        assert rain_window(snapshot, "7d").status.status == "invalid"
        assert rain_window(snapshot, "7d").total_in is None
        assert rain_window(snapshot, "30d").status.status == "prorated"
        assert rain_window(snapshot, "30d").total_in == approx(0.45 * 30 / 29)

    def test_the_archive_gap_status_is_untouched(self):
        # The rows are there: temperature and the rest still count the day.
        snapshot = build(nan_counter_archive(), None, self.NOW)

        assert {w.key: w.status for w in snapshot.gaps.windows}["7d"] == "valid"

    def test_the_recent_days_say_why(self):
        recent = build(nan_counter_archive(), None, self.NOW).rain_tab.recent
        jan_5 = (date(2026, 1, 5) - date.fromisoformat(recent.first_day)).days

        assert recent.rain_in[jan_5] is None
        assert recent.reason[jan_5] == "gap"

    def test_the_almanac_agrees_with_the_rain_tab(self):
        snapshot = build(nan_counter_archive(), None, self.NOW)

        for key in ("7d", "30d", "90d"):
            year = rolling(snapshot, key)
            assert year.status == rain_window(snapshot, key).status
            assert year.rain_in == rain_window(snapshot, key).total_in

    def test_a_live_counter_reads_for_today(self):
        # Today (Jan 9) has rows but no counter; the live reading has one.
        archive = build_archive(
            "2025-12-01T08:00:00Z", "2026-01-10T07:55:00Z", rain=JAN_RAIN
        )
        today = archive["dateutc"] >= ms("2026-01-09T08:00:00Z")
        archive.loc[today, "dailyrainin"] = np.nan
        live = {"dateutc": ms("2026-01-10T07:58:00Z"), "dailyrainin": 0.25}

        snapshot = build(archive, live, self.NOW)

        assert rain_window(snapshot, "today").status.status == "valid"
        assert windows(snapshot)["today"] == approx(0.25)


def test_the_callers_archive_is_not_changed():
    """Building a snapshot leaves the caller's archive as it was."""
    archive = nan_counter_archive()
    before = archive.copy()

    build(archive, None, "2026-01-10T07:59:00Z")

    pd.testing.assert_frame_equal(archive, before)


# ---------- one count of the days, one daily rain series (#196) ----------

# A year to Mon 2026-01-05 23:00 local. Rain on Dec 1, Dec 11 and Dec 31:
# two dry spells, 9 and 19 days. Dec 20 has rows but no counter reading.
YEAR_START = "2025-01-01T08:00:00Z"
YEAR_END = "2026-01-06T07:00:00Z"
YEAR_NOW = "2026-01-06T07:02:00Z"
DECEMBER_RAIN = [
    RainSpell("2025-12-01T18:00:00Z", 1, 0.12),
    RainSpell("2025-12-11T18:00:00Z", 1, 0.12),
    RainSpell("2025-12-31T18:00:00Z", 1, 0.12),
]


def without_counter(archive, first: str, last: str):
    """`archive` with no rain counter reading from `first` to `last` local."""
    start = from_local_date(date.fromisoformat(first), DEFAULT_TZ)
    end = from_local_date(date.fromisoformat(last) + timedelta(days=1), DEFAULT_TZ)
    archive = archive.copy()
    span = (archive["dateutc"] >= start) & (archive["dateutc"] < end)
    archive.loc[span, "dailyrainin"] = np.nan
    return archive


@pytest.fixture(scope="module")
def december_year():
    """The year with Dec 20's counter unread."""
    archive = build_archive(YEAR_START, YEAR_END, rain=DECEMBER_RAIN)
    return without_counter(archive, "2025-12-20", "2025-12-20")


def calm(snapshot):
    """The calm lead rule."""
    return next(r for r in snapshot.hero_lead.rules if r.kind == "calm")


class TestTheLeadAndTheRainTabAgree:
    def test_the_calm_leads_spells_are_the_rain_tabs(self, december_year):
        snapshot = build(december_year, None, YEAR_NOW)
        longest = snapshot.rain_tab.dry_spells.longest

        # The 19-day spell holds Dec 20, so neither side counts it.
        assert (longest.first_day, longest.last_day, longest.days) == (
            "2025-12-02",
            "2025-12-10",
            9,
        )
        assert snapshot.hero_lead.kind == "calm"
        assert snapshot.hero_lead.sample == 1
        assert calm(snapshot).reference == longest.days

    def test_the_unread_day_hatches_rain_windows_only(self, december_year):
        snapshot = build(
            december_year, None, "2025-12-24T07:02:00Z"
        )  # Dec 23, 23:02 local: Dec 20 is in the week

        periods = {p.status.key: p.status for p in snapshot.temperature_context.periods}

        assert rain_window(snapshot, "7d").status.status == "invalid"
        assert {w.key: w.status for w in snapshot.gaps.windows}["7d"] == "valid"
        assert periods["7d"].status == "valid"

    def test_solar_ignores_the_unread_counter(self):
        archive = nan_counter_archive()
        snapshot = build(
            archive, None, "2026-01-10T07:59:00Z", energy=energy_catalog(archive)
        )
        solar = {w.key: w.status for w in snapshot.solar_tab.windows}

        assert rain_window(snapshot, "7d").status.status == "invalid"
        assert solar["7d"].status == "valid"

    def test_the_usual_dry_spell_is_unavailable_over_a_counter_outage(self):
        # Forty days unread: the 365-day window is invalid for rain, though
        # the archive holds every day.
        archive = without_counter(
            build_archive(YEAR_START, YEAR_END, rain=DECEMBER_RAIN),
            "2025-10-01",
            "2025-11-09",
        )

        snapshot = build(archive, None, YEAR_NOW)

        assert {w.key: w.status for w in snapshot.gaps.windows}["365d"] == "valid"
        assert rain_window(snapshot, "365d").status.status == "invalid"
        assert calm(snapshot).reference is None
        assert calm(snapshot).missing == "no_comparison"

    def test_the_same_archive_with_its_counter_has_the_usual_dry_spell(self):
        archive = build_archive(YEAR_START, YEAR_END, rain=DECEMBER_RAIN)

        snapshot = build(archive, None, YEAR_NOW)

        assert rain_window(snapshot, "365d").status.status == "valid"
        assert calm(snapshot).reference is not None
        assert calm(snapshot).missing is None


class TestRainEventsAgreeOnTheUnreadDay:
    def test_a_season_reads_the_rain_tabs_status(self):
        archive = build_archive(
            "2025-10-01T07:00:00Z", "2026-01-10T08:00:00Z", rain=JAN_RAIN
        )
        archive = without_counter(archive, "2025-12-20", "2025-12-20")

        snapshot = build(archive, None, "2026-01-10T07:59:00Z", event_catalog(archive))
        season = next(s for s in snapshot.rain_events.seasons if s.water_year == 2026)
        curve = next(c for c in snapshot.rain_tab.water_years if c.water_year == 2026)

        assert season.status == curve.status
        assert season.status.status == "prorated"
        assert season.status.reason == "gap"


def test_todays_rain_is_the_live_counter_everywhere():
    # The archive's newest row today reads 0.25 in; the live reading 0.4 in.
    archive = build_archive(
        "2025-12-01T08:00:00Z", "2026-01-10T07:55:00Z", rain=JAN_RAIN
    )
    live = {"dateutc": ms("2026-01-10T07:58:00Z"), "dailyrainin": 0.4}

    snapshot = build(archive, live, "2026-01-10T07:59:00Z")

    assert snapshot.current.rain_today_in == 0.4
    assert windows(snapshot)["today"] == approx(0.4)
    assert snapshot.rain_tab.recent.rain_in[-1] == approx(0.4)
    assert rolling(snapshot, "7d").rain_in == approx(0.7)  # Jan 5 and today


def test_an_archive_with_no_rain_counter_reads_rain_unknown():
    archive = build_archive(
        "2025-12-01T08:00:00Z", "2026-01-10T08:00:00Z", rain=JAN_RAIN
    ).drop(columns=["dailyrainin"])

    snapshot = build(archive, None, "2026-01-10T07:59:00Z")

    assert {w.key: w.status for w in snapshot.gaps.windows}["7d"] == "valid"
    assert all(w.total_in is None for w in snapshot.rain_tab.windows)
    assert all(w.status.status == "invalid" for w in snapshot.rain_tab.windows)
    assert set(snapshot.rain_tab.recent.rain_in) == {None}
    assert snapshot.rain_tab.dry_spells.longest is None
    assert rolling(snapshot, "7d").rain_in is None


def test_the_days_are_counted_once_per_build(monkeypatch):
    counts = {"coverage": 0, "rain": 0}
    real_init = gaps.ArchiveCoverage.__init__
    real_extract = days.extract_daily_rainfall

    def counting_init(self, *args, **kwargs):
        counts["coverage"] += 1
        real_init(self, *args, **kwargs)

    def counting_extract(*args, **kwargs):
        counts["rain"] += 1
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(gaps.ArchiveCoverage, "__init__", counting_init)
    monkeypatch.setattr(days, "extract_daily_rainfall", counting_extract)
    archive = nan_counter_archive()

    build(archive, None, "2026-01-10T07:59:00Z", event_catalog(archive))

    assert counts == {"coverage": 1, "rain": 1}

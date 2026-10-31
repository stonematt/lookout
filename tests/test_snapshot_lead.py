"""
The hero lead (#79), through `build_snapshot`: the Adaptive Lead Rule.

The rules are checked in order and the first that holds leads:

1. rain, if an event is live or ended in the last 12 hours;
2. temperature, if it is at least 10° from the month median (this hour's
   median over the same calendar month in earlier years);
3. solar, if it is at least 400 W/m²;
4. otherwise calm: the days since it last rained against the usual dry spell
   (the median completed dry spell over the last 365 days).

Every rule's live value is in the snapshot, met or not, for "Why this
leads". The month median and the usual dry spell are comparisons over months
and windows, so they go through the gap status: an invalid one is never
computed, and its rule says "no_comparison".

Each lead also has a small golden document, `tests/golden/leads/<kind>.json`,
the `hero_lead` section the TypeScript side renders. Regenerate with
`LOOKOUT_REGEN_GOLDEN=1 python -m pytest tests/test_snapshot_lead.py
-p no:cacheprovider`.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

from lookout.snapshot.model import _plain
from tests.fixtures.ambient import to_ambient_records
from tests.fixtures.archive import RainSpell, build_archive
from tests.test_snapshot import (
    REGENERATE,
    build,
    dry_station,
    event_catalog,
    ms,
    raining_station,
    scenario_until,
)

LEAD_GOLDEN_DIR = Path(__file__).parent / "golden" / "leads"
HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS


def rule(snapshot, kind: str):
    """The lead rule of one kind."""
    return next(r for r in snapshot.hero_lead.rules if r.kind == kind)


def reading(archive: pd.DataFrame, when: str, **values) -> Dict:
    """The archive row at `when` as a live `lastData`, with `values` set."""
    row = to_ambient_records(archive[archive["dateutc"] == ms(when)])[0]
    row.update(values)
    return row


def assert_lead_golden(snapshot, kind: str) -> None:
    """Compare the `hero_lead` section with `tests/golden/leads/<kind>.json`."""
    path = LEAD_GOLDEN_DIR / f"{kind}.json"
    text = json.dumps(_plain(asdict(snapshot.hero_lead)), indent=2) + "\n"
    if REGENERATE:
        LEAD_GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert json.loads(text) == json.loads(path.read_text())


# ---------- the lead scenarios ----------

# The SCENARIO week, before event 2: event 1 (Sun 10:00-13:00 local) has
# ended; Tue 2026-01-06 is dry. The archive runs to 11:55 local, and the live
# reading is at noon (20:00Z).
QUIET_NOW = "2026-01-06T20:00:00Z"
QUIET_FULL = scenario_until("2026-01-06T20:05:00Z")


def quiet_week() -> pd.DataFrame:
    """The SCENARIO station to Tuesday 11:55 local: no event for over a day."""
    return QUIET_FULL[QUIET_FULL["dateutc"] < ms(QUIET_NOW)].reset_index(drop=True)


def quiet_live(**values) -> Dict:
    """The live reading at Tuesday noon, with `values` set."""
    return reading(QUIET_FULL, QUIET_NOW, **values)


def with_last_january(current: pd.DataFrame, tempf: float = 50.0) -> pd.DataFrame:
    """
    `current` with a whole January 2025 before it, flat at `tempf`.

    So January's month median is `tempf` at every hour.
    """
    earlier = build_archive("2025-01-01T08:00:00Z", "2025-02-01T08:00:00Z")
    earlier["tempf"] = tempf
    return pd.concat([current, earlier], ignore_index=True)


def solar_station(solar: float = 400.0):
    """Tuesday noon, no earlier January, the live radiation at `solar`."""
    archive = quiet_week()
    live = quiet_live(solarradiation=solar)
    return build(archive, live, "2026-01-06T20:02:00Z", event_catalog(archive))


def temperature_station(tempf: float = 60.0):
    """Tuesday noon over a flat-50° January 2025, the live reading at `tempf`."""
    current = quiet_week()
    archive = with_last_january(current)
    live = quiet_live(tempf=tempf, solarradiation=0.0)
    return build(archive, live, "2026-01-06T20:02:00Z", event_catalog(current))


# A year of rain every fifth day, an hour from 10:00 local (18:00Z): every
# completed dry spell is four days long.
YEAR_START = "2025-01-01T08:00:00Z"
YEAR_END = "2026-01-06T07:00:00Z"  # Mon 2026-01-05 23:00 local
LAST_RAIN_DAY = "2026-01-03"


def rain_every_fifth_day() -> pd.DataFrame:
    """The year of rain every fifth day, the last on Sat 2026-01-03."""
    days = pd.date_range(end=LAST_RAIN_DAY, periods=73, freq="5D")
    spells = tuple(
        RainSpell(f"{day.date()}T18:00:00Z", hours=1, inches_per_hour=0.12)
        for day in days
    )
    assert days[-1] == pd.Timestamp(LAST_RAIN_DAY)
    return build_archive(YEAR_START, YEAR_END, rain=spells)


@pytest.fixture(scope="module")
def year_of_rain() -> pd.DataFrame:
    return rain_every_fifth_day()


def calm_station(archive: pd.DataFrame):
    """Mon 23:00 local, two and a half days after the last rain; no catalog."""
    return build(archive, None, "2026-01-06T07:02:00Z")


# ---------- the order of the rules ----------


class TestTheLeadChoice:
    def test_rain_leads_while_an_event_is_live(self):
        snapshot = raining_station()

        assert snapshot.hero_lead.kind == "rain"
        assert rule(snapshot, "rain").met is True
        assert rule(snapshot, "rain").value == 0.0
        assert_lead_golden(snapshot, "rain")

    def test_temperature_leads_ten_degrees_from_the_month_median(self):
        snapshot = temperature_station(60.0)

        assert snapshot.hero_lead.kind == "temperature"
        temperature = rule(snapshot, "temperature")
        assert temperature.reference == 50.0
        assert temperature.value == 10.0
        assert temperature.compared == ("2025-01",)
        assert_lead_golden(snapshot, "temperature")

    def test_solar_leads_at_400_wm2(self):
        snapshot = solar_station(400.0)

        assert snapshot.hero_lead.kind == "solar"
        assert rule(snapshot, "solar").value == 400.0
        assert_lead_golden(snapshot, "solar")

    def test_calm_leads_otherwise(self, year_of_rain):
        snapshot = calm_station(year_of_rain)

        assert snapshot.hero_lead.kind == "calm"
        calm = rule(snapshot, "calm")
        # Last rain at 10:55 local Saturday; now 23:00 local Monday.
        assert calm.value == pytest.approx(2.5, abs=0.01)
        assert calm.reference == 4.0
        assert calm.compared == ("365d",)
        assert_lead_golden(snapshot, "calm")

    def test_rain_comes_before_temperature(self):
        full = scenario_until("2026-01-09T00:30:00Z")
        archive = with_last_january(full[full["dateutc"] < ms("2026-01-09T00:00Z")])
        live = reading(full, "2026-01-09T00:25:00Z", tempf=80.0)

        snapshot = build(archive, live, "2026-01-09T00:27:00Z", event_catalog(full))

        assert rule(snapshot, "temperature").met is True
        assert snapshot.hero_lead.kind == "rain"

    def test_temperature_comes_before_solar(self):
        current = quiet_week()
        archive = with_last_january(current)
        live = quiet_live(tempf=30.0, solarradiation=600.0)

        snapshot = build(archive, live, "2026-01-06T20:02:00Z")

        assert rule(snapshot, "solar").met is True
        assert snapshot.hero_lead.kind == "temperature"
        assert rule(snapshot, "temperature").value == -20.0

    def test_every_rule_carries_its_live_value_met_or_not(self):
        snapshot = solar_station(400.0)

        rules = {r.kind: r for r in snapshot.hero_lead.rules}
        assert list(rules) == ["rain", "temperature", "solar", "calm"]
        assert [r.met for r in rules.values()] == [False, False, True, False]
        # Event 1's counter last reported rain at 02:50Z Monday, 41 h before.
        assert rules["rain"].value == pytest.approx(41.17, abs=0.01)
        assert rules["rain"].threshold == 12.0
        assert rules["temperature"].threshold == 10.0
        assert rules["solar"].threshold == 400.0
        # Event 1's last rain fell at 12:55 local Sunday.
        assert rules["calm"].value == pytest.approx(1.96, abs=0.01)


# ---------- the boundaries ----------


class TestBoundaries:
    @pytest.fixture(scope="class")
    def first_event(self):
        """Event 1 and when its event counter last reported rain."""
        archive = scenario_until("2026-01-05T20:00:00Z")
        catalog = event_catalog(archive)
        ended_ms = int(pd.Timestamp(catalog["end_time"].iloc[0]).value // 10**6)
        return archive, catalog, ended_ms

    @pytest.mark.parametrize(
        "after_hours, leads", [(12.0, "rain"), (12.0 + 5 / 60, "calm")]
    )
    def test_rain_leads_until_exactly_12_hours_after_the_event(
        self, first_event, after_hours, leads
    ):
        archive, catalog, ended_ms = first_event
        observed_ms = ended_ms + int(after_hours * HOUR_MS)
        cut = archive[archive["dateutc"] <= observed_ms]
        now = pd.Timestamp(observed_ms + 60_000, unit="ms", tz="UTC").isoformat()

        snapshot = build(cut, None, now, catalog)

        assert rule(snapshot, "rain").value == pytest.approx(after_hours, abs=0.01)
        assert snapshot.hero_lead.kind == leads

    @pytest.mark.parametrize(
        "tempf, leads",
        [(60.0, "temperature"), (59.9, "calm"), (40.0, "temperature"), (40.1, "calm")],
    )
    def test_temperature_leads_from_exactly_10_degrees(self, tempf, leads):
        assert temperature_station(tempf).hero_lead.kind == leads

    @pytest.mark.parametrize("solar, leads", [(400.0, "solar"), (399.9, "calm")])
    def test_solar_leads_from_exactly_400_wm2(self, solar, leads):
        assert solar_station(solar).hero_lead.kind == leads


# ---------- invalid comparisons ----------


class TestInvalidComparisons:
    def test_temperature_cannot_lead_without_a_comparable_month(self):
        archive = quiet_week()
        live = quiet_live(tempf=90.0, solarradiation=0.0)

        snapshot = build(archive, live, "2026-01-06T20:02:00Z")

        temperature = rule(snapshot, "temperature")
        assert temperature.met is False
        assert temperature.value is None
        assert temperature.reference is None
        assert temperature.compared == ()
        assert temperature.missing == "no_comparison"
        assert snapshot.hero_lead.kind == "calm"

    def test_a_month_with_a_gap_is_not_compared(self):
        current = quiet_week()
        archive = with_last_january(current)
        # Drop five days of January 2025: over 10% missing, so invalid.
        hole = (archive["dateutc"] >= ms("2025-01-10T08:00:00Z")) & (
            archive["dateutc"] < ms("2025-01-15T08:00:00Z")
        )
        live = quiet_live(tempf=90.0, solarradiation=0.0)

        snapshot = build(archive[~hole], live, "2026-01-06T20:02:00Z")

        month = next(m for m in snapshot.gaps.months if m.key == "2025-01")
        assert month.status == "invalid"
        assert rule(snapshot, "temperature").missing == "no_comparison"
        assert snapshot.hero_lead.kind == "calm"

    def test_calm_still_leads_without_a_usual_dry_spell(self):
        snapshot = solar_station(0.0)

        calm = rule(snapshot, "calm")
        window = next(w for w in snapshot.gaps.windows if w.key == "365d")
        assert window.status == "invalid"
        assert snapshot.hero_lead.kind == "calm"
        assert calm.value is not None
        assert calm.reference is None
        assert calm.compared == ()
        assert calm.missing == "no_comparison"

    def test_a_gap_breaks_a_dry_spell_rather_than_lengthening_it(self, year_of_rain):
        # A day-long gap in one spell: the window is prorated (1 of 365 days)
        # and that spell is not counted, not read as longer; the rest are
        # still four days.
        whole = calm_station(year_of_rain).hero_lead.sample
        hole = (year_of_rain["dateutc"] >= ms("2025-06-01T07:00:00Z")) & (
            year_of_rain["dateutc"] < ms("2025-06-02T07:00:00Z")
        )

        snapshot = calm_station(year_of_rain[~hole])

        window = next(w for w in snapshot.gaps.windows if w.key == "365d")
        assert window.status == "prorated"
        assert rule(snapshot, "calm").reference == 4.0
        assert snapshot.hero_lead.sample == whole - 1


# ---------- what the hero chart is drawn from ----------


class TestTheHeroChart:
    def test_the_rain_chart_is_the_event_so_far_over_past_event_totals(self):
        snapshot = raining_station()
        lead = snapshot.hero_lead

        assert lead.unit == "in"
        assert lead.now == snapshot.active_rain_event.total_in
        assert lead.now_ms == snapshot.observed_at_ms
        # Events 1 and 2 are the comparison set: 0.36" and 0.48".
        assert lead.sample == 2
        assert (lead.usual_low, lead.usual_median, lead.usual_high) == (
            0.39,
            0.42,
            0.45,
        )
        trace = lead.trace
        assert trace.slot_minutes == 15
        assert trace.start_ms == ms("2026-01-08T22:00:00Z")
        assert trace.usual_median == ()
        # Archived to 23:55Z, then only the live total at 00:25Z.
        assert trace.values[3] == 0.12  # 22:45Z-23:00Z
        assert trace.values[8] is None  # 00:00Z-00:15Z, not archived
        assert trace.values[-1] == lead.now

    def test_the_temperature_chart_is_today_over_the_months_hours(self):
        snapshot = temperature_station(60.0)
        lead = snapshot.hero_lead
        trace = lead.trace

        assert lead.unit == "f"
        assert lead.now == 60.0
        assert lead.sample == 31
        assert (lead.usual_low, lead.usual_median, lead.usual_high) == (50, 50, 50)
        assert trace.slot_minutes == 60
        assert trace.start_ms == ms("2026-01-06T08:00:00Z")  # local midnight
        assert len(trace.values) == 24
        assert trace.usual_median == (50.0,) * 24
        assert trace.values[13] is None  # 1 pm, still to come
        assert trace.values[11] is not None

    def test_the_calm_chart_is_the_spell_against_usual_spells(self, year_of_rain):
        lead = calm_station(year_of_rain).hero_lead

        assert lead.unit == "days"
        assert lead.trace is None
        assert lead.now == pytest.approx(2.5, abs=0.01)
        assert lead.usual_median == 4.0
        assert lead.sample > 60


class TestTheGoldenScenarios:
    def test_the_dry_golden_is_calm_and_the_raining_golden_is_rain(self):
        assert dry_station().hero_lead.kind == "calm"
        assert raining_station().hero_lead.kind == "rain"


def test_a_snapshot_with_no_history_is_calm_with_nothing_known():
    empty = build_archive("2026-01-04T08:00:00Z", "2026-01-04T08:00:00Z")
    assert empty.empty

    snapshot = build(empty, None, "2026-01-06T20:02:00Z")

    assert snapshot.hero_lead.kind == "calm"
    assert all(r.value is None for r in snapshot.hero_lead.rules)
    assert rule(snapshot, "rain").missing == "no_event"

"""
Pins the shared synthetic archive builder in `tests/fixtures/archive.py`.

The other gate tests lean on these properties, so they are checked here once:
the archive's shape, the rain counters' resets, the gap and the carryover.
Expected values are worked by hand from the spell rates, not recomputed.
"""

import pandas as pd
import pytest

from tests.fixtures.archive import (
    FAKE_MAC,
    SCENARIO,
    STATION_TZ,
    Gap,
    RainSpell,
    build_archive,
    scenario_archive,
)


def _at(df: pd.DataFrame, local: str) -> pd.Series:
    """Return the row stamped at a station-local wall time."""
    ts = pd.Timestamp(local, tz=STATION_TZ)
    rows = df[df["date"] == ts]
    assert len(rows) == 1, f"no single row at {local}"
    return rows.iloc[0]


def test_archive_shape_matches_the_storj_archive():
    df = build_archive("2026-01-05T08:00:00Z", "2026-01-06T08:00:00Z")

    assert len(df) == 288
    assert df["dateutc"].dtype == "int64"
    assert str(df["date"].dtype) == f"datetime64[ns, {STATION_TZ}]"
    assert df["dateutc"].is_monotonic_decreasing  # newest first, like Storj
    assert df["dateutc"].iloc[0] == 1767686100000  # 2026-01-06 07:55Z
    assert df["dateutc"].iloc[-1] == 1767600000000  # 2026-01-05 08:00Z
    for col in (
        "tempf",
        "humidity",
        "solarradiation",
        "hourlyrainin",
        "eventrainin",
        "dailyrainin",
        "weeklyrainin",
        "monthlyrainin",
        "yearlyrainin",
        "lastRain",
    ):
        assert col in df.columns


def test_oldest_first_on_request():
    df = build_archive(
        "2026-01-05T08:00:00Z", "2026-01-05T09:00:00Z", newest_first=False
    )
    assert df["dateutc"].is_monotonic_increasing
    assert len(df) == 12


def test_overlapping_builds_agree_row_for_row():
    spell = (RainSpell("2026-01-05T18:00:00Z", hours=2, inches_per_hour=0.12),)
    wide = build_archive("2026-01-05T08:00:00Z", "2026-01-06T08:00:00Z", rain=spell)
    narrow = build_archive("2026-01-05T08:00:00Z", "2026-01-05T22:00:00Z", rain=spell)
    joined = wide.merge(narrow, on="dateutc", suffixes=("_w", "_n"))
    assert len(joined) == len(narrow)
    assert (joined["dailyrainin_w"] == joined["dailyrainin_n"]).all()
    assert (joined["tempf_w"] == joined["tempf_n"]).all()


def test_rain_spell_fills_the_counters():
    # 0.12 in/hr for 2h from 10:00 local: 24 steps of 0.01".
    df = build_archive(
        "2026-01-05T08:00:00Z",
        "2026-01-06T08:00:00Z",
        rain=(RainSpell("2026-01-05T18:00:00Z", hours=2, inches_per_hour=0.12),),
    )
    before = _at(df, "2026-01-05 09:55")
    assert before["dailyrainin"] == 0
    assert before["eventrainin"] == 0
    assert pd.isna(before["lastRain"])

    mid = _at(df, "2026-01-05 11:00")
    assert mid["hourlyrainin"] == pytest.approx(0.12)
    assert mid["dailyrainin"] == pytest.approx(0.13)

    after = _at(df, "2026-01-05 13:00")
    assert after["dailyrainin"] == pytest.approx(0.24)
    assert after["eventrainin"] == pytest.approx(0.24)
    assert after["weeklyrainin"] == pytest.approx(0.24)
    assert after["monthlyrainin"] == pytest.approx(0.24)
    assert after["yearlyrainin"] == pytest.approx(0.24)
    assert after["hourlyrainin"] == 0
    assert after["lastRain"] == pd.Timestamp("2026-01-05 11:55", tz=STATION_TZ)


def test_daily_counter_resets_at_local_midnight_and_weekly_keeps_going():
    df = build_archive(
        "2026-01-05T08:00:00Z",
        "2026-01-07T08:00:00Z",
        rain=(
            RainSpell("2026-01-05T18:00:00Z", hours=1, inches_per_hour=0.12),
            RainSpell("2026-01-06T18:00:00Z", hours=1, inches_per_hour=0.12),
        ),
    )
    assert _at(df, "2026-01-05 23:55")["dailyrainin"] == pytest.approx(0.12)
    assert _at(df, "2026-01-06 00:00")["dailyrainin"] == 0
    assert _at(df, "2026-01-06 23:55")["dailyrainin"] == pytest.approx(0.12)
    assert _at(df, "2026-01-06 23:55")["weeklyrainin"] == pytest.approx(0.24)


def test_event_counter_resets_after_the_dry_spell():
    df = build_archive(
        "2026-01-05T08:00:00Z",
        "2026-01-06T08:00:00Z",
        rain=(RainSpell("2026-01-05T18:00:00Z", hours=1, inches_per_hour=0.12),),
        event_reset_hours=6,
    )
    # Last rain step 10:55 local; the event holds six hours, then resets.
    assert _at(df, "2026-01-05 16:50")["eventrainin"] == pytest.approx(0.12)
    assert _at(df, "2026-01-05 16:55")["eventrainin"] == 0


def test_gap_drops_rows_but_the_station_kept_counting():
    df = build_archive(
        "2026-01-05T08:00:00Z",
        "2026-01-06T08:00:00Z",
        rain=(RainSpell("2026-01-05T18:00:00Z", hours=2, inches_per_hour=0.12),),
        gaps=(Gap("2026-01-05T18:30:00Z", "2026-01-05T19:30:00Z"),),
    )
    assert len(df) == 288 - 12
    stamps = pd.to_datetime(df["dateutc"], unit="ms", utc=True)
    assert not (
        (stamps >= "2026-01-05T18:30:00Z") & (stamps < "2026-01-05T19:30:00Z")
    ).any()
    # Rain that fell in the gap still shows up in the counter afterwards.
    assert _at(df, "2026-01-05 13:00")["dailyrainin"] == pytest.approx(0.24)


def test_carryover_gap_keeps_the_daily_counter_through_midnight():
    # 0.36" falls on the 5th; the station is offline across midnight and its
    # daily counter fails to reset, so the 6th starts at 0.36".
    df = build_archive(
        "2026-01-05T08:00:00Z",
        "2026-01-07T08:00:00Z",
        rain=(RainSpell("2026-01-05T18:00:00Z", hours=3, inches_per_hour=0.12),),
        gaps=(Gap("2026-01-06T07:00:00Z", "2026-01-06T09:00:00Z", carryover=True),),
    )
    day6 = df[df["date"].dt.date == pd.Timestamp("2026-01-06").date()]
    first = day6.sort_values("dateutc").iloc[0]
    assert first["date"] == pd.Timestamp("2026-01-06 01:00", tz=STATION_TZ)
    assert first["dailyrainin"] == pytest.approx(0.36)
    assert _at(df, "2026-01-06 23:55")["dailyrainin"] == pytest.approx(0.36)
    # The inflated total runs on into the weekly, monthly and yearly counters.
    assert _at(df, "2026-01-06 23:55")["weeklyrainin"] == pytest.approx(0.72)
    assert _at(df, "2026-01-06 23:55")["monthlyrainin"] == pytest.approx(0.72)


def test_carryover_gap_must_cross_local_midnight():
    with pytest.raises(ValueError, match="midnight"):
        build_archive(
            "2026-01-05T08:00:00Z",
            "2026-01-06T08:00:00Z",
            gaps=(Gap("2026-01-05T18:00:00Z", "2026-01-05T19:00:00Z", carryover=True),),
        )


def test_scenario_has_the_named_shapes():
    df = scenario_archive()
    stamps = pd.to_datetime(df["dateutc"], unit="ms", utc=True)

    assert stamps.min() == pd.Timestamp(SCENARIO.start)
    # Three separate rain events: eventrainin returns to zero between them.
    ordered = df.sort_values("dateutc")
    starts = (ordered["eventrainin"] > 0) & (ordered["eventrainin"].shift() == 0)
    assert starts.sum() == 3
    # Each gap is missing from the rows.
    for gap in SCENARIO.gaps:
        inside = (stamps >= pd.Timestamp(gap.start)) & (stamps < pd.Timestamp(gap.end))
        assert not inside.any()
    assert any(g.carryover for g in SCENARIO.gaps)
    assert any(not g.carryover for g in SCENARIO.gaps)


def test_fake_mac_is_not_a_real_station():
    assert FAKE_MAC == "00:11:22:33:44:55"

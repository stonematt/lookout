"""
archive.py: Build synthetic station archives for tests.

`build_archive(start, end, rain=..., gaps=...)` simulates a station reporting
every five minutes and returns a DataFrame shaped like the Storj archive:
`dateutc` int64 epoch ms, `date` tz-aware station-local, newest row first, and
the Ambient rain counters derived from the spells you describe.

Counter rules (the builder's own, chosen to be easy to work by hand):

- A row at time t receives `inches_per_hour * interval / 60` for every spell
  with `start <= t < start + hours`.
- `hourlyrainin` is the rain in the 60 minutes ending at t (rows in (t-1h, t]).
- `dailyrainin`, `weeklyrainin`, `monthlyrainin`, `yearlyrainin` accumulate
  and reset at station-local midnight, Sunday, the 1st, and 1 January.
- `eventrainin` accumulates while it rains and returns to 0 once the station
  has been dry for `event_reset_hours` (Ambient's own rule varies).
- `lastRain` is the local time of the last row that received rain.
- The station keeps counting through a `Gap`; the gap's rows are just absent.
- A `Gap(carryover=True)` must cross local midnight. The daily counter fails
  to reset there: the previous day's total is added to that day's daily
  counter and to the weekly, monthly and yearly counters until each one next
  resets. This mirrors the model `lookout.core.rain_corrections` corrects.

Every other column is a deterministic function of the timestamp, so two builds
over overlapping ranges with the same spells agree row for row.

`scenario_archive()` builds `SCENARIO`: a week with three rain events (one of
two spells), a carryover gap and a plain gap.
"""

from dataclasses import dataclass, field
from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd

FAKE_MAC = "00:11:22:33:44:55"
STATION_TZ = "America/Los_Angeles"

TimeLike = Union[str, pd.Timestamp]


@dataclass(frozen=True)
class RainSpell:
    """
    Steady rain from `start` for `hours` at `inches_per_hour`.

    :param start: ISO-8601 time; a naive time is read as UTC.
    :param hours: Duration of the spell.
    :param inches_per_hour: Rain rate while the spell lasts.
    """

    start: TimeLike
    hours: float
    inches_per_hour: float


@dataclass(frozen=True)
class Gap:
    """
    A stretch with no rows: those with `start <= t < end` are dropped.

    :param start: ISO-8601 time; a naive time is read as UTC.
    :param end: ISO-8601 time, exclusive.
    :param carryover: The daily counter fails to reset at the local midnight
        inside the gap.
    """

    start: TimeLike
    end: TimeLike
    carryover: bool = False


def _utc(value: TimeLike) -> pd.Timestamp:
    """Read a time as a UTC Timestamp; naive input is taken as UTC."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _weather(local: pd.DatetimeIndex) -> dict:
    """Deterministic non-rain columns, each a function of local time of day."""
    hour = local.hour + local.minute / 60.0
    diurnal = np.sin(2 * np.pi * (hour - 9) / 24)
    sun = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    tempf = np.round(50 + 10 * diurnal, 1)
    solar = np.round(800 * sun, 2)
    return {
        "tempinf": np.full(len(local), 68.0),
        "battin": np.full(len(local), 1.0),
        "humidityin": np.full(len(local), 45.0),
        "baromrelin": np.full(len(local), 30.0),
        "baromabsin": np.full(len(local), 29.9),
        "tempf": tempf,
        "battout": np.full(len(local), 1.0),
        "humidity": np.round(80 - 15 * diurnal, 0),
        "winddir": np.full(len(local), 180.0),
        "windspeedmph": np.full(len(local), 3.0),
        "windgustmph": np.full(len(local), 6.0),
        "solarradiation": solar,
        "uv": np.floor(solar / 100),
        "feelsLike": tempf,
        "dewPoint": np.round(tempf - 5, 1),
    }


def _carryover_offsets(
    local: pd.DatetimeIndex, daily: pd.Series, gaps: Sequence[Gap]
) -> dict:
    """Extra amount each counter carries for every carryover gap."""
    names = ("dailyrainin", "weeklyrainin", "monthlyrainin", "yearlyrainin")
    offsets = {name: np.zeros(len(local)) for name in names}
    day = local.normalize()
    for gap in gaps:
        if not gap.carryover:
            continue
        g0 = _utc(gap.start).tz_convert(STATION_TZ)
        g1 = _utc(gap.end).tz_convert(STATION_TZ)
        midnight = g1.normalize()
        if not g0 < midnight <= g1 - pd.Timedelta(microseconds=1):
            raise ValueError(f"Carryover gap {gap} must cross local midnight")
        before = local < midnight
        if not before.any():
            raise ValueError(f"Carryover gap {gap} starts before the archive")
        amount = float(daily[before].iloc[-1])
        after = local >= midnight
        offsets["dailyrainin"] += np.where(after & (day == midnight), amount, 0.0)
        for key, freq in (
            ("weeklyrainin", "W-SAT"),
            ("monthlyrainin", "M"),
            ("yearlyrainin", "Y"),
        ):
            period = local.tz_localize(None).to_period(freq)
            target = midnight.tz_localize(None).to_period(freq)
            offsets[key] += np.where(after & (period == target), amount, 0.0)
    return offsets


def _event_counter(
    utc: pd.DatetimeIndex, fallen: np.ndarray, reset_hours: float
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Event total per row, and the time of the last wet row (NaT before any)."""
    reset = pd.Timedelta(hours=reset_hours)
    event = np.zeros(len(utc))
    last = [pd.NaT] * len(utc)
    total, last_wet = 0.0, None
    for i, ts in enumerate(utc):
        if fallen[i] > 0:
            total += fallen[i]
            last_wet = ts
        elif last_wet is not None and ts - last_wet >= reset:
            total = 0.0
        event[i] = total
        last[i] = last_wet if last_wet is not None else pd.NaT
    return event, pd.DatetimeIndex(last, tz="UTC")


def build_archive(
    start: TimeLike,
    end: TimeLike,
    *,
    rain: Sequence[RainSpell] = (),
    gaps: Sequence[Gap] = (),
    interval_minutes: int = 5,
    event_reset_hours: float = 6.0,
    newest_first: bool = True,
) -> pd.DataFrame:
    """
    Simulate a station from `start` (inclusive) to `end` (exclusive).

    :param start: First row's time; a naive time is read as UTC.
    :param end: Time after the last row, exclusive.
    :param rain: Rain spells; overlapping spells add.
    :param gaps: Stretches with no rows; see `Gap`.
    :param interval_minutes: Reporting interval.
    :param event_reset_hours: Dry hours after which `eventrainin` returns to 0.
    :param newest_first: Sort newest row first, as the Storj archive is.
    :return: The archive DataFrame, index reset.
    """
    step = pd.Timedelta(minutes=interval_minutes)
    utc = pd.date_range(_utc(start), _utc(end) - step, freq=step)
    local = utc.tz_convert(STATION_TZ)

    fallen = np.zeros(len(utc))
    for spell in rain:
        s0 = _utc(spell.start)
        s1 = s0 + pd.Timedelta(hours=spell.hours)
        active = (utc >= s0) & (utc < s1)
        fallen += np.where(active, spell.inches_per_hour * interval_minutes / 60, 0)
    rain_series = pd.Series(fallen, index=utc)

    def _since_reset(key: np.ndarray) -> pd.Series:
        return rain_series.groupby(key).cumsum()

    naive = local.tz_localize(None)
    daily = _since_reset(naive.normalize().values)
    weekly = _since_reset(naive.to_period("W-SAT").astype(str).values)
    monthly = _since_reset(naive.to_period("M").astype(str).values)
    yearly = _since_reset(naive.year.values)

    hourly = rain_series.rolling(pd.Timedelta(hours=1), closed="right").sum()

    event, last_rain = _event_counter(utc, fallen, event_reset_hours)

    carried = _carryover_offsets(local, daily, gaps)

    df = pd.DataFrame(
        {
            "dateutc": (utc.asi8 // 1_000_000).astype("int64"),
            **_weather(local),
            "hourlyrainin": np.round(hourly.values, 3),
            "eventrainin": np.round(event, 3),
            "dailyrainin": np.round(daily.values + carried["dailyrainin"], 3),
            "weeklyrainin": np.round(weekly.values + carried["weeklyrainin"], 3),
            "monthlyrainin": np.round(monthly.values + carried["monthlyrainin"], 3),
            "yearlyrainin": np.round(yearly.values + carried["yearlyrainin"], 3),
            "lastRain": last_rain.tz_convert(STATION_TZ),
            "date": local,
        }
    )

    for gap in gaps:
        inside = (utc >= _utc(gap.start)) & (utc < _utc(gap.end))
        df = df[~inside[df.index]]

    df = df.sort_values("dateutc", ascending=not newest_first)
    return df.reset_index(drop=True)


@dataclass(frozen=True)
class Scenario:
    """The inputs `scenario_archive` builds from, for tests to refer to."""

    start: str
    end: str
    rain: Tuple[RainSpell, ...] = field(default_factory=tuple)
    gaps: Tuple[Gap, ...] = field(default_factory=tuple)


SCENARIO = Scenario(
    # Sunday 2026-01-04 00:00 local to Sunday 2026-01-11 00:00 local (PST).
    start="2026-01-04T08:00:00Z",
    end="2026-01-11T08:00:00Z",
    rain=(
        # Event 1: Sun 10:00-13:00 local, 0.36".
        RainSpell("2026-01-04T18:00:00Z", hours=3, inches_per_hour=0.12),
        # Event 2: Tue 20:00 to Wed 02:00 local, 0.36", across midnight ...
        RainSpell("2026-01-07T04:00:00Z", hours=6, inches_per_hour=0.06),
        # ... then 0.12" more at Wed 05:00 local, dry 3h so the same event.
        RainSpell("2026-01-07T13:00:00Z", hours=1, inches_per_hour=0.12),
        # Event 3: Thu 14:00-18:00 local, 0.48", carried over into Friday.
        RainSpell("2026-01-08T22:00:00Z", hours=4, inches_per_hour=0.12),
    ),
    gaps=(
        # Thu 23:00 to Fri 01:00 local: the daily counter does not reset.
        Gap("2026-01-09T07:00:00Z", "2026-01-09T09:00:00Z", carryover=True),
        # Sat 10:00-14:00 local: an ordinary outage.
        Gap("2026-01-10T18:00:00Z", "2026-01-10T22:00:00Z"),
    ),
)


def scenario_archive(newest_first: bool = True) -> pd.DataFrame:
    """
    Build the shared `SCENARIO` week: three rain events, a carryover, a gap.

    :param newest_first: Sort newest row first, as the Storj archive is.
    :return: The archive DataFrame.
    """
    return build_archive(
        SCENARIO.start,
        SCENARIO.end,
        rain=SCENARIO.rain,
        gaps=SCENARIO.gaps,
        newest_first=newest_first,
    )

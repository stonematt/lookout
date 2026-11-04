"""
sun_moon.py: The snapshot's sun and moon block, for the station-local day.

The sun comes from astral (sunrise and sunset at the station's coordinates,
on its local day in its zone). The moon does not: astral's `moon.phase`
adds a fixed 6.43 degrees and truncates the elongation to whole degrees,
which puts it about half a day ahead of the true phase. The moon here is
computed from the Sun-Moon elongation, with the formula in `elongation_deg`.

Offers:

- `sun_moon(station, clock)`: the `SunMoon` block, or None when the station
  has no coordinates or the sun does not rise or set that day.
- `moon_on(day, tz)`: the moon for one local day (`Moon`).
- `elongation_deg(when)`: the Moon's elongation from the Sun, 0 to 360.
"""

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

from astral import Observer
from astral.sun import sunrise, sunset

from lookout.snapshot.clock import StationClock
from lookout.snapshot.model import MoonPhase, Station, StationLocation, SunMoon
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

# The mean synodic month, new moon to new moon, in days.
SYNODIC_DAYS = 29.530588853

# The principal phases, by the elongation (degrees) at which each happens.
_PRINCIPAL: Tuple[Tuple[float, MoonPhase], ...] = (
    (0.0, "new"),
    (90.0, "first quarter"),
    (180.0, "full"),
    (270.0, "last quarter"),
)


@dataclass(frozen=True)
class Moon:
    """
    The moon on one station-local day.

    :param phase: The named phase.
    :param illumination_pct: The lit share of the disc at local noon, 0-100.
    :param days_to_full: Local days to the day of the next full moon; 0 on
        the day of a full moon.
    """

    phase: MoonPhase
    illumination_pct: int
    days_to_full: int


def elongation_deg(when: datetime) -> float:
    """
    The Moon's elongation from the Sun at `when`: 0 new, 90 first quarter,
    180 full, 270 last quarter.

    Meeus, Astronomical Algorithms (2nd ed.), eq. 47.2-47.4 for the mean
    elongation D, the Sun's mean anomaly M and the Moon's mean anomaly M',
    with T in Julian centuries from J2000:

        E = D + 6.289 sin M' - 2.100 sin M + 1.274 sin(2D - M')
              + 0.658 sin 2D + 0.214 sin 2M' + 0.110 sin D

    (the terms of eq. 48.4, where the phase angle is i = 180 - E). Good to
    about a degree, two hours of the Moon's motion; TT - UT is ignored.

    :param when: A timezone-aware time.
    :return: E, in degrees, 0 <= E < 360.
    """
    jd = when.timestamp() / 86400.0 + 2440587.5
    t = (jd - 2451545.0) / 36525.0
    d = 297.8501921 + 445267.1114034 * t - 0.0018819 * t * t
    m = 357.5291092 + 35999.0502909 * t
    m1 = 134.9633964 + 477198.8675055 * t + 0.0087414 * t * t
    d_r, m_r, m1_r = (math.radians(x % 360.0) for x in (d, m, m1))
    e = (
        d
        + 6.289 * math.sin(m1_r)
        - 2.100 * math.sin(m_r)
        + 1.274 * math.sin(2 * d_r - m1_r)
        + 0.658 * math.sin(2 * d_r)
        + 0.214 * math.sin(2 * m1_r)
        + 0.110 * math.sin(d_r)
    )
    return e % 360.0


def _illumination_pct(elongation: float) -> int:
    """
    The lit share of the disc: k = (1 + cos i) / 2 with i = 180 - E, which
    is k = (1 - cos E) / 2 (Meeus eq. 48.1).

    :param elongation: E, in degrees.
    :return: k as a whole percent.
    """
    return round(100 * (1 - math.cos(math.radians(elongation))) / 2)


def _ahead(start: float, target: float) -> float:
    """Degrees the Moon moves from elongation `start` to reach `target`."""
    return (target - start) % 360.0


def _next_full(after: datetime) -> datetime:
    """
    The next full moon (E = 180) at or after `after`.

    A mean-rate step, then two corrections at the true elongation.

    :param after: A timezone-aware time.
    :return: The full moon, to within the formula's two hours.
    """
    when = after + timedelta(
        days=_ahead(elongation_deg(after), 180.0) / 360.0 * SYNODIC_DAYS
    )
    for _ in range(2):
        miss = (180.0 - elongation_deg(when) + 180.0) % 360.0 - 180.0
        when += timedelta(days=miss / 360.0 * SYNODIC_DAYS)
    return when


def moon_on(day: date, tz: str = "UTC") -> Moon:
    """
    The moon on one local day.

    A principal phase (new, first quarter, full, last quarter) names the
    local day its instant falls on, as printed calendars do; the days
    between take the crescent or gibbous name of the elongation at noon.

    :param day: The station-local day.
    :param tz: The station's zone.
    :return: The moon.
    """
    zone = ZoneInfo(tz)
    start = datetime.combine(day, time(0), zone)
    end = datetime.combine(day + timedelta(days=1), time(0), zone)
    noon = datetime.combine(day, time(12), zone)
    e_start, e_end, e_noon = (elongation_deg(t) for t in (start, end, noon))
    span = _ahead(e_start, e_end)

    phase: Optional[MoonPhase] = None
    for target, name in _PRINCIPAL:
        if _ahead(e_start, target) < span:
            phase = name
    if phase is None:
        quadrant = int(e_noon // 90)
        phase = (
            "waxing crescent",
            "waxing gibbous",
            "waning gibbous",
            "waning crescent",
        )[quadrant]

    full_day = _next_full(start).astimezone(zone).date()
    return Moon(
        phase=phase,
        illumination_pct=_illumination_pct(e_noon),
        days_to_full=(full_day - day).days,
    )


def _sun_times(
    location: StationLocation, day: date, tz: str
) -> Tuple[datetime, datetime]:
    """
    Sunrise and sunset on one local day.

    :param location: The station's coordinates.
    :param day: The station-local day.
    :param tz: The station's zone.
    :return: Sunrise and sunset, timezone-aware.
    :raises ValueError: When the sun does not rise or set that day.
    """
    observer = Observer(latitude=location.lat, longitude=location.lon)
    zone = ZoneInfo(tz)
    return (
        sunrise(observer, date=day, tzinfo=zone),
        sunset(observer, date=day, tzinfo=zone),
    )


def _ms(when: datetime) -> int:
    """Epoch ms of a timezone-aware time."""
    return int(when.astimezone(timezone.utc).timestamp() * 1000)


def sun_moon(station: Station, clock: StationClock) -> Optional[SunMoon]:
    """
    The sun and moon for the station-local day of the build.

    :param station: The station; its coordinates are read from Ambient on
        every build.
    :param clock: The build's clock: its local day and zone.
    :return: The block, or None when the station has no coordinates, or the
        sun does not rise or set today or yesterday (polar day or night).
    """
    if station.location is None:
        return None
    today = clock.today
    try:
        rise, set_ = _sun_times(station.location, today, clock.tz)
        y_rise, y_set = _sun_times(
            station.location, today - timedelta(days=1), clock.tz
        )
    except ValueError as e:
        logger.info(f"No sunrise or sunset on {today}: {e}")
        return None
    daylight_s = (set_ - rise).total_seconds()
    yesterday_s = (y_set - y_rise).total_seconds()
    moon = moon_on(today, clock.tz)
    return SunMoon(
        sunrise_ms=_ms(rise),
        sunset_ms=_ms(set_),
        daylight_minutes=round(daylight_s / 60),
        daylight_change_minutes=round((daylight_s - yesterday_s) / 60),
        moon_phase=moon.phase,
        moon_illumination_pct=moon.illumination_pct,
        days_to_full_moon=moon.days_to_full,
    )

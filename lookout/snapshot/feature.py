"""
feature.py: The feature, picked by the Feature Rule (ADR-0009).

One rule decides what matters now; the Overview's feature tile shows it and
the widget view is derived from it (`lookout.snapshot.widget`). Checked in
order, the first that holds is the feature:

1. **rain**: an event is live (the build's `ActiveRainEvent`), or the
   newest Event Catalog event ended at most `RAIN_HOURS` before the build
   time (`StationClock.now_ms`). "Ended" is the catalog's `end_time`, as the
   lead's rain rule reads it (`lookout.snapshot.events.newest_event`).
   The feature names the event it holds on by `event_start_ms`, the key of
   its `rain_events` row: the live event when one is live, else that newest
   ended event, so the site never picks the event itself (ADR-0003).
2. **unusual**: a current reading is below the `LOW`th or above the `HIGH`th
   percentile of its readings at the newest observation's station-local
   hour, over the usual window for that date (`usual_window`, the ±7 days
   around it in every comparable earlier year). The readings and their
   archive columns are `READINGS`. A reading's percentile is its percentile
   rank: the share of the usual readings below it, a reading equal to it
   counting half, times 100 (so a gauge that never moves sits at 50). The
   reading furthest from the 50th wins; on an exact tie the first in
   `READINGS` wins. The rule is decided on the exact percentile; the one
   published is rounded to a whole number, but never into the band (at
   most `LOW` - 1 below it, at least `HIGH` + 1 above it). Skipped entirely unless the
   window's basis is "earlier_years": a recent-days band is too thin to call
   anything unusual. Skipped too unless the newest observation is fresh at
   the build time (`lookout.utils.dateutc.is_fresh` against
   `StationClock.now_ms`), so a silent station's last reading is never
   judged; the rule then falls through to dry spell, then quiet.
3. **dry_spell**: the snapshot's `DrySpellNow`, `days_since_rain` longer
   than `usual_days`; it cannot hold while either is None.
4. **quiet**: otherwise, temperature as a usual day.

Functions:
- feature(event_catalog, active, stitched, current, observed_at_ms, coverage,
  dry, clock): The `Feature` for one snapshot.
"""

from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from lookout.snapshot.clock import StationClock
from lookout.snapshot.current import to_ms
from lookout.snapshot.events import RAIN_HOURS, newest_event
from lookout.snapshot.gaps import ArchiveCoverage
from lookout.snapshot.model import (
    ActiveRainEvent,
    DrySpellNow,
    Feature,
    FeatureKind,
    FeatureReading,
    FeatureUnit,
    SnapshotConditions,
)
from lookout.snapshot.temperature import usual_readings, usual_window
from lookout.utils.dateutc import HOUR_MS, is_fresh, local_time

LOW = 10.0
"""A reading below this percentile of its usual is unusual."""

HIGH = 90.0
"""A reading above this percentile of its usual is unusual."""


class Judged(NamedTuple):
    """
    One reading the unusual rule judges.

    :param reading: Its name in the feature.
    :param column: Its archive column, for its usual readings.
    :param field: Its `SnapshotConditions` field, for its current value.
    :param unit: The unit of its value.
    """

    reading: FeatureReading
    column: str
    field: str
    unit: FeatureUnit


READINGS: Tuple[Judged, ...] = (
    Judged("temperature", "tempf", "temperature_f", "f"),
    Judged("humidity", "humidity", "humidity_pct", "pct"),
    Judged("wind_gust", "windgustmph", "wind_gust_mph", "mph"),
    Judged("pressure", "baromrelin", "barometer_inhg", "inhg"),
    Judged("solar", "solarradiation", "solar_wm2", "wm2"),
)
"""Every judged reading, in tie-break order: on an exact tie the first wins."""


def _bare(kind: FeatureKind, event_start_ms: Optional[int] = None) -> Feature:
    """
    A feature of `kind` without a reading: every rule but unusual.

    :param kind: The rule that holds.
    :param event_start_ms: For "rain", its event's `start_ms`.
    :return: The feature.
    """
    return Feature(
        kind=kind,
        reading=None,
        value=None,
        unit=None,
        percentile=None,
        event_start_ms=event_start_ms,
    )


def _rain(
    event_catalog: Optional[pd.DataFrame],
    active: Optional[ActiveRainEvent],
    clock: StationClock,
) -> Optional[int]:
    """
    The rain rule: an event is live, or the newest ended within `RAIN_HOURS`.

    :param event_catalog: The Event Catalog, or None.
    :param active: The active rain event, or None.
    :param clock: The build's clock; the hours are counted to its `now`.
    :return: The `start_ms` of the event the rule holds on (the live event,
        else the newest ended one), or None when it does not hold.
    """
    if active is not None:
        return active.start_ms
    newest = newest_event(event_catalog, clock.now_ms)
    if newest is None:
        return None
    if (clock.now_ms - to_ms(newest["end_time"])) / HOUR_MS > RAIN_HOURS:
        return None
    return to_ms(newest["start_time"])


def _percentile(value: float, usual: np.ndarray) -> float:
    """
    The percentile rank of `value` among `usual`: below, plus half of equal.

    :param value: The current reading.
    :param usual: Its usual readings, not empty.
    :return: 0 to 100.
    """
    below = int(np.count_nonzero(usual < value))
    equal = int(np.count_nonzero(usual == value))
    return 100.0 * (below + equal / 2) / len(usual)


def _published(rank: float) -> int:
    """
    An unusual rank as published: whole, and still outside the band.

    :param rank: The exact percentile rank, below `LOW` or above `HIGH`.
    :return: It rounded, kept below `LOW` or above `HIGH` (9.6 is 9, not 10).
    """
    whole = int(round(rank))
    if rank < LOW:
        return min(whole, int(LOW) - 1)
    return max(whole, int(HIGH) + 1)


def _unusual(
    stitched: pd.DataFrame,
    current: SnapshotConditions,
    observed_at_ms: Optional[int],
    coverage: ArchiveCoverage,
    clock: StationClock,
) -> Optional[Feature]:
    """
    The unusual rule: the reading furthest outside its usual, if any is.

    :param stitched: `stitch_live` output.
    :param current: Current conditions, live stitched.
    :param observed_at_ms: The newest observation, or None.
    :param coverage: The build's counted days.
    :param clock: The build's clock; the observation's age is counted to its
        `now`.
    :return: The feature, or None when no reading is unusual or the rule is
        skipped (no observation, a stale one, no earlier year).
    """
    if observed_at_ms is None or not is_fresh(observed_at_ms, clock.now_ms):
        return None
    observed = local_time(observed_at_ms, coverage.tz)
    window = usual_window(coverage, observed.date())
    if window.basis != "earlier_years":
        return None
    found: List[Tuple[float, Feature]] = []
    for judged in READINGS:
        value = getattr(current, judged.field)
        if value is None:
            continue
        usual = usual_readings(stitched, coverage.tz, window.days, judged.column)
        at_hour = usual[observed.hour]
        if len(at_hour) == 0:
            continue
        rank = _percentile(float(value), at_hour)
        if LOW <= rank <= HIGH:
            continue
        found.append(
            (
                abs(rank - 50.0),
                Feature(
                    kind="unusual",
                    reading=judged.reading,
                    value=value,
                    unit=judged.unit,
                    percentile=_published(rank),
                    event_start_ms=None,
                ),
            )
        )
    if not found:
        return None
    # `max` keeps the first of equals, so a tie goes to the earlier reading.
    return max(found, key=lambda item: item[0])[1]


def _dry_spell(dry: DrySpellNow) -> bool:
    """
    The dry spell rule: the current dry spell is longer than the usual one.

    :param dry: The snapshot's dry spell field.
    :return: True only when both values are known and the spell is longer.
    """
    if dry.days_since_rain is None or dry.usual_days is None:
        return False
    return dry.days_since_rain > dry.usual_days


def feature(
    event_catalog: Optional[pd.DataFrame],
    active: Optional[ActiveRainEvent],
    stitched: pd.DataFrame,
    current: SnapshotConditions,
    observed_at_ms: Optional[int],
    coverage: ArchiveCoverage,
    dry: DrySpellNow,
    clock: StationClock,
) -> Feature:
    """
    Pick the feature by the Feature Rule.

    :param event_catalog: The Event Catalog, or None.
    :param active: The active rain event, or None.
    :param stitched: `stitch_live` output.
    :param current: Current conditions, live stitched.
    :param observed_at_ms: The newest observation, or None.
    :param coverage: The build's counted days.
    :param dry: The snapshot's dry spell field, computed once in the build.
    :param clock: The build's clock.
    :return: The feature.
    """
    rain_start_ms = _rain(event_catalog, active, clock)
    if rain_start_ms is not None:
        return _bare("rain", rain_start_ms)
    unusual = _unusual(stitched, current, observed_at_ms, coverage, clock)
    if unusual is not None:
        return unusual
    if _dry_spell(dry):
        return _bare("dry_spell")
    return _bare("quiet")

"""
feature.py: The feature, picked by the Feature Rule (ADR-0009).

One rule decides what matters now; the Overview's feature tile shows it and
the widget view is derived from it (`lookout.snapshot.widget`). Checked in
order, the first that holds is the feature:

1. **rain**: an event is live (the build's `ActiveRainEvent`), or the
   newest Event Catalog event ended at most `RAIN_HOURS` before the build
   time (`StationClock.now_ms`). "Ended" is the catalog's `end_time`
   (`lookout.snapshot.events.newest_event`).
   The feature names the event it holds on by `event_start_ms`, the key of
   its `rain_events` row: the live event when one is live, else that newest
   ended event, so the site never picks the event itself (ADR-0003).
2. **unusual**: a current reading is below the `LOW`th or above the `HIGH`th
   percentile of its readings at the newest observation's station-local
   hour, over the usual window for that date (`usual_window`, the ±7 days
   around it in every comparable earlier year). The readings and their
   archive columns are `READINGS`: temperature, humidity, pressure and
   solar. Wind gust is not one (#533): it stays in the snapshot for the wind
   tile but cannot win the feature. A reading's percentile is its percentile
   rank: the share of the usual readings below it, a reading equal to it
   counting half, times 100 (so a gauge that never moves sits at 50). The
   reading furthest from the 50th wins; on an exact tie the first in
   `READINGS` wins. The rule is decided on the exact percentile; the one
   published is rounded to a whole number, but never into the band (at
   most `LOW` - 1 below it, at least `HIGH` + 1 above it). Skipped entirely
   unless the window's basis is "earlier_years": a recent-days band is too
   thin to call anything unusual. Skipped too unless the newest observation
   is fresh at the build time (`lookout.utils.dateutc.is_fresh` against
   `StationClock.now_ms`), so a silent station's last reading is never
   judged; the rule then falls through to dry spell, then quiet.
3. **dry_spell**: the snapshot's `DrySpellNow`, `days_since_rain` longer
   than `usual_days`; it cannot hold while either is None.
4. **quiet**: otherwise, temperature as a usual day.

**The hold** (#533). A feature's subject is its kind, and for unusual its
reading; `since_ms` is the build time of the build that first chose it.
When the rule picks a different subject from the last published feature's,
and that subject was chosen under `HOLD_HOURS` before the build time, the
last subject is kept and marked `held`, unless the rule now picks rain,
which takes over at once. Rain itself is never held: its rule's
`RAIN_HOURS` after an event already is its hold. A held feature's values
are recomputed this build: a held unusual reading carries its value and
percentile as judged now (rounded plainly when back inside the band, with
`outside_usual` False), and keeps the `direction` it was chosen with
(`above` or `below`, #584) however far the percentile has moved; a held
reading published before the field has none. A held unusual reading that
cannot be judged this build (no value, a stale observation, no earlier
year, no usual readings at the hour) is not held: the rule's own pick
stands. A last feature without `since_ms` (published before the hold) is no
last feature.

Functions:
- feature(event_catalog, active, stitched, current, observed_at_ms, coverage,
  dry, clock, previous): The `Feature` for one snapshot.
- previous_feature(dashboard): The last published feature, from the last
  published dashboard document, or None.
"""

from dataclasses import replace
from datetime import date
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, get_args

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
    FeatureDirection,
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

HOLD_HOURS = 3
"""A subject chosen under this many hours ago is kept (#533)."""


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
    Judged("pressure", "baromrelin", "barometer_inhg", "inhg"),
    Judged("solar", "solarradiation", "solar_wm2", "wm2"),
)
"""Every judged reading, in tie-break order: on an exact tie the first wins."""

READING_BY_NAME: Dict[str, Judged] = {judged.reading: judged for judged in READINGS}
"""Each judged reading by its name in the feature, for the hold."""


class Ranked(NamedTuple):
    """
    One reading judged this build.

    :param rank: Its exact percentile rank, 0 to 100.
    :param feature: Its unusual feature, `percentile` rounded for publishing.
    """

    rank: float
    feature: Feature


class Judging(NamedTuple):
    """
    What the unusual rule judges against this build.

    :param hour: The newest observation's station-local hour.
    :param days: The usual window's days.
    """

    hour: int
    days: Tuple[date, ...]


def _bare(kind: FeatureKind, since_ms: int, held: bool) -> Feature:
    """
    A feature of `kind` without a reading or an event: dry spell, quiet.

    :param kind: "dry_spell" or "quiet".
    :param since_ms: When its subject was first chosen.
    :param held: Whether the hold keeps it.
    :return: The feature.
    """
    return Feature(
        kind=kind,
        reading=None,
        value=None,
        unit=None,
        percentile=None,
        event_start_ms=None,
        since_ms=since_ms,
        held=held,
        outside_usual=None,
    )


def _rain_feature(
    event_catalog: Optional[pd.DataFrame],
    active: Optional[ActiveRainEvent],
    clock: StationClock,
) -> Optional[Feature]:
    """
    The rain rule: an event is live, or the newest ended within `RAIN_HOURS`.

    :param event_catalog: The Event Catalog, or None.
    :param active: The active rain event, or None.
    :param clock: The build's clock; the hours are counted to its `now`.
    :return: The rain feature, chosen now, naming the event the rule holds
        on (the live event, else the newest ended one) by its `start_ms`;
        None when the rule does not hold.
    """
    if active is not None:
        start_ms: Optional[int] = active.start_ms
    else:
        newest = newest_event(event_catalog, clock.now_ms)
        if newest is None:
            return None
        if (clock.now_ms - to_ms(newest["end_time"])) / HOUR_MS > RAIN_HOURS:
            return None
        start_ms = to_ms(newest["start_time"])
    return Feature(
        kind="rain",
        reading=None,
        value=None,
        unit=None,
        percentile=None,
        event_start_ms=start_ms,
        since_ms=clock.now_ms,
        held=False,
        outside_usual=None,
    )


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


def _outside(rank: float) -> bool:
    """
    Whether a percentile rank is outside the usual band.

    :param rank: The exact percentile rank.
    :return: True below `LOW` or above `HIGH`.
    """
    return rank < LOW or rank > HIGH


def _side(rank: float) -> FeatureDirection:
    """
    The side of the usual band a percentile rank is on.

    :param rank: The exact percentile rank.
    :return: "above" over the 50th, else "below".
    """
    return "above" if rank > 50.0 else "below"


def _published(rank: float) -> int:
    """
    A percentile rank as published: whole, and on the side of the band it is.

    :param rank: The exact percentile rank.
    :return: It rounded; below `LOW` or above `HIGH` it is kept outside the
        band (9.6 is 9, not 10), inside it it is rounded plainly.
    """
    whole = int(round(rank))
    if rank < LOW:
        return min(whole, int(LOW) - 1)
    if rank > HIGH:
        return max(whole, int(HIGH) + 1)
    return whole


def _judging(
    observed_at_ms: Optional[int],
    coverage: ArchiveCoverage,
    clock: StationClock,
) -> Optional[Judging]:
    """
    The hour and window the unusual rule judges against, when it applies.

    :param observed_at_ms: The newest observation, or None.
    :param coverage: The build's counted days.
    :param clock: The build's clock; the observation's age is counted to its
        `now`.
    :return: The judging; None when nothing can be judged this build (no
        observation, a stale one, no earlier year).
    """
    if observed_at_ms is None or not is_fresh(observed_at_ms, clock.now_ms):
        return None
    observed = local_time(observed_at_ms, coverage.tz)
    window = usual_window(coverage, observed.date())
    if window.basis != "earlier_years":
        return None
    return Judging(hour=observed.hour, days=window.days)


def _judge(
    judged: Judged,
    stitched: pd.DataFrame,
    current: SnapshotConditions,
    coverage: ArchiveCoverage,
    judging: Judging,
    since_ms: int,
    held: bool,
    kept: Optional[FeatureDirection] = None,
) -> Optional[Ranked]:
    """
    One reading judged against its usual readings at the hour.

    :param judged: The reading.
    :param stitched: `stitch_live` output.
    :param current: Current conditions, live stitched.
    :param coverage: The build's counted days.
    :param judging: The hour and window.
    :param since_ms: When its subject was first chosen.
    :param held: Whether the hold keeps it.
    :param kept: When held, the side it was chosen on (kept as published).
    :return: Its exact percentile rank and its unusual feature; None when it
        has no current value or no usual readings at the hour.
    """
    value = getattr(current, judged.field)
    if value is None:
        return None
    usual = usual_readings(stitched, coverage.tz, judging.days, judged.column)
    at_hour = usual[judging.hour]
    if len(at_hour) == 0:
        return None
    rank = _percentile(float(value), at_hour)
    # A fresh choice is outside the band, so its side is its rank's side; a
    # held one keeps the side it was chosen on, however far it has moved.
    direction = kept if held else _side(rank)
    return Ranked(
        rank=rank,
        feature=Feature(
            kind="unusual",
            reading=judged.reading,
            value=value,
            unit=judged.unit,
            percentile=_published(rank),
            event_start_ms=None,
            since_ms=since_ms,
            held=held,
            outside_usual=_outside(rank),
            direction=direction,
        ),
    )


def _unusual(
    stitched: pd.DataFrame,
    current: SnapshotConditions,
    coverage: ArchiveCoverage,
    judging: Optional[Judging],
    clock: StationClock,
) -> Optional[Feature]:
    """
    The unusual rule: the reading furthest outside its usual, if any is.

    :param stitched: `stitch_live` output.
    :param current: Current conditions, live stitched.
    :param coverage: The build's counted days.
    :param judging: The hour and window, or None when the rule is skipped.
    :param clock: The build's clock; a feature it picks is chosen now.
    :return: The feature, or None when no reading is unusual or the rule is
        skipped.
    """
    if judging is None:
        return None
    found: List[Tuple[float, Feature]] = []
    for judged in READINGS:
        ranked = _judge(
            judged,
            stitched,
            current,
            coverage,
            judging,
            since_ms=clock.now_ms,
            held=False,
        )
        if ranked is None or not _outside(ranked.rank):
            continue
        found.append((abs(ranked.rank - 50.0), ranked.feature))
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


def _subject(picked: Feature) -> Tuple[str, Optional[str]]:
    """
    A feature's subject: its kind, and for unusual its reading.

    :param picked: The feature.
    :return: (kind, reading).
    """
    return (picked.kind, picked.reading)


def _held(
    previous: Feature,
    stitched: pd.DataFrame,
    current: SnapshotConditions,
    coverage: ArchiveCoverage,
    judging: Optional[Judging],
) -> Optional[Feature]:
    """
    The last subject, kept by the hold, with this build's values.

    :param previous: The last published feature.
    :param stitched: `stitch_live` output.
    :param current: Current conditions, live stitched.
    :param coverage: The build's counted days.
    :param judging: The unusual rule's hour and window, or None.
    :return: The held feature; None when the subject cannot be held: rain,
        or an unusual reading that cannot be judged this build.
    """
    if previous.kind in ("dry_spell", "quiet"):
        return _bare(previous.kind, previous.since_ms, True)
    judged = READING_BY_NAME.get(previous.reading or "")
    if previous.kind != "unusual" or judging is None or judged is None:
        return None
    ranked = _judge(
        judged,
        stitched,
        current,
        coverage,
        judging,
        since_ms=previous.since_ms,
        held=True,
        kept=previous.direction,
    )
    return None if ranked is None else ranked.feature


def feature(
    event_catalog: Optional[pd.DataFrame],
    active: Optional[ActiveRainEvent],
    stitched: pd.DataFrame,
    current: SnapshotConditions,
    observed_at_ms: Optional[int],
    coverage: ArchiveCoverage,
    dry: DrySpellNow,
    clock: StationClock,
    previous: Optional[Feature] = None,
) -> Feature:
    """
    Pick the feature by the Feature Rule, then apply the hold.

    :param event_catalog: The Event Catalog, or None.
    :param active: The active rain event, or None.
    :param stitched: `stitch_live` output.
    :param current: Current conditions, live stitched.
    :param observed_at_ms: The newest observation, or None.
    :param coverage: The build's counted days.
    :param dry: The snapshot's dry spell field, computed once in the build.
    :param clock: The build's clock; the hold counts to its `now`.
    :param previous: The last published feature (`previous_feature`), or
        None: no hold.
    :return: The feature.
    """
    judging = _judging(observed_at_ms, coverage, clock)
    picked = _rain_feature(event_catalog, active, clock)
    if picked is None:
        picked = _unusual(stitched, current, coverage, judging, clock)
    if picked is None:
        kind: FeatureKind = "dry_spell" if _dry_spell(dry) else "quiet"
        picked = _bare(kind, clock.now_ms, False)
    if previous is None:
        return picked
    if _subject(picked) == _subject(previous):
        return replace(picked, since_ms=previous.since_ms)
    age_ms = clock.now_ms - previous.since_ms
    # A choice time after the build (a clock gone back) holds nothing, so a
    # hold can never outlast its hours.
    if picked.kind == "rain" or not 0 <= age_ms < HOLD_HOURS * HOUR_MS:
        return picked
    held = _held(previous, stitched, current, coverage, judging)
    return picked if held is None else held


def previous_feature(dashboard: Optional[Dict[str, Any]]) -> Optional[Feature]:
    """
    The last published feature, as the hold reads it.

    :param dashboard: The last published dashboard document
        (`Snapshot.dashboard()`), or None when there is none.
    :return: Its feature; None when there is no document, it is not a JSON
        object, it has no feature, or its feature has no `since_ms`
        (published before the hold) or a kind or reading this rule does not
        know (an old "wind_gust"). None
        means no hold: a document that parsed as JSON is never a failed read
        (ADR-0004); only the hold is lost.
    """
    if not isinstance(dashboard, dict):
        return None
    published = dashboard.get("feature")
    if not isinstance(published, dict):
        return None
    since_ms = published.get("since_ms")
    if not isinstance(since_ms, int) or isinstance(since_ms, bool):
        return None
    if published.get("kind") not in get_args(FeatureKind):
        return None
    reading = published.get("reading")
    if reading is not None and reading not in READING_BY_NAME:
        return None
    direction = published.get("direction")
    return Feature(
        kind=published["kind"],
        reading=reading,
        value=published.get("value"),
        unit=published.get("unit"),
        percentile=published.get("percentile"),
        event_start_ms=published.get("event_start_ms"),
        since_ms=since_ms,
        held=bool(published.get("held")),
        outside_usual=published.get("outside_usual"),
        direction=direction if direction in get_args(FeatureDirection) else None,
    )

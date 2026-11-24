"""
events.py: The Rain Events tab's section of the snapshot, and event curves.

Every event is keyed by its start time (`start_ms`), the first reading with
event rain. The Event Catalog's `event_id` is not stable: an ongoing event
is re-detected, under a new id, on every refresh run. The start is. So the
curve file of a completed event and the page's `?event=` link both use it,
and a link opened while an event is live still finds the event once it has
completed and its curve file is written.

Functions:
- rain_events_section(event_catalog, stitched, active, observed_at_ms,
  clock, year_status): The tab's section, with the page's headline counts
  and its four-water-year scatter (#666).
- completed_curves(event_catalog, archive, skip, limit): Curves of completed
  events, newest first, for the refresh run to write once each.
- newest_event(event_catalog, observed_at_ms): The catalog event that ended
  last, up to a time: the feature's rain rule reads it against
  `RAIN_HOURS`.
"""

from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)

import numpy as np
import pandas as pd

from lookout.core.event_category import event_category
from lookout.core.rain_counter import rain_increments
from lookout.core.rain_events import (
    filter_events,
    is_event_ongoing,
    summarize_events,
)
from lookout.core.rainfall_analysis import water_year, water_year_start
from lookout.snapshot.clock import StationClock
from lookout.snapshot.current import to_ms
from lookout.snapshot.model import (
    ActiveRainEvent,
    EventCounts,
    EventCurve,
    EventSeason,
    EventStats,
    GapStatus,
    RainEventRow,
    RainEvents,
    ScatterYear,
)
from lookout.snapshot.values import number
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

RAIN_HOURS = 12.0
"""An event live or ended at most this many hours ago is rain now: the rain
rule of the feature, counted to the build time."""


# A reading more than this many minutes after the one before follows a gap
# (the Event Catalog's quality rule and the Streamlit rate chart's).
GAP_MINUTES = 10.0

NO_DATA = "invalid"

SCATTER_YEARS = 4
"""The scatter's water years: this one and the three before it."""

RECENT_MS = 7 * 24 * 3600 * 1000
"""The headline's recent window: the 7 days before the build time."""


class _History:
    """The stitched or archived readings, oldest first, sliced by time."""

    def __init__(self, readings: pd.DataFrame) -> None:
        """
        Sort the readings once so each event's span is a binary search.

        :param readings: Rows with `dateutc` (ms), in any order.
        """
        ordered = readings.sort_values("dateutc") if not readings.empty else readings
        self.t = ordered["dateutc"].to_numpy(dtype="int64") if len(ordered) else None
        self.frame = ordered.reset_index(drop=True)

    def span(self, start_ms: int, end_ms: Optional[int]) -> pd.DataFrame:
        """
        The readings from `start_ms` to `end_ms`, both inclusive.

        :param start_ms: First time.
        :param end_ms: Last time; None for every reading from the start on.
        :return: Those rows, oldest first.
        """
        if self.t is None:
            return self.frame.iloc[0:0]
        lo = int(np.searchsorted(self.t, start_ms, side="left"))
        hi = (
            len(self.t)
            if end_ms is None
            else int(np.searchsorted(self.t, end_ms, side="right"))
        )
        return self.frame.iloc[lo:hi]


def _ms(value: Any) -> int:
    """A catalog time (a Timestamp or anything `to_datetime` reads) in ms."""
    return int(pd.Timestamp(pd.to_datetime(value, utc=True)).value // 10**6)


def _curve(readings: pd.DataFrame, start_ms: int, end_ms: int) -> EventCurve:
    """
    One event's curve from its readings, oldest first.

    The rate between readings is the rain the reading adds to `dailyrainin`
    (`rain_counter.rain_increments`: a counter drop is the reset, and the
    reading after it holds only new rain) over the time between them, as
    the Event Catalog's peak rate computes it (#654). The first reading
    reads 0. A reading after a gap has no rate.

    :param readings: The event's readings.
    :param start_ms: The event's start.
    :param end_ms: Its end.
    :return: The curve.
    """
    t = readings["dateutc"].to_numpy(dtype="int64")
    minutes = np.diff(t, prepend=t[0] - 5 * 60_000) / 60_000
    fallen = rain_increments(readings["dailyrainin"]).fillna(0).to_numpy()
    rate = fallen / (minutes / 60)
    after_gap = minutes > GAP_MINUTES
    return EventCurve(
        start_ms=start_ms,
        end_ms=end_ms,
        t_ms=tuple(int(x) for x in t),
        event_in=tuple(number(x) for x in readings["eventrainin"]),
        rate_in_hr=tuple(None if gap else number(r) for r, gap in zip(rate, after_gap)),
        gap_minutes=tuple(
            float(m) if gap else None for m, gap in zip(minutes, after_gap)
        ),
    )


def _stats(events: pd.DataFrame) -> EventStats:
    """
    `summarize_events` as the snapshot's dataclass, rounded for JSON.

    The median is over completed events only: the live event is never part
    of the median it is measured against (#375).
    """
    summary = summarize_events(events)
    completed = events[~events["live"]] if "live" in events else events
    return EventStats(
        count=int(summary["count"]),
        total_in=round(float(summary["total_in"]), 3),
        median_in=number(summarize_events(completed)["median_in"]),
        mean_duration_minutes=number(summary["mean_duration_minutes"], 1),
    )


def _live_catalog(
    event_catalog: pd.DataFrame,
    active: Optional[ActiveRainEvent],
    observed_at_ms: Optional[int],
) -> pd.DataFrame:
    """
    The catalog with the live event's total and duration from the live
    reading, so every view shows the same number (the One Number Rule).

    :param event_catalog: The Event Catalog.
    :param active: The active rain event, or None.
    :param observed_at_ms: The newest observation.
    :return: A copy, with `start_ms` and `live` (the active rain event)
        columns. There is no second "open" flag: an event the catalog still
        holds open but the live reading has ended is not live (#147).
    """
    events = event_catalog.copy()
    events["start_time"] = pd.to_datetime(events["start_time"], utc=True)
    events["end_time"] = pd.to_datetime(events["end_time"], utc=True)
    events["start_ms"] = [_ms(t) for t in events["start_time"]]
    events["live"] = False
    if active is not None:
        live = events["start_ms"] == active.start_ms
        events.loc[live, "live"] = True
        events.loc[live, "total_rainfall"] = active.total_in
        events.loc[live, "duration_minutes"] = active.duration_minutes
        if observed_at_ms is not None:
            events.loc[live, "end_time"] = pd.Timestamp(
                observed_at_ms, unit="ms", tz="UTC"
            )
    return events.sort_values("start_time", ascending=False).reset_index(drop=True)


def _row(event: pd.Series, rank: int, history: _History, tz: str) -> RainEventRow:
    """One listed event."""
    start_ms = int(event["start_ms"])
    end_ms = _ms(event["end_time"])
    has_curve = not history.span(start_ms, end_ms).empty
    completeness = number(event.get("data_completeness"))
    quality = event.get("quality_rating")
    total_in = float(event["total_rainfall"])
    peak = number(event.get("max_hourly_rate"))
    return RainEventRow(
        start_ms=start_ms,
        end_ms=end_ms,
        duration_minutes=float(event["duration_minutes"]),
        total_in=total_in,
        peak_rate_in_hr=peak,
        category=event_category(total_in, peak),
        quality=quality if isinstance(quality, str) and has_curve else NO_DATA,
        completeness_pct=None if completeness is None else round(completeness * 100, 1),
        max_gap_minutes=number(event.get("max_gap_minutes"), 1),
        water_year=water_year(event["start_time"].tz_convert(tz).date()),
        rank=rank,
        live=bool(event["live"]),
        has_curve=has_curve,
    )


def rain_events_section(
    event_catalog: Optional[pd.DataFrame],
    stitched: pd.DataFrame,
    active: Optional[ActiveRainEvent],
    observed_at_ms: Optional[int],
    clock: StationClock,
    year_status: Callable[[int], GapStatus],
) -> RainEvents:
    """
    The Rain Events tab: every event, its summaries, the live event's curve.

    :param event_catalog: The Event Catalog, or None.
    :param stitched: The archive with the live reading stitched in.
    :param active: The active rain event, or None: which event is live.
    :param observed_at_ms: The newest observation.
    :param clock: The station clock: its zone says which water year an
        event starts in.
    :param year_status: A water year's rain gap status, judged as the Rain
        tab's water-year curves are (`CountedDays.rain.water_year_status`),
        so a season reads against the same status, and every season has one.
    :return: The section.
    """
    this_year = water_year(clock.today)
    if event_catalog is None or event_catalog.empty:
        return RainEvents(
            events=(),
            all_time=_stats(pd.DataFrame()),
            seasons=(),
            open_curve=None,
            counts=EventCounts(water_year=this_year, events=0, last_7_days=0),
            scatter=(),
        )

    events = _live_catalog(event_catalog, active, observed_at_ms)
    history = _History(stitched)
    totals = events["total_rainfall"].to_numpy(dtype=float)
    rows = tuple(
        _row(event, int((totals > totals[i]).sum()) + 1, history, clock.tz)
        for i, event in events.iterrows()
    )

    years = sorted({row.water_year for row in rows}, reverse=True)
    seasons = tuple(
        EventSeason(
            water_year=year,
            stats=_stats(
                filter_events(
                    events,
                    start=pd.Timestamp(water_year_start(year), tz=clock.tz),
                    end=pd.Timestamp(water_year_start(year + 1), tz=clock.tz),
                )
            ),
            status=year_status(year),
        )
        for year in years
    )

    # The open curve is the live event's: the hero's one "raining now"
    # decision. An event the catalog still holds open after the live reading
    # has ended gets its curve file once the catalog closes it, from an
    # archive that has caught up (`completed_curves` writes each file once).
    open_curve = None
    live_now = events[events["live"]]
    if not live_now.empty:
        start_ms = int(live_now.iloc[0]["start_ms"])
        readings = history.span(start_ms, None)
        if not readings.empty:
            open_curve = _curve(readings, start_ms, int(readings["dateutc"].iloc[-1]))

    recent_from = clock.now_ms - RECENT_MS
    counts = EventCounts(
        water_year=this_year,
        events=sum(1 for row in rows if row.water_year == this_year),
        last_7_days=sum(
            1 for row in rows if recent_from <= row.start_ms <= clock.now_ms
        ),
    )
    return RainEvents(
        events=rows,
        all_time=_stats(events),
        seasons=seasons,
        open_curve=open_curve,
        counts=counts,
        scatter=_scatter(event_catalog, clock.tz, this_year),
    )


def _scatter(
    event_catalog: pd.DataFrame, tz: str, this_year: int
) -> Tuple[ScatterYear, ...]:
    """
    The scatter's rows from the Event Catalog alone: each completed event's
    start, catalog total and category, per water year, newest first.

    :param event_catalog: The Event Catalog, not empty.
    :param tz: The station's zone: which water year an event starts in.
    :param this_year: The build time's water year, the first row.
    :return: This water year and the three before it, none before the water
        year of the catalog's oldest event; a year with no completed event
        is an empty row.
    """
    events = event_catalog.copy()
    events["start_time"] = pd.to_datetime(events["start_time"], utc=True)
    events = events.sort_values("start_time", ascending=False)
    events["water_year"] = [
        water_year(start.tz_convert(tz).date()) for start in events["start_time"]
    ]
    oldest = int(events["water_year"].min())
    points: Dict[int, Tuple[List[int], List[float], List[str]]] = {
        year: ([], [], [])
        for year in range(this_year, this_year - SCATTER_YEARS, -1)
        if year >= oldest or year == this_year
    }
    for _, event in events.iterrows():
        row = points.get(int(event["water_year"]))
        if row is None or is_event_ongoing(event):
            continue
        total_in = float(event["total_rainfall"])
        peak = number(event.get("max_hourly_rate"))
        row[0].append(_ms(event["start_time"]))
        row[1].append(round(total_in, 2))
        row[2].append(event_category(total_in, peak))
    return tuple(
        ScatterYear(
            water_year=year,
            start_ms=tuple(starts),
            total_in=tuple(totals),
            category=tuple(categories),
        )
        for year, (starts, totals, categories) in points.items()
    )


def _completed(
    event_catalog: pd.DataFrame, skip: Collection[int]
) -> Iterator[Tuple[int, int]]:
    """(start_ms, end_ms) of completed events not in `skip`, newest first."""
    events = event_catalog.copy()
    events["start_time"] = pd.to_datetime(events["start_time"], utc=True)
    events = events.sort_values("start_time", ascending=False)
    for _, event in events.iterrows():
        if is_event_ongoing(event):
            continue
        start_ms = _ms(event["start_time"])
        if start_ms not in skip:
            yield start_ms, _ms(event["end_time"])


def completed_curves(
    event_catalog: Optional[pd.DataFrame],
    archive: pd.DataFrame,
    skip: Collection[int] = (),
    limit: Optional[int] = None,
    failed: Optional[List[int]] = None,
) -> List[EventCurve]:
    """
    The curves of completed events, newest first, for curve files.

    An event with no archive readings in its range gets no curve (the tab
    shows its totals only). Only the curves returned are computed, so a
    run that skips every stored one does no slicing beyond the new ones.
    An event whose curve fails to compute is logged at ERROR and left out,
    so it never blocks the older ones (#160); the next run retries it. Its
    start is appended to `failed` when given, so the caller can fail the run
    (#233).

    :param event_catalog: The Event Catalog, or None.
    :param archive: The archive.
    :param skip: Starts (ms) to leave out: those already stored.
    :param limit: At most this many curves; None for all.
    :param failed: Collects the starts (ms) whose curve failed to build.
    :return: The curves.
    """
    if event_catalog is None or event_catalog.empty:
        return []
    history = _History(archive)
    curves: List[EventCurve] = []
    for start_ms, end_ms in _completed(event_catalog, set(skip)):
        if limit is not None and len(curves) >= limit:
            break
        readings = history.span(start_ms, end_ms)
        if readings.empty:
            continue
        try:
            curves.append(_curve(readings, start_ms, end_ms))
        except Exception as e:
            logger.exception(f"No curve for the event starting {start_ms}: {e}")
            if failed is not None:
                failed.append(start_ms)
    return curves


def newest_event(
    event_catalog: Optional[pd.DataFrame], observed_at_ms: Optional[int]
) -> Optional[pd.Series]:
    """
    The catalog event that ended last, up to `observed_at_ms`.

    :param event_catalog: The Event Catalog, or None.
    :param observed_at_ms: The time to look back from: the build time for
        the feature.
    :return: The event's catalog row, or None when none has ended.
    """
    if event_catalog is None or event_catalog.empty or observed_at_ms is None:
        return None
    ends = event_catalog["end_time"].map(to_ms)
    ended = ends[ends.notna() & (ends <= observed_at_ms)]
    if ended.empty:
        return None
    return event_catalog.loc[ended.astype("int64").idxmax()]

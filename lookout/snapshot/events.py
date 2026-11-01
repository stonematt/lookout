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
  year_status): The tab's section.
- completed_curves(event_catalog, archive, skip, limit): Curves of completed
  events, newest first, for the refresh run to write once each.
"""

from typing import Any, Collection, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from lookout.core.rain_events import (
    filter_events,
    is_event_ongoing,
    summarize_events,
)
from lookout.core.rainfall_analysis import water_year, water_year_start
from lookout.snapshot.model import (
    ActiveRainEvent,
    EventCurve,
    EventSeason,
    EventStats,
    GapStatus,
    RainEventRow,
    RainEvents,
)
from lookout.snapshot.values import number
from lookout.utils.dateutc import DEFAULT_TZ


# A reading more than this many minutes after the one before follows a gap
# (the Event Catalog's quality rule and the Streamlit rate chart's).
GAP_MINUTES = 10.0

NO_DATA = "invalid"


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

    The rate between readings is the rise in `dailyrainin` (negative steps,
    the midnight reset, read as 0) over the time between them, as the
    Event Catalog's peak rate and the Streamlit rate chart compute it. The
    first reading reads 0. A reading after a gap has no rate.

    :param readings: The event's readings.
    :param start_ms: The event's start.
    :param end_ms: Its end.
    :return: The curve.
    """
    t = readings["dateutc"].to_numpy(dtype="int64")
    minutes = np.diff(t, prepend=t[0] - 5 * 60_000) / 60_000
    daily = readings["dailyrainin"].astype(float)
    fallen = daily.diff().clip(lower=0).fillna(0).to_numpy()
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
    """`summarize_events` as the snapshot's dataclass, rounded for JSON."""
    summary = summarize_events(events)
    return EventStats(
        count=int(summary["count"]),
        total_in=round(float(summary["total_in"]), 3),
        median_in=number(summary["median_in"]),
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


def _row(event: pd.Series, rank: int, history: _History) -> RainEventRow:
    """One listed event."""
    start_ms = int(event["start_ms"])
    end_ms = _ms(event["end_time"])
    has_curve = not history.span(start_ms, end_ms).empty
    completeness = number(event.get("data_completeness"))
    quality = event.get("quality_rating")
    return RainEventRow(
        start_ms=start_ms,
        end_ms=end_ms,
        duration_minutes=float(event["duration_minutes"]),
        total_in=float(event["total_rainfall"]),
        peak_rate_in_hr=number(event.get("max_hourly_rate")),
        quality=quality if isinstance(quality, str) and has_curve else NO_DATA,
        completeness_pct=None if completeness is None else round(completeness * 100, 1),
        max_gap_minutes=number(event.get("max_gap_minutes"), 1),
        water_year=water_year(event["start_time"].tz_convert(DEFAULT_TZ).date()),
        rank=rank,
        live=bool(event["live"]),
        has_curve=has_curve,
    )


def rain_events_section(
    event_catalog: Optional[pd.DataFrame],
    stitched: pd.DataFrame,
    active: Optional[ActiveRainEvent],
    observed_at_ms: Optional[int],
    year_status: Mapping[int, GapStatus],
) -> RainEvents:
    """
    The Rain Events tab: every event, its summaries, the live event's curve.

    :param event_catalog: The Event Catalog, or None.
    :param stitched: The archive with the live reading stitched in.
    :param active: The active rain event, or None: which event is live.
    :param observed_at_ms: The newest observation.
    :param year_status: Each water year's rain gap status, from the Rain
        tab's water-year curves, so a season reads against the same status.
    :return: The section.
    """
    if event_catalog is None or event_catalog.empty:
        return RainEvents(
            events=(), all_time=_stats(pd.DataFrame()), seasons=(), open_curve=None
        )

    events = _live_catalog(event_catalog, active, observed_at_ms)
    history = _History(stitched)
    totals = events["total_rainfall"].to_numpy(dtype=float)
    rows = tuple(
        _row(event, int((totals > totals[i]).sum()) + 1, history)
        for i, event in events.iterrows()
    )

    years = sorted({row.water_year for row in rows}, reverse=True)
    seasons = tuple(
        EventSeason(
            water_year=year,
            stats=_stats(
                filter_events(
                    events,
                    start=pd.Timestamp(water_year_start(year), tz=DEFAULT_TZ),
                    end=pd.Timestamp(water_year_start(year + 1), tz=DEFAULT_TZ),
                )
            ),
            status=year_status.get(year),
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

    return RainEvents(
        events=rows,
        all_time=_stats(events),
        seasons=seasons,
        open_curve=open_curve,
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
) -> List[EventCurve]:
    """
    The curves of completed events, newest first, for curve files.

    An event with no archive readings in its range gets no curve (the tab
    shows its totals only). Only the curves returned are computed, so a
    run that skips every stored one does no slicing beyond the new ones.

    :param event_catalog: The Event Catalog, or None.
    :param archive: The archive.
    :param skip: Starts (ms) to leave out: those already stored.
    :param limit: At most this many curves; None for all.
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
        if not readings.empty:
            curves.append(_curve(readings, start_ms, end_ms))
    return curves

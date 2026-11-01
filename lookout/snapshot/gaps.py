"""
gaps.py: The gap status of months and windows (spec #66 "Gap status").

Each month and each window gets one status, the One Gap Status Rule:

- "valid": every day in it can be counted.
- "prorated": up to 10% of its days cannot be counted. A total over the
  counted days is scaled to the whole window (`prorate`) and shown with ≈.
- "invalid": no value is computed for it (`comparable`), and the site hatches
  it with the reason in place.

Days are station-local calendar days (ADR-0002), 23 or 25 hours long across
daylight saving. A day is judged by how many of its five-minute slots the
archive holds, up to `now` for today:

- "ok": at least 80% of its slots (`LOW_COVERAGE`).
- "low": some rows, under 80%: a low-coverage day.
- "gap": no rows at all: a gap in the record.
- "not_archived": the day runs past the archive's newest row by more than
  `ARCHIVE_LAG` (a span after the archive ends).
- "before_archive": the day ended before the archive's first row.
- "future": the day is after today.

A window's status then follows, in this order of reasons:

1. any day still to come: invalid, "in_progress". A calendar month is in
   progress until its last day has ended, so a month that is still running is
   never compared with complete ones.
2. any day after the archive ends: invalid, "not_archived".
3. any day before the archive began: invalid, "before_archive".
4. "gap" and "low" days are the window's missing days. None: valid. Up to 10%
   of its days (`MAX_MISSING_SHARE`): prorated. More: invalid. The reason is
   "gap" when any day has no rows, else "low_coverage".

So a single day, or a week, with any missing day is invalid; prorating starts
with windows of ten days or more.

Pass the archive as stored, not the history with the live reading stitched
in: the status describes what the archive holds. Stitching today's live
value into a valid or prorated window is the caller's own step.

The API, for every section that compares months or windows:

- `ArchiveCoverage(archive, now)`: reads the archive once, then answers
  `.window(first_day, last_day)`, `.month(year, month)`, `.rolling(days)`,
  `.water_year()` and `.counted_days(first_day, last_day)`;
  `.excluding(missing)` judges days a derived record lacks as missing.
- `window_status(archive, first_day, last_day, now)` and
  `month_status(archive, year, month, now)`: one-off shorthands.
- `comparable(status, compute)`: `compute()`, or None when invalid; it never
  runs `compute` for an invalid window.
- `prorate(status, total)`: a total over the counted days, scaled to the
  whole window; None when invalid.
- `snapshot_gaps(archive, now)`: the `Gaps` section `build_snapshot` adds.
"""

from copy import copy
from dataclasses import replace
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Mapping, Optional, Tuple, TypeVar

import pandas as pd

from lookout.core.rainfall_analysis import water_year, water_year_start
from lookout.snapshot.model import Gaps, GapStatus
from lookout.utils.dateutc import DEFAULT_TZ

T = TypeVar("T")

VALID = "valid"
PRORATED = "prorated"
INVALID = "invalid"

INTERVAL_MS = 5 * 60 * 1000
"""The archive's reporting interval: Ambient's five minutes."""

LOW_COVERAGE = 0.8
"""A day holding under this share of its five-minute slots is low-coverage."""

MAX_MISSING_SHARE = 0.10
"""A window missing more than this share of its days is invalid."""

ARCHIVE_LAG = pd.Timedelta(hours=1)
"""How far the archive may trail `now` before today is "not archived yet"."""

STANDARD_WINDOWS: Tuple[Tuple[str, int, int], ...] = (
    ("today", 0, 0),
    ("yesterday", 1, 1),
    ("7d", 6, 0),
    ("30d", 29, 0),
    ("90d", 89, 0),
    ("365d", 364, 0),
)
"""The windows the snapshot carries, as (key, first day, last day), each day
counted back from today. The water year follows them."""


def standard_window_keys() -> Tuple[str, ...]:
    """
    The keys of `STANDARD_WINDOWS`, in order; the water year is not among them.

    Read at call time, so a section's keys follow the windows and not a copy.
    """
    return tuple(key for key, _, _ in STANDARD_WINDOWS)


_OK, _LOW, _GAP = "ok", "low", "gap"
_NOT_ARCHIVED, _BEFORE_ARCHIVE, _FUTURE = "not_archived", "before_archive", "future"
_MISSING_KINDS = frozenset({_GAP, _LOW, _NOT_ARCHIVED, _BEFORE_ARCHIVE})


class ArchiveCoverage:
    """
    How much of each station-local day the archive holds, as of `now`.

    :param archive: The station's archive, `dateutc` in epoch ms.
    :param now: The build time, timezone-aware.
    :param tz: The station-local zone.
    :raises ValueError: If `now` has no timezone.
    """

    def __init__(
        self, archive: pd.DataFrame, now: datetime, tz: str = DEFAULT_TZ
    ) -> None:
        now = pd.Timestamp(now)
        if now.tzinfo is None:
            raise ValueError("ArchiveCoverage needs a timezone-aware `now`")
        self.tz = tz
        self.now_ms = int(now.value // 10**6)
        self.today: date = now.tz_convert(tz).date()

        stamps = archive["dateutc"] if "dateutc" in archive else pd.Series([])
        stamps = pd.Series(stamps, dtype="int64")
        stamps = stamps[stamps <= self.now_ms]
        self.empty = stamps.empty
        self.oldest_ms = None if self.empty else int(stamps.min())
        self.newest_ms = None if self.empty else int(stamps.max())

        # One count per five-minute slot, so a duplicate row is not counted twice.
        slots = pd.Series((stamps // INTERVAL_MS).unique() * INTERVAL_MS)
        local_days = pd.to_datetime(slots, unit="ms", utc=True).dt.tz_convert(tz)
        self._slots: Dict[date, int] = (
            local_days.dt.date.value_counts().to_dict() if len(slots) else {}
        )
        self._kinds: Dict[date, str] = {}

    @property
    def first_day(self) -> Optional[date]:
        """The archive's first station-local day, or None when it is empty."""
        if self.empty:
            return None
        return (
            pd.Timestamp(self.oldest_ms, unit="ms", tz="UTC").tz_convert(self.tz).date()
        )

    def _midnight_ms(self, day: date) -> int:
        """Epoch ms of local midnight starting `day`."""
        return int(pd.Timestamp(day).tz_localize(self.tz).value // 10**6)

    def day_kind(self, day: date) -> str:
        """
        Judge one station-local day (see the module docstring).

        :param day: The day.
        :return: "ok", "low", "gap", "not_archived", "before_archive" or
            "future".
        """
        if day in self._kinds:
            return self._kinds[day]
        kind = self._judge(day)
        self._kinds[day] = kind
        return kind

    def _judge(self, day: date) -> str:
        """Compute `day_kind` for a day not yet judged."""
        if day > self.today:
            return _FUTURE
        if self.empty:
            return _NOT_ARCHIVED
        start = self._midnight_ms(day)
        end = min(self._midnight_ms(day + timedelta(days=1)), self.now_ms)
        if self._midnight_ms(day + timedelta(days=1)) <= self.oldest_ms:
            return _BEFORE_ARCHIVE
        if end > self.newest_ms + ARCHIVE_LAG.value // 10**6:
            return _NOT_ARCHIVED
        # Within the allowed lag, the slots after the newest row are not due.
        due = min(end, self.newest_ms + INTERVAL_MS) - start
        expected = -(-due // INTERVAL_MS) if due > 0 else 0
        held = self._slots.get(day, 0)
        if expected == 0:
            return _OK
        if held == 0:
            return _GAP
        return _OK if held / expected >= LOW_COVERAGE else _LOW

    def excluding(self, missing: Mapping[date, str]) -> "ArchiveCoverage":
        """
        This coverage with some counted days judged missing after all.

        For a section whose values come from a record derived from the
        archive (the energy catalog) that lacks days the archive holds: those
        days are judged as `missing` says, so a window over them is prorated
        or invalid by the same rule, never counted as zero.

        :param missing: Day -> the kind to judge it as: "gap", "low",
            "not_archived" or "before_archive". Only "ok" days change; the
            copy shares the read-only slot counts and has its own kinds.
        :return: A new coverage; this one is unchanged.
        :raises ValueError: For any other kind.
        """
        unknown = set(missing.values()) - _MISSING_KINDS
        if unknown:
            raise ValueError(f"Not a missing day's kind: {sorted(unknown)}")
        other = copy(self)
        other._kinds = dict(self._kinds)
        for day, kind in missing.items():
            if self.day_kind(day) == _OK:
                other._kinds[day] = kind
        return other

    def counted_days(self, first_day: date, last_day: date) -> Tuple[date, ...]:
        """
        The days a valid or prorated window's total is summed over.

        :param first_day: The window's first local day.
        :param last_day: Its last local day, inclusive.
        :return: Its "ok" days, oldest first.
        """
        return tuple(d for d in _days(first_day, last_day) if self.day_kind(d) == _OK)

    def window(
        self, first_day: date, last_day: date, key: Optional[str] = None
    ) -> GapStatus:
        """
        The gap status of the local days `first_day` to `last_day`.

        :param first_day: The first day.
        :param last_day: The last day, inclusive.
        :param key: Its name in the snapshot; "first..last" when None.
        :return: Its status.
        """
        days = _days(first_day, last_day)
        kinds = [self.day_kind(d) for d in days]
        status, reason = _judge_window(kinds)
        return GapStatus(
            key=key or f"{first_day.isoformat()}..{last_day.isoformat()}",
            first_day=first_day.isoformat(),
            last_day=last_day.isoformat(),
            days=len(days),
            missing_days=sum(kind != _OK for kind in kinds),
            status=status,
            reason=reason,
        )

    def month(self, year: int, month: int) -> GapStatus:
        """
        The gap status of one station-local calendar month.

        A month is in progress, so invalid, until its last day has ended.

        :param year: The year.
        :param month: The month, 1 to 12.
        :return: Its status, keyed "YYYY-MM".
        """
        first = date(year, month, 1)
        last = _next_month(first) - timedelta(days=1)
        status = self.window(first, last, key=f"{year:04d}-{month:02d}")
        if last >= self.today:
            return replace(status, status=INVALID, reason="in_progress")
        return status

    def rolling(self, days: int, key: Optional[str] = None) -> GapStatus:
        """
        The gap status of the `days` local days ending today.

        :param days: How many days, today included.
        :param key: Its name in the snapshot.
        :return: Its status.
        """
        return self.window(self.today - timedelta(days=days - 1), self.today, key)

    def water_year(self, key: str = "water_year") -> GapStatus:
        """
        The gap status of the water year so far: Oct 1 to today.

        :param key: Its name in the snapshot.
        :return: Its status.
        """
        start = water_year_start(water_year(self.today))
        return self.window(start, self.today, key)

    def months(self) -> Tuple[GapStatus, ...]:
        """
        Every month from the archive's first through the current one.

        :return: Their statuses, oldest first; empty for an empty archive.
        """
        first = self.first_day
        if first is None:
            return ()
        statuses: List[GapStatus] = []
        month = first.replace(day=1)
        while month <= self.today:
            statuses.append(self.month(month.year, month.month))
            month = _next_month(month)
        return tuple(statuses)

    def standard_windows(self) -> Tuple[GapStatus, ...]:
        """
        The snapshot's windows: `STANDARD_WINDOWS`, then the water year.

        :return: Their statuses, in that order.
        """
        windows = [
            self.window(
                self.today - timedelta(days=first_back),
                self.today - timedelta(days=last_back),
                key,
            )
            for key, first_back, last_back in STANDARD_WINDOWS
        ]
        return (*windows, self.water_year())


def window_status(
    archive: pd.DataFrame,
    first_day: date,
    last_day: date,
    now: datetime,
    key: Optional[str] = None,
) -> GapStatus:
    """
    The gap status of one window of station-local days.

    For many windows over one archive, build one `ArchiveCoverage` instead.

    :param archive: The station's archive as stored.
    :param first_day: The window's first day.
    :param last_day: Its last day, inclusive.
    :param now: The build time, timezone-aware.
    :param key: Its name; "first..last" when None.
    :return: Its status.
    """
    return ArchiveCoverage(archive, now).window(first_day, last_day, key)


def month_status(
    archive: pd.DataFrame, year: int, month: int, now: datetime
) -> GapStatus:
    """
    The gap status of one station-local calendar month.

    :param archive: The station's archive as stored.
    :param year: The year.
    :param month: The month, 1 to 12.
    :param now: The build time, timezone-aware.
    :return: Its status, keyed "YYYY-MM".
    """
    return ArchiveCoverage(archive, now).month(year, month)


def comparable(status: GapStatus, compute: Callable[[], T]) -> Optional[T]:
    """
    Compute a window's value only when its comparison is allowed.

    :param status: The window's gap status.
    :param compute: Computes the value; never called for an invalid window.
    :return: Its value, or None when the window is invalid.
    """
    if status.status == INVALID:
        return None
    return compute()


def prorate(status: GapStatus, total: float) -> Optional[float]:
    """
    Scale a total over a window's counted days to the whole window.

    :param status: The window's gap status.
    :param total: The sum over its counted days (`counted_days`).
    :return: The total for all its days; as given when valid; None when
        invalid.
    """
    if status.status == INVALID:
        return None
    if status.status == VALID:
        return total
    return total * status.days / (status.days - status.missing_days)


def snapshot_gaps(archive: pd.DataFrame, now: datetime) -> Gaps:
    """
    The snapshot's gap section: every month and every standard window.

    :param archive: The station's archive as stored.
    :param now: The build time, timezone-aware.
    :return: The `Gaps` section.
    """
    coverage = ArchiveCoverage(archive, now)
    return Gaps(months=coverage.months(), windows=coverage.standard_windows())


def _judge_window(kinds: List[str]) -> Tuple[str, Optional[str]]:
    """
    A window's status and reason from its days' kinds (module docstring).

    :param kinds: Each day's kind.
    :return: (status, reason).
    """
    for kind, reason in (
        (_FUTURE, "in_progress"),
        (_NOT_ARCHIVED, "not_archived"),
        (_BEFORE_ARCHIVE, "before_archive"),
    ):
        if kind in kinds:
            return INVALID, reason
    missing = sum(kind in (_GAP, _LOW) for kind in kinds)
    if missing == 0:
        return VALID, None
    reason = "gap" if _GAP in kinds else "low_coverage"
    if missing <= MAX_MISSING_SHARE * len(kinds):
        return PRORATED, reason
    return INVALID, reason


def _days(first_day: date, last_day: date) -> List[date]:
    """Every day from `first_day` to `last_day`, inclusive."""
    return [
        first_day + timedelta(days=i) for i in range((last_day - first_day).days + 1)
    ]


def _next_month(day: date) -> date:
    """The first day of the month after `day`'s."""
    return date(day.year + day.month // 12, day.month % 12 + 1, 1)

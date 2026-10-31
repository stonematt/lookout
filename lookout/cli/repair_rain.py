"""
repair_rain.py: Restore the raw rain counters in the damaged windows (#95).

Until #95, every catchup run subtracted each known rain carryover from the
stored archive again, so the rows from a carryover day to the end of its
year hold counters that are too low (clipped at 0 at worst). Rows from 2026
on are not affected. The repair reuses gap fill: for each window it drops
the archive's rows in it, refills it with `fill_archive_gap` (raw, as
Ambient reports it), and cross-checks the refilled rain counters against the
protected raw backup. A dry run is the default.

Usage:
    python -m lookout.cli.repair_rain [--bucket lookout] [--write]
        [--window FIRST LAST ...] [--tolerance 0.001]
"""

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

import lookout.api.awn_controller as awn
from lookout.cli.diagnostics import emit, load_station
from lookout.config import DEFAULT_BUCKET
from lookout.storage.storj import backup_and_save_history, get_df_from_s3
from lookout.utils.dateutc import DEFAULT_TZ, from_local_date

# Station-local days, first and last inclusive: from each carryover day the
# owner found damaged to the end of its year (owner check on #95).
DEFAULT_WINDOWS: List[Tuple[str, str]] = [
    ("2024-12-16", "2024-12-31"),
    ("2025-10-14", "2025-12-31"),
]
# The newest backup with raw counters, kept out of `.backups/` rotation.
REFERENCE_KEY = "backups/{mac}.raw-pre-95_20251222.parquet"
RAIN_COLUMNS = ("dailyrainin", "weeklyrainin", "monthlyrainin", "yearlyrainin")
DEFAULT_TOLERANCE = 0.001  # inches
# One Ambient page brings about this many new five-minute readings; the
# slack covers pages spent seeking over the station's own outages.
PAGE_ROWS = 246
SLACK_PAGES = 10
FIVE_MIN_MS = 5 * 60 * 1000

RUN_NOTES = """\
How to run it (after the #95 fix is deployed, with live secrets, on a host
that can reach Ambient and Storj; never while a catchup or refresh cron run
could save the archive at the same time):

  1. Dry run, the default. Refills both windows in memory, prints each day's
     rows and day-end rain counters before -> after, and the cross-check
     against the raw backup. Writes nothing:

         PYTHONPATH=. python -m lookout.cli.repair_rain

  2. Read the diff. Every day should show "rows N -> N" (or more after), no
     "short" line, and "0 over tolerance". The exit status is 1 if not.

  3. The real run. Same checks; if any day refills short or the cross-check
     fails, nothing is written. Otherwise it backs the live archive up to
     .backups/<MAC>/<today>/ and saves the repaired archive:

         PYTHONPATH=. python -m lookout.cli.repair_rain --write

  A window can be given instead of the defaults, as station-local days
  (repeatable): --window 2025-10-14 2025-12-31. The fill waits a second per
  Ambient page; the 2025 window is about 90 pages.
"""


@dataclass
class WindowRefill:
    """
    One window refilled in memory.

    :param first_day: Its first station-local day, ISO.
    :param last_day: Its last station-local day, ISO, inclusive.
    :param dropped: The archive rows in the window, as stored.
    :param refilled: The rows gap fill brought back for it, raw.
    :param status: The fill's status (`GapFill.status`).
    """

    first_day: str
    last_day: str
    dropped: pd.DataFrame
    refilled: pd.DataFrame
    status: str


@dataclass
class DayDiff:
    """
    One day of a window, before and after the refill.

    :param day: The station-local day, ISO.
    :param rows_before: Rows the archive held for it.
    :param rows_after: Rows the refill brought back.
    :param before: Day-end rain counters as stored (daily is the day's
        highest); None when there were no rows.
    :param after: The same after the refill.
    """

    day: str
    rows_before: int
    rows_after: int
    before: Optional[Dict[str, float]]
    after: Optional[Dict[str, float]]

    @property
    def short(self) -> bool:
        """True when the refill brought back fewer rows than were dropped."""
        return self.rows_after < self.rows_before


@dataclass
class CrossCheck:
    """
    The refilled rows against the raw reference, matched on `dateutc`.

    :param compared: Rows found in both.
    :param over: Rows where a rain counter differs by more than the tolerance.
    :param max_diff: The largest difference seen, inches.
    :param unchecked: Refilled rows the reference does not hold (it ends
        2025-12-22, so the last days of 2025 cannot be checked).
    """

    compared: int
    over: int
    max_diff: float
    unchecked: int = 0


def window_bounds(first_day: str, last_day: str) -> Tuple[int, int]:
    """
    A window's `dateutc` bounds: local midnight starting `first_day` to
    local midnight ending `last_day`. Station-local days, because the stored
    `date` column is station-local and `apply_corrections` dates rows by it.

    :param first_day: The first station-local day, ISO.
    :param last_day: The last day, inclusive.
    :return: Start (inclusive) and end (exclusive), epoch ms.
    """
    end_day = (pd.Timestamp(last_day) + pd.Timedelta(days=1)).date()
    return from_local_date(first_day), from_local_date(end_day)


def refill_window(
    device: Dict,
    archive: pd.DataFrame,
    first_day: str,
    last_day: str,
    sleep: bool = True,
) -> Tuple[pd.DataFrame, WindowRefill]:
    """
    Drop a window's rows and refill them through gap fill, in memory.

    Only rows inside the window are taken from the fill, so rows outside it
    stay exactly as stored even where a fetched page reached past it.

    :param device: The station's `/devices` entry.
    :param archive: The archive, `dateutc` in epoch ms.
    :param first_day: The window's first station-local day.
    :param last_day: Its last day, inclusive.
    :param sleep: Wait a second per Ambient page.
    :return: The repaired archive and what the window lost and got back.
    """
    start_ms, end_ms = window_bounds(first_day, last_day)
    inside = (archive["dateutc"] >= start_ms) & (archive["dateutc"] < end_ms)
    dropped = archive[inside]
    kept = archive[~inside]

    pages = math.ceil((end_ms - start_ms) / FIVE_MIN_MS / PAGE_ROWS) + SLACK_PAGES
    outcome = awn.fill_archive_gap(
        device,
        kept,
        pd.to_datetime(start_ms, unit="ms", utc=True),
        pd.to_datetime(end_ms, unit="ms", utc=True),
        pages=pages,
        sleep=sleep,
    )
    filled = outcome.history_df
    refilled = filled[(filled["dateutc"] >= start_ms) & (filled["dateutc"] < end_ms)]
    repaired = awn.combine_full_history(kept, refilled)
    return repaired, WindowRefill(
        first_day, last_day, dropped, refilled.copy(), outcome.status
    )


def _local_day(df: pd.DataFrame) -> pd.Series:
    """Each row's station-local day, ISO."""
    stamps = pd.to_datetime(df["dateutc"], unit="ms", utc=True)
    return stamps.dt.tz_convert(DEFAULT_TZ).dt.strftime("%Y-%m-%d")


def _day_end(rows: pd.DataFrame) -> Optional[Dict[str, float]]:
    """A day's rain counters at its last reading; daily at its highest."""
    if rows.empty:
        return None
    last = rows.sort_values("dateutc").iloc[-1]
    values = {col: float(last[col]) for col in RAIN_COLUMNS if col in rows}
    if "dailyrainin" in rows:
        values["dailyrainin"] = float(rows["dailyrainin"].max())
    return values


def day_diffs(refill: WindowRefill) -> List[DayDiff]:
    """
    Every day of the window, before and after.

    :param refill: The window refilled.
    :return: One `DayDiff` per station-local day, oldest first.
    """
    days = pd.date_range(refill.first_day, refill.last_day, freq="D")
    before_day = _local_day(refill.dropped)
    after_day = _local_day(refill.refilled)
    diffs = []
    for day in days.strftime("%Y-%m-%d"):
        before = refill.dropped[before_day == day]
        after = refill.refilled[after_day == day]
        diffs.append(
            DayDiff(day, len(before), len(after), _day_end(before), _day_end(after))
        )
    return diffs


def cross_check(
    refilled: pd.DataFrame, reference: pd.DataFrame, tolerance: float
) -> CrossCheck:
    """
    Compare the refilled rain counters with the raw reference.

    :param refilled: The refilled rows.
    :param reference: The protected raw backup; may be empty.
    :param tolerance: Largest difference allowed, inches.
    :return: The comparison.
    """
    columns = [c for c in RAIN_COLUMNS if c in refilled and c in reference]
    if reference.empty or not columns:
        return CrossCheck(0, 0, 0.0, len(refilled))
    merged = refilled[["dateutc", *columns]].merge(
        reference[["dateutc", *columns]], on="dateutc", suffixes=("", "_ref")
    )
    unchecked = len(refilled) - len(merged)
    if merged.empty:
        return CrossCheck(0, 0, 0.0, unchecked)
    diffs = pd.concat(
        [(merged[c] - merged[c + "_ref"]).abs() for c in columns], axis=1
    ).max(axis=1)
    return CrossCheck(
        len(merged), int((diffs > tolerance).sum()), float(diffs.max()), unchecked
    )


def _pair(
    before: Optional[Dict[str, float]], after: Optional[Dict[str, float]], col: str
) -> str:
    """`before -> after` for one counter, `-` where a side has no rows."""

    def one(values: Optional[Dict[str, float]]) -> str:
        return "-" if values is None or col not in values else f"{values[col]:.3f}"

    return f"{one(before)} -> {one(after)}"


def day_line(diff: DayDiff) -> str:
    """
    One day of the report.

    :param diff: The day.
    :return: Rows and day-end counters, before -> after, and how short.
    """
    counters = "  ".join(
        f"{col.replace('rainin', '')} {_pair(diff.before, diff.after, col)}"
        for col in RAIN_COLUMNS
    )
    line = f"{diff.day}  rows {diff.rows_before} -> {diff.rows_after}  {counters}"
    if diff.short:
        line += f"  short by {diff.rows_before - diff.rows_after}"
    return line


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Read the command line.

    :param argv: Arguments, without the program name; `sys.argv` when None.
    :return: The parsed arguments; `window` is a list of (first, last).
    """
    parser = argparse.ArgumentParser(
        prog="python -m lookout.cli.repair_rain",
        description=(
            "Restore the raw rain counters in the windows #95 damaged: drop "
            "each window's rows, refill them through gap fill, cross-check "
            "against the raw backup. Dry run unless --write."
        ),
        epilog=RUN_NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket name")
    parser.add_argument(
        "--window",
        nargs=2,
        action="append",
        metavar=("FIRST", "LAST"),
        help="Station-local days to repair, inclusive; repeatable. "
        "Default: 2024-12-16 2024-12-31 and 2025-10-14 2025-12-31.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Largest cross-check difference allowed, inches (default 0.001)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Back up the live archive to .backups/ and save the repair",
    )
    args = parser.parse_args(argv)
    windows = args.window or DEFAULT_WINDOWS
    args.window = [(first, last) for first, last in windows]
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    The shell entry point.

    :param argv: Arguments, without the program name; `sys.argv` when None.
    :return: 0 when the repair checks out (and, with --write, was saved);
        1 when a day refilled short, the cross-check failed, or the raw
        reference is missing.
    """
    args = parse_args(argv)
    device, archive = load_station(args.bucket)
    mac = device["macAddress"]
    reference_key = REFERENCE_KEY.format(mac=mac)
    reference = get_df_from_s3(args.bucket, reference_key, file_type="parquet")

    problems: List[str] = []
    if reference.empty:
        problems.append(f"the raw reference {reference_key} is missing or empty")

    repaired = archive
    for first, last in args.window:
        repaired, refill = refill_window(device, repaired, first, last)
        emit(f"Window {first} to {last}: gap fill {refill.status}")
        for diff in day_diffs(refill):
            emit(day_line(diff))
            if diff.short:
                problems.append(f"{diff.day} refilled short")
        check = cross_check(refill.refilled, reference, args.tolerance)
        emit(
            f"Cross-check against {reference_key}: {check.compared} rows compared, "
            f"{check.over} over tolerance (max diff {check.max_diff:.3f}), "
            f"{check.unchecked} not in the reference"
        )
        if check.over:
            problems.append(f"{check.over} rows over tolerance in {first}..{last}")
        if not check.compared and not reference.empty:
            problems.append(f"no refilled row in {first}..{last} could be checked")
        emit()

    for problem in problems:
        emit(f"Problem: {problem}")
    if not args.write:
        emit("Dry run: nothing written. Rerun with --write to back up and save.")
        return 1 if problems else 0
    if problems:
        emit("Nothing written: fix the problems above first.")
        return 1

    backup_and_save_history(df=repaired, device=device, bucket=args.bucket)
    emit(f"Live archive backed up to .backups/{mac}/; repaired archive saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

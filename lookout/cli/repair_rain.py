"""
repair_rain.py: Restore the raw rain counters in the damaged windows (#95).

Until #95, every catchup run subtracted each known rain carryover from the
stored archive again, so the rows from a carryover day to the end of its
year hold counters that are too low (clipped at 0 at worst). Rows from 2026
on are not affected.

For each window the repair:

1. restores the rain counters of every row the protected raw backup holds
   (matched on `dateutc`) from the backup, at its 3-decimal precision, and
   leaves every other column as stored;
2. drops the rest of the window's rows (those past the backup's end, and
   any inside its range that it lacks, which are reported) and refills them
   from Ambient through gap fill, raw;
3. cross-checks the refilled rows the backup holds at 2 decimals, the
   precision Ambient serves now, and checks the restored rows equal the
   backup exactly.

A dry run is the default.

Usage:
    python -m lookout.cli.repair_rain [--bucket lookout] [--write]
        [--window FIRST LAST ...] [--tolerance 0.001]
"""

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import lookout.api.awn_controller as awn
from lookout.cli.diagnostics import emit, load_station
from lookout.config import DEFAULT_BUCKET
from lookout.core import archive_merge
from lookout.storage.storj import backup_and_save_history, read_df_from_s3
from lookout.utils.dateutc import DEFAULT_TZ, INTERVAL_MS, from_local_date

# Station-local days, first and last inclusive: from each carryover day the
# owner found damaged to the end of its year (owner check on #95).
DEFAULT_WINDOWS: List[Tuple[str, str]] = [
    ("2024-12-16", "2024-12-31"),
    ("2025-10-14", "2025-12-31"),
]
# The newest backup with raw counters, kept out of `.backups/` rotation.
REFERENCE_KEY = "backups/{mac}.raw-pre-95_20251222.parquet"
# Every rain counter: the four `apply_corrections` touches, plus the event
# and hourly totals, so a restored row's rain columns all come from one place.
RAIN_COLUMNS = (
    "dailyrainin",
    "weeklyrainin",
    "monthlyrainin",
    "yearlyrainin",
    "eventrainin",
    "hourlyrainin",
)
# The counters the per-day report shows.
REPORT_COLUMNS = ("dailyrainin", "weeklyrainin", "monthlyrainin", "yearlyrainin")
DEFAULT_TOLERANCE = 0.001  # inches, after rounding both sides to 2 decimals
# One Ambient page brings about this many new five-minute readings; the
# slack covers pages spent seeking over the station's own outages.
PAGE_ROWS = 246
SLACK_PAGES = 10
# Rows the backup lacks that lie closer together than this share one refill.
STRAY_JOIN_MS = 60 * 60 * 1000
# How many rows the backup lacks to list by time.
STRAYS_LISTED = 10

RUN_NOTES = """\
How to run it (after the #95 fix is deployed, with live secrets, on a host
that can reach Ambient and Storj; never while a catchup or refresh cron run
could save the archive at the same time):

  1. Dry run, the default. Repairs both windows in memory and writes
     nothing. For each window it prints how many rows were restored from
     the raw backup and how many were refilled from Ambient, any row inside
     the backup's range that the backup lacks (refilled from Ambient), each
     day's rows and day-end rain counters before -> after, the cross-check
     of the refilled rows, and the check that restored rows equal the
     backup:

         PYTHONPATH=. python -m lookout.cli.repair_rain

  2. Read it. Every day should show "rows N -> N" (or more after) with no
     "short" line, the cross-check "0 mismatched", and "0 differ from the
     backup". The exit status is 1 if not.

  3. The real run. Same checks; if any day refills short or a check fails,
     nothing is written. Otherwise it backs the live archive up to
     .backups/<MAC>/<today>/ and saves the repaired archive:

         PYTHONPATH=. python -m lookout.cli.repair_rain --write

  The backup ends 2025-12-22, so only the last days of 2025 are refilled
  from Ambient (about 3 pages, waiting a second per page). A window can be
  given instead of the defaults, as station-local days (repeatable):
  --window 2025-10-14 2025-12-31.
"""

Numeric = Union[pd.Series, pd.DataFrame]


@dataclass
class WindowRepair:
    """
    One window repaired in memory.

    :param first_day: Its first station-local day, ISO.
    :param last_day: Its last station-local day, ISO, inclusive.
    :param dropped: The archive rows in the window, as stored.
    :param restored: Window rows whose rain counters came from the backup.
    :param refilled: Rows gap fill brought back for the rest, raw.
    :param strays: Window rows inside the backup's range that it lacks, as
        stored; they were dropped and refilled.
    :param statuses: Each gap fill's status (`GapFill.status`).
    :param replaced: What the replace swapped in the archive.
    """

    first_day: str
    last_day: str
    dropped: pd.DataFrame
    restored: pd.DataFrame
    refilled: pd.DataFrame
    strays: pd.DataFrame
    statuses: List[str] = field(default_factory=list)
    replaced: archive_merge.ReplaceReport = field(
        default_factory=archive_merge.ReplaceReport
    )

    @property
    def after(self) -> pd.DataFrame:
        """The window's rows after the repair."""
        return pd.concat([self.restored, self.refilled], ignore_index=True)


@dataclass
class DayDiff:
    """
    One day of a window, before and after the repair.

    :param day: The station-local day, ISO.
    :param rows_before: Rows the archive held for it.
    :param rows_after: Rows it holds after the repair.
    :param before: Day-end rain counters as stored (daily is the day's
        highest); None when there were no rows.
    :param after: The same after the repair.
    """

    day: str
    rows_before: int
    rows_after: int
    before: Optional[Dict[str, float]]
    after: Optional[Dict[str, float]]

    @property
    def short(self) -> bool:
        """True when the repair left fewer rows than were stored."""
        return self.rows_after < self.rows_before


@dataclass
class CrossCheck:
    """
    The refilled rows against the raw reference, matched on `dateutc`.

    :param compared: Rows found in both.
    :param over: Rows where a rain counter differs by more than the
        tolerance at 2 decimals, or is missing on one side only.
    :param max_diff: The largest difference seen at 2 decimals, inches.
    :param unchecked: Refilled rows the reference does not hold.
    """

    compared: int
    over: int
    max_diff: float
    unchecked: int = 0


def round_half_up(values: Numeric) -> Numeric:
    """
    Round to 2 decimals, halves up, as Ambient serves rain counters now.

    The epsilon keeps a stored 3-decimal half (1.005, held as 1.00499...)
    rounding up as its decimal form does.

    :param values: Inches.
    :return: The same shape, to the hundredth.
    """
    return np.floor(values * 100 + 0.5 + 1e-6) / 100


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


def restore_from_reference(rows: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """
    Set the rain counters of `rows` from the reference, matched on `dateutc`.

    :param rows: Archive rows, every one held by the reference.
    :param reference: The raw backup.
    :return: A copy of `rows`; its rain columns the reference also has are
        the reference's values, every other column as it was.
    """
    out = rows.copy()
    columns = [c for c in RAIN_COLUMNS if c in out and c in reference]
    source = reference.drop_duplicates("dateutc", keep="last").set_index("dateutc")
    for col in columns:
        out[col] = out["dateutc"].map(source[col]).to_numpy()
    return out


def refill_spans(
    strays: Iterable[int], start_ms: int, end_ms: int, covered: Optional[Tuple]
) -> List[Tuple[int, int]]:
    """
    The stretches of a window to refill from Ambient.

    :param strays: `dateutc` of window rows inside the backup's range that
        the backup lacks.
    :param start_ms: The window's start, inclusive.
    :param end_ms: Its end, exclusive.
    :param covered: The backup's first and last `dateutc`, or None when
        there is no backup.
    :return: (start, end exclusive) pairs, epoch ms, oldest first: the
        window outside the backup's range, and each cluster of strays.
    """
    if covered is None:
        return [(start_ms, end_ms)]
    lo, hi = covered
    spans = []
    if lo > start_ms:
        spans.append((start_ms, min(lo, end_ms)))
    cluster: List[int] = []
    for ms in sorted(strays):
        if cluster and ms - cluster[-1] >= STRAY_JOIN_MS:
            spans.append((cluster[0], cluster[-1] + 1))
            cluster = []
        cluster.append(ms)
    if cluster:
        spans.append((cluster[0], cluster[-1] + 1))
    if hi + 1 < end_ms:
        spans.append((max(hi + 1, start_ms), end_ms))
    return sorted(spans)


def repair_window(
    device: Dict,
    archive: pd.DataFrame,
    reference: pd.DataFrame,
    first_day: str,
    last_day: str,
    sleep: bool = True,
) -> Tuple[pd.DataFrame, WindowRepair]:
    """
    Restore a window from the backup and refill the rest, in memory.

    Rows outside the window stay exactly as stored, even where a fetched
    page reached past it.

    :param device: The station's `/devices` entry.
    :param archive: The archive, `dateutc` in epoch ms.
    :param reference: The protected raw backup; may be empty.
    :param first_day: The window's first station-local day.
    :param last_day: Its last day, inclusive.
    :param sleep: Wait a second per Ambient page.
    :return: The repaired archive and what the window lost and got back.
    """
    start_ms, end_ms = window_bounds(first_day, last_day)
    inside = (archive["dateutc"] >= start_ms) & (archive["dateutc"] < end_ms)
    window = archive[inside]

    covered = None
    in_range = pd.Series(False, index=window.index)
    matched = in_range
    if not reference.empty:
        covered = (int(reference["dateutc"].min()), int(reference["dateutc"].max()))
        in_range = window["dateutc"].between(*covered)
        matched = in_range & window["dateutc"].isin(reference["dateutc"])
    restored = restore_from_reference(window[matched], reference)
    strays = window[in_range & ~matched]

    base = archive_merge.replace(archive, start_ms, end_ms, restored).archive
    statuses = []
    parts = []
    for span_start, span_end in refill_spans(
        strays["dateutc"], start_ms, end_ms, covered
    ):
        pages = (
            math.ceil((span_end - span_start) / INTERVAL_MS / PAGE_ROWS) + SLACK_PAGES
        )
        outcome = awn.fill_archive_gap(
            device,
            base,
            pd.to_datetime(span_start, unit="ms", utc=True),
            pd.to_datetime(span_end, unit="ms", utc=True),
            pages=pages,
            sleep=sleep,
        )
        statuses.append(outcome.status)
        filled = outcome.history_df
        parts.append(
            filled[
                (filled["dateutc"] >= span_start)
                & (filled["dateutc"] < span_end)
                & ~filled["dateutc"].isin(restored["dateutc"])
            ]
        )
    refilled = pd.concat(parts, ignore_index=True) if parts else window.iloc[0:0]
    refilled = refilled.drop_duplicates("dateutc", keep="last")
    after = pd.concat([restored, refilled], ignore_index=True)
    repaired, swapped = archive_merge.replace(archive, start_ms, end_ms, after)
    return repaired, WindowRepair(
        first_day,
        last_day,
        window,
        restored,
        refilled.reset_index(drop=True),
        strays,
        statuses,
        swapped,
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
    values = {col: float(last[col]) for col in REPORT_COLUMNS if col in rows}
    if "dailyrainin" in rows:
        values["dailyrainin"] = float(rows["dailyrainin"].max())
    return values


def day_diffs(repair: WindowRepair) -> List[DayDiff]:
    """
    Every day of the window, before and after.

    :param repair: The window repaired.
    :return: One `DayDiff` per station-local day, oldest first.
    """
    days = pd.date_range(repair.first_day, repair.last_day, freq="D")
    after_rows = repair.after
    before_day = _local_day(repair.dropped)
    after_day = _local_day(after_rows)
    diffs = []
    for day in days.strftime("%Y-%m-%d"):
        before = repair.dropped[before_day == day]
        after = after_rows[after_day == day]
        diffs.append(
            DayDiff(day, len(before), len(after), _day_end(before), _day_end(after))
        )
    return diffs


def cross_check(
    refilled: pd.DataFrame, reference: pd.DataFrame, tolerance: float
) -> CrossCheck:
    """
    Compare refilled rain counters with the raw reference, at 2 decimals.

    Ambient serves 2 decimals now and the reference holds 3, so both sides
    are rounded (halves up) before comparing. A value missing on one side
    only is a mismatch; missing on both is not.

    :param refilled: The rows refilled from Ambient.
    :param reference: The protected raw backup; may be empty.
    :param tolerance: Largest difference allowed after rounding, inches.
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
    got = round_half_up(merged[columns].astype(float))
    want = round_half_up(merged[[c + "_ref" for c in columns]].astype(float))
    want.columns = columns
    diffs = (got - want).abs()
    one_sided = got.isna() ^ want.isna()
    bad = ((diffs > tolerance) | one_sided).any(axis=1)
    max_diff = float(np.nan_to_num(diffs.to_numpy(), nan=0.0).max())
    return CrossCheck(len(merged), int(bad.sum()), max_diff, unchecked)


def verify_restored(
    repaired: pd.DataFrame, reference: pd.DataFrame, dateutcs: Iterable[int]
) -> Tuple[int, int]:
    """
    Check the restored rows of the repaired archive equal the backup exactly.

    :param repaired: The repaired archive.
    :param reference: The protected raw backup.
    :param dateutcs: The restored rows' `dateutc`.
    :return: (rows checked, rows whose rain counters differ). A row the
        repaired archive lost, or NaN on one side only, counts as differing.
    """
    wanted = pd.Index(list(dateutcs))
    if wanted.empty:
        return 0, 0
    columns = [c for c in RAIN_COLUMNS if c in repaired and c in reference]
    got = repaired.drop_duplicates("dateutc").set_index("dateutc")
    want = reference.drop_duplicates("dateutc").set_index("dateutc")
    got = got.reindex(wanted)[columns]
    want = want.reindex(wanted)[columns]
    same = (got == want) | (got.isna() & want.isna())
    lost = ~wanted.isin(repaired["dateutc"])
    differ = ~same.all(axis=1).to_numpy() | lost
    return len(wanted), int(differ.sum())


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
        for col in REPORT_COLUMNS
    )
    line = f"{diff.day}  rows {diff.rows_before} -> {diff.rows_after}  {counters}"
    if diff.short:
        line += f"  short by {diff.rows_before - diff.rows_after}"
    return line


def _rows(n: int) -> str:
    """`1 row`, `2 rows`."""
    return f"{n} row" if n == 1 else f"{n} rows"


def report_window(
    repair: WindowRepair,
    repaired: pd.DataFrame,
    reference: pd.DataFrame,
    reference_key: str,
    tolerance: float,
) -> List[str]:
    """
    Print one window's report.

    :param repair: The window repaired.
    :param repaired: The repaired archive.
    :param reference: The protected raw backup; may be empty.
    :param reference_key: Its key, for the report.
    :param tolerance: The cross-check tolerance.
    :return: The window's problems; empty when it checks out.
    """
    first, last = repair.first_day, repair.last_day
    problems = []
    fills = ", ".join(repair.statuses) or "not needed"
    emit(
        f"Window {first} to {last}: restored {len(repair.restored)} from the "
        f"backup, refilled {len(repair.refilled)} from Ambient (gap fill: {fills})"
    )
    if not repair.strays.empty:
        emit(
            f"{_rows(len(repair.strays))} in the backup's range but not in it; "
            "dropped and refilled from Ambient:"
        )
        stamps = pd.to_datetime(repair.strays["dateutc"], unit="ms", utc=True)
        for stamp in stamps.sort_values().iloc[:STRAYS_LISTED]:
            emit(f"  {stamp.isoformat()}")
        if len(stamps) > STRAYS_LISTED:
            emit(f"  ... and {len(stamps) - STRAYS_LISTED} more")
    emit(f"Archive {repair.replaced.describe()}")
    for diff in day_diffs(repair):
        emit(day_line(diff))
        if diff.short:
            problems.append(f"{diff.day} refilled short")
    check = cross_check(repair.refilled, reference, tolerance)
    emit(
        f"Cross-check of refilled rows against {reference_key}, at 2 decimals: "
        f"{check.compared} compared, {check.over} mismatched "
        f"(max diff {check.max_diff:.3f}), {check.unchecked} not in the reference"
    )
    if check.over:
        problems.append(f"{check.over} refilled rows mismatched in {first}..{last}")
    checked, differ = verify_restored(repaired, reference, repair.restored["dateutc"])
    emit(f"{checked} restored rows checked, {differ} differ from the backup")
    if differ:
        problems.append(f"{differ} restored rows differ from the backup")
    emit()
    return problems


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Read the command line.

    :param argv: Arguments, without the program name; `sys.argv` when None.
    :return: The parsed arguments; `window` is a list of (first, last).
    """
    parser = argparse.ArgumentParser(
        prog="python -m lookout.cli.repair_rain",
        description=(
            "Restore the raw rain counters in the windows #95 damaged: set "
            "the rain columns of rows the raw backup holds from the backup, "
            "refill the rest through gap fill, and check both. Dry run "
            "unless --write."
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
        help="Largest difference allowed when checking refilled rows "
        "against the backup, after rounding both to 2 decimals, inches "
        "(default 0.001)",
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
        1 when a day refilled short, a check failed, or the raw reference is
        missing.
    :raises StorageReadError: If the archive or the raw reference read
        failed; nothing is written then.
    """
    args = parse_args(argv)
    device, archive = load_station(args.bucket)
    mac = device["macAddress"]
    reference_key = REFERENCE_KEY.format(mac=mac)
    # A failed read raises (ADR-0004): the repair never compares against a
    # reference it could not read.
    reference = read_df_from_s3(args.bucket, reference_key, file_type="parquet")
    if reference is None:
        emit(f"Problem: the raw reference {reference_key} is missing or empty")
        emit("Nothing written: fix the problems above first.")
        return 1

    problems: List[str] = []
    if reference.empty:
        problems.append(f"the raw reference {reference_key} is missing or empty")

    repaired = archive
    for first, last in args.window:
        repaired, repair = repair_window(device, repaired, reference, first, last)
        problems += report_window(
            repair, repaired, reference, reference_key, args.tolerance
        )

    for problem in problems:
        emit(f"Problem: {problem}")
    if not args.write:
        emit("Dry run: nothing written. Rerun with --write to back up and save.")
        return 1 if problems else 0
    if problems:
        emit("Nothing written: fix the problems above first.")
        return 1

    # An explicit repair: the save may drop or blank what it corrects (#190).
    backup_and_save_history(
        df=repaired,
        device=device,
        bucket=args.bucket,
        stored=archive,
        repair=True,
    )
    emit(f"Live archive backed up to .backups/{mac}/; repaired archive saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

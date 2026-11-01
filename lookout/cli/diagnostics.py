"""
diagnostics.py: Archive diagnostics for the pinned station, as plain text.

The Streamlit Diagnostics tab's maintenance work, for an agent's terminal. It
reads secrets only through `lookout.secrets_loader` (env vars, then
`~/.config/lookout/secrets.toml`) and never imports Streamlit.

Usage:
    python -m lookout.cli.diagnostics <command> [options]

Commands:
    gaps            List archive gaps (--threshold minutes, --days N or
                    --since/--until dates).
    fill-gap        Fill one gap, picked by --index (a row of `gaps`, same
                    window flags) or by --start and --end, then save the
                    archive after a backup. --dry-run fetches and writes
                    nothing. Exits 1 when Ambient has no data for the gap.
    coverage        Samples per hour per day, as a text grid (--days N).
    status          The archive's newest record against the device's live
                    reading, and the lag between them.
    last-record     The device's live reading.
    energy-catalog  Regenerate the solar energy catalog from the archive.
                    Reports only; --save backs up the stored catalog and
                    writes the new one. --save --dry-run says what --save
                    would do; --dry-run alone is refused.
    rain-catalog    Rebuild the Event Catalog from the archive. A dry run by
                    default: the stored catalog's read outcome (found, missing
                    or failed) and event count, the detected event count, and
                    the events added or dropped. --save backs up the stored
                    object as it is (a corrupt catalog is kept), then writes;
                    it refuses to write when the backup fails.

Every command takes --bucket. Day windows end at the archive's newest record,
not the wall clock, so a stalled station still shows its last good days.
Exit status: 0 done, 1 refused or failed.
"""

import argparse
import sys
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

import lookout.api.awn_controller as awn
from lookout.api.ambient_client import StationNotFoundError, get_station_device
from lookout.config import DEFAULT_BUCKET
from lookout.core import energy_catalog_func
from lookout.core.energy_catalog import EnergyCatalog
from lookout.core.rain_events import RainEventCatalog
from lookout.storage.storj import (
    BackupError,
    StorageReadError,
    backup_and_save_history,
    load_archive,
)
from lookout.utils.dateutc import DEFAULT_TZ
from lookout.utils.log_util import app_logger

logger = app_logger(__name__, log_file="diagnostics.log")

DEFAULT_THRESHOLD_MINUTES = 30
DEFAULT_GAP_DAYS = 84
DEFAULT_COVERAGE_DAYS = 14
MAX_FILL_PAGES = 20
STAMP = "%Y-%m-%d %H:%M"


# ---------- helpers ----------


def emit(line: str = "") -> None:
    """
    Print one line of the report.

    :param line: The text.
    """
    print(line)


def to_ms(stamp: pd.Timestamp) -> int:
    """
    A timestamp as epoch milliseconds.

    :param stamp: A timezone-aware timestamp.
    :return: Milliseconds since the epoch.
    """
    return int(stamp.value // 10**6)


def to_utc(ms: int) -> pd.Timestamp:
    """
    An epoch-ms value as a UTC timestamp.

    :param ms: Epoch milliseconds.
    :return: The timestamp, UTC.
    """
    return pd.to_datetime(int(ms), unit="ms", utc=True)


def human_duration(minutes: float) -> str:
    """
    A span as minutes, hours or days, whichever reads best.

    :param minutes: The span in minutes.
    :return: E.g. "45 minutes", "2.5 hours", "3.0 days".
    """
    if minutes < 60:
        return f"{minutes:.0f} minutes"
    if minutes < 1440:
        return f"{minutes / 60:.1f} hours"
    return f"{minutes / 1440:.1f} days"


def load_station(bucket: str) -> Tuple[Dict, pd.DataFrame]:
    """
    The pinned station and its archive, normalized (ascending `dateutc`).

    Read strictly (#160): `fill-gap` and the rain repair save over the
    archive, and hand it to the save as `stored`, which guards against it.

    :param bucket: The bucket.
    :return: The station's `/devices` entry and its archive.
    :raises StationNotFoundError: If `STATION_MAC` is not on the account.
    :raises StorageReadError: If the archive is missing, unreadable, or has
        no valid rows.
    """
    device = get_station_device()
    # Straight from Storj: the dev-mode local cache would write a file here.
    return device, load_archive(bucket, device["macAddress"])


def station_tz(device: Dict) -> str:
    """
    The station's time zone name.

    :param device: The station's `/devices` entry.
    :return: Its `lastData.tz`, or DEFAULT_TZ when absent.
    """
    return (device.get("lastData") or {}).get("tz") or DEFAULT_TZ


def window(
    archive: pd.DataFrame,
    days: Optional[int],
    since: Optional[str],
    until: Optional[str],
) -> pd.DataFrame:
    """
    The archive rows inside a window.

    :param archive: The normalized archive.
    :param days: The window's length ending at the newest record; ignored
        when `since` is given.
    :param since: First day of the window (UTC date), inclusive.
    :param until: Last day (UTC date), inclusive; the newest record when None.
    :return: The rows in the window.
    """
    newest = to_utc(archive["dateutc"].max())
    end = (
        pd.Timestamp(until, tz="UTC") + pd.Timedelta(days=1)
        if until
        else newest + pd.Timedelta(milliseconds=1)
    )
    if since:
        start = pd.Timestamp(since, tz="UTC")
    else:
        start = newest - pd.Timedelta(days=days or DEFAULT_GAP_DAYS)
    start_ms = to_ms(start)
    end_ms = to_ms(end)
    return archive[(archive["dateutc"] >= start_ms) & (archive["dateutc"] < end_ms)]


def find_gaps(rows: pd.DataFrame, threshold_minutes: float) -> pd.DataFrame:
    """
    The stretches between consecutive records longer than a threshold.

    :param rows: Archive rows, any order.
    :param threshold_minutes: The smallest gap to list.
    :return: One row per gap, oldest first: `start`, `end` (UTC timestamps
        of the records either side) and `minutes`.
    """
    ms = rows["dateutc"].sort_values().reset_index(drop=True)
    minutes = ms.diff() / 60000
    found = minutes > threshold_minutes
    return pd.DataFrame(
        {
            "start": [to_utc(v) for v in ms.shift(1)[found]],
            "end": [to_utc(v) for v in ms[found]],
            "minutes": minutes[found].round(1).tolist(),
        }
    )


def add_window_args(parser: argparse.ArgumentParser) -> None:
    """
    Add the gap-window flags `gaps` and `fill-gap` share.

    :param parser: The subcommand's parser.
    """
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD_MINUTES,
        help="Smallest gap to list, in minutes (default %(default)s)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_GAP_DAYS,
        help="Window length ending at the newest record (default %(default)s)",
    )
    parser.add_argument("--since", help="Window start, YYYY-MM-DD UTC; beats --days")
    parser.add_argument("--until", help="Window end, YYYY-MM-DD UTC, inclusive")


def gaps_in(archive: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    The gaps in the window the flags describe.

    :param archive: The normalized archive.
    :param args: Parsed arguments with the window flags.
    :return: `find_gaps` over the window.
    """
    return find_gaps(window(archive, args.days, args.since, args.until), args.threshold)


# ---------- commands ----------


def cmd_gaps(args: argparse.Namespace) -> int:
    """
    List archive gaps.

    :param args: Parsed arguments.
    :return: The exit status.
    """
    device, archive = load_station(args.bucket)
    gaps = gaps_in(archive, args)
    emit(f"Station {device['macAddress']}, {len(archive)} archive rows")
    emit(f"Gaps over {args.threshold} minutes: {len(gaps)}")
    if gaps.empty:
        return 0
    emit("index  start (UTC)       end (UTC)         minutes  length")
    for index, gap in gaps.iterrows():
        emit(
            f"{index:>5}  {gap['start']:{STAMP}}  {gap['end']:{STAMP}}  "
            f"{gap['minutes']:>7}  {human_duration(gap['minutes'])}"
        )
    emit(f"Longest: {human_duration(gaps['minutes'].max())}")
    return 0


def resolve_gap(
    archive: pd.DataFrame, args: argparse.Namespace
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    The gap `fill-gap` was asked for.

    :param archive: The normalized archive.
    :param args: Parsed arguments: `--index`, or `--start` and `--end`.
    :return: Start and end (UTC), or None after saying why not.
    """
    if args.index is not None:
        gaps = gaps_in(archive, args)
        if args.index not in gaps.index:
            emit(f"No gap {args.index}: the window holds {len(gaps)} (see `gaps`).")
            return None
        gap = gaps.loc[args.index]
        return gap["start"], gap["end"]
    if args.start and args.end:
        return pd.to_datetime(args.start, utc=True), pd.to_datetime(args.end, utc=True)
    emit("Pick a gap: --index N, or --start and --end.")
    return None


def cmd_fill_gap(args: argparse.Namespace) -> int:
    """
    Fill one gap from Ambient, back up the archive and save it.

    :param args: Parsed arguments.
    :return: The exit status.
    """
    device, archive = load_station(args.bucket)
    picked = resolve_gap(archive, args)
    if picked is None:
        return 1
    start, end = picked
    emit(f"Gap: {start:{STAMP}} to {end:{STAMP}} UTC ({end - start})")

    if args.dry_run:
        before = archive[archive["dateutc"] < to_ms(start)]
        baseline = (
            f"{to_utc(before['dateutc'].max()):{STAMP}} UTC"
            if not before.empty
            else "none (the fill would be refused)"
        )
        emit("Dry run: nothing fetched, nothing written.")
        emit(f"Would fetch Ambient history after {baseline}, up to the gap's end.")
        emit(f"At most {MAX_FILL_PAGES} pages, then merge (raw), back up, save.")
        return 0

    outcome = awn.fill_archive_gap(device, archive, start, end)
    if outcome.status == "refused":
        emit("Refused: the gap is inverted or no archive data precedes it.")
        return 1
    if outcome.skipped:
        emit(
            "Ambient had no new data for this gap (0 added, 0 filled, "
            "0 updated). Nothing saved."
        )
        return 1

    backup_and_save_history(
        df=outcome.history_df,
        device=device,
        bucket=args.bucket,
        stored=archive,
    )
    report = outcome.report
    emit(
        f"Merged {report.added} added, {report.filled} filled, "
        f"{report.updated} updated; archive backed up and saved."
    )
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    """
    Print samples per hour per day.

    :param args: Parsed arguments.
    :return: The exit status.
    """
    device, archive = load_station(args.bucket)
    if archive.empty:
        emit("The archive is empty.")
        return 1
    tz = station_tz(device)
    rows = window(archive, args.days, None, None)
    local = pd.to_datetime(rows["dateutc"], unit="ms", utc=True).dt.tz_convert(tz)
    counts = (
        pd.DataFrame({"day": local.dt.date, "hour": local.dt.hour})
        .groupby(["day", "hour"])
        .size()
    )
    days = pd.date_range(min(counts.index)[0], max(counts.index)[0], freq="D").date
    emit(f"Samples per hour, station time ({tz}); '.' is none; last {args.days} days")
    emit("date        " + "".join(f"{h:>3}" for h in range(24)) + "  total")
    for day in days:
        cells = [int(counts.get((day, h), 0)) for h in range(24)]
        emit(
            f"{day}  "
            + "".join(f"{c:>3}" if c else "  ." for c in cells)
            + f"  {sum(cells):>5}"
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """
    The archive's newest record against the device's live reading.

    :param args: Parsed arguments.
    :return: The exit status.
    """
    device, archive = load_station(args.bucket)
    live_ms = (device.get("lastData") or {}).get("dateutc")
    emit(f"Station {device['macAddress']}")
    if archive.empty:
        emit("Archive: empty")
    else:
        newest = int(archive["dateutc"].max())
        emit(f"Archive: {len(archive)} rows, newest {to_utc(newest):{STAMP}} UTC")
    if live_ms is None:
        emit("Device: no live reading")
        return 1
    emit(f"Device: last reading {to_utc(live_ms):{STAMP}} UTC")
    if not archive.empty:
        lag = (live_ms - newest) / 60000
        emit(
            f"Archive lag: {human_duration(abs(lag))}" + (" (ahead)" if lag < 0 else "")
        )
    return 0


def cmd_last_record(args: argparse.Namespace) -> int:
    """
    Print the device's live reading.

    :param args: Parsed arguments.
    :return: The exit status.
    """
    device = get_station_device()
    last = device.get("lastData") or {}
    if not last:
        emit("The device reports no live reading.")
        return 1
    emit(f"Station {device['macAddress']}")
    if "dateutc" in last:
        emit(f"reading time: {to_utc(last['dateutc']):{STAMP}} UTC")
    for key in sorted(last):
        emit(f"{key}: {last[key]}")
    return 0


def cmd_energy_catalog(args: argparse.Namespace) -> int:
    """
    Regenerate the energy catalog; save it only with --save.

    The stored catalog is read by the reader contract (#183): a failed or
    corrupt read raises, so the command reports it and never saves over a
    catalog it could not read.

    :param args: Parsed arguments.
    :return: The exit status; 1 for --dry-run without --save, which would
        otherwise do nothing a plain report does not.
    :raises StorageReadError: If the stored catalog's read failed or it is
        corrupt.
    """
    if args.dry_run and not args.save:
        emit("Refused: --dry-run only applies with --save.")
        return 1
    device, archive = load_station(args.bucket)
    if archive.empty:
        emit("The archive is empty.")
        return 1
    stored = EnergyCatalog(device["macAddress"], bucket=args.bucket).load_catalog()
    if "date" not in archive.columns:
        archive["date"] = pd.to_datetime(
            archive["dateutc"], unit="ms", utc=True
        ).dt.tz_convert(station_tz(device))
    periods = energy_catalog_func.detect_and_calculate_periods(
        archive, bucket=args.bucket, auto_save=False
    )
    if periods.empty:
        emit("No energy periods could be calculated from the archive.")
        return 1
    count = "none" if stored is None else f"{len(stored)} periods"
    emit(f"Stored catalog: {count}")
    emit(
        f"Regenerated: {len(periods)} periods, "
        f"{periods['period_start'].min():{STAMP}} to "
        f"{periods['period_end'].max():{STAMP}}, "
        f"{periods['energy_kwh'].sum():.1f} kWh"
    )
    if not args.save:
        emit("Not saved. Pass --save to back up the stored catalog and replace it.")
        return 0
    if args.dry_run:
        emit("Dry run: would back up the stored catalog and save the regenerated one.")
        return 0
    if not energy_catalog_func.save_energy_catalog(periods, args.bucket):
        emit("Save failed (the backup or the write); the stored catalog is unchanged.")
        return 1
    emit("Stored catalog backed up and replaced.")
    return 0


def cmd_rain_catalog(args: argparse.Namespace) -> int:
    """
    Rebuild the Event Catalog from the archive; write it only with --save.

    The way out of a corrupt catalog, which stops the refresh run (ADR-0004).
    --save backs up the stored object as it is, then writes; when the backup
    fails nothing is written.

    :param args: Parsed arguments.
    :return: The exit status; 1 when nothing was detected or the backup or
        write failed.
    """
    device, archive = load_station(args.bucket)
    store = RainEventCatalog(device["macAddress"], bucket=args.bucket)
    stored: Optional[pd.DataFrame] = None
    try:
        stored = store.load_catalog()
        if stored is None:
            emit("Stored catalog: missing")
        else:
            emit(f"Stored catalog: found, {len(stored)} events")
    except StorageReadError as e:
        emit(f"Stored catalog: failed ({e})")
    detected = store.detect_and_catalog_events(archive, auto_save=False)
    if detected.empty:
        emit("No rain events could be detected from the archive.")
        return 1
    emit(f"Detected: {len(detected)} events")
    if stored is None:
        old_starts: set = set()
    else:
        old_starts = set(pd.to_datetime(stored["start_time"], utc=True))
    new_starts = set(pd.to_datetime(detected["start_time"], utc=True))
    added, dropped = len(new_starts - old_starts), len(old_starts - new_starts)
    emit(f"Added: {added}, Dropped: {dropped}")
    if not args.save:
        emit("Not saved. Pass --save to back up the stored catalog and replace it.")
        return 0
    try:
        backup = store.backup_catalog()
    except BackupError as e:
        emit(f"Backup failed ({e}); the stored catalog is unchanged.")
        return 1
    if not store.save_catalog(detected):
        emit("Save failed; the stored catalog may be unchanged.")
        return 1
    if backup is None:
        emit("No catalog was stored (no backup); the rebuilt one is saved.")
    else:
        emit(f"Stored catalog backed up to {backup} and replaced.")
    return 0


# ---------- the shell ----------


def build_parser() -> argparse.ArgumentParser:
    """
    The command line.

    :return: The parser, one subparser per command.
    """
    parser = argparse.ArgumentParser(description="Archive diagnostics.")
    commands = parser.add_subparsers(dest="command", required=True)

    def command(
        name: str, handler: Callable[[argparse.Namespace], int], help_: str
    ) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_)
        sub.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket name")
        sub.set_defaults(handler=handler)
        return sub

    gaps = command("gaps", cmd_gaps, "List archive gaps")
    add_window_args(gaps)

    fill = command("fill-gap", cmd_fill_gap, "Fill one gap, then save the archive")
    add_window_args(fill)
    fill.add_argument("--index", type=int, help="A row of `gaps` (same window flags)")
    fill.add_argument("--start", help="Gap start, ISO time, UTC")
    fill.add_argument("--end", help="Gap end, ISO time, UTC")
    fill.add_argument("--dry-run", action="store_true", help="Fetch and write nothing")

    cover = command("coverage", cmd_coverage, "Samples per hour per day")
    cover.add_argument("--days", type=int, default=DEFAULT_COVERAGE_DAYS)

    command("status", cmd_status, "Archive newest record against the device")
    command("last-record", cmd_last_record, "The device's live reading")

    energy = command("energy-catalog", cmd_energy_catalog, "Regenerate the catalog")
    energy.add_argument("--save", action="store_true", help="Back up and save it")
    energy.add_argument(
        "--dry-run", action="store_true", help="With --save: write nothing"
    )

    rain = command("rain-catalog", cmd_rain_catalog, "Rebuild the Event Catalog")
    rain.add_argument("--save", action="store_true", help="Back up and save it")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    The shell entry point.

    :param argv: Arguments, without the program name; `sys.argv` when None.
    :return: The exit status.
    """
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except StationNotFoundError as e:
        emit(f"Station not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"{args.command} failed: {e}")
        emit(f"{args.command} failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

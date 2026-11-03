"""
stations.py: the station inventory's one writer, as a command (ADR-0006).

Usage:
    python -m lookout.cli.stations scan [--bucket lookout]
    python -m lookout.cli.stations add <MAC> [--bucket lookout]

Commands:
    scan  List every device on the Ambient account (MAC and Ambient name),
          marking which are in the station inventory, and any inventory
          station the account no longer lists. Writes nothing.
    add   Add one station. It must be on the account and not in the
          inventory. Then it gets an archive: one stored for the MAC is
          adopted as it is, never rewritten; with none stored, the first
          archive is built from Ambient's history, paging back from now
          until Ambient returns no older readings, and saved through the
          archive save. The inventory is backed up before the archive step,
          so a failed backup writes nothing, and the station is appended
          after it, so a failed backfill leaves the inventory as it was.
          The first add seeds the inventory, and that station is the
          default station. Running `add` again adopts an archive that a
          failed inventory write left behind.

`add` is an archive writer: it takes the archive-writer lock (#244) before
any work, and exits 1 naming the lock file when another writer holds it.
`scan` takes no lock. Every Ambient request goes through the client's
throttle (a second before each). Exit status: 0 done, 1 refused or failed.
"""

import argparse
import sys
from typing import Callable, Dict, List, Optional

import pandas as pd

import lookout.api.awn_controller as awn
from lookout.api.ambient_client import StationNotFoundError, find_station, get_devices
from lookout.config import DEFAULT_BUCKET, HISTORY_FILE_TYPE
from lookout.core import archive_merge
from lookout.storage import station_inventory, storj
from lookout.utils.archive_lock import ArchiveLockHeld, writer_lock
from lookout.utils.dateutc import normalize
from lookout.utils.log_util import app_logger

logger = app_logger(__name__, log_file="stations.log")

STAMP = "%Y-%m-%d %H:%M"


class Refused(RuntimeError):
    """The command was refused before it wrote anything."""


def emit(line: str = "") -> None:
    """
    Print one line of the command's report to stdout.

    :param line: The text.
    """
    print(line, flush=True)


def device_name(device: Dict) -> str:
    """
    A device's Ambient name.

    :param device: A `/devices` entry.
    :return: `info.name`, or "(unnamed)".
    """
    return str((device.get("info") or {}).get("name") or "(unnamed)")


def account_devices() -> List[Dict]:
    """
    The account's devices, refusing an empty list.

    :return: The `/devices` body.
    :raises Refused: If Ambient lists no device: an API error, wrong keys,
        or no station on the account, none of which `scan` can tell apart.
    """
    devices = get_devices()
    if not devices:
        raise Refused(
            "Ambient returned no devices (API error, wrong keys, or no station "
            "on the account)"
        )
    return devices


def span(archive: pd.DataFrame) -> str:
    """
    An archive's reach, for the report.

    :param archive: A normalized archive.
    :return: "<n> readings, <oldest> to <newest> UTC".
    """
    oldest = pd.to_datetime(int(archive["dateutc"].min()), unit="ms", utc=True)
    newest = pd.to_datetime(int(archive["dateutc"].max()), unit="ms", utc=True)
    return (
        f"{len(archive)} readings, {oldest.strftime(STAMP)} to "
        f"{newest.strftime(STAMP)} UTC"
    )


def cmd_scan(args: argparse.Namespace) -> int:
    """
    List the account's devices and mark inventory membership; write nothing.

    :param args: Parsed arguments.
    :return: 0.
    :raises Refused: If Ambient lists no device.
    :raises StorageReadError: If the inventory read failed or it is corrupt.
    """
    devices = account_devices()
    inventory = station_inventory.read_inventory(args.bucket)

    emit(f"Devices on the Ambient account: {len(devices)}")
    listed = set()
    for device in devices:
        mac = str(device.get("macAddress", ""))
        position = inventory.position(mac) if inventory else None
        if position is None:
            mark = "not in the inventory"
        else:
            listed.add(mac.upper())
            mark = f"in the inventory, #{position}"
            if position == 1:
                mark += " (default station)"
        emit(f"  {mac:<17}  {device_name(device):<24}  {mark}")

    if inventory is None:
        emit("No station inventory is stored yet; the first `add` seeds it.")
        return 0
    missing = [m for m in inventory.macs if m.upper() not in listed]
    for mac in missing:
        emit(f"  {mac:<17}  {'(not on the account)':<24}  in the inventory")
    emit(f"Station inventory: {', '.join(inventory.macs)}")
    return 0


def give_archive(bucket: str, device: Dict) -> None:
    """
    Make sure a station has an archive: adopt a stored one, or build it.

    :param bucket: The bucket.
    :param device: The station's `/devices` entry.
    :raises StorageReadError: If the archive read failed, or a stored
        archive is corrupt or holds no valid reading; nothing is written.
    :raises Refused: If Ambient holds no reading for the station.
    :raises AmbientRequestError: If a history page request failed.
    :raises StorageWriteError: If the save failed.
    """
    mac = device["macAddress"]
    key = f"{mac}.{HISTORY_FILE_TYPE}"
    stored = storj.read_df_from_s3(bucket, key, file_type=HISTORY_FILE_TYPE)
    if stored is not None:
        archive = normalize(stored)
        if archive.empty:
            raise storj.StorageReadError(
                f"s3://{bucket}/{key} has no rows with a valid dateutc; "
                "not adopting it"
            )
        emit(f"Adopted the stored archive, unchanged: {span(archive)}.")
        return

    emit(f"No archive stored for {mac}; building it from Ambient's history.")
    fetched = awn.fetch_full_history(device)
    archive, report = archive_merge.merge(pd.DataFrame(), fetched)
    logger.info(f"First archive merge: {report.describe()}")
    if archive.empty:
        raise Refused(f"Ambient holds no readings for {mac}; nothing to archive")
    storj.save_archive(archive, bucket, mac, stored=pd.DataFrame())
    emit(f"Built and saved the first archive: {span(archive)}.")


def cmd_add(args: argparse.Namespace) -> int:
    """
    Add a station: check it, give it an archive, then append it.

    :param args: Parsed arguments; `args.mac` is the station.
    :return: 0 when added.
    :raises Refused: If Ambient lists no device or not this MAC, the MAC is
        already in the inventory, or Ambient holds no reading for it.
    :raises StorageReadError: If the inventory or archive read failed.
    :raises BackupError: If the inventory backup failed.
    :raises StorageWriteError: If a save failed.
    """
    devices = account_devices()
    try:
        device = find_station(devices, args.mac)
    except StationNotFoundError:
        listed = ", ".join(str(d.get("macAddress")) for d in devices)
        raise Refused(
            f"{args.mac.strip().upper()} is not on the Ambient account "
            f"(it lists: {listed})"
        ) from None
    mac = device["macAddress"]
    name = device_name(device)
    inventory = station_inventory.read_inventory(args.bucket)
    if inventory is not None and mac in inventory:
        raise Refused(
            f"{mac} ({name}) is already in the station inventory, "
            f"#{inventory.position(mac)}"
        )

    # Backup, then the archive, then the inventory write: a failed backup
    # writes nothing, and a failed backfill leaves the inventory as it was.
    written = station_inventory.append_station(
        args.bucket,
        mac,
        inventory,
        before_write=lambda: give_archive(args.bucket, device),
    )
    if inventory is None:
        emit(f"Seeded the station inventory with {mac} ({name}), the default station.")
    else:
        emit(
            f"Added {mac} ({name}) to the station inventory, #{len(written.macs)}; "
            "the previous inventory is backed up under backups/."
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """
    The command line.

    :return: The parser, one subparser per command.
    """
    parser = argparse.ArgumentParser(description="The station inventory.")
    commands = parser.add_subparsers(dest="command", required=True)

    def command(
        name: str, handler: Callable[[argparse.Namespace], int], help_: str
    ) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_)
        sub.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket name")
        sub.set_defaults(handler=handler)
        return sub

    command("scan", cmd_scan, "List the account's devices; write nothing")
    add = command("add", cmd_add, "Add a station to the inventory")
    add.add_argument("mac", help="The station's MAC, as `scan` lists it")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    The shell entry point; `add` holds the archive-writer lock.

    :param argv: Arguments, without the program name; `sys.argv` when None.
    :return: The exit status: 0 done, 1 refused or failed.
    """
    args = build_parser().parse_args(argv)
    try:
        with writer_lock(saves=args.command == "add"):
            return args.handler(args)
    except ArchiveLockHeld as e:
        emit(f"{e}; {args.command} did nothing.")
        return 1
    except Refused as e:
        emit(f"Refused: {e}")
        return 1
    except Exception as e:
        logger.exception(f"{args.command} failed: {e}")
        emit(f"{args.command} failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

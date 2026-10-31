"""
refresh.py: One refresh run for the pinned station, as the VPS cron runs it.

Runs catchup, loads the catalogs, builds the dashboard snapshot and publishes
`snapshot/<MAC>.dashboard.json` and `snapshot/<MAC>.context.json` on Storj.
It reads secrets only through `lookout.secrets_loader` (env vars, then
`~/.config/lookout/secrets.toml`) and never imports Streamlit.

Usage:
    PYTHONPATH=. python lookout/cli/refresh.py [--bucket lookout] [--pages 20]
    python -m lookout.cli.refresh [--bucket lookout] [--pages 20]

The steps, each its own function, run in this order by `run`:

1. `load_station()`: the pinned station's `/devices` entry; its `lastData` is
   the live reading.
2. `update_archive(device, bucket, pages)`: catchup; the archive as saved.
3. `load_catalogs(station, bucket)`: the Event Catalog and the energy
   catalog as stored.
4. `update_catalogs(station, bucket, archive, catalogs)`: both catalogs
   brought up to the caught-up archive. The Event Catalog is saved only when
   an ongoing event completed in this run; the energy catalog only when the
   stored one is more than `ENERGY_SAVE_AGE` old.
5. `build(station, device, archive, catalogs, now)`: `build_snapshot`.
6. `publish(snapshot, bucket)`: both documents under `snapshot/`.
7. `publish_curves(station, archive, catalogs, bucket)`: a curve file for
   each completed event that has none yet, under `snapshot/<MAC>.events/`,
   at most `curves.CURVE_BACKFILL_LIMIT` per run (#84).

`main` is the shell entry point: it runs `run` and turns the outcome into an
exit status, 0 or 1, and sends the Healthchecks heartbeats (#74) around `run`.
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import pandas as pd

from lookout.api.ambient_client import get_station_device
from lookout.cli import catchup
from lookout.config import DEFAULT_BUCKET
from lookout.core.energy_catalog import EnergyCatalog
from lookout.core.energy_catalog_func import CATALOG_TYPE as ENERGY_TYPE
from lookout.core.rain_events import RainEventCatalog
from lookout.snapshot import Snapshot, Station, build_snapshot
from lookout.snapshot.curves import publish_event_curves
from lookout.snapshot.publish import publish_snapshot
from lookout.utils.healthchecks import Heartbeat
from lookout.storage.storj import backup_and_save_catalog
from lookout.utils.log_util import app_logger

logger = app_logger(__name__, log_file="refresh.log")

DEFAULT_PAGES = 20
ENERGY_SAVE_AGE = pd.Timedelta(days=2)


@dataclass
class Catalogs:
    """
    The catalogs one run builds the snapshot from.

    :param events: The Event Catalog; None when none is stored.
    :param energy: The energy catalog; None when none is stored.
    """

    events: Optional[pd.DataFrame] = None
    energy: Optional[pd.DataFrame] = None


def utc_now() -> datetime:
    """
    The wall clock, UTC; the default clock for `run`.

    :return: Now, timezone-aware.
    """
    return datetime.now(timezone.utc)


# ---------- the steps ----------


def load_station() -> Dict:
    """
    Step 1: the pinned station, with its live reading.

    :return: Its `/devices` entry.
    :raises StationNotFoundError: If `STATION_MAC` is not on the account.
    """
    device = get_station_device()
    logger.info(f"Refreshing {device.get('macAddress')}")
    return device


def update_archive(device: Dict, bucket: str, pages: int) -> pd.DataFrame:
    """
    Step 2: catchup, unchanged from the catchup CLI.

    :param device: The station's `/devices` entry.
    :param bucket: The bucket.
    :param pages: Maximum Ambient pages to fetch.
    :return: The archive as saved.
    """
    return catchup.update_archive(device, bucket, pages)


def load_catalogs(station: Station, bucket: str) -> Catalogs:
    """
    Step 3: the catalogs as stored.

    :param station: The station.
    :param bucket: The bucket.
    :return: The catalogs; a field is None when none is stored.
    """
    events = RainEventCatalog(station.mac, bucket=bucket).load_catalog()
    energy = EnergyCatalog(station.mac, bucket=bucket).load_catalog()
    return Catalogs(
        events=None if events.empty else events,
        energy=None if energy.empty else energy,
    )


def update_catalogs(
    station: Station, bucket: str, archive: pd.DataFrame, catalogs: Catalogs
) -> Catalogs:
    """
    Step 4: bring both catalogs up to the caught-up archive.

    The Event Catalog is saved only when its last event was ongoing in the
    stored catalog and has completed (`eventrainin` back to 0), as
    `RainEventCatalog.check_and_auto_save_ongoing_event` decides. A catalog
    detected afresh because none is stored is not saved. The energy catalog
    is saved only when a stored one is more than `ENERGY_SAVE_AGE` old.

    :param station: The station.
    :param bucket: The bucket.
    :param archive: The archive after catchup.
    :param catalogs: The catalogs as stored.
    :return: The updated catalogs, in memory; the stored ones are untouched
        unless a save rule above fired.
    """
    event_store = RainEventCatalog(station.mac, bucket=bucket)
    stored = catalogs.events
    events = event_store.update_catalog_with_new_data(
        archive, pd.DataFrame() if stored is None else stored.copy()
    )
    if stored is not None:
        event_store.check_and_auto_save_ongoing_event(stored, events)

    energy_store = EnergyCatalog(station.mac, bucket=bucket)
    energy = energy_store.update_catalog_with_new_data(
        archive, pd.DataFrame() if catalogs.energy is None else catalogs.energy.copy()
    )
    if catalogs.energy is not None and (
        energy_store.get_catalog_age() > ENERGY_SAVE_AGE
    ):
        backup_and_save_catalog(energy, ENERGY_TYPE, bucket, station.mac)
        logger.info(f"Energy catalog saved (stored one older than {ENERGY_SAVE_AGE})")

    return Catalogs(events=None if events.empty else events, energy=energy)


def build(
    station: Station,
    device: Dict,
    archive: pd.DataFrame,
    catalogs: Catalogs,
    now: datetime,
) -> Snapshot:
    """
    Step 5: the snapshot.

    :param station: The station.
    :param device: The station's `/devices` entry; `lastData` is live.
    :param archive: The archive after catchup.
    :param catalogs: The catalogs.
    :param now: The build time, timezone-aware.
    :return: The snapshot.
    """
    return build_snapshot(
        archive=archive,
        live=device.get("lastData"),
        event_catalog=catalogs.events,
        energy_catalog=catalogs.energy,
        station=station,
        now=now,
    )


def publish(snapshot: Snapshot, bucket: str) -> List[str]:
    """
    Step 6: write both documents under `snapshot/`.

    :param snapshot: The snapshot.
    :param bucket: The bucket.
    :return: The keys written.
    """
    return publish_snapshot(snapshot, bucket)


def publish_curves(
    station: Station, archive: pd.DataFrame, catalogs: Catalogs, bucket: str
) -> List[str]:
    """
    Step 7: write the curve files completed events do not have yet.

    Each is written once, by the first run that sees its event completed
    (or, for events completed before this step existed, by a later run of
    the bounded backfill), and never rewritten.

    :param station: The station.
    :param archive: The archive after catchup.
    :param catalogs: The updated catalogs.
    :param bucket: The bucket.
    :return: The keys written.
    """
    return publish_event_curves(station, catalogs.events, archive, bucket)


# ---------- the run ----------


def run(
    bucket: str = DEFAULT_BUCKET,
    pages: int = DEFAULT_PAGES,
    clock: Callable[[], datetime] = utc_now,
) -> Snapshot:
    """
    One refresh run: every step, in order. Any failure raises.

    :param bucket: The bucket.
    :param pages: Maximum Ambient pages catchup fetches.
    :param clock: Returns the build time; injected so tests can fix it.
    :return: The published snapshot.
    """
    device = load_station()
    station = Station.from_device(device)
    archive = update_archive(device, bucket, pages)
    catalogs = load_catalogs(station, bucket)
    catalogs = update_catalogs(station, bucket, archive, catalogs)
    snapshot = build(station, device, archive, catalogs, clock())
    publish(snapshot, bucket)
    publish_curves(station, archive, catalogs, bucket)
    return snapshot


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Read the command line.

    :param argv: Arguments, without the program name; `sys.argv` when None.
    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Catch up the archive and publish the dashboard snapshot."
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket name")
    parser.add_argument(
        "--pages",
        type=int,
        default=DEFAULT_PAGES,
        help="Max number of Ambient pages catchup fetches",
    )
    return parser.parse_args(argv)


def main(
    argv: Optional[List[str]] = None,
    clock: Callable[[], datetime] = utc_now,
) -> int:
    """
    The shell entry point, wrapped in the Healthchecks heartbeats.

    `/start` goes out first; the exit status goes out from a `finally` after
    the run, so a failed run (or an interrupt) reports a non-zero status at
    once. The station heartbeat goes out after a published run. A heartbeat
    that cannot be sent never changes the exit status.

    :param argv: Arguments, without the program name; `sys.argv` when None.
    :param clock: Returns the build time; injected so tests can fix it.
    :return: The exit status: 0 when the snapshot was published, else 1.
    """
    args = parse_args(argv)
    heartbeat = Heartbeat.from_secrets()
    heartbeat.start()
    status = 1
    try:
        snapshot = run(bucket=args.bucket, pages=args.pages, clock=clock)
        status = 0
        heartbeat.station(
            snapshot.station.name,
            snapshot.station.mac,
            snapshot.observed_at_ms,
            snapshot.built_at_ms,
        )
    except Exception as e:
        logger.exception(f"Refresh failed: {e}")
    finally:
        heartbeat.finish(status)
    return status


if __name__ == "__main__":
    sys.exit(main())

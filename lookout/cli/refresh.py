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
2. `update_archive(device, bucket, pages)`: catchup; the archive as saved,
   raw.
3. `correct_rain(station, bucket, archive, live)`: the stored rain
   corrections applied once, in memory (#95), to the archive and to the
   live reading alike (#118). Every later step reads these; the archive in
   storage stays raw. A stored catalog that cannot be read fails the run
   (#160), rather than publish the rain uncorrected.
4. `load_catalogs(station, bucket)`: the Event Catalog and the energy
   catalog as stored. An Event Catalog read that fails (or finds it
   corrupt) fails the run before anything is written (#180, ADR-0004); a
   missing one is detected from the archive in step 5 and saved.
5. `update_catalogs(station, bucket, archive, catalogs)`: both catalogs
   brought up to the caught-up archive. The Event Catalog is saved only when
   an event completed since the last save; the energy catalog only when the
   stored one is more than `ENERGY_SAVE_AGE` old. A catalog whose update
   raises is logged at ERROR and kept as stored, and marked stale (#118):
   one catalog's bug never stops the snapshot from publishing.
6. `build(station, live, archive, catalogs, now)`: `build_snapshot`, with
   the stale catalogs marked on it.
7. `publish(snapshot, bucket)`: both documents under `snapshot/`.
8. `publish_curves(station, archive, catalogs, bucket)`: a curve file for
   each completed event that has none yet, under `snapshot/<MAC>.events/`,
   at most `curves.CURVE_BACKFILL_LIMIT` per run (#84).

`main` is the shell entry point: it runs `run` and turns the outcome into an
exit status, 0 or 1, and sends the Healthchecks heartbeats (#74) around `run`.
"""

import argparse
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from lookout.api.ambient_client import get_station_device
from lookout.cli import catchup
from lookout.config import DEFAULT_BUCKET
from lookout.core.energy_catalog import EnergyCatalog
from lookout.core.energy_catalog_func import CATALOG_TYPE as ENERGY_TYPE
from lookout.core.rain_corrections import correct_counters, read_catalog
from lookout.core.rain_events import RainEventCatalog
from lookout.snapshot import Snapshot, StaleCatalog, Station, build_snapshot
from lookout.snapshot.model import CatalogName
from lookout.snapshot.current import to_ms
from lookout.snapshot.curves import publish_event_curves
from lookout.snapshot.publish import publish_snapshot
from lookout.utils.healthchecks import Heartbeat
from lookout.storage.storj import backup_and_save_catalog
from lookout.utils.dateutc import DEFAULT_TZ
from lookout.utils.log_util import app_logger

logger = app_logger(__name__, log_file="refresh.log")

DEFAULT_PAGES = 20
ENERGY_SAVE_AGE = pd.Timedelta(days=2)
RAIN_COUNTERS = ("dailyrainin", "weeklyrainin", "monthlyrainin", "yearlyrainin")


@dataclass
class Catalogs:
    """
    The catalogs one run builds the snapshot from.

    :param events: The Event Catalog; None when none is stored.
    :param energy: The energy catalog; None when none is stored.
    :param stale: The catalogs whose update failed this run and are as last
        stored ("events", "energy"); empty when both updated.
    """

    events: Optional[pd.DataFrame] = None
    energy: Optional[pd.DataFrame] = None
    stale: Tuple[CatalogName, ...] = ()


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
    :return: The archive as saved: raw, no rain corrections applied.
    """
    return catchup.update_archive(device, bucket, pages)


def correct_rain(
    station: Station,
    bucket: str,
    archive: pd.DataFrame,
    live: Optional[Dict],
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Step 3: the stored rain corrections, applied once to the raw archive and
    to the live reading.

    The archive in storage is always raw (#95), so this is the one place a
    refresh run corrects it; every later step reads what it returns.
    `build_snapshot` stays pure and applies nothing itself. The live reading
    gets the same corrections by the same rules (`correct_counters`), dated
    the way the archive is (#118).

    A catalog that is stored but cannot be read raises, so the run fails
    and the last good snapshot stays up (#160): publishing the rain
    uncorrected would show a carryover as rain, with nothing marking it.

    :param station: The station.
    :param bucket: The bucket holding `<MAC>.rain_corrections.json`.
    :param archive: The archive as saved, raw.
    :param live: The live `lastData`, raw, or None.
    :return: The corrected archive and live reading, in memory; `archive`
        and `live` themselves when the station has no corrections.
    :raises StorageReadError: If the stored catalog cannot be read.
    """
    corrections = (read_catalog(station.mac, bucket) or {}).get("corrections")
    if not corrections:
        return archive, live
    return correct_counters(archive, corrections), correct_live(
        live, archive, corrections
    )


def correct_live(
    live: Optional[Dict], archive: pd.DataFrame, corrections: List[Dict]
) -> Optional[Dict]:
    """
    The live reading with the rain corrections applied to its counters.

    It is corrected as a one-row frame through `correct_counters`, so it gets
    exactly the archive's rules. Its day is its `dateutc` in the archive's
    `date` time zone (the station's, `DEFAULT_TZ` without an archive), the
    calendar day the archive's rows are corrected by.

    :param live: The live `lastData`, raw, or None.
    :param archive: The raw archive, for its time zone.
    :param corrections: The stored corrections.
    :return: A corrected copy; `live` itself when it has no time or no
        counters.
    """
    if not live or live.get("dateutc") is None:
        return live
    present = [col for col in RAIN_COUNTERS if _is_reading(live.get(col))]
    if not present:
        return live

    tz = DEFAULT_TZ
    if "date" in archive.columns and isinstance(
        archive["date"].dtype, pd.DatetimeTZDtype
    ):
        tz = archive["date"].dt.tz
    observed = pd.Timestamp(to_ms(live["dateutc"]), unit="ms", tz="UTC")
    row = pd.DataFrame(
        [
            {
                "date": observed.tz_convert(tz),
                **{col: float(live[col]) for col in present},
            }
        ]
    )
    for col in RAIN_COUNTERS:
        if col not in row.columns:
            row[col] = float("nan")

    corrected = correct_counters(row, corrections).iloc[0]
    return {**live, **{col: float(corrected[col]) for col in present}}


def _is_reading(value: object) -> bool:
    """
    Whether a live value is a usable counter reading: a finite number.

    :param value: A `lastData` value.
    :return: True for an int or float that is not NaN or infinite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return bool(pd.notna(value)) and abs(value) != float("inf")


def load_catalogs(station: Station, bucket: str) -> Catalogs:
    """
    Step 4: the catalogs as stored.

    The Event Catalog is read once and the run acts on its read outcome
    (#180): failed raises, so the run exits 1 before any catalog or
    snapshot write and the last good snapshot stays up; missing is None,
    and step 5 detects it from the archive and saves it.

    :param station: The station.
    :param bucket: The bucket.
    :return: The catalogs; a field is None when none is stored (or the
        stored one holds no event).
    :raises StorageReadError: If the Event Catalog read failed or it is
        corrupt.
    """
    events = RainEventCatalog(station.mac, bucket=bucket).load_catalog()
    energy = EnergyCatalog(station.mac, bucket=bucket).load_catalog()
    return Catalogs(
        events=None if events is None or events.empty else events,
        energy=None if energy.empty else energy,
    )


def update_catalogs(
    station: Station, bucket: str, archive: pd.DataFrame, catalogs: Catalogs
) -> Catalogs:
    """
    Step 5: bring both catalogs up to the caught-up archive.

    The Event Catalog is saved, after a backup of the stored one, only when
    the updated catalog holds a completed event (`eventrainin` back to 0)
    the stored one has no completed event for, matched by start: a stored
    ongoing event that completed, or an event that started since the last
    save and has completed. With none stored, the first run with a
    completed event seeds it. `RainEventCatalog.save_if_events_completed`
    decides. The energy catalog is saved only when a stored one is more
    than `ENERGY_SAVE_AGE` old.

    Each catalog is updated on its own (#118): an error in one's update is
    logged at ERROR with the traceback, that catalog stays as stored (the
    last good one) and is named in `stale`, and the other still updates. A
    failed save is logged at ERROR and the updated catalog is still used (a
    later run's save rule fires again). The run goes on to publish and
    exits 0.

    :param station: The station.
    :param bucket: The bucket.
    :param archive: The archive after catchup, rain corrected (`correct_rain`).
    :param catalogs: The catalogs as stored.
    :return: The updated catalogs, in memory; the stored ones are untouched
        unless a save rule above fired.
    """
    stale: List[CatalogName] = []
    try:
        events = update_event_catalog(station, bucket, archive, catalogs.events)
    except Exception as e:
        logger.exception(f"Event Catalog not updated (using the stored one): {e}")
        events, stale = catalogs.events, stale + ["events"]
    try:
        energy = update_energy_catalog(station, bucket, archive, catalogs.energy)
    except Exception as e:
        logger.exception(f"Energy catalog not updated (using the stored one): {e}")
        energy, stale = catalogs.energy, stale + ["energy"]
    return Catalogs(events=events, energy=energy, stale=tuple(stale))


def update_event_catalog(
    station: Station,
    bucket: str,
    archive: pd.DataFrame,
    stored: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """
    The Event Catalog brought up to the archive, saved when an event completed.

    :param station: The station.
    :param bucket: The bucket.
    :param archive: The archive after catchup, rain corrected.
    :param stored: The Event Catalog as stored, or None.
    :return: The updated catalog; None when it holds no event.
    """
    event_store = RainEventCatalog(station.mac, bucket=bucket)
    events = event_store.update_catalog_with_new_data(
        archive, pd.DataFrame() if stored is None else stored.copy()
    )
    try:
        event_store.save_if_events_completed(stored, events)
    except Exception as e:
        logger.exception(f"Event Catalog not saved (the update is used): {e}")
    return None if events.empty else events


def update_energy_catalog(
    station: Station,
    bucket: str,
    archive: pd.DataFrame,
    stored: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    The energy catalog brought up to the archive, saved when the stored one
    is more than `ENERGY_SAVE_AGE` old.

    :param station: The station.
    :param bucket: The bucket.
    :param archive: The archive after catchup, rain corrected.
    :param stored: The energy catalog as stored, or None.
    :return: The updated catalog.
    """
    energy_store = EnergyCatalog(station.mac, bucket=bucket)
    energy = energy_store.update_catalog_with_new_data(
        archive, pd.DataFrame() if stored is None else stored.copy()
    )
    try:
        if stored is not None and energy_store.get_catalog_age() > ENERGY_SAVE_AGE:
            backup_and_save_catalog(energy, ENERGY_TYPE, bucket, station.mac)
            logger.info(
                f"Energy catalog saved (stored one older than {ENERGY_SAVE_AGE})"
            )
    except Exception as e:
        logger.exception(f"Energy catalog not saved (the update is used): {e}")
    return energy


def stale_catalogs(catalogs: Catalogs) -> Tuple[StaleCatalog, ...]:
    """
    The snapshot's mark for each catalog the run could not update.

    :param catalogs: The catalogs the snapshot is built from.
    :return: One `StaleCatalog` per name in `catalogs.stale`, dated by the
        newest time the stored catalog covers.
    """
    newest = {
        "events": (catalogs.events, "end_time"),
        "energy": (catalogs.energy, "period_end"),
    }
    marks = []
    for name in catalogs.stale:
        frame, column = newest[name]
        as_of = None
        if (
            frame is not None
            and column in frame.columns
            and frame[column].notna().any()
        ):
            as_of = to_ms(pd.to_datetime(frame[column], utc=True).max())
        marks.append(StaleCatalog(catalog=name, as_of_ms=as_of))
    return tuple(marks)


def build(
    station: Station,
    live: Optional[Dict],
    archive: pd.DataFrame,
    catalogs: Catalogs,
    now: datetime,
) -> Snapshot:
    """
    Step 6: the snapshot, with the catalogs this run could not update
    marked stale on it (`build_snapshot` itself never marks one).

    :param station: The station.
    :param live: The live `lastData`, rain corrected (`correct_rain`).
    :param archive: The archive after catchup, rain corrected (`correct_rain`).
    :param catalogs: The catalogs.
    :param now: The build time, timezone-aware.
    :return: The snapshot.
    """
    snapshot = build_snapshot(
        archive=archive,
        live=live,
        event_catalog=catalogs.events,
        energy_catalog=catalogs.energy,
        station=station,
        now=now,
    )
    if not catalogs.stale:
        return snapshot
    return replace(snapshot, stale_catalogs=stale_catalogs(catalogs))


def publish(snapshot: Snapshot, bucket: str) -> List[str]:
    """
    Step 7: write both documents under `snapshot/`.

    :param snapshot: The snapshot.
    :param bucket: The bucket.
    :return: The keys written.
    """
    return publish_snapshot(snapshot, bucket)


def publish_curves(
    station: Station, archive: pd.DataFrame, catalogs: Catalogs, bucket: str
) -> List[str]:
    """
    Step 8: write the curve files completed events do not have yet.

    Each is written once, by the first run that sees its event completed
    (or, for events completed before this step existed, by a later run of
    the bounded backfill), and never rewritten.

    :param station: The station.
    :param archive: The archive after catchup, rain corrected (`correct_rain`).
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
    One refresh run: every step, in order. A failure raises (a failed
    Event Catalog read among them, #180), except a catalog's update
    failing (that catalog is used as stored and marked stale) and a failure writing curve files after the snapshot published:
    those are logged at ERROR only.

    :param bucket: The bucket.
    :param pages: Maximum Ambient pages catchup fetches.
    :param clock: Returns the build time; injected so tests can fix it.
    :return: The published snapshot.
    """
    device = load_station()
    station = Station.from_device(device)
    archive, live = correct_rain(
        station, bucket, update_archive(device, bucket, pages), device.get("lastData")
    )
    catalogs = load_catalogs(station, bucket)
    catalogs = update_catalogs(station, bucket, archive, catalogs)
    snapshot = build(station, live, archive, catalogs, clock())
    publish(snapshot, bucket)
    try:
        publish_curves(station, archive, catalogs, bucket)
    except Exception as e:
        # The snapshot is out; curve files are retried by the next run (the
        # bounded backfill writes whatever is missing), so this is not a
        # failed run.
        logger.exception(f"Event curve files not written (snapshot published): {e}")
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

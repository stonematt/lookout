"""
refresh.py: One refresh run for every station in the station inventory, as
the VPS cron runs it (#271).

For each station it runs catchup, loads the catalogs, builds the dashboard
snapshot and publishes `snapshot/<MAC>.dashboard.json` and
`snapshot/<MAC>.context.json` on Storj; then the station index,
`snapshot/stations.json`. It reads secrets only through
`lookout.secrets_loader` (env vars, then `~/.config/lookout/secrets.toml`)
and never imports Streamlit.

Usage:
    PYTHONPATH=. python lookout/cli/refresh.py [--bucket lookout] [--pages 20]
    python -m lookout.cli.refresh [--bucket lookout] [--pages 20]

`run` works in three phases, each step its own function:

1. `load_stations(bucket)`: the station inventory (ADR-0006), then one
   `/devices` call, each inventory station picked out of it by MAC, never
   by the account's device order; a device not in the inventory is
   ignored. A missing or unreadable inventory, or an empty device list,
   fails the whole run before anything is written (ADR-0004).
2. The Ambient phase, in inventory order: `fetch_delta(device, bucket,
   pages)` reads the station's newest archived reading (only the `dateutc`
   column) and fetches what Ambient holds after it. Every Ambient call of
   the run is made here, through the one throttled client, before any
   station's archive is loaded. No threads.
3. The station phase, in inventory order: `refresh_station` runs the
   per-station pipeline. Each station's archive and frames are released
   before the next station loads, so the memory peak stays at one
   station's size.
   a. The archive: opened and the fetched readings merged in through the
      archive store (`lookout.storage.archive_store`), which backs up and
      saves only when the merge changed something; the archive as merged,
      raw. A station with no archive fails; it is never rebuilt.
   b. `correct_rain(station, bucket, archive, live)`: the stored rain
      corrections applied once, in memory (#95), to the archive and to the
      live reading alike (#118). Every later step reads these; the archive
      in storage stays raw. A stored catalog that cannot be read fails the
      station (#160), rather than publish the rain uncorrected.
   c. `load_catalogs(station, bucket)`: the Event Catalog and the energy
      catalog as stored. A read that fails (or finds one corrupt) fails
      the station before its catalogs or snapshot are written (#180,
      ADR-0004); a missing one is detected from the archive in step d.
   d. `update_catalogs(station, bucket, archive, catalogs)`: both catalogs
      brought up to the caught-up archive. The Event Catalog is saved only
      when an event completed since the last save; the energy catalog only
      when the stored one is more than `ENERGY_SAVE_AGE` old. A catalog
      whose update raises is logged at ERROR and kept as stored, and marked
      stale (#118): one catalog's bug never stops the snapshot publishing.
   e. `read_previous(station, bucket)`: the feature of the station's last
      published dashboard snapshot, for the Feature Rule's hold (#533); a
      snapshot read, never the archive. Missing (or published before the
      hold) means no hold; a read that fails, or a document that cannot be
      parsed, fails the station before its snapshot is written (ADR-0004),
      so the last published snapshot stays up. Then
      `build(station, live, archive, catalogs, now, previous)`:
      `build_snapshot`, with the stale catalogs marked on it.
   f. `publish(snapshot, bucket)`: both documents under `snapshot/`.
   g. `publish_curves(station, archive, catalogs, bucket)`: a curve file
      for each completed event that has none yet, under
      `snapshot/<MAC>.events/`, at most `curves.CURVE_BACKFILL_LIMIT` per
      run (#84).

A station's failure, in either phase, is caught, logged with its MAC, and
the other stations still run; `run` returns each station's `StationRun`.

4. `publish_index(bucket, runs, clock)`, from `main` once `run` returns: the
   station index, `snapshot/stations.json` (#272,
   `lookout.snapshot.station_index`), last, after every station. A station
   that published gets a fresh entry; one that failed keeps its entry from
   the stored index, and one that has never published is not listed (a run
   where no station has an entry writes none). An index that cannot be
   built or written fails the run (exit 1) and leaves the stored one as it
   was; each station keeps its own outcome.
Every log line written while a station is worked on carries its name and
MAC (`lookout.utils.log_util.station_context`).

Each step runs through `_step`, which logs "Peak memory after <step>: N MB"
(#280): the process's peak RSS so far, which only rises, so the step where it
jumps set the peak. `fetch_delta` logs `ambient_fetch`, and the archive
store its own three (archive_load, archive_merge, archive_save).

`main` is the shell entry point: it runs `run` and turns the outcome into an
exit status, 1 when the run or any station failed, and sends the
Healthchecks heartbeats (#74) around `run`: one cron check for the whole
run, and each published station's own station check. It ends with one
summary line per station: published or failed, the age of its newest
reading, and its MAC. First it takes the archive-writer lock (#244,
`lookout.utils.archive_lock`), one lock for the whole run; when another
archive writer holds it, the run logs one line and exits 0 with no
heartbeat, so only the holder's pings reach the `lookout-refresh` check. A
lock that cannot be opened at all (an OSError) is a failed run, not a skip.
Every run it starts ends by writing each station's status document (#268,
`lookout.snapshot.status`): that station's outcome, the run's start and end,
and the code version and git SHA it ran on, served by the site at `/status`.
The code is read once as the run starts, so a `git pull` by a later cron
tick, while this run still works, never names code this run did not run. A
skipped run writes none, as it sends no heartbeat: the holder's run writes
its own.
"""

import argparse
import contextlib
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import (
    Callable,
    ContextManager,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import pandas as pd

import lookout
from lookout.api.ambient_client import (
    StationNotFoundError,
    find_station,
    get_devices,
)
from lookout.cli import catchup
from lookout.config import DEFAULT_BUCKET
from lookout.core.energy_catalog import EnergyCatalog
from lookout.core.energy_catalog_func import CATALOG_TYPE as ENERGY_TYPE
from lookout.core.rain_corrections import correct_counters, read_catalog
from lookout.core.rain_events import RainEventCatalog
from lookout.snapshot import Snapshot, StaleCatalog, Station, build_snapshot
from lookout.snapshot.feature import previous_feature
from lookout.snapshot.model import CatalogName, Feature
from lookout.snapshot.current import to_ms
from lookout.snapshot.curves import publish_event_curves
from lookout.snapshot.publish import publish_snapshot, snapshot_keys
from lookout.snapshot.station_index import (
    IndexEntry,
    next_index,
    publish_station_index,
    read_station_index,
)
from lookout.snapshot.status import (
    CodeVersion,
    RunStatus,
    code_version,
    publish_status,
)
from lookout.storage import archive_store
from lookout.storage.station_inventory import load_inventory, read_inventory
from lookout.storage.storj import (
    archive_newest_ms,
    backup_and_save_catalog,
    read_json_from_s3,
)
from lookout.utils.archive_lock import ArchiveLockHeld, hold_archive_lock
from lookout.utils.healthchecks import Heartbeat
from lookout.utils.log_util import app_logger, station_context
from lookout.utils.memory_utils import log_peak_rss

logger = app_logger(__name__, log_file="refresh.log")

T = TypeVar("T")

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


@dataclass
class StationRun:
    """
    One station's part of a refresh run.

    :param mac: The station's MAC, as the station inventory spells it.
    :param device: Its `/devices` entry; None when the account lacks it.
    :param station: The station read from `device`; None until the Ambient
        phase reads it.
    :param fetched: What the Ambient phase fetched for it; released (None)
        once the station phase has merged it.
    :param archive_newest_ms: Its newest archived reading before the run,
        epoch ms; None when unread.
    :param archive_start_ms: Its archive's first reading, epoch ms; None
        until the station phase loads the archive.
    :param snapshot: The snapshot it published; None when it published none.
    :param error: Why it failed; None when it did not. A curve step failure
        (`CurveStepFailed`) comes with a published snapshot.
    """

    mac: str
    device: Optional[Dict] = None
    station: Optional[Station] = None
    fetched: Optional[pd.DataFrame] = None
    archive_newest_ms: Optional[int] = None
    archive_start_ms: Optional[int] = None
    snapshot: Optional[Snapshot] = None
    error: Optional[Exception] = None

    @property
    def name(self) -> Optional[str]:
        """
        The station's Ambient name.

        :return: The name; None when unnamed or not on the account.
        """
        return ((self.device or {}).get("info") or {}).get("name")

    @property
    def ok(self) -> bool:
        """
        Whether the station's part of the run finished clean.

        :return: True when it published and nothing failed.
        """
        return self.error is None and self.snapshot is not None

    def newest_ms(self) -> Optional[int]:
        """
        The newest reading the run knows of for the station.

        :return: The published snapshot's newest reading; else the live
            reading's time; else the newest archived reading; None when the
            run knows none.
        """
        if self.snapshot is not None:
            return self.snapshot.observed_at_ms
        live = ((self.device or {}).get("lastData") or {}).get("dateutc")
        if live is not None:
            return to_ms(live)
        return self.archive_newest_ms


def utc_now() -> datetime:
    """
    The wall clock, UTC; the default clock for `run`.

    :return: Now, timezone-aware.
    """
    return datetime.now(timezone.utc)


# ---------- the steps ----------


def load_stations(bucket: str) -> List[StationRun]:
    """
    Step 1: the station inventory, and each station's `/devices` entry.

    One `/devices` call serves every station. Each inventory station is
    picked out of it by MAC, never by the account's device order, and a
    device the inventory does not list is ignored. An inventory station the
    account does not list fails on its own (logged with its MAC); the other
    stations still run.

    :param bucket: The bucket holding the station inventory.
    :return: One `StationRun` per inventory station, in inventory order.
    :raises StorageReadError: If the inventory is missing, unreadable or
        corrupt (ADR-0004, ADR-0006).
    :raises StationNotFoundError: If Ambient lists no device at all (an API
        error or wrong keys), which no station could run past.
    """
    inventory = load_inventory(bucket)
    devices = get_devices()
    if not devices:
        raise StationNotFoundError(
            "Ambient returned no devices (API error, wrong keys, or no station "
            "on the account)"
        )
    runs = []
    for mac in inventory.macs:
        station_run = StationRun(mac=mac)
        try:
            station_run.device = find_station(devices, mac)
        except StationNotFoundError as e:
            with station_context(None, mac):
                logger.error(f"Station {mac} failed: {e}")
            station_run.error = e
        runs.append(station_run)
    logger.info(f"Refreshing {', '.join(r.mac for r in runs)}")
    return runs


def fetch_delta(device: Dict, bucket: str, pages: int) -> Tuple[pd.DataFrame, int]:
    """
    Step 2, for one station: what Ambient holds after its newest archived
    reading.

    Only the archive's `dateutc` column is read here (`archive_newest_ms`),
    so every station's Ambient calls come before any archive is loaded.

    :param device: The station's `/devices` entry.
    :param bucket: The bucket.
    :param pages: Maximum Ambient pages to fetch.
    :return: The fetched readings (raw) and the newest archived reading's
        `dateutc`.
    :raises StorageReadError: If the station has no archive, or it cannot be
        read: the station fails and is never rebuilt (ADR-0004).
    :raises CatchupStalled: If Ambient holds readings past a gap the fetch
        could not cross.
    """
    newest = archive_newest_ms(bucket, device["macAddress"])
    return catchup.fetch_delta(device, newest, pages), newest


def correct_rain(
    station: Station,
    bucket: str,
    archive: pd.DataFrame,
    live: Optional[Dict],
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Step 3b: the stored rain corrections, applied once to the raw archive and
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
        live, station.tz, corrections
    )


def correct_live(
    live: Optional[Dict], tz: str, corrections: List[Dict]
) -> Optional[Dict]:
    """
    The live reading with the rain corrections applied to its counters.

    It is corrected as a one-row frame through `correct_counters`, so it gets
    exactly the archive's rules. Its day is its `dateutc` in the station's
    zone (ADR-0005), the calendar day the archive's rows are corrected by.

    :param live: The live `lastData`, raw, or None.
    :param tz: The station's zone.
    :param corrections: The stored corrections.
    :return: A corrected copy; `live` itself when it has no time or no
        counters.
    """
    if not live or live.get("dateutc") is None:
        return live
    present = [col for col in RAIN_COUNTERS if _is_reading(live.get(col))]
    if not present:
        return live

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
    Step 3c: the catalogs as stored.

    Each catalog is read once and the run acts on its read outcome (#180,
    #183): failed raises, so the run exits 1 before any catalog or snapshot
    write and the last good snapshot stays up; missing is None, and step 3d
    detects it from the archive (the Event Catalog is then saved).

    :param station: The station.
    :param bucket: The bucket.
    :return: The catalogs; a field is None when none is stored (or the
        stored one is empty).
    :raises StorageReadError: If either catalog's read failed or it is
        corrupt.
    """
    events = RainEventCatalog(station.mac, bucket=bucket).load_catalog()
    energy = EnergyCatalog(station.mac, bucket=bucket).load_catalog()
    return Catalogs(
        events=None if events is None or events.empty else events,
        energy=None if energy is None or energy.empty else energy,
    )


def update_catalogs(
    station: Station, bucket: str, archive: pd.DataFrame, catalogs: Catalogs
) -> Catalogs:
    """
    Step 3d: bring both catalogs up to the caught-up archive.

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


def read_previous(station: Station, bucket: str) -> Optional[Feature]:
    """
    Step 3e, first: the feature of the station's last published dashboard
    snapshot, for the Feature Rule's hold (#533). A snapshot read: the
    archive is not read here.

    :param station: The station.
    :param bucket: The bucket.
    :return: The last published feature; None when there is no dashboard
        snapshot, or its feature predates the hold (no `since_ms`).
    :raises StorageReadError: If the read fails or the document cannot be
        parsed (ADR-0004): the station fails before its snapshot is written,
        so the last published one stays up.
    """
    key = snapshot_keys(station.mac)["dashboard"]
    return previous_feature(read_json_from_s3(bucket, key))


def build(
    station: Station,
    live: Optional[Dict],
    archive: pd.DataFrame,
    catalogs: Catalogs,
    now: datetime,
    previous: Optional[Feature] = None,
) -> Snapshot:
    """
    Step 3e: the snapshot, with the catalogs this run could not update
    marked stale on it (`build_snapshot` itself never marks one).

    :param station: The station.
    :param live: The live `lastData`, rain corrected (`correct_rain`).
    :param archive: The archive after catchup, rain corrected (`correct_rain`).
    :param catalogs: The catalogs.
    :param now: The build time, timezone-aware.
    :param previous: The last published feature (`read_previous`), or None:
        nothing is held.
    :return: The snapshot.
    """
    snapshot = build_snapshot(
        archive=archive,
        live=live,
        event_catalog=catalogs.events,
        energy_catalog=catalogs.energy,
        station=station,
        now=now,
        previous_feature=previous,
    )
    if not catalogs.stale:
        return snapshot
    return replace(snapshot, stale_catalogs=stale_catalogs(catalogs))


def publish(snapshot: Snapshot, bucket: str) -> List[str]:
    """
    Step 3f: write both documents under `snapshot/`.

    :param snapshot: The snapshot.
    :param bucket: The bucket.
    :return: The keys written.
    """
    return publish_snapshot(snapshot, bucket)


def publish_curves(
    station: Station, archive: pd.DataFrame, catalogs: Catalogs, bucket: str
) -> List[str]:
    """
    Step 3g: write the curve files completed events do not have yet.

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


class CurveStepFailed(RuntimeError):
    """
    The curve step failed after the snapshot published.

    :param snapshot: The published snapshot, so the run still reports the
        station's heartbeat; the exit status is still 1.
    """

    def __init__(self, snapshot: Snapshot) -> None:
        super().__init__("event curve files not written (snapshot published)")
        self.snapshot = snapshot


def _step(name: str, fn: Callable[..., T], *args: object) -> T:
    """
    Run one step of the run and log the process's peak RSS after it (#280).

    Every step of `run` goes through here, so what must happen around each
    step (the peak line today) has one place to grow. A step that raises
    logs nothing and the exception passes through.

    :param name: The step's name, as the log line shows it.
    :param fn: The step's function.
    :param args: Its arguments.
    :return: What the step returned.
    """
    result = fn(*args)
    log_peak_rss(name, logger)
    return result


def refresh_station(
    station_run: StationRun, bucket: str, clock: Callable[[], datetime]
) -> Snapshot:
    """
    Step 3, for one station: the per-station pipeline, from the archive
    save to the curve files.

    The fetched readings are taken off `station_run` before the archive loads, and
    nothing here outlives the call but the snapshot, so the station's
    archive and frames are released before the next station loads.

    :param station_run: The station, after the Ambient phase (`device`, `station`
        and `fetched` set).
    :param bucket: The bucket.
    :param clock: Returns the build time.
    :return: The published snapshot.
    :raises CurveStepFailed: If the curve step failed after the snapshot
        published; it carries the snapshot.
    """
    device, station, fetched = (
        station_run.device,
        station_run.station,
        station_run.fetched,
    )
    station_run.fetched = None
    # The archive store logs its own peak lines, one per part (#280), so
    # this skips `_step`'s.
    session = archive_store.open(bucket, device["macAddress"])
    session.merge(fetched)
    archive = session.archive
    del fetched, session
    station_run.archive_start_ms = int(archive["dateutc"].min())
    archive, live = _step(
        "correct_rain", correct_rain, station, bucket, archive, device.get("lastData")
    )
    catalogs = _step("load_catalogs", load_catalogs, station, bucket)
    catalogs = _step(
        "update_catalogs", update_catalogs, station, bucket, archive, catalogs
    )
    previous = _step("read_previous", read_previous, station, bucket)
    snapshot = _step(
        "build", build, station, live, archive, catalogs, clock(), previous
    )
    _step("publish", publish, snapshot, bucket)
    try:
        _step("publish_curves", publish_curves, station, archive, catalogs, bucket)
    except Exception as e:
        # The snapshot is out and the next run's bounded backfill writes
        # whatever is missing, but a persistent curve problem must not stay
        # green: the station reports failure (#182).
        raise CurveStepFailed(snapshot) from e
    return snapshot


def _station_failed(station_run: StationRun, error: Exception) -> None:
    """
    Record a station's failure and log it with its traceback.

    Called inside the station's `station_context`, so the line carries its
    name and MAC too.

    :param station_run: The station.
    :param error: Why it failed.
    """
    station_run.error, station_run.fetched = error, None
    logger.error(f"Station {station_run.mac} failed: {error}", exc_info=error)


def run(
    bucket: str = DEFAULT_BUCKET,
    pages: int = DEFAULT_PAGES,
    clock: Callable[[], datetime] = utc_now,
) -> List[StationRun]:
    """
    One refresh run: every inventory station, in inventory order.

    The inventory and the device list come first (`load_stations`); a
    failure there raises, as nothing could run past it. Then the Ambient
    phase fetches every station's delta, and the station phase runs each
    station's pipeline (`refresh_station`). A station's failure in either
    phase is caught and logged with its MAC (`StationRun.error`), and the
    other stations still run. A failed station is skipped in the station
    phase. A catalog's update failing is not a failure: that catalog is
    used as stored and marked stale.

    :param bucket: The bucket.
    :param pages: Maximum Ambient pages each station's fetch takes.
    :param clock: Returns the build time; injected so tests can fix it.
    :return: Each station's part of the run, in inventory order.
    :raises StorageReadError: If the station inventory is missing,
        unreadable or corrupt.
    :raises StationNotFoundError: If Ambient lists no device.
    """
    runs = _step("load_stations", load_stations, bucket)
    for station_run in runs:  # the Ambient phase: every Ambient call of the run
        if station_run.error is not None:
            continue
        with station_context(station_run.name, station_run.mac):
            try:
                # With no zone the station fails here, before any write.
                station_run.station = Station.from_device(station_run.device)
                station_run.fetched, station_run.archive_newest_ms = fetch_delta(
                    station_run.device, bucket, pages
                )
            except Exception as e:
                _station_failed(station_run, e)
    for station_run in runs:  # the station phase: one station's archive at a time
        if station_run.error is not None:
            continue
        with station_context(station_run.name, station_run.mac):
            try:
                station_run.snapshot = refresh_station(station_run, bucket, clock)
            except CurveStepFailed as e:
                # Published, so the station heartbeat still goes out; the
                # station still counts as failed.
                station_run.snapshot, station_run.error = e.snapshot, e
                logger.error(
                    f"Station {station_run.mac} failed: event curve files not "
                    f"written: {e.__cause__}",
                    exc_info=e.__cause__,
                )
            except Exception as e:
                _station_failed(station_run, e)
    return runs


def publish_index(
    bucket: str, runs: Sequence[StationRun], clock: Callable[[], datetime]
) -> bool:
    """
    Step 4: the station index, published last, after every station (#272).

    Each station that published this run gets a fresh entry; one that did
    not keeps its entry from the stored index (`next_index`), so a failed
    station never leaves the site, and one that has never published is
    never listed. An index that would list no station is not written, and
    that is no failure of its own: every station failed, and they fail the
    run. Never raises: a station index that cannot be built or written is
    logged at ERROR and fails the run, while every station keeps its own
    outcome.

    :param bucket: The bucket.
    :param runs: Each station's part of the run, in inventory order.
    :param clock: Returns the publish time.
    :return: False when the index was needed and not written; else True.
    """
    published = {
        r.mac: IndexEntry(
            mac=r.mac,
            name=r.snapshot.station.name,
            tz=r.snapshot.station.tz,
            newest_reading_ms=r.snapshot.observed_at_ms,
            archive_start_ms=r.archive_start_ms,
        )
        for r in runs
        if r.snapshot is not None
    }
    try:
        index = next_index(
            [r.mac for r in runs],
            published,
            lambda: read_station_index(bucket),
            to_ms(clock()),
        )
        if not index.stations:
            logger.warning("Station index not published: no station has published")
            return True
        publish_station_index(index, bucket)
    except Exception as e:
        logger.error(f"Station index not published: {e}", exc_info=e)
        return False
    return True


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
    code: Optional[Callable[[], CodeVersion]] = None,
) -> int:
    """
    The shell entry point, wrapped in the Healthchecks heartbeats.

    `/start` goes out first; the exit status goes out from a `finally` after
    the run, so a failed run (or an interrupt) reports a non-zero status at
    once. This cron check is one check for the whole run, however many
    stations it holds. Each station that published then pings its own
    station check. A heartbeat that cannot be sent never changes the exit
    status.

    Each station's status document (#268) is written from the same
    `finally`, so every run that starts writes them, however it ends. A
    status write that fails is logged and never changes the exit status
    either. Last comes one summary line per station.

    The archive-writer lock is taken before anything else and released when
    the run ends, however it ends: one lock for every station. When another
    archive writer holds it, the run is skipped: one log line, no
    heartbeat, no status document, exit 0. A lock that cannot be opened (an
    OSError: say, a lock directory that cannot be made) fails the run
    before any work: it pings failure and writes a failed status.

    The code the status documents name is read once, as the run starts.

    :param argv: Arguments, without the program name; `sys.argv` when None.
    :param clock: Returns the build time, and the run's start and end;
        injected so tests can fix it.
    :param code: Returns the code version and SHA the status documents
        name, called once as the run starts;
        `lookout.snapshot.status.code_version` when None.
    :return: The exit status: 0 when every station published clean or the
        run was skipped for a held lock, else 1 (a station's curve step
        failure after publish included).
    """
    args = parse_args(argv)
    lock_error: Optional[OSError] = None
    lock: ContextManager[object] = contextlib.nullcontext()
    try:
        lock = hold_archive_lock()
    except ArchiveLockHeld as e:
        logger.warning(f"{e}; this refresh run is skipped, no heartbeat sent")
        return 0
    except OSError as e:
        lock_error = e
    with lock:
        started = clock()
        ran_on = code_at_start(code or code_version)
        status = 1
        runs: Optional[List[StationRun]] = None
        error: Optional[str] = None
        heartbeat: Optional[Heartbeat] = None
        try:
            heartbeat = Heartbeat.from_secrets()
            heartbeat.start()
            if lock_error is not None:
                # No lock, so no work: failed like any other run that raises.
                raise lock_error
            runs = run(bucket=args.bucket, pages=args.pages, clock=clock)
            indexed = _step("publish_index", publish_index, args.bucket, runs, clock)
            status = 0 if indexed and all(r.ok for r in runs) else 1
            for station_run in runs:
                if station_run.snapshot is not None:
                    with station_context(station_run.name, station_run.mac):
                        heartbeat.station(
                            station_run.snapshot.station.name,
                            station_run.snapshot.station.mac,
                            station_run.snapshot.observed_at_ms,
                            station_run.snapshot.built_at_ms,
                        )
        except Exception as e:
            error = type(e).__name__
            logger.exception(f"Refresh failed: {e}")
        except BaseException as e:
            # An interrupt: still reported below, then it propagates.
            error = type(e).__name__
            raise
        finally:
            if heartbeat is not None:
                heartbeat.finish(status)
            write_statuses(args.bucket, ran_on, started, clock, runs, error)
            if runs is not None:
                log_summary(runs, clock)
        return status


def code_at_start(code: Callable[[], CodeVersion]) -> CodeVersion:
    """
    The code the run starts on, read once; never raises (#268).

    :param code: Returns the code version and SHA.
    :return: Its answer; when it raises, the imported version with no SHA.
    """
    try:
        return code()
    except Exception as e:
        logger.warning(f"No code version for the status document: {e!r}")
        return CodeVersion(version=lookout.__version__, sha=None)


def write_statuses(
    bucket: str,
    code: CodeVersion,
    started: datetime,
    clock: Callable[[], datetime],
    runs: Optional[Sequence[StationRun]],
    error: Optional[str],
) -> None:
    """
    Write each station's status document; never raises (#268).

    Each document names its own station's outcome: "ok" when it published
    clean, "failed" with its own error otherwise, so one station's failure
    never marks another's document failed. When the run failed before it
    had its stations (`runs` is None: the lock, the inventory read or the
    device list), every station the inventory lists gets a failed document
    naming the run's error; with no readable inventory there is no station
    to write one for, and that is logged.

    The end time is read here, inside the guard, so nothing about the
    status documents can raise out of `main`'s `finally` and replace the
    run's own outcome.

    :param bucket: The bucket.
    :param code: The code the run started on (`code_at_start`).
    :param started: The run's start.
    :param clock: Returns now: the run's end.
    :param runs: Each station's part of the run; None when the run failed
        before it had them.
    :param error: The run's failure's exception class name, or None.
    """
    try:
        ended = to_ms(clock())
        if runs is None:
            outcomes = [(mac, error, None) for mac in _inventory_macs(bucket)]
        else:
            outcomes = [
                (
                    station_run.mac,
                    None if station_run.ok else type(station_run.error).__name__,
                    (
                        None
                        if station_run.snapshot is None
                        else station_run.snapshot.built_at_ms
                    ),
                )
                for station_run in runs
            ]
        for mac, failure, built_at_ms in outcomes:
            publish_status(
                RunStatus(
                    station_mac=mac,
                    code=code,
                    started_at_ms=to_ms(started),
                    ended_at_ms=ended,
                    outcome="ok" if failure is None else "failed",
                    error=failure,
                    built_at_ms=built_at_ms,
                ),
                bucket,
            )
    except Exception as e:
        logger.error(f"Status document not written: {type(e).__name__}: {e}")


def _inventory_macs(bucket: str) -> Tuple[str, ...]:
    """
    The inventory's stations, for a run that failed before it read them.

    :param bucket: The bucket.
    :return: The MACs; empty, with an ERROR line, when the inventory is
        missing or cannot be read.
    """
    try:
        inventory = read_inventory(bucket)
    except Exception as e:
        logger.error(f"No status document written: {type(e).__name__}: {e}")
        return ()
    if inventory is None:
        logger.error("No status document written: no station inventory")
        return ()
    return inventory.macs


def log_summary(runs: Sequence[StationRun], clock: Callable[[], datetime]) -> None:
    """
    The run's last lines: one per station, published or failed, with the
    age of its newest reading and its MAC.

    :param runs: Each station's part of the run.
    :param clock: Returns now, to age the readings by.
    """
    now_ms = to_ms(clock())
    for station_run in runs:
        name = station_run.name or "unnamed"
        newest = station_run.newest_ms()
        age = "unknown" if newest is None else f"{_age(now_ms - newest)} old"
        if station_run.ok:
            logger.info(f"{station_run.mac} {name}: published, newest reading {age}")
            continue
        outcome = "published, then failed" if station_run.snapshot else "failed"
        error = type(station_run.error).__name__
        logger.error(
            f"{station_run.mac} {name}: {outcome} ({error}), newest reading {age}"
        )


def _age(ms: int) -> str:
    """
    A reading's age, for the summary.

    :param ms: The age, milliseconds.
    :return: Minutes under two hours, hours under two days, else days.
    """
    minutes = ms / 60_000
    if minutes < 120:
        return f"{minutes:.0f} min"
    if minutes < 2 * 1440:
        return f"{minutes / 60:.1f} h"
    return f"{minutes / 1440:.1f} days"


if __name__ == "__main__":
    sys.exit(main())

"""
curves.py: Write each completed rain event's curve file to Storj, once.

Key: `snapshot/<MAC>.events/<start_ms>.json`, one `EventCurve.document` per
completed event, keyed by the event's start (see `lookout.snapshot.events`
for why not its catalog id). Under `snapshot/`, so Vercel's read-only
credential reaches it.

A curve file is immutable: a completed event's readings never change. Each
refresh run lists the stored curve files (one LIST call, paginated at 1000
keys) and writes only the completed events missing from it, newest first and
at most `CURVE_BACKFILL_LIMIT` per run. In steady state a run writes nothing;
the run that first sees an event completed writes its one file. The first run
against an existing catalog backfills its events over a few runs (about 185
events: 8 runs, 40 minutes at the 5-minute cron) instead of in one burst.
"""

import json
from typing import Any, List, Optional, Set

import pandas as pd

from lookout.snapshot.events import completed_curves
from lookout.snapshot.model import Station
from lookout.snapshot.publish import SNAPSHOT_PREFIX
from lookout.storage.storj import StorageReadError, StorageWriteError, get_s3_client
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

# Curve files one run writes at most; the rest wait for the next run.
CURVE_BACKFILL_LIMIT = 24


def curve_prefix(mac: str) -> str:
    """
    The Storj prefix of one station's curve files.

    :param mac: The station's MAC address.
    :return: `snapshot/<MAC>.events/`.
    """
    return f"{SNAPSHOT_PREFIX}{mac}.events/"


def curve_key(mac: str, start_ms: int) -> str:
    """
    The Storj key of one event's curve file.

    :param mac: The station's MAC address.
    :param start_ms: The event's start, its key.
    :return: `snapshot/<MAC>.events/<start_ms>.json`.
    """
    return f"{curve_prefix(mac)}{start_ms}.json"


def stored_curve_starts(client: Any, bucket: str, mac: str) -> Set[int]:
    """
    The starts of the events whose curve files are already stored.

    :param client: An S3 client.
    :param bucket: The bucket.
    :param mac: The station's MAC address.
    :return: Their starts (ms); keys that are not `<int>.json` are ignored.
    :raises StorageReadError: If the listing failed.
    """
    prefix = curve_prefix(mac)
    starts: Set[int] = set()
    try:
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        ):
            for item in page.get("Contents", []):
                name = item["Key"].removeprefix(prefix)
                stem = name.removesuffix(".json") if name.endswith(".json") else ""
                if stem.isdigit():
                    starts.add(int(stem))
    except Exception as e:
        raise StorageReadError(f"s3://{bucket}/{prefix} listing failed: {e}") from e
    return starts


def publish_event_curves(
    station: Station,
    event_catalog: Optional[pd.DataFrame],
    archive: pd.DataFrame,
    bucket: str,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Write the curve files of completed events that have none yet.

    Each event is written on its own (#160): one whose document or write
    fails is logged at ERROR and skipped, and the others are still written;
    the next run retries it. Once the rest are written, it raises
    `StorageWriteError` naming the failures, and a failed listing raises
    `StorageReadError`: the refresh run then exits 1 (the snapshot has
    published).

    :param station: The station.
    :param event_catalog: The Event Catalog, or None.
    :param archive: The archive after catchup.
    :param bucket: The bucket.
    :param limit: At most this many files; `CURVE_BACKFILL_LIMIT` when None.
    :return: The keys written, newest event first.
    :raises StorageReadError: If listing the stored curve files failed.
    :raises StorageWriteError: If any curve file was not written.
    """
    if event_catalog is None or event_catalog.empty:
        return []
    client = get_s3_client()
    stored = stored_curve_starts(client, bucket, station.mac)
    curves = completed_curves(
        event_catalog,
        archive,
        skip=stored,
        limit=CURVE_BACKFILL_LIMIT if limit is None else limit,
    )
    written = []
    failed = []
    for curve in curves:
        key = curve_key(station.mac, curve.start_ms)
        try:
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(curve.document(station), allow_nan=False),
                ContentType="application/json",
            )
        except Exception as e:
            logger.exception(f"Curve file {key} not written: {e}")
            failed.append(key)
            continue
        written.append(key)
    if written:
        prefix = curve_prefix(station.mac)
        logger.info(f"Wrote {len(written)} event curve file(s) under {prefix}")
    if failed:
        raise StorageWriteError(
            f"{len(failed)} curve file(s) not written: {', '.join(failed)}"
        )
    return written

"""
station_index.py: The station index, `snapshot/stations.json` (#272).

The published list of stations the site can show, in station inventory
order: the first is the default station. The refresh run writes it last,
after every station's snapshot, so it never lists a station whose run failed
before its first publish. A station whose run failed keeps its entry from the
last good index, so one bad run never drops it from the site. The site reads
it (never the station inventory, which sits outside `snapshot/`) to know
which `?station=` it serves and which station is the default.

The document:

- `schema_version`: `STATION_INDEX_SCHEMA_VERSION`, versioned on its own like
  the snapshot contract (`lookout/snapshot/model.py`) but separate from it:
  the dashboard's `SCHEMA_VERSION` does not cover this document. A shape
  change bumps it here and in `web/src/stations.ts` together.
- `kind`: always "station_index".
- `built_at_ms`: when the run that wrote it published it, epoch ms.
- `stations`: one entry per station, in inventory order: `mac`, `name` (its
  Ambient name, or null), `tz` (its IANA zone), `newest_reading_ms` (its
  snapshot's newest reading, or null) and `archive_start_ms` (its archive's
  first reading, or null).

Offers `IndexEntry`, `StationIndex`, `next_index` (the index a run publishes,
from what it published and the stored index), `read_station_index` (by the
reader contract, ADR-0004) and `publish_station_index`.
"""

import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from lookout.snapshot.publish import SNAPSHOT_PREFIX
from lookout.storage import storj
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

STATION_INDEX_KEY = f"{SNAPSHOT_PREFIX}stations.json"
STATION_INDEX_SCHEMA_VERSION = 1
KIND = "station_index"


@dataclass(frozen=True)
class IndexEntry:
    """
    One station in the station index.

    :param mac: Its MAC, as the station inventory spells it.
    :param name: Its Ambient name; None when unnamed.
    :param tz: Its IANA time zone (ADR-0005).
    :param newest_reading_ms: Its newest reading, epoch ms; None when unknown.
    :param archive_start_ms: Its archive's first reading, epoch ms; None when
        unknown.
    """

    mac: str
    name: Optional[str]
    tz: str
    newest_reading_ms: Optional[int]
    archive_start_ms: Optional[int]

    @classmethod
    def from_document(cls, doc: object) -> "IndexEntry":
        """
        Read one entry of a stored index.

        :param doc: The entry as parsed from JSON.
        :return: The entry.
        :raises ValueError: If it lacks a field or a field has the wrong type.
        """
        if not isinstance(doc, dict):
            raise ValueError("an entry is not an object")
        mac, name, tz = doc.get("mac"), doc.get("name"), doc.get("tz")
        if not isinstance(mac, str) or not mac or not isinstance(tz, str):
            raise ValueError("an entry lacks its mac or tz")
        if name is not None and not isinstance(name, str):
            raise ValueError(f"{mac}: name is not a string")
        times = {}
        for field in ("newest_reading_ms", "archive_start_ms"):
            value = doc.get(field)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise ValueError(f"{mac}: {field} is not epoch ms")
            times[field] = value
        return cls(mac=mac, name=name, tz=tz, **times)


@dataclass(frozen=True)
class StationIndex:
    """
    The station index as published.

    :param built_at_ms: When it was published, epoch ms.
    :param stations: Its entries, in inventory order.
    """

    built_at_ms: int
    stations: Tuple[IndexEntry, ...]

    def document(self) -> Dict[str, Any]:
        """
        The document, as written to Storj and served at `/stations`.

        :return: The JSON-ready dict.
        """
        return {
            "schema_version": STATION_INDEX_SCHEMA_VERSION,
            "kind": KIND,
            "built_at_ms": self.built_at_ms,
            "stations": [asdict(entry) for entry in self.stations],
        }

    def entry(self, mac: str) -> Optional[IndexEntry]:
        """
        A station's entry, by MAC in any case.

        :param mac: The station.
        :return: Its entry; None when the index does not list it.
        """
        wanted = mac.upper()
        return next((e for e in self.stations if e.mac.upper() == wanted), None)


def _parse(body: object, where: str) -> StationIndex:
    """
    Check a stored index's shape and build it.

    :param body: The parsed JSON.
    :param where: The object's location, for the error.
    :return: The index.
    :raises StorageReadError: If it is not a station index of this schema
        version: a corrupt index counts as failed (ADR-0004).
    """
    if not isinstance(body, dict) or body.get("kind") != KIND:
        raise storj.StorageReadError(f"{where} is not a station index")
    if body.get("schema_version") != STATION_INDEX_SCHEMA_VERSION:
        raise storj.StorageReadError(
            f"{where} has schema_version {body.get('schema_version')!r}, "
            f"not {STATION_INDEX_SCHEMA_VERSION}"
        )
    built, stations = body.get("built_at_ms"), body.get("stations")
    if (
        not isinstance(built, int)
        or isinstance(built, bool)
        or not isinstance(stations, list)
        or not stations
    ):
        raise storj.StorageReadError(f"{where} lacks built_at_ms or stations")
    try:
        entries = tuple(IndexEntry.from_document(doc) for doc in stations)
    except ValueError as e:
        raise storj.StorageReadError(f"{where}: {e}") from e
    return StationIndex(built_at_ms=built, stations=entries)


def read_station_index(bucket: str) -> Optional[StationIndex]:
    """
    Read the stored station index, by the reader contract (ADR-0004).

    :param bucket: The bucket.
    :return: The index when found; None when none is stored.
    :raises StorageReadError: If the read failed or the stored object is not
        a station index of this schema version.
    """
    body = storj.read_json_from_s3(bucket, STATION_INDEX_KEY)
    if body is None:
        return None
    return _parse(body, f"s3://{bucket}/{STATION_INDEX_KEY}")


def next_index(
    macs: Sequence[str],
    published: Mapping[str, IndexEntry],
    stored: Callable[[], Optional[StationIndex]],
    built_at_ms: int,
) -> StationIndex:
    """
    The station index a run publishes.

    Each inventory station, in inventory order: its entry from this run when
    it published, else its entry in the stored index, else none at all, so
    the index never lists a station that has never published. A station the
    inventory no longer lists leaves the index. The stored index is read only
    when some station did not publish.

    :param macs: The station inventory, in order.
    :param published: The entries of the stations that published this run,
        by MAC.
    :param stored: Reads the stored index (`read_station_index`).
    :param built_at_ms: The publish time.
    :return: The index.
    :raises StorageReadError: If the stored index was needed and could not be
        read: publishing without it would drop a station that failed.
    """
    last: Optional[StationIndex] = None
    if any(mac not in published for mac in macs):
        last = stored()
    entries = []
    for mac in macs:
        entry = published.get(mac) or (last.entry(mac) if last else None)
        if entry is not None:
            entries.append(entry)
    return StationIndex(built_at_ms=built_at_ms, stations=tuple(entries))


def publish_station_index(index: StationIndex, bucket: str) -> None:
    """
    Write the station index.

    Like `publish_snapshot`, a failed write raises, so the run fails.

    :param index: The index.
    :param bucket: The bucket.
    :raises Exception: Whatever the S3 client raises on a failed write.
    """
    storj.get_s3_client().put_object(
        Bucket=bucket,
        Key=STATION_INDEX_KEY,
        Body=json.dumps(index.document(), allow_nan=False),
        ContentType="application/json",
    )
    logger.info(f"Published s3://{bucket}/{STATION_INDEX_KEY}")

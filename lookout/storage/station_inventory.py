"""
station_inventory.py: the station inventory on Storj (ADR-0006).

The station inventory is one small JSON object at `INVENTORY_KEY`, an ordered
list of station MACs: `{"stations": ["<MAC>", ...]}`. Its order is the
priority order, and the first station is the default station. Only the
`stations add` command writes it, backing it up first; every other job only
reads it. The site never reads it (it reads the station index under
`snapshot/`), so it sits outside `snapshot/`.

Functions:
- read_inventory(bucket): the inventory by the reader contract (ADR-0004):
  found returns a `StationInventory`, missing returns None, failed (or
  corrupt) raises `StorageReadError`.
- load_inventory(bucket): the same read for a job that needs the inventory,
  where missing is an error too.
- append_station(bucket, mac, stored, before_write): back the inventory up,
  run `before_write` (the add's archive step), then write it with `mac`
  appended.

Classes:
- StationInventory: the MACs in priority order; `default` is the first.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional, Tuple

from lookout.storage import storj
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

INVENTORY_KEY = "station_inventory.json"
BACKUP_PREFIX = "backups/station_inventory"


@dataclass(frozen=True)
class StationInventory:
    """
    The stations Lookout archives, in priority order.

    :param macs: The station MACs, as Ambient's device list spells them;
        never empty, no MAC twice.
    """

    macs: Tuple[str, ...]

    @property
    def default(self) -> str:
        """
        The default station: the first in the inventory.

        :return: Its MAC.
        """
        return self.macs[0]

    def __contains__(self, mac: object) -> bool:
        """
        Whether a MAC is in the inventory, ignoring case.

        :param mac: The MAC to look for.
        :return: True when listed.
        """
        return str(mac).strip().upper() in {m.upper() for m in self.macs}

    def position(self, mac: str) -> Optional[int]:
        """
        A station's place in the priority order, counting from 1.

        :param mac: The MAC to look for, any case.
        :return: Its position, or None when it is not listed.
        """
        wanted = mac.strip().upper()
        for index, listed in enumerate(self.macs, start=1):
            if listed.upper() == wanted:
                return index
        return None

    def to_json(self) -> dict:
        """
        The stored form.

        :return: `{"stations": [<MAC>, ...]}`.
        """
        return {"stations": list(self.macs)}


def _parse(body: object, where: str) -> StationInventory:
    """
    Check a stored inventory's shape and build it.

    :param body: The parsed JSON.
    :param where: The object's location, for the error.
    :return: The inventory.
    :raises StorageReadError: If the object is not a non-empty list of
        distinct MAC strings under `stations`: a corrupt inventory counts
        as failed (ADR-0004).
    """
    stations = body.get("stations") if isinstance(body, dict) else None
    if not isinstance(stations, list) or not stations:
        raise storj.StorageReadError(f"{where} holds no `stations` list")
    if not all(isinstance(m, str) and m.strip() for m in stations):
        raise storj.StorageReadError(f"{where} lists a station that is not a MAC")
    macs = tuple(m.strip() for m in stations)
    if len({m.upper() for m in macs}) != len(macs):
        raise storj.StorageReadError(f"{where} lists a station twice")
    return StationInventory(macs)


def read_inventory(bucket: str) -> Optional[StationInventory]:
    """
    Read the station inventory, by the reader contract (ADR-0004).

    :param bucket: The bucket.
    :return: The inventory when found; None when none is stored.
    :raises StorageReadError: If the read failed or the stored object is
        corrupt (unparseable, or not a non-empty list of distinct MACs).
    """
    body = storj.read_json_from_s3(bucket, INVENTORY_KEY)
    if body is None:
        return None
    return _parse(body, f"s3://{bucket}/{INVENTORY_KEY}")


def load_inventory(bucket: str) -> StationInventory:
    """
    Read the station inventory for a job that cannot run without it.

    A missing inventory is an error, as a missing archive is (ADR-0004,
    ADR-0006): it is never taken for "no stations".

    :param bucket: The bucket.
    :return: The inventory.
    :raises StorageReadError: If it is missing, the read failed, or it is
        corrupt.
    """
    inventory = read_inventory(bucket)
    if inventory is None:
        raise storj.StorageReadError(
            f"s3://{bucket}/{INVENTORY_KEY} is missing; seed it with "
            "`stations add <MAC>`"
        )
    return inventory


def append_station(
    bucket: str,
    mac: str,
    stored: Optional[StationInventory],
    before_write: Callable[[], None] = lambda: None,
) -> StationInventory:
    """
    Back the stored inventory up, run `before_write`, then append `mac`.

    The order is the guarantee. The backup, a timestamped copy under
    `backups/`, comes before anything else, so a failed backup runs and
    writes nothing; with no inventory stored there is nothing to copy, and
    the write seeds the inventory. `before_write` is where `stations add`
    gives the station its archive: if it raises, the inventory is left as
    it was.

    :param bucket: The bucket.
    :param mac: The station to append, as Ambient's device list spells it.
    :param stored: The inventory as the caller read it (`read_inventory`);
        None when none is stored.
    :param before_write: Work that must succeed before the station is
        listed.
    :return: The inventory as written.
    :raises ValueError: If `mac` is already listed; nothing runs then.
    :raises BackupError: If the backup failed; nothing runs then.
    :raises StorageWriteError: If the write failed.
    """
    if stored is not None and mac in stored:
        raise ValueError(f"{mac} is already in the station inventory")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = storj.backup_object(bucket, INVENTORY_KEY, f"{BACKUP_PREFIX}_{stamp}.json")
    if backup is None and stored is not None:
        # The caller read an inventory that the copy then found missing.
        raise storj.BackupError(
            f"s3://{bucket}/{INVENTORY_KEY} vanished before its backup"
        )
    before_write()
    inventory = StationInventory((stored.macs if stored else ()) + (mac,))
    if not storj.save_json_to_s3(inventory.to_json(), bucket, INVENTORY_KEY):
        raise storj.StorageWriteError(f"Failed to save s3://{bucket}/{INVENTORY_KEY}")
    logger.info(f"Station inventory: {', '.join(inventory.macs)}")
    return inventory

"""
publish.py: Write a snapshot's two documents to Storj under `snapshot/`.

Keys, next to the archive in the same bucket and keyed by station MAC:

- `snapshot/<MAC>.dashboard.json`: `Snapshot.dashboard()`, for `/dashboard`.
- `snapshot/<MAC>.context.json`: `Snapshot.context()`, for `/current`.

Vercel's read-only Storj credential is limited to the `snapshot/` prefix.
"""

import json
from typing import Dict, List

from lookout.snapshot.model import Snapshot
from lookout.storage.storj import get_s3_client
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

SNAPSHOT_PREFIX = "snapshot/"


def snapshot_keys(mac: str) -> Dict[str, str]:
    """
    The Storj keys of one station's snapshot documents.

    :param mac: The station's MAC address.
    :return: {"dashboard": key, "context": key}.
    """
    return {
        "dashboard": f"{SNAPSHOT_PREFIX}{mac}.dashboard.json",
        "context": f"{SNAPSHOT_PREFIX}{mac}.context.json",
    }


def publish_snapshot(snapshot: Snapshot, bucket: str) -> List[str]:
    """
    Write the dashboard document, then the context block.

    Unlike `storj.save_json_to_s3`, a failed write raises, so the refresh run
    fails (and its heartbeat reports it) instead of logging and going on.

    :param snapshot: The snapshot to publish.
    :param bucket: The bucket the archive lives in.
    :return: The keys written, in order.
    :raises Exception: Whatever the S3 client raises on a failed write.
    """
    documents = {
        "dashboard": snapshot.dashboard(),
        "context": snapshot.context(),
    }
    client = get_s3_client()
    written = []
    for kind, key in snapshot_keys(snapshot.station.mac).items():
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(documents[kind], allow_nan=False),
            ContentType="application/json",
        )
        logger.info(f"Published s3://{bucket}/{key}")
        written.append(key)
    return written

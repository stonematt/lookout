"""
status.py: The refresh run's status document, `snapshot/<MAC>.status.json` (#268).

The refresh run writes one per station in the station inventory at the end of
every run it starts, success or failure, so each station's outcome and the
code the run was on can be read from outside without ssh to the VPS. The
site serves the one its Vercel `STATION_MAC` names at `/status`. It is its
own document, not part of the dashboard snapshot: the dashboard contract
and its `SCHEMA_VERSION` do not cover it.

Its fields:

- `kind`: always "status".
- `station_mac`: the station the document reports on.
- `version`: the release version (`VERSION`, ADR-0007) the run's code carries.
- `sha`: the checkout's `git rev-parse HEAD`, or null when git cannot say.
- `started_at_ms`, `ended_at_ms`: the run's start and end, epoch ms.
- `outcome`: this station's part of the run: "ok" when it published clean,
  "failed" when it, or the whole run, failed. A run with one failed station
  exits 1, while the other stations' documents still say "ok" (#271).
- `error`: the failure's exception class name, or null. Only the name: an
  exception's message can carry a URL or a key, and this document is public.
- `built_at_ms`: the `built_at_ms` of the snapshot this run published for
  the station, or null when it published none.

Offers `CodeVersion`, `code_version` (version and SHA, read at run time from
the checkout), `RunStatus`, `status_key` and `publish_status`, which never
raises: a status that cannot be written is logged and the run's own outcome
stands.

It sits in `lookout.snapshot` and under the `snapshot/` prefix because that
prefix is all Vercel's read-only Storj credential can read.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

import lookout
from lookout.snapshot.publish import SNAPSHOT_PREFIX
from lookout.storage.storj import get_s3_client
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

REPO_ROOT = Path(lookout.__file__).resolve().parent.parent
GIT_TIMEOUT_S = 10

Outcome = Literal["ok", "failed"]


@dataclass(frozen=True)
class CodeVersion:
    """
    Which code a run is on.

    :param version: The release version, from `VERSION`.
    :param sha: The checkout's HEAD commit, 40 hex; None when unknown.
    """

    version: str
    sha: Optional[str]


def code_version(root: Optional[Path] = None) -> CodeVersion:
    """
    The code this process runs: the release version and the checkout's HEAD.

    :param root: The checkout to ask git about; the one this package lives in
        when None.
    :return: The version, and the SHA, or None when git fails (not a git
        checkout, no git, or no answer in `GIT_TIMEOUT_S`).
    """
    return CodeVersion(version=lookout.__version__, sha=git_head(root or REPO_ROOT))


def git_head(root: Path) -> Optional[str]:
    """
    `git rev-parse HEAD` in a directory.

    :param root: The directory.
    :return: The 40-hex commit, or None when git fails or answers otherwise.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning(f"No git SHA for the status document: {type(e).__name__}")
        return None
    sha = out.stdout.strip()
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        logger.warning("No git SHA for the status document: unexpected answer")
        return None
    return sha


@dataclass(frozen=True)
class RunStatus:
    """
    How one refresh run ended for one station.

    :param station_mac: The station.
    :param code: The code it ran on.
    :param started_at_ms: Its start, epoch ms.
    :param ended_at_ms: Its end, epoch ms.
    :param outcome: "ok" when the station published clean, else "failed".
    :param error: The failure's exception class name; None when none raised.
    :param built_at_ms: The published snapshot's `built_at_ms`; None when it
        published none.
    """

    station_mac: str
    code: CodeVersion
    started_at_ms: int
    ended_at_ms: int
    outcome: Outcome
    error: Optional[str] = None
    built_at_ms: Optional[int] = None

    def document(self) -> Dict:
        """
        The status document, as written to Storj and served at `/status`.

        :return: The JSON-ready dict.
        """
        return {
            "kind": "status",
            "station_mac": self.station_mac,
            "version": self.code.version,
            "sha": self.code.sha,
            "started_at_ms": self.started_at_ms,
            "ended_at_ms": self.ended_at_ms,
            "outcome": self.outcome,
            "error": self.error,
            "built_at_ms": self.built_at_ms,
        }


def status_key(mac: str) -> str:
    """
    The Storj key of one station's status document.

    :param mac: The station's MAC address.
    :return: `snapshot/<MAC>.status.json`.
    """
    return f"{SNAPSHOT_PREFIX}{mac}.status.json"


def publish_status(status: RunStatus, bucket: str) -> None:
    """
    Write the status document, never raising.

    It is written after the run has decided its outcome, so a failed write
    is logged at ERROR and changes nothing else: a good run stays good, and
    a failed run's own error is the one it reports.

    :param status: How the run ended.
    :param bucket: The bucket the snapshot lives in.
    """
    key = status_key(status.station_mac)
    try:
        get_s3_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(status.document(), allow_nan=False),
            ContentType="application/json",
        )
    except Exception as e:
        logger.error(f"Status document not written to {key}: {type(e).__name__}: {e}")
        return
    logger.info(f"Published s3://{bucket}/{key} ({status.outcome})")

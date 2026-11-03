"""
memory_harness.py: The refresh run's memory, step by step, measured offline.

Runs the refresh run's steps (`lookout.cli.refresh`) against a local archive
parquet, with Storj and Ambient faked at the boundary, and prints the peak
RSS and the tracemalloc peak of each step (#281, spec #234). The memory
levers of #234 take their before and after numbers from it.

Usage:
    python -m lookout.cli.memory_harness data/<MAC>.parquet [--new 1]
        [--tz America/Los_Angeles] [--pages 20] [--events PATH]
        [--energy PATH] [--corrections PATH] [--no-tracemalloc] [--json PATH]

How a measurement runs:

- The archive file is only read. The newest `--new` readings are held back
  (1 by default, as a cron run finds), and the rest is seeded into a moto
  bucket as the stored archive, newest first; the optional catalogs and
  rain corrections are seeded beside it under their usual keys.
- Ambient is faked at `lookout.api.ambient_client`'s `requests`: `/devices`
  answers one station whose `lastData` is the newest reading, and the
  history endpoint pages over the newest readings, as Ambient does. No
  request waits out the rate limit.
- The archive is split in one short-lived child process and the steps run
  in another, so the parent never holds the archive: Linux carries a
  process's RSS high-water mark across the `exec` that starts a child, and
  the steps' child would otherwise start at the split's peak. The steps'
  child reports its mark before the first step (`baseline_rss_mb`): the
  imports, and the stored archive's bytes the moto bucket keeps resident
  for the whole run, which real Storj does not. Its secrets are fakes in
  the environment, its secrets file a path that does not exist, and every
  internet socket connect raises: a write lands in the moto bucket or
  nowhere, and Ambient is never called.
- Each step runs under `lookout.utils.memory_utils.track_step`, which gives
  the RSS high-water mark after it and the tracemalloc peak during it. The
  RSS mark only rises, so the step where it jumps is the one that set it.
  tracemalloc slows the run and adds its own overhead to the RSS, so take
  the RSS figures from a `--no-tracemalloc` run and the traced peaks from a
  run with it.

Needs `requirements-dev.txt` (moto). Numbers on macOS are relative only,
never the #234 gate. Never run it on itri alongside a cron run.

Offers `measure` (the harness, returning a `HarnessReport`), `format_report`,
`main` (the shell entry point) and `STEPS` (the step names, in run order).
"""

import argparse
import gc
import json
import multiprocessing
import os
import socket
import sys
import tempfile
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple

import pandas as pd

from lookout.utils.memory_utils import StepMemory, peak_rss_mb, track_step

HARNESS_MAC = "00:00:00:00:00:00"
HARNESS_BUCKET = "memory-harness"
FAKE_STORJ_ENDPOINT = "https://storj.invalid"
FAKE_ENV = {
    "AMBIENT_API_KEY": "memory-harness",
    "AMBIENT_APPLICATION_KEY": "memory-harness",
    "AMBIENT_ENDPOINT": "https://ambient.invalid/v1",
    "HC_PING_KEY": "memory-harness",
    "STORJ_ACCESS_KEY_ID": "memory-harness",
    "STORJ_SECRET_ACCESS_KEY": "memory-harness",
    "STORJ_ENDPOINT_URL": FAKE_STORJ_ENDPOINT,
    "MOTO_S3_CUSTOM_ENDPOINTS": FAKE_STORJ_ENDPOINT,
    "STATION_MAC": HARNESS_MAC,
    "DEV_MODE": "false",
}
STEPS = (
    "load_station",
    "update_archive",
    "correct_rain",
    "load_catalogs",
    "update_catalogs",
    "build",
    "publish",
    "publish_curves",
)
# One Ambient page; history beyond the held-back readings by this much, so
# catchup's first page overlaps the stored archive as it does live.
PAGE = 288
# Archive columns Ambient's history endpoint does not send.
ARCHIVE_ONLY = ("passkey", "loc", "time", "value_bin", "direction_bin", "hour")
STORED_FILE = "stored.parquet"
HISTORY_FILE = "history.parquet"


@dataclass
class HarnessReport:
    """
    One measurement.

    :param steps: Each refresh step's figures, in run order.
    :param baseline_rss_mb: The steps' process's peak RSS before the first
        step: its imports and the seeded moto bucket.
    :param readings: Readings in the local archive.
    :param new: Readings held back from the stored archive for Ambient.
    :param tracemalloc: Whether tracemalloc was on.
    :param fake_writes: The keys the run wrote, every one in the moto bucket.
    :param connections_attempted: Internet socket connects tried (and
        refused); 0 means nothing tried to leave the process.
    """

    steps: List[StepMemory]
    baseline_rss_mb: float
    readings: int
    new: int
    tracemalloc: bool
    fake_writes: List[str] = field(default_factory=list)
    connections_attempted: int = 0


# ---------- the parent: prepare, then measure in a child ----------


def measure(
    archive: Path,
    tz: Optional[str] = None,
    new: int = 1,
    pages: Optional[int] = None,
    events: Optional[Path] = None,
    energy: Optional[Path] = None,
    corrections: Optional[Path] = None,
    trace: bool = True,
) -> HarnessReport:
    """
    Run the refresh steps on a local archive and measure each one.

    :param archive: The local archive parquet; only read.
    :param tz: The station's zone; the zone of the archive's `date` column
        when None.
    :param new: Readings held back for Ambient to serve, at least 1.
    :param pages: Ambient pages catchup may fetch; the refresh run's default
        when None.
    :param events: A stored Event Catalog parquet to seed, or None.
    :param energy: A stored energy catalog parquet to seed, or None.
    :param corrections: A stored rain corrections JSON to seed, or None.
    :param trace: Whether to run tracemalloc alongside the RSS.
    :return: The report.
    :raises ValueError: If `new` is out of range or no zone is known.
    """
    with tempfile.TemporaryDirectory(prefix="memory-harness-") as workdir:
        context = multiprocessing.get_context("spawn")
        # A fresh process per task, so the split's peak never reaches the steps.
        with context.Pool(1, maxtasksperchild=1) as pool:
            readings, zone = pool.apply(
                _prepare, (Path(archive), Path(workdir), new, tz)
            )
            config = {
                "workdir": workdir,
                "tz": zone,
                "new": new,
                "readings": readings,
                "pages": pages,
                "trace": trace,
                "events": str(events) if events else None,
                "energy": str(energy) if energy else None,
                "corrections": str(corrections) if corrections else None,
            }
            return pool.apply(_run_steps, (config,))


def _prepare(
    archive: Path, workdir: Path, new: int, tz: Optional[str]
) -> Tuple[int, str]:
    """
    Split the local archive into the stored archive and Ambient's history.

    :param archive: The local archive parquet.
    :param workdir: Where the two files go.
    :param new: Readings held back from the stored archive.
    :param tz: The zone given, or None to read it from `date`.
    :return: The archive's reading count and the station's zone.
    :raises ValueError: If `new` is out of range or no zone is known.
    """
    df = pd.read_parquet(archive, engine="pyarrow")
    if len(df) < 2:
        raise ValueError("The archive needs at least 2 readings to hold one back")
    if not 1 <= new < len(df):
        raise ValueError(f"--new must be 1 to {len(df) - 1}; got {new}")
    zone = tz or _archive_zone(df)
    df = df.sort_values("dateutc", ignore_index=True)
    stored = df.iloc[: len(df) - new].iloc[::-1].reset_index(drop=True)
    stored.to_parquet(workdir / STORED_FILE, engine="pyarrow", index=False)
    history = df.tail(new + PAGE).reset_index(drop=True)
    history.to_parquet(workdir / HISTORY_FILE, engine="pyarrow", index=False)
    return len(df), zone


def _archive_zone(df: pd.DataFrame) -> str:
    """
    The station's zone, from the archive's tz-aware `date` column.

    :param df: The archive.
    :return: The zone's name.
    :raises ValueError: If `date` holds no zone.
    """
    zone = getattr(df["date"].dtype, "tz", None) if "date" in df.columns else None
    if zone is None:
        raise ValueError("The archive's date column has no zone; pass --tz")
    return str(zone)


# ---------- the child: fake the boundary, run the steps ----------


def _run_steps(config: Dict[str, Any]) -> HarnessReport:
    """
    In the child process: fake Storj and Ambient, then run and measure.

    Every `lookout` import here is deferred until the fake environment and
    secrets file are in place, so nothing reads a real secret or endpoint.

    :param config: What `measure` prepared.
    :return: The report.
    """
    workdir = Path(config["workdir"])
    # Relative import paths (`PYTHONPATH=.`) must survive the chdir below.
    sys.path[:] = [os.path.abspath(entry) for entry in sys.path]
    # The refresh logger opens refresh.log in the working directory.
    os.chdir(workdir)
    os.environ.update(FAKE_ENV)
    attempts = _block_internet()

    from lookout import secrets_loader

    secrets_loader.SECRETS_FILE = workdir / "no-secrets.toml"

    try:
        from moto import mock_aws
    except ImportError as e:  # pragma: no cover - dev requirements absent
        raise RuntimeError("The memory harness needs requirements-dev.txt") from e

    with mock_aws():
        client = _seed_bucket(config)
        before = _object_tags(client)
        history = pd.read_parquet(workdir / HISTORY_FILE, engine="pyarrow")
        device = _fake_ambient(history, config["tz"])
        newest_ms = int(history["dateutc"].max())
        del history
        gc.collect()
        baseline = peak_rss_mb()
        steps = _measured_run(device, newest_ms, config)
        after = _object_tags(client)

    return HarnessReport(
        steps=steps,
        baseline_rss_mb=baseline,
        readings=config["readings"],
        new=config["new"],
        tracemalloc=config["trace"],
        fake_writes=sorted(k for k, tag in after.items() if before.get(k) != tag),
        connections_attempted=len(attempts),
    )


def _block_internet() -> List[object]:
    """
    Make every internet socket connect in this process raise.

    :return: The list each attempted address is appended to.
    """
    attempts: List[object] = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def refuse(address: object) -> NoReturn:
        """Count the attempt and refuse it. :param address: Its target."""
        attempts.append(address)
        raise ConnectionRefusedError(f"memory harness: no network ({address})")

    def connect(sock: socket.socket, address: object) -> None:
        """`socket.connect`, refused for internet sockets.

        :param sock: The socket. :param address: Its target."""
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            refuse(address)
        real_connect(sock, address)

    def connect_ex(sock: socket.socket, address: object) -> int:
        """`socket.connect_ex`, refused for internet sockets.

        :param sock: The socket. :param address: Its target."""
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            refuse(address)
        return real_connect_ex(sock, address)

    def create_connection(address: object, *args: object, **kwargs: object) -> NoReturn:
        """`socket.create_connection`, always refused.

        :param address: Its target. :param args: Ignored. :param kwargs: Ignored."""
        refuse(address)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.create_connection = create_connection
    return attempts


def _seed_bucket(config: Dict[str, Any]) -> Any:
    """
    Create the moto bucket and store the archive and optional seeds in it.

    :param config: What `measure` prepared.
    :return: An S3 client on the moto bucket.
    """
    from lookout.core.energy_catalog import EnergyCatalog
    from lookout.core.rain_events import RainEventCatalog
    from lookout.storage import storj

    client = storj.get_s3_client()
    client.create_bucket(Bucket=HARNESS_BUCKET)
    seeds = {
        f"{HARNESS_MAC}.parquet": Path(config["workdir"]) / STORED_FILE,
        RainEventCatalog(HARNESS_MAC).catalog_path: config["events"],
        EnergyCatalog(HARNESS_MAC).catalog_path: config["energy"],
        f"{HARNESS_MAC}.rain_corrections.json": config["corrections"],
    }
    for key, path in seeds.items():
        if path:
            client.put_object(
                Bucket=HARNESS_BUCKET, Key=key, Body=Path(path).read_bytes()
            )
    return client


def _object_tags(client: Any) -> Dict[str, str]:
    """
    Each object in the moto bucket, by key, with its ETag and time.

    :param client: An S3 client on the moto bucket.
    :return: Key to a tag that changes when the object is rewritten.
    """
    tags = {}
    pages = client.get_paginator("list_objects_v2").paginate(Bucket=HARNESS_BUCKET)
    for page in pages:
        for item in page.get("Contents", []):
            tags[item["Key"]] = f"{item['ETag']}@{item['LastModified'].isoformat()}"
    return tags


def _fake_ambient(history: pd.DataFrame, tz: str) -> Dict:
    """
    Answer the Ambient client from `history`, with no rate-limit waits.

    :param history: The newest readings, any order.
    :param tz: The station's zone, sent as `lastData.tz`.
    :return: The `/devices` entry the fake serves.
    """
    from lookout.api import ambient_client, awn_controller

    newest_first = history.sort_values("dateutc", ascending=False, ignore_index=True)
    device = {
        "macAddress": HARNESS_MAC,
        "lastData": {**_ambient_records(newest_first.head(1))[0], "tz": tz},
        "info": {"name": "Memory harness"},
    }

    def get(url: str, params: Optional[Dict[str, Any]] = None) -> SimpleNamespace:
        """Ambient's answer to one request.

        :param url: The request URL. :param params: Its query parameters."""
        params = params or {}
        if url.endswith("/devices"):
            body: List[Dict] = [device]
        else:
            rows = newest_first
            if params.get("endDate") is not None:
                rows = rows[rows["dateutc"] < int(params["endDate"])]
            body = _ambient_records(rows.head(int(params.get("limit", PAGE))))
        return SimpleNamespace(status_code=200, text="", json=lambda: body)

    no_wait = _without_sleep()
    ambient_client.requests = SimpleNamespace(get=get)
    ambient_client.time = no_wait
    awn_controller.time = no_wait
    return device


def _without_sleep() -> SimpleNamespace:
    """
    The `time` module with `sleep` a no-op.

    :return: A stand-in for `time`.
    """
    names = {name: getattr(time, name) for name in dir(time) if name[0] != "_"}
    return SimpleNamespace(**{**names, "sleep": lambda _seconds: None})


def _ambient_records(df: pd.DataFrame) -> List[Dict]:
    """
    Archive rows as Ambient history records.

    :param df: Archive rows, in the order wanted.
    :return: One dict per row: archive-only columns and NaN values dropped,
        times as ISO-8601 UTC strings, numpy scalars as Python values.
    """
    sent = df.drop(columns=[c for c in ARCHIVE_ONLY if c in df.columns])
    records = []
    for row in sent.to_dict("records"):
        record = {}
        for key, value in row.items():
            if value is None or (not isinstance(value, str) and pd.isna(value)):
                continue
            if isinstance(value, pd.Timestamp):
                value = value.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z")
            elif hasattr(value, "item"):
                value = value.item()
            record[key] = value
        records.append(record)
    return records


def _measured_run(
    device: Dict, newest_ms: int, config: Dict[str, Any]
) -> List[StepMemory]:
    """
    The refresh run's steps, in `refresh.run`'s order, each one measured.

    Each step is called through its public function in
    `lookout.cli.refresh`; the build time is a minute after the newest
    reading, as a cron run builds it.

    :param device: The `/devices` entry the fake Ambient serves.
    :param newest_ms: The newest reading's `dateutc`.
    :param config: What `measure` prepared.
    :return: Each step's figures, in run order.
    """
    from lookout.cli import refresh
    from lookout.snapshot import Station

    bucket = HARNESS_BUCKET
    pages = config["pages"] or refresh.DEFAULT_PAGES
    now = datetime.fromtimestamp(newest_ms / 1000 + 60, tz=timezone.utc)
    steps: List[StepMemory] = []

    def step(name: str, fn: Callable[[], Any]) -> Any:
        """Run and measure one step.

        :param name: Its name. :param fn: The step. :return: Its result."""
        with track_step(name) as record:
            result = fn()
        steps.append(record)
        return result

    if config["trace"]:
        tracemalloc.start()
    try:
        device = step("load_station", refresh.load_station)
        station = Station.from_device(device)
        archive = step(
            "update_archive", lambda: refresh.update_archive(device, bucket, pages)
        )
        archive, live = step(
            "correct_rain",
            lambda: refresh.correct_rain(
                station, bucket, archive, device.get("lastData")
            ),
        )
        catalogs = step("load_catalogs", lambda: refresh.load_catalogs(station, bucket))
        catalogs = step(
            "update_catalogs",
            lambda: refresh.update_catalogs(station, bucket, archive, catalogs),
        )
        snapshot = step(
            "build", lambda: refresh.build(station, live, archive, catalogs, now)
        )
        step("publish", lambda: refresh.publish(snapshot, bucket))
        step(
            "publish_curves",
            lambda: refresh.publish_curves(station, archive, catalogs, bucket),
        )
    finally:
        if config["trace"]:
            tracemalloc.stop()
    return steps


# ---------- output ----------


def format_report(report: HarnessReport) -> str:
    """
    The report as a table, one line per step.

    :param report: The report.
    :return: The text.
    """

    def mb(value: Optional[float]) -> str:
        """A figure for the table. :param value: MB, or None. :return: Text."""
        return "-" if value is None else f"{value:,.1f}"

    lines = [
        f"Archive: {report.readings:,} readings, {report.new:,} held back for "
        f"Ambient; tracemalloc {'on' if report.tracemalloc else 'off'}; "
        f"peak RSS before the first step {mb(report.baseline_rss_mb)} MB",
        f"{'step':<16}{'seconds':>9}{'peak RSS MB':>14}"
        f"{'traced peak MB':>17}{'traced held MB':>17}",
    ]
    for s in report.steps:
        lines.append(
            f"{s.step:<16}{s.seconds:>9.2f}{mb(s.rss_peak_mb):>14}"
            f"{mb(s.tracemalloc_peak_mb):>17}{mb(s.tracemalloc_held_mb):>17}"
        )
    lines.append(
        f"Writes, all to the fake bucket: {len(report.fake_writes)}; "
        f"network connects refused: {report.connections_attempted}"
    )
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Read the command line.

    :param argv: Arguments, without the program name; `sys.argv` when None.
    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Measure each refresh step's memory on a local archive, "
        "with Storj and Ambient faked."
    )
    parser.add_argument("archive", type=Path, help="Local archive parquet")
    parser.add_argument("--tz", help="Station zone (default: the date column's)")
    parser.add_argument(
        "--new", type=int, default=1, help="Readings Ambient serves as new"
    )
    parser.add_argument("--pages", type=int, help="Max Ambient pages catchup fetches")
    parser.add_argument("--events", type=Path, help="Event Catalog parquet to seed")
    parser.add_argument("--energy", type=Path, help="Energy catalog parquet to seed")
    parser.add_argument(
        "--corrections", type=Path, help="Rain corrections JSON to seed"
    )
    parser.add_argument(
        "--no-tracemalloc", action="store_true", help="Measure the RSS alone"
    )
    parser.add_argument("--json", type=Path, help="Also write the report here")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """
    The shell entry point: measure, print the table, optionally write JSON.

    :param argv: Arguments, without the program name; `sys.argv` when None.
    :return: The exit status, 0.
    """
    args = parse_args(argv)
    report = measure(
        args.archive,
        tz=args.tz,
        new=args.new,
        pages=args.pages,
        events=args.events,
        energy=args.energy,
        corrections=args.corrections,
        trace=not args.no_tracemalloc,
    )
    print(format_report(report))
    if args.json:
        args.json.write_text(json.dumps(asdict(report), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

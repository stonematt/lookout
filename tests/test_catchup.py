"""
Characterize one catchup run end to end: `lookout.cli.catchup.main`.

The station's archive sits in the moto bucket, the Ambient API answers from
recorded pages, and the clock is fixed. A run must append the new readings to
`<mac>.parquet` with no duplicate `dateutc`, keep the schema, and back the
archive up first. Where today's behaviour looks wrong it is pinned as is and
the test says so.

Reusable setup: the `catchup_station` fixture. Call it to seed a station and
get a `CatchupStation` back; `.run()` is one catchup run and `.stored()` reads
the archive back. Later refresh-run tests (spec #66, seam 3) build on it: in
their module, `from tests.test_catchup import catchup_station`, seed the
station the same way, then run the refresh command instead of `.run()`.
"""

import datetime as dt
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from lookout.api import awn_controller as awn
from lookout.core import rain_corrections
from lookout.storage import storj
from tests.fixtures.ambient import (
    AmbientReplay,
    devices_body,
    load_history,
    to_ambient_records,
)
from tests.fixtures.archive import FAKE_MAC, build_archive, scenario_archive
from tests.fixtures.real import AMBIENT_END, load_real_slice
from tests.fixtures.storage import BUCKET, FROZEN_DAY, bucket_keys

ARCHIVE_KEY = f"{FAKE_MAC}.parquet"
BACKUP_KEY = f".backups/{FAKE_MAC}/{FROZEN_DAY}/{ARCHIVE_KEY}"
HOUR_MS = 3600 * 1000
FIVE_MIN_MS = 5 * 60 * 1000
# The recorded history: 36 readings, 05:00Z to 07:55Z on 2025-11-19, right
# after the real slice's newest row (04:55Z).
RECORDED_ROWS = 36


def ms(when: str) -> int:
    """
    Epoch ms for an ISO-8601 UTC time.

    :param when: The time; a naive one is read as UTC.
    :return: Milliseconds since the epoch.
    """
    return int(pd.Timestamp(when).value // 10**6)


def reporting(df: pd.DataFrame) -> List[Dict]:
    """
    A `/devices` body whose `lastData` is the newest row of `df`.

    :param df: Archive rows, newest first.
    :return: The one-device body.
    """
    return devices_body(to_ambient_records(df.head(1))[0])


def frozen_clock(now: pd.Timestamp) -> type:
    """
    A `datetime.datetime` whose `now()` is fixed.

    :param now: The moment to report, tz-aware.
    :return: A subclass to install as `awn_controller.datetime`.
    """
    fixed = now.tz_convert("UTC").to_pydatetime()

    class _Frozen(dt.datetime):
        @classmethod
        def now(cls, tz: Optional[dt.tzinfo] = None) -> dt.datetime:
            return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)

    return _Frozen


@dataclass
class CatchupStation:
    """
    The fixture station, seeded for a catchup run.

    :param s3: Client on the moto gateway.
    :param replay: The fake Ambient API; `.calls` lists each request.
    :param archive: The archive as seeded in the bucket, newest first.
    :param now: The frozen clock the run sees.
    """

    s3: object
    replay: AmbientReplay
    archive: pd.DataFrame
    now: pd.Timestamp

    def run(self, pages: int = 20, dry_run: bool = False) -> None:
        """
        Run catchup once, as the cron job does (`--bucket lookout`).

        :param pages: `--pages`.
        :param dry_run: `--dry-run`.
        """
        from lookout.cli import catchup

        catchup.main(BUCKET, dry_run=dry_run, pages=pages, repair_archive=False)

    def stored(self, key: str = ARCHIVE_KEY) -> pd.DataFrame:
        """
        Read a parquet back from the bucket.

        :param key: Its key; the archive by default.
        :return: The frame, empty if the key is missing.
        """
        return storj.get_df_from_s3(BUCKET, key, file_type="parquet")

    def keys(self) -> List[str]:
        """
        Every key in the bucket.

        :return: The keys, sorted.
        """
        return bucket_keys(self.s3)


def recorded_pages(overlap_hours: float = 2.0) -> List[Dict]:
    """
    The recorded history, plus the real slice's last hours served again.

    Ambient pages end at `endDate` and run backwards, so a real page reaches
    into readings the archive already holds; the overlap makes dedup matter.

    :param overlap_hours: How much of the archive the pages repeat.
    :return: History records, any order.
    """
    real = load_real_slice()
    newest = int(real["dateutc"].max())
    tail = real[real["dateutc"] > newest - int(overlap_hours * HOUR_MS)]
    return to_ambient_records(tail) + load_history()


@pytest.fixture
def catchup_station(
    monkeypatch: pytest.MonkeyPatch,
    fake_s3: object,
    ambient_replay: AmbientReplay,
    frozen_today: str,
) -> Callable[..., CatchupStation]:
    """
    Seed the fixture station for a catchup run.

    Pins `STATION_MAC` to `FAKE_MAC`, drops the controller's one-second
    sleeps, and reads the archive from the bucket, not a dev-mode local
    cache. `backup_data` stamps `FROZEN_DAY`.

    :return: `seed(archive=None, history=None, devices=None, now=None)`,
        which writes the archive to the bucket, loads the fake Ambient API,
        freezes the controller's clock and returns a `CatchupStation`. By
        default the archive is the real slice, the history is
        `recorded_pages()`, `/devices` is the recorded body (its `lastData`
        is the newest recorded reading) and now is `AMBIENT_END`.
    """
    monkeypatch.setenv("STATION_MAC", FAKE_MAC)
    monkeypatch.setattr(awn, "time", SimpleNamespace(sleep=lambda _s: None))
    # Dev mode reads st.secrets directly, which raises with no
    # .streamlit/secrets.toml (held finding); pin the production branch.
    monkeypatch.setattr(awn, "is_dev_mode", lambda: False)

    def seed(
        archive: Optional[pd.DataFrame] = None,
        history: Optional[List[Dict]] = None,
        devices: Optional[List[Dict]] = None,
        now: Optional[str] = None,
    ) -> CatchupStation:
        archive = load_real_slice() if archive is None else archive
        clock = pd.Timestamp(now or AMBIENT_END)
        monkeypatch.setattr(awn, "datetime", frozen_clock(clock))
        if devices is not None:
            ambient_replay.devices = devices
        records = recorded_pages() if history is None else history
        ambient_replay.history = sorted(
            records, key=lambda r: r["dateutc"], reverse=True
        )
        storj.save_df_to_s3(archive, BUCKET, ARCHIVE_KEY)
        return CatchupStation(fake_s3, ambient_replay, archive, clock)

    return seed


def history_calls(station: CatchupStation) -> List[Dict]:
    """
    The params of each history request the run made.

    :param station: The station after a run.
    :return: One params dict per `/devices/{mac}` call, in order.
    """
    return [params for path, params in station.replay.calls if path != "/devices"]


# ---------- the run against recorded pages ----------


class TestCatchupRun:
    def test_appends_the_recorded_readings_with_no_duplicates(self, catchup_station):
        station = catchup_station()

        station.run()

        stored = station.stored()
        assert len(stored) == len(station.archive) + RECORDED_ROWS
        assert stored["dateutc"].is_unique
        assert stored["dateutc"].is_monotonic_decreasing
        assert int(stored["dateutc"].max()) == ms("2025-11-19T07:55:00Z")
        assert int(stored["dateutc"].min()) == int(station.archive["dateutc"].min())

    def test_keeps_the_archive_schema(self, catchup_station):
        station = catchup_station()

        station.run()

        stored = station.stored()
        assert list(stored.columns) == list(station.archive.columns)
        assert stored.dtypes.astype(str).to_dict() == (
            station.archive.dtypes.astype(str).to_dict()
        )

    def test_rows_before_the_overlap_are_kept_as_they_were(self, catchup_station):
        station = catchup_station()
        overlap_from = int(station.archive["dateutc"].max()) - 2 * HOUR_MS

        station.run()

        stored = station.stored()
        before = station.archive[station.archive["dateutc"] <= overlap_from]
        after = stored[stored["dateutc"] <= overlap_from]
        assert_frame_equal(after.reset_index(drop=True), before.reset_index(drop=True))

    def test_new_rows_carry_the_readings_ambient_sent(self, catchup_station):
        station = catchup_station()

        station.run()

        stored = station.stored().set_index("dateutc")
        for record in load_history():
            row = stored.loc[record["dateutc"]]
            assert row["tempf"] == record["tempf"]
            assert row["yearlyrainin"] == record["yearlyrainin"]
            assert row["date"] == pd.Timestamp(record["date"])
            # Archive-only columns stay empty on rows Ambient sent.
            assert pd.isna(row["passkey"]) and pd.isna(row["hour"])

    def test_backs_the_archive_up_before_writing(self, catchup_station):
        station = catchup_station()

        station.run()

        assert station.keys() == [BACKUP_KEY, ARCHIVE_KEY]
        assert_frame_equal(station.stored(BACKUP_KEY), station.archive)

    def test_the_archive_backup_leaves_the_catalogs_out(self, catchup_station, fake_s3):
        station = catchup_station()
        catalog_key = f"{FAKE_MAC}.event_catalog.parquet"
        fake_s3.put_object(Bucket=BUCKET, Key=catalog_key, Body=b"x")

        station.run()

        assert station.keys() == sorted([BACKUP_KEY, ARCHIVE_KEY, catalog_key])

    def test_asks_ambient_for_one_page_ending_now(self, catchup_station):
        station = catchup_station()

        station.run()

        assert history_calls(station) == [
            {
                "apiKey": "test-ambient-api-key",
                "applicationKey": "test-ambient-application-key",
                "limit": 250,
                "endDate": ms(AMBIENT_END),
            }
        ]

    def test_a_second_run_with_nothing_new_changes_nothing(self, catchup_station):
        station = catchup_station()
        station.run()
        first = station.stored()

        station.run()

        assert_frame_equal(station.stored(), first)
        # Same day: the backup is skipped, so it still holds the original.
        assert_frame_equal(station.stored(BACKUP_KEY), station.archive)

    def test_dry_run_writes_nothing(self, catchup_station):
        station = catchup_station()

        station.run(dry_run=True)

        assert station.keys() == [ARCHIVE_KEY]
        assert_frame_equal(station.stored(), station.archive)


# ---------- the stored order (#133) ----------


def newest_first(df: pd.DataFrame) -> pd.DataFrame:
    """
    The frame's rows, newest `dateutc` first, index reset.

    :param df: Archive rows, any order.
    :return: The rows reordered.
    """
    return df.sort_values("dateutc", ascending=False).reset_index(drop=True)


class TestStoredOrder:
    """Whatever the run, the archive in storage is newest first (#133)."""

    def test_a_run_with_nothing_new_stores_newest_first(self, catchup_station):
        archive = load_real_slice()
        station = catchup_station(
            archive=archive, history=[], devices=reporting(archive)
        )

        station.run()

        stored = station.stored()
        assert stored["dateutc"].is_monotonic_decreasing
        assert_frame_equal(stored, newest_first(archive))

    def test_a_run_with_new_rows_stores_newest_first(self, catchup_station):
        station = catchup_station()

        station.run()

        stored = station.stored()
        assert stored["dateutc"].is_monotonic_decreasing
        assert len(stored) == len(station.archive) + RECORDED_ROWS

    def test_repair_archive_stores_newest_first_with_the_same_rows(
        self, catchup_station
    ):
        from lookout.cli import catchup

        archive = load_real_slice()
        oldest_first = archive.iloc[::-1].reset_index(drop=True)
        station = catchup_station(archive=oldest_first)

        catchup.main(BUCKET, dry_run=False, pages=20, repair_archive=True)

        stored = station.stored()
        assert stored["dateutc"].is_monotonic_decreasing
        assert_frame_equal(stored, newest_first(archive))


# ---------- paging over a longer outage ----------


OUTAGE_START = "2026-01-02T00:00:00Z"
OUTAGE_END = "2026-01-04T00:00:00Z"
OUTAGE_ROWS = 2 * 288  # two days of five-minute readings
# One page ends 247 intervals past the cursor, exclusive; the cursor's own
# reading is already archived, so a page brings 246 new ones.
PAGE_ROWS = 246


@pytest.fixture
def after_outage(catchup_station) -> CatchupStation:
    """
    A synthetic archive two days behind a station still reporting.

    :return: The seeded station; now is `OUTAGE_END`.
    """
    archive = build_archive("2026-01-01T00:00:00Z", OUTAGE_START)
    live = build_archive(OUTAGE_START, OUTAGE_END)
    return catchup_station(
        archive=archive,
        history=to_ambient_records(live),
        devices=reporting(live),
        now=OUTAGE_END,
    )


class TestPaging:
    def test_pages_forward_until_it_reaches_now(self, after_outage):
        after_outage.run()

        stored = after_outage.stored()
        assert len(stored) == len(after_outage.archive) + OUTAGE_ROWS
        assert stored["dateutc"].is_unique
        assert int(stored["dateutc"].max()) == ms(OUTAGE_END) - FIVE_MIN_MS
        assert len(history_calls(after_outage)) == -(-OUTAGE_ROWS // PAGE_ROWS)

    def test_the_pages_argument_caps_the_run(self, after_outage):
        """One page brings `PAGE_ROWS` readings past the archive, not 250."""
        after_outage.run(pages=1)

        stored = after_outage.stored()
        assert len(stored) == len(after_outage.archive) + PAGE_ROWS
        assert int(stored["dateutc"].max()) == (
            ms(OUTAGE_START) + (PAGE_ROWS - 1) * FIVE_MIN_MS
        )


# ---------- rain corrections ----------


def test_each_run_keeps_the_stored_rain_counters_raw(catchup_station):
    """
    Catchup saves the archive raw (#95): corrections are applied on read.

    With one carryover on 2026-01-09 in the catalog, three runs in a row
    leave that day's stored weekly counter at its raw value, and the rows
    each run fetched are stored raw too.
    """
    full = scenario_archive()
    rain_corrections.save_catalog(
        {"corrections": rain_corrections.detect_carryovers(full)}, FAKE_MAC
    )
    station = catchup_station(
        archive=full[full["dateutc"] < ms("2026-01-10T00:00:00Z")],
        history=to_ambient_records(full),
        devices=reporting(full),
        now="2026-01-11T08:00:00Z",
    )
    noon = ms("2026-01-09T20:00:00Z")  # Friday 12:00 local, archived
    fetched = ms("2026-01-10T14:00:00Z")  # Saturday 06:00 local, fetched
    raw = full.set_index("dateutc")

    for _ in range(3):
        station.run()
        stored = station.stored().set_index("dateutc")
        for row in (noon, fetched):
            for col in ("dailyrainin", "weeklyrainin", "monthlyrainin"):
                assert stored.loc[row, col] == pytest.approx(raw.loc[row, col])


# ---------- archive safety (#160) ----------


class FlakyStorj:
    """
    The moto client, with chosen calls failing the way a Storj outage does.

    :param s3: The moto client every other call goes to.
    :param read: A key whose `get_object` raises.
    :param write: A key whose `upload_fileobj` raises.
    :param copy: Whether `copy_object` (the backup) raises.
    """

    def __init__(
        self,
        s3: object,
        read: Optional[str] = None,
        write: Optional[str] = None,
        copy: bool = False,
    ) -> None:
        self.s3 = s3
        self.read = read
        self.write = write
        self.copy = copy

    def __getattr__(self, name: str) -> object:
        return getattr(self.s3, name)

    def get_object(self, **kwargs: object) -> Dict:
        if kwargs["Key"] == self.read:
            raise OSError("storj read timed out")
        return self.s3.get_object(**kwargs)

    def upload_fileobj(self, **kwargs: object) -> None:
        if kwargs["Key"] == self.write:
            raise OSError("storj write refused")
        return self.s3.upload_fileobj(**kwargs)

    def copy_object(self, **kwargs: object) -> Dict:
        if self.copy:
            raise OSError("storj copy refused")
        return self.s3.copy_object(**kwargs)


def flaky_storj(monkeypatch: pytest.MonkeyPatch, s3: object, **fail: object) -> None:
    """
    Route every `storj` client through a `FlakyStorj`.

    :param monkeypatch: The test's monkeypatch.
    :param s3: The moto client.
    :param fail: `FlakyStorj` keyword arguments.
    """
    flaky = FlakyStorj(s3, **fail)
    monkeypatch.setattr(storj, "get_s3_client", lambda: flaky)


def stored_bytes(station: CatchupStation, key: str = ARCHIVE_KEY) -> bytes:
    """
    The raw bytes of a stored object.

    :param station: The seeded station.
    :param key: Its key; the archive by default.
    :return: The body.
    """
    return station.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


class TestArchiveSafety:
    def test_a_failed_archive_read_leaves_the_archive_and_fails_the_run(
        self, catchup_station, monkeypatch
    ):
        station = catchup_station()
        before = stored_bytes(station)
        flaky_storj(monkeypatch, station.s3, read=ARCHIVE_KEY)

        with pytest.raises(storj.StorageReadError):
            station.run()

        assert stored_bytes(station) == before

    def test_an_empty_archive_read_leaves_the_archive_and_fails_the_run(
        self, catchup_station
    ):
        station = catchup_station(archive=load_real_slice().head(0))
        before = stored_bytes(station)

        with pytest.raises(storj.StorageReadError):
            station.run()

        assert stored_bytes(station) == before

    def test_an_unreadable_archive_body_leaves_it_and_fails_the_run(
        self, catchup_station
    ):
        station = catchup_station()
        station.s3.put_object(Bucket=BUCKET, Key=ARCHIVE_KEY, Body=b"not parquet")

        with pytest.raises(storj.StorageReadError):
            station.run()

        assert stored_bytes(station) == b"not parquet"

    def test_a_save_that_would_shrink_the_archive_is_refused(
        self, catchup_station, monkeypatch
    ):
        station = catchup_station()
        before = stored_bytes(station)
        monkeypatch.setattr(
            awn,
            "get_history_since_last_archive",
            lambda device, archive, **_: archive.iloc[1:],
        )

        with pytest.raises(storj.ArchiveShrinkError):
            station.run()

        assert stored_bytes(station) == before

    def test_a_failed_archive_save_fails_the_run(self, catchup_station, monkeypatch):
        station = catchup_station()
        flaky_storj(monkeypatch, station.s3, write=ARCHIVE_KEY)

        with pytest.raises(storj.StorageWriteError):
            station.run()

        assert len(station.stored()) == len(station.archive)

    def test_a_failed_backup_fails_the_run_before_anything_is_saved(
        self, catchup_station, monkeypatch
    ):
        station = catchup_station()
        before = stored_bytes(station)
        flaky_storj(monkeypatch, station.s3, copy=True)

        with pytest.raises(storj.BackupError):
            station.run()

        assert stored_bytes(station) == before
        assert BACKUP_KEY not in station.keys()


class TestLongOutage:
    """#160: an outage past the forward seek's ~208 h reach (#132)."""

    OUTAGE_START = "2026-01-02T00:00:00Z"
    RESUME = "2026-01-14T00:00:00Z"  # 288 hours of silence

    def seed(self, catchup_station, now: str) -> CatchupStation:
        archive = build_archive("2026-01-01T00:00:00Z", self.OUTAGE_START)
        after = build_archive(self.RESUME, now)
        live = pd.concat([after, archive], ignore_index=True)
        return catchup_station(
            archive=archive,
            history=to_ambient_records(live),
            devices=reporting(after),
            now=now,
        )

    def test_catchup_crosses_a_288_hour_outage_in_one_run(self, catchup_station):
        station = self.seed(catchup_station, "2026-01-14T12:00:00Z")

        station.run()

        stored = station.stored()
        assert len(stored) == len(station.archive) + 12 * 12
        assert int(stored["dateutc"].max()) == ms("2026-01-14T11:55:00Z")
        assert stored["dateutc"].is_unique

    def test_a_crossing_out_of_pages_fails_the_run_and_saves_nothing(
        self, catchup_station
    ):
        station = self.seed(catchup_station, "2026-01-18T00:00:00Z")
        before = stored_bytes(station)

        with pytest.raises(awn.CatchupStalled):
            station.run(pages=12)

        assert stored_bytes(station) == before

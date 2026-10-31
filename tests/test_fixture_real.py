"""
Pins the frozen real slice and the recorded Ambient responses, and that no
committed fixture carries a station identifier or key.
"""

import re
from pathlib import Path

import pandas as pd

from lookout.api import ambient_client
from lookout.config import DEFAULT_MAC_ADDRESS
from tests.conftest import FAKE_SECRETS
from tests.fixtures.ambient import load_devices, load_history
from tests.fixtures.archive import FAKE_MAC, STATION_TZ
from tests.fixtures.make_real_fixtures import FAKE_PASSKEY
from tests.fixtures.real import SLICE_END, SLICE_START, load_real_slice

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _ms(iso: str) -> int:
    return pd.Timestamp(iso).value // 1_000_000


def test_real_slice_keeps_the_archive_schema():
    df = load_real_slice()

    assert len(df) == 520
    assert df["dateutc"].dtype == "int64"
    assert str(df["date"].dtype) == f"datetime64[ns, {STATION_TZ}]"
    assert str(df["lastRain"].dtype) == f"datetime64[ns, {STATION_TZ}]"
    assert df["dateutc"].is_monotonic_decreasing
    assert df["dateutc"].min() >= _ms(SLICE_START)
    assert df["dateutc"].max() < _ms(SLICE_END)
    assert len(df.columns) == 55


def test_real_slice_has_rain_and_a_gap():
    df = load_real_slice().sort_values("dateutc")
    assert df["dailyrainin"].max() > 0.1
    steps = df["dateutc"].diff().dropna() / 60_000
    assert steps.max() >= 90  # the 94-minute upload gap on 2025-11-18


def test_history_runs_on_from_the_slice_newest_first():
    history = load_history()
    stamps = [r["dateutc"] for r in history]

    assert len(history) == 36
    assert stamps == sorted(stamps, reverse=True)
    assert min(stamps) >= _ms(SLICE_END)
    assert history[0]["date"].endswith(".000Z")
    assert "passkey" not in history[0]


def test_devices_body_names_the_fake_station():
    (device,) = load_devices()
    assert device["macAddress"] == FAKE_MAC
    assert device["info"]["name"] == "Fixture Station"
    assert device["lastData"]["dateutc"] == load_history()[0]["dateutc"]
    assert device["lastData"]["tz"] == STATION_TZ


def test_fixtures_hold_no_station_identifier_or_key():
    real_macs = {DEFAULT_MAC_ADDRESS, DEFAULT_MAC_ADDRESS.replace(":", "-")}
    hex32 = re.compile(rb"[0-9a-f]{32}")
    for path in FIXTURE_DIR.rglob("*"):
        if path.suffix not in (".json", ".parquet"):
            continue
        blob = path.read_bytes()
        for mac in real_macs:
            assert mac.encode() not in blob, path
        for found in hex32.findall(blob):
            assert found == FAKE_PASSKEY.encode(), path
        for value in FAKE_SECRETS.values():
            assert value.encode() not in blob, path


def test_replay_serves_the_ambient_client(ambient_replay):
    devices = ambient_client.get_devices()
    assert devices[0]["macAddress"] == FAKE_MAC

    stamps = [r["dateutc"] for r in load_history()]
    page = ambient_client.get_device_history_raw(FAKE_MAC, limit=5)
    assert [r["dateutc"] for r in page] == stamps[:5]

    # endDate is exclusive: the next page starts just past the last one.
    older = ambient_client.get_device_history_raw(
        FAKE_MAC, limit=3, end_date=page[-1]["dateutc"]
    )
    assert [r["dateutc"] for r in older] == stamps[5:8]

    assert ambient_client.get_device_history_raw("AA:AA:AA:AA:AA:AA") == []

    path, params = ambient_replay.calls[1]
    assert path == f"/devices/{FAKE_MAC}"
    assert params["apiKey"] == FAKE_SECRETS["AMBIENT_API_KEY"]
    assert params["limit"] == 5

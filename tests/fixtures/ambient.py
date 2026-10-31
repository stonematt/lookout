"""
ambient.py: Replay recorded Ambient Weather responses without the network.

`ambient/devices.json` and `ambient/history.json` are the `/devices` and
`/devices/{mac}` bodies for the fixture station, reconstructed from the real
archive in the three hours after the real slice ends (see
`make_real_fixtures.py`). The MAC is `FAKE_MAC`; no key appears in them.

`AmbientReplay` stands in for `requests.get` inside
`lookout.api.ambient_client`. `/devices` returns the devices body;
`/devices/{mac}` returns history records with `dateutc < endDate` (when
given), newest first, at most `limit` of them. An unknown MAC returns `[]`.
Every call is kept in `calls` as `(path, params)`.

`to_ambient_records(df)` and `devices_body(last_data)` turn archive rows (real
or from `build_archive`) into those bodies, e.g. to serve a live span that
runs ahead of a synthetic archive with a matching `lastData`.
"""

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from tests.fixtures.archive import FAKE_MAC, STATION_TZ

AMBIENT_DIR = Path(__file__).parent / "ambient"


# Columns the archive carries that Ambient's history endpoint does not send.
ARCHIVE_ONLY = ("passkey", "loc", "time", "value_bin", "direction_bin", "hour")


def _iso_z(ts: pd.Timestamp) -> str:
    """Format a time as Ambient does: UTC with milliseconds and a Z."""
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z")


def to_ambient_records(df: pd.DataFrame) -> List[Dict]:
    """
    Reshape archive rows as Ambient history records.

    :param df: Archive rows, real or from `build_archive`, in the order wanted.
    :return: One dict per row: archive-only columns and NaN fields dropped,
        timestamps as ISO-8601 UTC strings, numpy scalars as Python values.
    """
    records = []
    for row in df.drop(columns=[c for c in ARCHIVE_ONLY if c in df.columns]).to_dict(
        "records"
    ):
        record = {}
        for key, value in row.items():
            if value is None or (not isinstance(value, str) and pd.isna(value)):
                continue
            if isinstance(value, pd.Timestamp):
                value = _iso_z(value)
            elif hasattr(value, "item"):
                value = value.item()
            record[key] = value
        records.append(record)
    return records


def devices_body(last_data: Dict) -> List[Dict]:
    """
    Build the `/devices` body for the invented fixture station.

    :param last_data: The record to report as `lastData`; `tz` is added.
    :return: A one-device list under `FAKE_MAC`.
    """
    return [
        {
            "macAddress": FAKE_MAC,
            "lastData": {**last_data, "tz": STATION_TZ},
            "info": {
                "name": "Fixture Station",
                "coords": {
                    "coords": {"lat": 45.0, "lon": -123.0},
                    "address": "Fixture Address",
                    "location": "Fixture",
                    "elevation": 50.0,
                    "geo": {"type": "Point", "coordinates": [-123.0, 45.0]},
                },
            },
        }
    ]


def load_devices() -> List[Dict]:
    """
    Read the recorded `/devices` body.

    :return: A fresh copy each call.
    """
    with open(AMBIENT_DIR / "devices.json") as f:
        return json.load(f)


def load_history() -> List[Dict]:
    """
    Read the recorded history records.

    :return: The records newest first; a fresh copy each call.
    """
    with open(AMBIENT_DIR / "history.json") as f:
        return json.load(f)


class FakeResponse:
    """The slice of `requests.Response` the Ambient client reads."""

    def __init__(self, body: object) -> None:
        self._body = body
        self.status_code = 200
        self.text = json.dumps(body)

    def json(self) -> object:
        """
        Decode the body.

        :return: A deep copy of the body.
        """
        return copy.deepcopy(self._body)


class AmbientReplay:
    """
    A `requests.get` stand-in serving recorded Ambient bodies.

    :param devices: The `/devices` body; defaults to the recorded one.
    :param history: History records for the first device; defaults to the
        recorded ones. Any order; they are served newest first.
    """

    def __init__(
        self,
        devices: Optional[List[Dict]] = None,
        history: Optional[List[Dict]] = None,
    ) -> None:
        self.devices = load_devices() if devices is None else devices
        records = load_history() if history is None else history
        self.history = sorted(records, key=lambda r: r["dateutc"], reverse=True)
        self.calls: List[Tuple[str, Dict]] = []

    def __call__(
        self, url: str, params: Optional[Dict] = None, **_: object
    ) -> FakeResponse:
        """
        Answer one GET the way the Ambient REST API would.

        :param url: The full request URL.
        :param params: Query params; `endDate` and `limit` are honoured.
        :return: The response.
        """
        params = dict(params or {})
        path = "/devices" + url.split("/devices", 1)[1] if "/devices" in url else url
        self.calls.append((path, params))
        if path == "/devices":
            return FakeResponse(self.devices)
        mac = path.removeprefix("/devices/")
        if not self.devices or mac != self.devices[0]["macAddress"]:
            return FakeResponse([])
        records = self.history
        if params.get("endDate") is not None:
            end = int(params["endDate"])
            records = [r for r in records if r["dateutc"] < end]
        return FakeResponse(records[: int(params.get("limit", 288))])

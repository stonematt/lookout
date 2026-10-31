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
"""

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

AMBIENT_DIR = Path(__file__).parent / "ambient"


def load_devices() -> List[Dict]:
    """The recorded `/devices` body; a fresh copy each call."""
    with open(AMBIENT_DIR / "devices.json") as f:
        return json.load(f)


def load_history() -> List[Dict]:
    """The recorded history records, newest first; a fresh copy each call."""
    with open(AMBIENT_DIR / "history.json") as f:
        return json.load(f)


class FakeResponse:
    """The slice of `requests.Response` the Ambient client reads."""

    def __init__(self, body: object, status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code
        self.text = json.dumps(body)

    def json(self) -> object:
        """Return a copy of the body."""
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

    def __call__(self, url: str, params: Optional[Dict] = None, **_: object):
        """Answer one GET the way the Ambient REST API would."""
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

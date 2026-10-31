"""
TEMPORARY golden tests of today's Streamlit load path.

Delete this module and `tests/golden/load_path_*.json` when Streamlit
is retired: #72's `build_snapshot` golden tests replace them (spec #66,
"Testing Decisions"). Every test here carries the `temporary` mark, so
`pytest -m "not temporary"` runs the suite without them.

`lookout.core.data_processing.load_or_update_data` is driven on a fake session
state (a plain dict) the way `streamlit_app.py` calls it, against the real
archive slice in the moto bucket and the recorded Ambient pages, with the
controller's clock fixed at `AMBIENT_END`. Each scenario's outcome is summed
up as plain JSON (history, max-`dateutc` bookkeeping, trends, the Event
Catalog and the energy catalog) and compared with its golden file. Values the
code stamps from the wall clock or a fresh UUID (`event_id`, `created_at`,
`updated_at`) are left out.

Regenerate the golden files, then review the diff:

    LOOKOUT_REGEN_GOLDEN=1 python -m pytest tests/test_load_path_golden.py \
        -p no:cacheprovider
"""

import json
import os
import re
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from lookout.config import DEFAULT_MAC_ADDRESS
from lookout.core import data_processing as lo_dp
from lookout.core.rain_events import RainEventCatalog
from tests.fixtures.ambient import load_devices
from tests.fixtures.archive import FAKE_MAC
from tests.conftest import FAKE_SECRETS
from tests.fixtures.real import FAKE_PASSKEY, load_real_slice
from tests.fixtures.storage import BUCKET
from tests.test_catchup import ARCHIVE_KEY, CatchupStation
from tests.test_catchup import catchup_station  # noqa: F401 (a fixture)

# Not under tests/fixtures/: test_fixture_real.py counts the files there.
GOLDEN_DIR = Path(__file__).parent / "golden"
REGENERATE = os.environ.get("LOOKOUT_REGEN_GOLDEN") == "1"
# streamlit_app.py's refresh window: 6 minutes to 3 days.
SHORT_MINUTES = 6
LONG_MINUTES = 3 * 24 * 60
# Fields stamped from the wall clock or a fresh UUID on every run.
UNSTABLE_EVENT_FIELDS = ("event_id", "created_at", "updated_at")

with warnings.catch_warnings():
    # No pytest config registers marks; this module is the only user.
    warnings.simplefilter("ignore", pytest.PytestUnknownMarkWarning)
    pytestmark = pytest.mark.temporary


# ---------- summarising the session ----------


def _plain(value: object) -> object:
    """
    One value as JSON: rounded floats, ISO times, None for missing.

    :param value: A cell or scalar from the session state.
    :return: A JSON-serialisable value.
    """
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, str):
        return value
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6)
    return str(value)


def _history(df: pd.DataFrame) -> Dict:
    """
    The loaded history: size, order, range, schema and newest row.

    :param df: `history_df` from the session.
    :return: A JSON-ready summary.
    """
    return {
        "rows": len(df),
        "dateutc_unique": bool(df["dateutc"].is_unique),
        "newest_first": bool(df["dateutc"].is_monotonic_decreasing),
        "max_dateutc": int(df["dateutc"].max()),
        "min_dateutc": int(df["dateutc"].min()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "newest_row": _plain(df.iloc[0].to_dict()),
    }


def _events(df: pd.DataFrame) -> list:
    """
    The Event Catalog, one dict per event, oldest first.

    :param df: `rain_events_catalog` from the session.
    :return: The events, unstable fields dropped.
    """
    events = df.drop(columns=[c for c in UNSTABLE_EVENT_FIELDS if c in df.columns])
    events = events.sort_values("start_time") if "start_time" in events else events
    return [_plain(row) for row in events.to_dict("records")]


def _energy(df: pd.DataFrame) -> Dict:
    """
    The energy catalog in brief: periods, span and total.

    :param df: `energy_catalog` from the session.
    :return: A JSON-ready summary.
    """
    if df.empty:
        return {"periods": 0}
    return {
        "periods": len(df),
        "columns": list(df.columns),
        "first_period_start": _plain(df["period_start"].min()),
        "last_period_end": _plain(df["period_end"].max()),
        "total_kwh": _plain(df["energy_kwh"].sum()),
    }


def summarise(state: Dict) -> Dict:
    """
    What the load path left in the session, as plain JSON.

    :param state: The fake session state.
    :return: The summary compared with a golden file.
    """
    return {
        "history": _history(state["history_df"]),
        "bookkeeping": {
            key: _plain(state[key])
            for key in (
                "history_max_dateutc",
                "archive_max_dateutc",
                "cloud_max_dateutc",
                "session_counter",
            )
        },
        "trends": {key: state.get(key) for key in ("temp_trend", "barom_trend")},
        "rain_events_catalog": _events(state["rain_events_catalog"]),
        "energy_catalog": _energy(state["energy_catalog"]),
    }


def assert_matches_golden(summary: Dict, name: str) -> None:
    """
    Compare a summary with `tests/golden/<name>.json`.

    With `LOOKOUT_REGEN_GOLDEN=1` the file is rewritten instead.

    :param summary: The scenario's summary.
    :param name: The golden file's stem.
    """
    path = GOLDEN_DIR / f"{name}.json"
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if REGENERATE:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert json.loads(text) == json.loads(path.read_text())


# ---------- driving the load path ----------


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> Dict:
    """
    A fake Streamlit session state: a plain dict the code under test reads.

    :return: The dict.
    """
    state: Dict = {}
    monkeypatch.setattr(st, "session_state", state)
    return state


def device_at(dateutc: Optional[int] = None) -> Dict:
    """
    The recorded `/devices` entry, optionally reporting an older reading.

    :param dateutc: `lastData.dateutc` to report; the recorded one if None.
    :return: The device dict, as `streamlit_app.py` hands it on.
    """
    (device,) = load_devices()
    if dateutc is not None:
        device["lastData"] = {**device["lastData"], "dateutc": dateutc}
    return device


def load(state: Dict, device: Dict) -> None:
    """
    One script run's data step, as `streamlit_app.py` performs it.

    :param state: The fake session state.
    :param device: The pinned station.
    """
    state["device"] = device
    lo_dp.load_or_update_data(
        device=device,
        bucket=BUCKET,
        file_type="parquet",
        auto_update=True,
        short_minutes=SHORT_MINUTES,
        long_minutes=LONG_MINUTES,
    )


@pytest.fixture
def station(
    catchup_station: Callable[..., CatchupStation],  # noqa: F811 (the fixture)
) -> CatchupStation:
    """
    The real slice in the bucket and the recorded pages served.

    :return: The seeded station; its clock is fixed at `AMBIENT_END`.
    """
    return catchup_station()


def archive_max(station: CatchupStation) -> int:
    """
    The seeded archive's newest `dateutc`.

    :param station: The seeded station.
    :return: Epoch ms.
    """
    return int(station.archive["dateutc"].max())


# ---------- scenarios ----------


def test_initial_load(station, session):
    """First script run: lastData matches the archive, so nothing is fetched."""
    load(session, device_at(archive_max(station)))

    summary = summarise(session)
    assert summary["bookkeeping"]["history_max_dateutc"] == archive_max(station)
    assert summary["history"]["rows"] == len(load_real_slice())
    assert station.replay.calls == []
    assert station.keys() == [ARCHIVE_KEY]  # catalogs are built, not saved
    assert_matches_golden(summary, "load_path_initial")


def test_incremental_update(station, session):
    """
    A later run whose lastData is three hours ahead fetches and stitches.

    The archive max stays at the archive; the history max moves to the
    newest fetched reading, and the trends are taken again from it. The
    Event Catalog is not rebuilt: it is already in the session.
    """
    load(session, device_at(archive_max(station)))
    catalog_before = session["rain_events_catalog"]
    assert (session["temp_trend"], session["barom_trend"]) == ("→", "→")
    # The newest reading jumps, so fresh trends differ from the first load's.
    newest = station.replay.history[0]
    newest["tempf"] += 10.0
    newest["baromrelin"] += 0.5

    load(session, device_at())

    summary = summarise(session)
    assert summary["trends"] == {"temp_trend": "↑", "barom_trend": "↑"}
    assert summary["bookkeeping"]["archive_max_dateutc"] == archive_max(station)
    assert summary["bookkeeping"]["history_max_dateutc"] == (
        device_at()["lastData"]["dateutc"]
    )
    assert session["rain_events_catalog"] is catalog_before
    assert station.keys() == [ARCHIVE_KEY]
    assert_matches_golden(summary, "load_path_incremental")


def test_initial_load_updates_a_stored_event_catalog(station, session):
    """
    A stored catalog whose last event was still ongoing is re-detected from
    that event's start over the loaded history, and is not saved back.
    """
    real = station.archive
    cut = int(pd.Timestamp("2025-11-18T06:00:00Z").value // 10**6)
    catalog = RainEventCatalog(FAKE_MAC)
    stored = catalog.detect_and_catalog_events(real[real["dateutc"] < cut])
    stored_key = catalog.catalog_path

    load(session, device_at(archive_max(station)))

    summary = summarise(session)
    assert len(stored) == 1 and bool(stored.iloc[0]["flags"]["ongoing"])
    assert summary["rain_events_catalog"][0]["flags"]["ongoing"] is False
    assert_matches_golden(summary, "load_path_stored_catalog")
    # The stored copy is still the one detected before the load.
    assert len(catalog.load_catalog()) == 1
    assert sorted(station.keys()) == sorted([ARCHIVE_KEY, stored_key])


def test_golden_files_hold_no_station_identifier_or_key():
    """The leak scan `test_fixture_real.py` runs on tests/fixtures/, here."""
    texts = [p.read_text() for p in sorted(GOLDEN_DIR.glob("*.json"))]
    assert len(texts) == 3
    for text in texts:
        assert DEFAULT_MAC_ADDRESS.lower() not in text.lower()
        assert DEFAULT_MAC_ADDRESS.replace(":", "-").lower() not in text.lower()
        for found in re.findall(r"[0-9a-fA-F]{32}", text):
            assert found == FAKE_PASSKEY
        for value in FAKE_SECRETS.values():
            assert value not in text

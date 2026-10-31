"""
Characterizes catalog storage against a moto-faked Storj gateway: every
catalog's key path, and load, save, backup and exists for the generic
`catalog_utils` store and the Event Catalog (`RainEventCatalog`).

Event detection and catalog merging are pinned elsewhere; this is storage only.
"""

import re

import pandas as pd
import pytest
import streamlit as st
from pandas.testing import assert_frame_equal

from lookout.config import DEFAULT_MAC_ADDRESS
from lookout.core import catalog_utils, day_periods, rain_corrections
from lookout.core.energy_catalog import EnergyCatalog
from lookout.core.rain_events import RainEventCatalog
from lookout.storage import storj
from tests.fixtures.archive import FAKE_MAC
from tests.fixtures.storage import BUCKET, bucket_keys


@pytest.fixture
def station(monkeypatch):
    """Put the fake station in Streamlit session state, as the UI does."""
    monkeypatch.setitem(st.session_state, "device", {"macAddress": FAKE_MAC})
    return FAKE_MAC


@pytest.fixture
def no_station(monkeypatch):
    """Make sure no device is in session state."""
    if "device" in st.session_state:
        monkeypatch.delitem(st.session_state, "device")


def _periods() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period_start": pd.to_datetime(
                ["2026-01-05 08:00", "2026-01-05 09:00"], utc=True
            ),
            "energy_kwh": [0.4, 0.7],
        }
    )


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "start_time": ["2026-01-04T18:00:00Z", "2026-01-07T04:00:00Z"],
            "end_time": ["2026-01-05T02:50:00Z", "2026-01-07T19:50:00Z"],
            "total_rainfall": [0.36, 0.48],
        }
    )


# ---------- key paths ----------


def test_catalog_key_paths():
    assert catalog_utils.get_catalog_path(FAKE_MAC, "energy") == (
        "00:11:22:33:44:55.energy_catalog.parquet"
    )
    assert RainEventCatalog(FAKE_MAC).catalog_path == (
        "00:11:22:33:44:55.event_catalog.parquet"
    )
    assert EnergyCatalog(FAKE_MAC).catalog_path == (
        "00:11:22:33:44:55.energy_catalog.parquet"
    )
    assert day_periods.get_catalog_path(FAKE_MAC) == (
        "00:11:22:33:44:55.day_periods_catalog.parquet"
    )


def test_rain_corrections_catalog_lives_at_mac_dot_rain_corrections_json(fake_s3):
    assert rain_corrections.catalog_exists(FAKE_MAC, BUCKET) is False
    catalog = {"version": 1, "corrections": []}

    assert rain_corrections.save_catalog(catalog, FAKE_MAC, BUCKET) is True

    assert bucket_keys(fake_s3) == ["00:11:22:33:44:55.rain_corrections.json"]
    assert rain_corrections.catalog_exists(FAKE_MAC, BUCKET) is True
    assert rain_corrections.load_catalog(FAKE_MAC, BUCKET) == catalog


# ---------- catalog_utils: the station comes from session state ----------


def test_generic_catalog_is_keyed_by_the_session_station(fake_s3, station):
    assert catalog_utils.save_catalog(_periods(), "energy", BUCKET) is True
    assert bucket_keys(fake_s3) == [f"{FAKE_MAC}.energy_catalog.parquet"]


def test_generic_catalog_falls_back_to_the_default_station(fake_s3, no_station):
    assert catalog_utils.save_catalog(_periods(), "energy", BUCKET) is True
    assert bucket_keys(fake_s3) == [f"{DEFAULT_MAC_ADDRESS}.energy_catalog.parquet"]


def test_generic_catalog_exists_load_and_save(fake_s3, station):
    assert catalog_utils.catalog_exists("energy", BUCKET) is False
    assert catalog_utils.load_catalog("energy", BUCKET).empty

    assert catalog_utils.save_catalog(_periods(), "energy", BUCKET) is True

    assert catalog_utils.catalog_exists("energy", BUCKET) is True
    loaded = catalog_utils.load_catalog("energy", BUCKET)
    # save stamps two metadata columns onto the stored copy.
    assert list(loaded.columns) == [
        "period_start",
        "energy_kwh",
        "updated_at",
        "catalog_version",
    ]
    assert (loaded["catalog_version"] == "1.0").all()
    assert str(loaded["updated_at"].dtype).startswith("datetime64")
    assert_frame_equal(loaded[["period_start", "energy_kwh"]], _periods())


def test_generic_catalog_refuses_an_empty_save(fake_s3, station):
    assert catalog_utils.save_catalog(pd.DataFrame(), "energy", BUCKET) is False
    assert bucket_keys(fake_s3) == []


def test_generic_catalog_backup_copies_to_a_dated_folder(
    fake_s3, station, frozen_today
):
    catalog_utils.save_catalog(_periods(), "energy", BUCKET)

    result = catalog_utils.backup_catalog(_periods(), "energy", BUCKET)

    prefix = f"{FAKE_MAC}.energy_catalog"
    assert result == f"backed_up_{prefix}"
    assert f".backups/{prefix}/{frozen_today}/{prefix}.parquet" in bucket_keys(fake_s3)


def test_catalog_save_with_backup_writes_both(fake_s3, station, frozen_today):
    prefix = f"{FAKE_MAC}.energy_catalog"
    storj.save_df_to_s3(_periods().iloc[:1], BUCKET, f"{prefix}.parquet")

    storj.backup_and_save_catalog(_periods(), "energy", bucket=BUCKET)

    assert bucket_keys(fake_s3) == [
        f".backups/{prefix}/{frozen_today}/{prefix}.parquet",
        f"{prefix}.parquet",
    ]
    stored = storj.get_df_from_s3(BUCKET, f"{prefix}.parquet", "parquet")
    assert_frame_equal(stored, _periods())


# ---------- the Event Catalog ----------


def test_event_catalog_exists_load_and_save(fake_s3):
    catalog = RainEventCatalog(FAKE_MAC, bucket=BUCKET)
    assert catalog.catalog_exists() is False
    assert catalog.load_catalog().empty

    assert catalog.save_catalog(_events()) is True

    assert catalog.catalog_exists() is True
    assert bucket_keys(fake_s3) == [f"{FAKE_MAC}.event_catalog.parquet"]
    loaded = catalog.load_catalog()
    # Event times are stored as UTC timestamps, whatever they arrived as.
    assert str(loaded["start_time"].dtype) == "datetime64[ns, UTC]"
    assert loaded["start_time"].iloc[1] == pd.Timestamp("2026-01-07T04:00:00Z")
    assert loaded["event_id"].tolist() == ["e1", "e2"]
    assert loaded["total_rainfall"].tolist() == [0.36, 0.48]


def test_event_catalog_refuses_an_empty_save(fake_s3):
    assert (
        RainEventCatalog(FAKE_MAC, bucket=BUCKET).save_catalog(pd.DataFrame()) is False
    )
    assert bucket_keys(fake_s3) == []


def test_event_catalog_backup_is_a_timestamped_copy(fake_s3):
    catalog = RainEventCatalog(FAKE_MAC, bucket=BUCKET)
    assert catalog.backup_catalog() is None  # nothing to back up yet
    catalog.save_catalog(_events())

    path = catalog.backup_catalog()

    pattern = rf"backups/{re.escape(FAKE_MAC)}\.event_catalog_\d{{8}}_\d{{6}}\.parquet"
    assert re.fullmatch(pattern, path)
    assert path in bucket_keys(fake_s3)
    original = fake_s3.get_object(Bucket=BUCKET, Key=catalog.catalog_path)
    copy = fake_s3.get_object(Bucket=BUCKET, Key=path)
    assert copy["Body"].read() == original["Body"].read()

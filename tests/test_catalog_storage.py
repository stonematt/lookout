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

from lookout.core import catalog_utils, day_periods, rain_corrections
from lookout.core.energy_catalog import EnergyCatalog
from lookout.core.rain_events import RainEventCatalog
from lookout.storage import storj
from tests.fixtures.archive import FAKE_MAC
from tests.fixtures.storage import BUCKET, bucket_keys


@pytest.fixture
def station(monkeypatch):
    """Pin the fake station; nothing is put in session state."""
    monkeypatch.setenv("STATION_MAC", FAKE_MAC)
    return FAKE_MAC


@pytest.fixture
def no_station(monkeypatch):
    """Leave STATION_MAC unset."""
    monkeypatch.delenv("STATION_MAC", raising=False)


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
    assert rain_corrections.read_catalog(FAKE_MAC, BUCKET) is None
    catalog = {"version": 1, "corrections": []}

    assert rain_corrections.save_catalog(catalog, FAKE_MAC, BUCKET) is True

    assert bucket_keys(fake_s3) == ["00:11:22:33:44:55.rain_corrections.json"]
    assert rain_corrections.read_catalog(FAKE_MAC, BUCKET) == catalog


def _day_periods() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05", "2026-01-06"], utc=True),
            "sunrise_utc": pd.to_datetime(
                ["2026-01-05 15:50", "2026-01-06 15:50"], utc=True
            ),
        }
    )


def test_day_periods_catalog_exists_only_once_saved(fake_s3):
    assert day_periods.catalog_exists(FAKE_MAC, BUCKET) is False

    assert day_periods.save_day_periods_catalog(_day_periods(), FAKE_MAC, BUCKET)

    assert day_periods.catalog_exists(FAKE_MAC, BUCKET) is True


def test_missing_day_periods_catalog_loads_empty_by_the_no_catalog_path(
    fake_s3, caplog
):
    caplog.set_level("INFO")

    assert day_periods.load_day_periods_catalog(FAKE_MAC, BUCKET).empty

    assert f"No day periods catalog found for {FAKE_MAC}" in caplog.text
    assert "Failed to load DataFrame" not in caplog.text  # no read was tried


def test_day_periods_catalog_listing_follows_existence(fake_s3):
    assert day_periods.list_day_periods_catalogs(FAKE_MAC, BUCKET) == []

    day_periods.save_day_periods_catalog(_day_periods(), FAKE_MAC, BUCKET)

    assert day_periods.list_day_periods_catalogs(FAKE_MAC, BUCKET) == [
        "00:11:22:33:44:55.day_periods_catalog.parquet"
    ]


# ---------- catalog_utils: the station is the pinned one ----------


def test_generic_catalog_is_keyed_by_the_pinned_station(fake_s3, station, monkeypatch):
    # Whatever device the UI put in session state, the pin decides the key.
    monkeypatch.setitem(st.session_state, "device", {"macAddress": "AA:BB"})
    assert catalog_utils.save_catalog(_periods(), "energy", BUCKET) is True
    assert bucket_keys(fake_s3) == [f"{FAKE_MAC}.energy_catalog.parquet"]


def test_generic_catalog_unpinned_uses_the_default_station(fake_s3, no_station):
    assert catalog_utils.save_catalog(_periods(), "energy", BUCKET) is True
    assert bucket_keys(fake_s3) == ["98:CD:AC:22:0D:E5.energy_catalog.parquet"]


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


def test_catalog_save_with_backup_takes_the_station_as_a_parameter(
    fake_s3, no_station, frozen_today
):
    other = "AA:BB:CC:DD:EE:FF"

    storj.backup_and_save_catalog(
        _periods(), "energy", bucket=BUCKET, mac_address=other
    )

    assert bucket_keys(fake_s3) == [f"{other}.energy_catalog.parquet"]


# ---------- the Event Catalog ----------


def test_event_catalog_load_is_none_until_saved(fake_s3):
    catalog = RainEventCatalog(FAKE_MAC, bucket=BUCKET)
    assert catalog.load_catalog() is None

    assert catalog.save_catalog(_events()) is True

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

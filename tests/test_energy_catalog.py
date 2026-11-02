"""
Tests for EnergyCatalog class
"""

from unittest.mock import patch

import pandas as pd
import pytest

from lookout.core import energy_catalog_func
from lookout.core.energy_catalog import EnergyCatalog
from lookout.storage import storj
from tests.fixtures.storage import BUCKET, bucket_keys
from tests.fixtures.storage import flaky_storj


class TestEnergyCatalog:
    """Test EnergyCatalog functionality"""

    @pytest.fixture
    def sample_data(self):
        """Create sample solar data for testing"""
        dates = pd.date_range(
            "2023-01-01", periods=100, freq="5min", tz="America/Los_Angeles"
        )
        data = {
            "date": dates,  # TZ-aware datetime column (matches real data)
            "dateutc": [
                int(dt.timestamp() * 1000) for dt in dates
            ],  # Convert to milliseconds
            "solarradiation": [
                i % 1000 for i in range(100)
            ],  # Mock solar radiation values
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def catalog(self):
        """Create EnergyCatalog instance for testing"""
        return EnergyCatalog("TEST_MAC", "parquet")

    def test_catalog_creation(self, catalog):
        """Test EnergyCatalog initialization"""
        assert catalog.mac_address == "TEST_MAC"
        assert catalog.file_type == "parquet"
        assert catalog.bucket == "lookout"
        assert catalog.catalog_path == "TEST_MAC.energy_catalog.parquet"

    def test_a_saved_catalog_loads_back_in_pacific_time(
        self, fake_s3, catalog, sample_data
    ):
        periods = catalog.detect_and_calculate_periods(sample_data, auto_save=False)
        assert catalog.save_catalog(periods) is True

        loaded = catalog.load_catalog()

        assert len(loaded) == len(periods)
        assert str(loaded["period_start"].dt.tz) == "America/Los_Angeles"
        assert loaded["energy_kwh"].sum() == pytest.approx(periods["energy_kwh"].sum())

    def test_a_missing_catalog_loads_as_none(self, fake_s3, catalog):
        assert catalog.load_catalog() is None

    def test_a_failed_read_raises(self, fake_s3, catalog, sample_data, monkeypatch):
        catalog.save_catalog(
            catalog.detect_and_calculate_periods(sample_data, auto_save=False)
        )
        flaky_storj(monkeypatch, fake_s3, read=catalog.catalog_path)

        with pytest.raises(storj.StorageReadError):
            catalog.load_catalog()

    def test_a_corrupt_catalog_raises(self, fake_s3, catalog):
        fake_s3.put_object(Bucket=BUCKET, Key=catalog.catalog_path, Body=b"x")

        with pytest.raises(storj.StorageReadError):
            catalog.load_catalog()

    def test_saving_an_empty_catalog_writes_nothing(self, fake_s3, catalog):
        assert catalog.save_catalog(pd.DataFrame()) is False
        assert bucket_keys(fake_s3) == []

    def test_detect_and_calculate_periods(self, catalog, sample_data):
        """Test period detection and calculation"""
        with patch.object(catalog, "save_catalog") as mock_save:
            result = catalog.detect_and_calculate_periods(sample_data, auto_save=False)

            assert not result.empty
            assert "period_start" in result.columns
            assert "period_end" in result.columns
            assert "energy_kwh" in result.columns
            mock_save.assert_not_called()

    def test_auto_save_backs_up_the_stored_catalog_as_it_is_then_saves(
        self, fake_s3, catalog, sample_data
    ):
        fake_s3.put_object(Bucket=BUCKET, Key=catalog.catalog_path, Body=b"corrupt")

        result = catalog.detect_and_calculate_periods(sample_data, auto_save=True)

        backups = [k for k in bucket_keys(fake_s3) if k.startswith("backups/")]
        assert len(backups) == 1
        backup = fake_s3.get_object(Bucket=BUCKET, Key=backups[0])["Body"].read()
        assert backup == b"corrupt"
        assert len(catalog.load_catalog()) == len(result)

    def test_auto_save_with_none_stored_saves_without_a_backup(
        self, fake_s3, catalog, sample_data
    ):
        result = catalog.detect_and_calculate_periods(sample_data, auto_save=True)

        assert bucket_keys(fake_s3) == [catalog.catalog_path]
        assert len(catalog.load_catalog()) == len(result)

    def test_a_failed_backup_stops_the_save(
        self, fake_s3, catalog, sample_data, monkeypatch
    ):
        fake_s3.put_object(Bucket=BUCKET, Key=catalog.catalog_path, Body=b"stored")
        flaky_storj(monkeypatch, fake_s3, copy=True)

        with pytest.raises(storj.BackupError):
            catalog.detect_and_calculate_periods(sample_data, auto_save=True)

        assert bucket_keys(fake_s3) == [catalog.catalog_path]
        body = fake_s3.get_object(Bucket=BUCKET, Key=catalog.catalog_path)["Body"]
        assert body.read() == b"stored"

    def test_update_catalog_with_new_data(self, catalog, sample_data):
        """Test updating catalog with new data"""
        # Create existing catalog (first half of data)
        existing_data = sample_data.iloc[:50]
        existing_catalog = catalog.detect_and_calculate_periods(
            existing_data, auto_save=False
        )

        # Update with full dataset
        result = catalog.update_catalog_with_new_data(sample_data, existing_catalog)

        assert not result.empty
        assert len(result) >= len(existing_catalog)

    def test_update_catalog_with_empty_existing(self, catalog, sample_data):
        """Test updating when no existing catalog"""
        result = catalog.update_catalog_with_new_data(sample_data, pd.DataFrame())

        assert not result.empty
        assert "period_start" in result.columns

    def test_a_stored_catalog_is_young(self, fake_s3, catalog, sample_data):
        catalog.save_catalog(
            catalog.detect_and_calculate_periods(sample_data, auto_save=False)
        )

        assert catalog.get_catalog_age() < pd.Timedelta(minutes=5)

    def test_with_none_stored_the_age_is_very_large(self, fake_s3, catalog):
        assert catalog.get_catalog_age() == pd.Timedelta(days=999)


class TestUpdateRunAfterRun:
    """
    The refresh run updates the catalog every five minutes, so its newest
    period is usually still filling. Each run must leave the catalog equal to
    one computed from the whole archive at once.
    """

    @staticmethod
    def station(first: str, rows: int) -> pd.DataFrame:
        """Five-minute rows at a steady 600 W/m², newest first."""
        dates = pd.date_range(
            first, periods=rows, freq="5min", tz="America/Los_Angeles"
        )
        return pd.DataFrame(
            {
                "date": dates,
                "dateutc": [int(d.timestamp() * 1000) for d in dates],
                "solarradiation": [600.0] * rows,
            }
        ).iloc[::-1]

    def energy(self, periods: pd.DataFrame) -> dict:
        """Each period's start (as a local "HH:MM") -> its kWh, rounded."""
        return {
            p.strftime("%H:%M"): round(e, 6)
            for p, e in zip(periods["period_start"], periods["energy_kwh"])
        }

    @staticmethod
    def class_update(archive: pd.DataFrame, stored: pd.DataFrame) -> pd.DataFrame:
        """The refresh run's update: `EnergyCatalog.update_catalog_with_new_data`."""
        return EnergyCatalog("TEST_MAC").update_catalog_with_new_data(archive, stored)

    @staticmethod
    def func_update(archive: pd.DataFrame, stored: pd.DataFrame) -> pd.DataFrame:
        """The Streamlit app's update: `energy_catalog_func.update_energy_catalog`."""
        return energy_catalog_func.update_energy_catalog(archive, stored)

    @pytest.mark.parametrize("update", ["class_update", "func_update"])
    @pytest.mark.parametrize(
        "first, expected",
        [
            # 600 W/m² over a whole period is 0.15 kWh/m². The newest period
            # has no row after 10:55 yet, so it holds its 10 minutes: 0.1.
            (
                "2026-06-01 10:00",  # 10:00 .. 10:55
                {"10:00": 0.15, "10:15": 0.15, "10:30": 0.15, "10:45": 0.1},
            ),
            # Rows at :02, :07, :12: each period's first three minutes come
            # from the interval that starts in the period before it. 10:00
            # starts at its 10:02 row (0.13), 10:45 ends at 10:57 (0.12).
            (
                "2026-06-01 10:02",  # 10:02 .. 10:57
                {"10:00": 0.13, "10:15": 0.15, "10:30": 0.15, "10:45": 0.12},
            ),
        ],
        ids=["on-the-period-grid", "off-the-period-grid"],
    )
    def test_a_period_still_filling_is_completed_by_later_runs(
        self, update: str, first: str, expected: dict
    ):
        catalog = EnergyCatalog("TEST_MAC")
        archive = self.station(first, 12)

        stored = catalog.detect_and_calculate_periods(
            archive.iloc[-2:], auto_save=False
        )
        for newest in range(3, 13):  # one five-minute run at a time
            stored = getattr(self, update)(archive.iloc[-newest:], stored.copy())

        assert self.energy(stored) == expected
        whole = catalog.detect_and_calculate_periods(archive, auto_save=False)
        assert self.energy(stored) == self.energy(whole)

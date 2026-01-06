"""
Tests for EnergyCatalog class
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime, timezone

from lookout.core.energy_catalog import EnergyCatalog


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

    @patch("lookout.core.energy_catalog.get_s3_client")
    def test_catalog_exists_true(self, mock_get_client, catalog):
        """Test catalog_exists when catalog exists"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.head_object.return_value = {
            "LastModified": datetime.now(timezone.utc)
        }

        assert catalog.catalog_exists() is True
        mock_client.head_object.assert_called_once_with(
            Bucket="lookout", Key="TEST_MAC.energy_catalog.parquet"
        )

    @patch("lookout.core.energy_catalog.get_s3_client")
    def test_catalog_exists_false(self, mock_get_client, catalog):
        """Test catalog_exists when catalog doesn't exist"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.head_object.side_effect = Exception("Not found")

        assert catalog.catalog_exists() is False

    @patch("lookout.core.energy_catalog.get_s3_client")
    def test_load_catalog_success(self, mock_get_client, catalog, sample_data):
        """Test successful catalog loading"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock the parquet data
        mock_body = MagicMock()
        mock_body.read.return_value = sample_data.to_parquet()
        mock_client.get_object.return_value = {"Body": mock_body}

        with patch("pandas.read_parquet", return_value=sample_data) as mock_read:
            result = catalog.load_catalog()
            assert not result.empty
            mock_read.assert_called_once()

    @patch("lookout.core.energy_catalog.get_s3_client")
    def test_load_catalog_not_exists(self, mock_get_client, catalog):
        """Test loading catalog when it doesn't exist"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.head_object.side_effect = Exception("Not found")

        result = catalog.load_catalog()
        assert result.empty

    @patch("lookout.core.energy_catalog.get_s3_client")
    def test_save_catalog_success(self, mock_get_client, catalog, sample_data):
        """Test successful catalog saving"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        with patch("pandas.DataFrame.to_parquet") as mock_to_parquet:
            with patch("io.BytesIO") as mock_bytesio:
                mock_buffer = MagicMock()
                mock_bytesio.return_value = mock_buffer
                mock_buffer.getvalue.return_value = b"parquet_data"

                result = catalog.save_catalog(sample_data)
                assert result is True
                mock_client.put_object.assert_called_once()

    @patch("lookout.core.energy_catalog.get_s3_client")
    def test_save_catalog_empty(self, mock_get_client, catalog):
        """Test saving empty catalog"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        result = catalog.save_catalog(pd.DataFrame())
        assert result is False
        mock_client.put_object.assert_not_called()

    def test_detect_and_calculate_periods(self, catalog, sample_data):
        """Test period detection and calculation"""
        with patch.object(catalog, "save_catalog") as mock_save:
            result = catalog.detect_and_calculate_periods(sample_data, auto_save=False)

            assert not result.empty
            assert "period_start" in result.columns
            assert "period_end" in result.columns
            assert "energy_kwh" in result.columns
            mock_save.assert_not_called()

    def test_detect_and_calculate_periods_auto_save(self, catalog, sample_data):
        """Test period detection with auto-save"""
        with patch.object(catalog, "save_catalog", return_value=True) as mock_save:
            result = catalog.detect_and_calculate_periods(sample_data, auto_save=True)

            assert not result.empty
            mock_save.assert_called_once_with(result)

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

    def test_get_catalog_age_exists(self, catalog):
        """Test getting catalog age when it exists"""
        with patch.object(catalog, "catalog_exists", return_value=True):
            with patch("lookout.core.energy_catalog.get_s3_client") as mock_get_client:
                mock_client_instance = MagicMock()
                mock_get_client.return_value = mock_client_instance
                mock_client_instance.head_object.return_value = {
                    "LastModified": datetime.now(timezone.utc)
                }

                age = catalog.get_catalog_age()
                assert isinstance(age, pd.Timedelta)

    def test_get_catalog_age_not_exists(self, catalog):
        """Test getting catalog age when it doesn't exist"""
        with patch.object(catalog, "catalog_exists", return_value=False):
            age = catalog.get_catalog_age()
            assert age == pd.Timedelta(days=999)

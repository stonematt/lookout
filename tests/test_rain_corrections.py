"""
Unit tests for rain corrections CLI.

Tests rain carryover detection and catalog management CLI.
"""

import unittest.mock as mock
import pytest
import pandas as pd

from lookout.cli.rain_corrections_cli import (
    get_primary_device,
    load_archive_s3,
    handle_detect,
    handle_generate,
    handle_show,
)


class TestDeviceDiscovery:
    """Test device auto-discovery functionality."""

    @mock.patch('lookout.cli.rain_corrections_cli.ambient_client.get_devices')
    def test_single_device_discovery(self, mock_get_devices):
        """Test discovery when one device available."""
        mock_devices = [{
            'macAddress': '98:CD:AC:22:0D:E5',
            'info': {'name': 'Test Station'}
        }]
        mock_get_devices.return_value = mock_devices

        device, mac, name = get_primary_device()
        assert mac == '98:CD:AC:22:0D:E5'
        assert name == 'Test Station'
        assert device == mock_devices[0]

    @mock.patch('lookout.cli.rain_corrections_cli.ambient_client.get_devices')
    def test_multiple_devices_uses_first(self, mock_get_devices):
        """Test that first device is used when multiple devices exist."""
        mock_devices = [
            {'macAddress': '98:CD:AC:22:0D:E5', 'info': {'name': 'First Station'}},
            {'macAddress': '12:34:56:78:9A:BC', 'info': {'name': 'Second Station'}}
        ]
        mock_get_devices.return_value = mock_devices

        device, mac, name = get_primary_device()
        assert mac == '98:CD:AC:22:0D:E5'
        assert name == 'First Station'
        assert device == mock_devices[0]

    @mock.patch('lookout.cli.rain_corrections_cli.ambient_client.get_devices')
    def test_no_devices_error(self, mock_get_devices):
        """Test error handling when no devices found."""
        mock_get_devices.return_value = []

        device, mac, name = get_primary_device()
        assert device is None
        assert mac is None
        assert name is None





class TestArchiveLoading:
    """Test S3 archive loading."""

    @mock.patch('lookout.storage.storj.get_df_from_s3')
    def test_successful_load(self, mock_get_df_from_s3):
        """Test successful archive loading from S3."""
        mock_df = pd.DataFrame({'test': [1, 2, 3]})
        mock_get_df_from_s3.return_value = mock_df

        from lookout.cli.rain_corrections_cli import load_archive_s3
        result = load_archive_s3('98:CD:AC:22:0D:E5', 'lookout')
        assert not result.empty
        mock_get_df_from_s3.assert_called_once_with('lookout', '98:CD:AC:22:0D:E5.parquet', 'parquet')

    @mock.patch('lookout.storage.storj.get_df_from_s3')
    def test_load_error(self, mock_get_df_from_s3):
        """Test handling of S3 loading errors."""
        mock_get_df_from_s3.side_effect = Exception("S3 failed")

        from lookout.cli.rain_corrections_cli import load_archive_s3
        result = load_archive_s3('98:CD:AC:22:0D:E5', 'lookout')
        assert result.empty


class TestCliCommands:
    """Test CLI command handlers."""

    @mock.patch('builtins.print')
    @mock.patch('lookout.cli.rain_corrections_cli.get_primary_device')
    def test_detect_no_device(self, mock_get_primary_device, mock_print):
        """Test detect command when no device found."""
        mock_get_primary_device.return_value = (None, None, None)

        args = mock.Mock()
        handle_detect(args)

        mock_print.assert_called_with("❌ No devices found in Ambient account.")

    @mock.patch('builtins.print')
    @mock.patch('lookout.cli.rain_corrections_cli.load_archive_s3')
    @mock.patch('lookout.cli.rain_corrections_cli.get_primary_device')
    def test_detect_empty_archive(self, mock_get_primary_device, mock_load_archive_s3, mock_print):
        """Test detect command with empty archive."""
        mock_get_primary_device.return_value = ({'macAddress': '98:CD:AC:22:0D:E5'}, '98:CD:AC:22:0D:E5', 'Test Station')
        mock_load_archive_s3.return_value = pd.DataFrame()

        args = mock.Mock()
        handle_detect(args)

        calls = mock_print.call_args_list
        assert "❌ Archive is empty or not found." in str(calls)

    @mock.patch('builtins.print')
    @mock.patch('lookout.core.rain_corrections.detect_carryovers')
    @mock.patch('lookout.cli.rain_corrections_cli.load_archive_s3')
    @mock.patch('lookout.cli.rain_corrections_cli.get_primary_device')
    def test_detect_no_carryovers(self, mock_get_primary_device, mock_load_archive_s3,
                                  mock_detect_carryovers, mock_print):
        """Test detect command when no carryovers found."""
        mock_get_primary_device.return_value = ({'macAddress': '98:CD:AC:22:0D:E5'}, '98:CD:AC:22:0D:E5', 'Test Station')
        mock_load_archive_s3.return_value = pd.DataFrame({'test': [1]})
        mock_detect_carryovers.return_value = []

        args = mock.Mock()
        handle_detect(args)

        calls = mock_print.call_args_list
        assert "✅ No carryovers detected." in str(calls)

    @mock.patch('builtins.print')
    @mock.patch('lookout.core.rain_corrections.detect_carryovers')
    @mock.patch('lookout.cli.rain_corrections_cli.load_archive_s3')
    @mock.patch('lookout.cli.rain_corrections_cli.get_primary_device')
    def test_detect_with_carryovers(self, mock_get_primary_device, mock_load_archive_s3,
                                    mock_detect_carryovers, mock_print):
        """Test detect command with carryovers found."""
        mock_get_primary_device.return_value = ({'macAddress': '98:CD:AC:22:0D:E5'}, '98:CD:AC:22:0D:E5', 'Test Station')
        mock_load_archive_s3.return_value = pd.DataFrame({'test': [1]})

        # Mock the known carryover data
        mock_carryovers = [
            {
                'affected_date': '2024-12-16',
                'carryover_amount': 0.063,
                'gap_start': '2024-12-15 18:25:00',
                'gap_end': '2024-12-16 00:17:00'
            }
        ]
        mock_detect_carryovers.return_value = mock_carryovers

        args = mock.Mock()
        handle_detect(args)

        calls = mock_print.call_args_list
        assert "🚨 Found 1 carryover(s):" in str(calls)
        assert "2024-12-16: 0.063\"" in str(calls)

    @mock.patch('builtins.print')
    @mock.patch('lookout.core.rain_corrections.save_catalog')
    @mock.patch('lookout.core.rain_corrections.detect_carryovers')
    @mock.patch('lookout.cli.rain_corrections_cli.load_archive_s3')
    @mock.patch('lookout.cli.rain_corrections_cli.get_primary_device')
    def test_generate_dry_run(self, mock_get_primary_device, mock_load_archive_s3,
                              mock_detect_carryovers, mock_save_catalog, mock_print):
        """Test generate command with dry-run flag."""
        mock_get_primary_device.return_value = ({'macAddress': '98:CD:AC:22:0D:E5'}, '98:CD:AC:22:0D:E5', 'Test Station')
        mock_load_archive_s3.return_value = pd.DataFrame({'test': [1]})
        mock_detect_carryovers.return_value = [
            {
                'affected_date': '2024-12-16',
                'carryover_amount': 0.063,
                'gap_start': '2024-12-15 18:25:00',
                'gap_end': '2024-12-16 00:17:00'
            }
        ]
        mock_save_catalog.return_value = True

        args = mock.Mock()
        args.dry_run = True
        args.bucket = 'lookout'

        handle_generate(args)

        calls = mock_print.call_args_list
        assert "🔒 --dry-run: Catalog not saved." in str(calls)
        mock_save_catalog.assert_not_called()

    @mock.patch('builtins.print')
    @mock.patch('lookout.core.rain_corrections.save_catalog')
    @mock.patch('lookout.core.rain_corrections.detect_carryovers')
    @mock.patch('lookout.cli.rain_corrections_cli.load_archive_s3')
    @mock.patch('lookout.cli.rain_corrections_cli.get_primary_device')
    def test_generate_save_success(self, mock_get_primary_device, mock_load_archive_s3,
                                   mock_detect_carryovers, mock_save_catalog, mock_print):
        """Test generate command saving to S3 successfully."""
        mock_get_primary_device.return_value = ({'macAddress': '98:CD:AC:22:0D:E5'}, '98:CD:AC:22:0D:E5', 'Test Station')
        mock_load_archive_s3.return_value = pd.DataFrame({'test': [1]})
        mock_detect_carryovers.return_value = [
            {
                'affected_date': '2024-12-16',
                'carryover_amount': 0.063,
                'gap_start': '2024-12-15 18:25:00',
                'gap_end': '2024-12-16 00:17:00'
            }
        ]
        mock_save_catalog.return_value = True

        args = mock.Mock()
        args.dry_run = False
        args.bucket = 'lookout'

        handle_generate(args)

        calls = mock_print.call_args_list
        assert "💾 Saved catalog with 1 corrections to S3." in str(calls)
        mock_save_catalog.assert_called_once()

    @mock.patch('builtins.print')
    @mock.patch('lookout.cli.rain_corrections_cli.get_primary_device')
    def test_show_no_device(self, mock_get_primary_device, mock_print):
        """Test show command when no device found."""
        mock_get_primary_device.return_value = (None, None, None)

        args = mock.Mock()
        handle_show(args)

        mock_print.assert_called_with("❌ No devices found in Ambient account.")

    @mock.patch('builtins.print')
    @mock.patch('lookout.core.rain_corrections.load_catalog')
    @mock.patch('lookout.core.rain_corrections.catalog_exists')
    @mock.patch('lookout.cli.rain_corrections_cli.get_primary_device')
    def test_show_no_catalog(self, mock_get_primary_device, mock_catalog_exists,
                             mock_load_catalog, mock_print):
        """Test show command when no catalog exists."""
        mock_get_primary_device.return_value = ({'macAddress': '98:CD:AC:22:0D:E5'}, '98:CD:AC:22:0D:E5', 'Test Station')
        mock_catalog_exists.return_value = False

        args = mock.Mock()
        args.bucket = 'lookout'

        handle_show(args)

        calls = mock_print.call_args_list
        assert "📭 No corrections catalog found." in str(calls)
        mock_load_catalog.assert_not_called()

    @mock.patch('builtins.print')
    @mock.patch('lookout.core.rain_corrections.load_catalog')
    @mock.patch('lookout.core.rain_corrections.catalog_exists')
    @mock.patch('lookout.cli.rain_corrections_cli.get_primary_device')
    def test_show_with_catalog(self, mock_get_primary_device, mock_catalog_exists,
                               mock_load_catalog, mock_print):
        """Test show command displaying catalog."""
        mock_get_primary_device.return_value = ({'macAddress': '98:CD:AC:22:0D:E5'}, '98:CD:AC:22:0D:E5', 'Test Station')
        mock_catalog_exists.return_value = True

        mock_catalog_data = {
            'version': '1.0',
            'updated_at': '2025-12-20T20:00:00Z',
            'corrections': [
                {
                    'affected_date': '2024-12-16',
                    'carryover_amount': 0.063,
                    'gap_start': '2024-12-15 18:25:00',
                    'gap_end': '2024-12-16 00:17:00'
                }
            ]
        }
        mock_load_catalog.return_value = mock_catalog_data

        args = mock.Mock()
        args.bucket = 'lookout'

        handle_show(args)

        calls = mock_print.call_args_list
        assert "Version: 1.0" in str(calls)
        assert "Corrections: 1" in str(calls)
        assert "2024-12-16: 0.063\"" in str(calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
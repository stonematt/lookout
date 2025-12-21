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
from lookout.core.rain_corrections import apply_corrections
from lookout.api.awn_controller import combine_full_history, get_history_since_last_archive


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


class TestApplyCorrections:
    """Test rain carryover corrections application."""

    @mock.patch('lookout.core.rain_corrections.load_catalog')
    def test_no_catalog_returns_original_df(self, mock_load_catalog):
        """Test returns original DataFrame when no catalog exists."""
        mock_load_catalog.return_value = {}

        original_df = pd.DataFrame({'dailyrainin': [1.0, 2.0]})
        result = apply_corrections(original_df, '98:CD:AC:22:0D:E5')

        assert result.equals(original_df)

    @mock.patch('lookout.core.rain_corrections.load_catalog')
    def test_single_correction_applied_correctly(self, mock_load_catalog):
        """Test correct subtraction amounts and period boundaries."""
        mock_catalog = {
            'corrections': [{
                'affected_date': '2024-12-16',
                'carryover_amount': 0.063
            }]
        }
        mock_load_catalog.return_value = mock_catalog

        df = pd.DataFrame({
            'date': ['2024-12-16 00:00:00', '2024-12-17 00:00:00'],
            'dailyrainin': [0.1, 0.05],
            'weeklyrainin': [1.0, 0.5],
            'monthlyrainin': [10.0, 5.0],
            'yearlyrainin': [50.0, 25.0]
        })

        result = apply_corrections(df, '98:CD:AC:22:0D:E5')

        # Verify daily correction on affected date only
        assert result.loc[0, 'dailyrainin'] == pytest.approx(0.037, abs=1e-10)  # 0.1 - 0.063
        assert result.loc[1, 'dailyrainin'] == 0.05  # unchanged

    @mock.patch('lookout.core.rain_corrections.load_catalog')
    def test_negative_values_clipped_to_zero(self, mock_load_catalog):
        """Test clipping prevents negative rain values."""
        mock_catalog = {
            'corrections': [{
                'affected_date': '2024-12-16',
                'carryover_amount': 1.0  # Large correction
            }]
        }
        mock_load_catalog.return_value = mock_catalog

        df = pd.DataFrame({
            'date': ['2024-12-16 00:00:00'],
            'dailyrainin': [0.5],
            'weeklyrainin': [0.5],
            'monthlyrainin': [0.5],
            'yearlyrainin': [0.5]
        })

        result = apply_corrections(df, '98:CD:AC:22:0D:E5')

        # All values should be clipped to 0, not negative
        assert result.loc[0, 'dailyrainin'] == 0.0
        assert result.loc[0, 'weeklyrainin'] == 0.0
        assert result.loc[0, 'monthlyrainin'] == 0.0
        assert result.loc[0, 'yearlyrainin'] == 0.0


class TestCombineFullHistoryIntegration:
    """Test integration of corrections in combine_full_history."""

    def test_combine_full_history_is_pure_merge(self):
        """Test that combine_full_history is now a pure merge function."""
        # Create DataFrames with required 'dateutc' column for combine_full_history
        archive_df = pd.DataFrame({'dateutc': [1640995200000], 'test': [1]})  # 2022-01-01 00:00:00 UTC
        interim_df = pd.DataFrame({'dateutc': [1641081600000], 'test': [2]})  # 2022-01-02 00:00:00 UTC

        result = combine_full_history(archive_df, interim_df)

        # Should return combined data without applying any corrections
        assert len(result) == 2
        # Check that both values are present (order may vary due to sorting)
        assert 1 in result['test'].values
        assert 2 in result['test'].values

    def test_combine_full_history_handles_empty_interim(self):
        """Test combine_full_history with empty interim DataFrame."""
        archive_df = pd.DataFrame({'dateutc': [1640995200000], 'test': [1]})
        interim_df = pd.DataFrame()

        result = combine_full_history(archive_df, interim_df)

        # Should return archive data unchanged
        assert len(result) == 1
        assert result.iloc[0]['test'] == 1


class TestCorrectionsIntegration:
    """Test corrections integration in get_history_since_last_archive."""

    @mock.patch('lookout.api.awn_controller.apply_corrections')
    @mock.patch('lookout.api.awn_controller.combine_full_history')
    @mock.patch('lookout.api.awn_controller.validate_archive')
    def test_corrections_applied_in_get_history_since_last_archive(self, mock_validate_archive, mock_combine_full_history, mock_apply_corrections):
        """Test that corrections are applied at the end of get_history_since_last_archive."""
        # Mock validation to pass
        mock_validate_archive.return_value = True

        # Mock combine_full_history to return combined data
        combined_df = pd.DataFrame({'dateutc': [1640995200000], 'test': [1]})
        mock_combine_full_history.return_value = combined_df

        # Mock apply_corrections to return corrected data
        corrected_df = pd.DataFrame({'dateutc': [1640995200000], 'test': [2]})
        mock_apply_corrections.return_value = corrected_df

        device = {'macAddress': '98:CD:AC:22:0D:E5'}
        archive_df = pd.DataFrame({'dateutc': [1640995200000], 'test': [1]})

        result = get_history_since_last_archive(device, archive_df, limit=1, pages=1)

        # Verify corrections were called with correct parameters
        mock_apply_corrections.assert_called_once_with(combined_df, '98:CD:AC:22:0D:E5', "lookout")

        # Verify the corrected data was returned
        assert result.equals(corrected_df)

    @mock.patch('lookout.api.awn_controller.apply_corrections')
    @mock.patch('lookout.api.awn_controller.combine_full_history')
    @mock.patch('lookout.api.awn_controller.validate_archive')
    def test_corrections_fallback_on_error(self, mock_validate_archive, mock_combine_full_history, mock_apply_corrections):
        """Test fallback to uncorrected data when corrections fail."""
        # Mock validation to pass
        mock_validate_archive.return_value = True

        # Mock combine_full_history to return combined data
        combined_df = pd.DataFrame({'dateutc': [1640995200000], 'test': [1]})
        mock_combine_full_history.return_value = combined_df

        # Mock apply_corrections to raise exception
        mock_apply_corrections.side_effect = Exception("Correction failed")

        device = {'macAddress': '98:CD:AC:22:0D:E5'}
        archive_df = pd.DataFrame({'dateutc': [1640995200000], 'test': [1]})

        result = get_history_since_last_archive(device, archive_df, limit=1, pages=1)

        # Verify corrections were attempted
        mock_apply_corrections.assert_called_once()

        # Verify the uncorrected combined data was returned as fallback
        assert result.equals(combined_df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
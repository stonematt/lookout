"""
Tests for solar_data_transformer.py

Tests all pure data transformation functions to ensure they work correctly
without Plotly dependencies.
"""

import pandas as pd
import pytest
from lookout.core.solar_data_transformer import (
    prepare_month_day_heatmap_data,
    prepare_day_column_data,
    prepare_day_15min_heatmap_data,
    prepare_15min_bar_data,
    _aggregate_to_daily,
    _aggregate_to_hourly,
)


class TestPrepareMonthDayHeatmapData:
    """Test prepare_month_day_heatmap_data function."""

    def test_basic_month_day_pivot(self):
        """Test basic month/day pivot table creation."""
        # Create sample data for multiple months
        periods_df = pd.DataFrame({
            'period_start': pd.date_range('2023-01-01', periods=96, freq='15min'),  # Jan 1
            'period_end': pd.date_range('2023-01-01 00:15:00', periods=96, freq='15min'),
            'energy_kwh': [0.1] * 96  # 9.6 kWh total for Jan 1
        })
        # Add Feb 1 data
        feb_data = pd.DataFrame({
            'period_start': pd.date_range('2023-02-01', periods=96, freq='15min'),
            'period_end': pd.date_range('2023-02-01 00:15:00', periods=96, freq='15min'),
            'energy_kwh': [0.2] * 96  # 19.2 kWh total for Feb 1
        })
        periods_df = pd.concat([periods_df, feb_data], ignore_index=True)

        result = prepare_month_day_heatmap_data(periods_df)

        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # 2 months
        assert len(result.columns) == 31  # 31 days

        # Check month order (newest first)
        assert result.index.tolist() == ['2023-02', '2023-01']

        # Check day columns exist
        assert all(str(i) in result.columns for i in range(1, 32))

        # Check values (should be daily totals)
        assert abs(result.loc['2023-01', '1'] - 9.6) < 1e-10  # Jan 1: 96 * 0.1
        assert abs(result.loc['2023-02', '1'] - 19.2) < 1e-10  # Feb 1: 96 * 0.2

    def test_empty_dataframe(self):
        """Test handling of empty input."""
        result = prepare_month_day_heatmap_data(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_missing_days_filled_with_nan(self):
        """Test that missing days are filled with NaN."""
        # Only data for day 15
        periods_df = pd.DataFrame({
            'period_start': pd.date_range('2023-01-15', periods=96, freq='15min'),
            'period_end': pd.date_range('2023-01-15 00:15:00', periods=96, freq='15min'),
            'energy_kwh': [0.1] * 96
        })

        result = prepare_month_day_heatmap_data(periods_df)

        # Day 15 should have value
        assert abs(result.loc['2023-01', '15'] - 9.6) < 1e-10

        # Other days should be NaN
        assert pd.isna(result.loc['2023-01', '1'])
        assert pd.isna(result.loc['2023-01', '31'])

    def test_data_aggregation_accuracy(self):
        """Test that energy values are correctly aggregated."""
        periods_df = pd.DataFrame({
            'period_start': pd.date_range('2023-01-01', periods=4, freq='15min'),  # 1 hour
            'period_end': pd.date_range('2023-01-01 00:15:00', periods=4, freq='15min'),
            'energy_kwh': [0.1, 0.2, 0.3, 0.4]  # Total: 1.0 kWh
        })

        result = prepare_month_day_heatmap_data(periods_df)

        assert abs(result.loc['2023-01', '1'] - 1.0) < 1e-10


class TestPrepareDayColumnData:
    """Test prepare_day_column_data function."""

    def test_basic_hourly_aggregation(self):
        """Test basic hourly aggregation for a specific date."""
        # Create data for 2023-01-01 with different values per hour
        periods_df = pd.DataFrame({
            'period_start': pd.date_range('2023-01-01', periods=96, freq='15min'),  # Full day
            'period_end': pd.date_range('2023-01-01 00:15:00', periods=96, freq='15min'),
            'energy_kwh': [0.1] * 96  # 0.4 kWh per hour (4 * 0.1)
        })

        result = prepare_day_column_data(periods_df, '2023-01-01')

        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 24  # 24 hours
        assert list(result.columns) == ['hour', 'hourly_kwh']

        # Check all hours present
        assert result['hour'].tolist() == list(range(24))

        # Check values (0.4 kWh per hour)
        assert all(result['hourly_kwh'] == 0.4)

    def test_zeros_for_missing_date(self):
        """Test zeros result when date has no data."""
        periods_df = pd.DataFrame({
            'period_start': pd.date_range('2023-01-01', periods=96, freq='15min'),
            'period_end': pd.date_range('2023-01-01 00:15:00', periods=96, freq='15min'),
            'energy_kwh': [0.1] * 96
        })

        result = prepare_day_column_data(periods_df, '2023-01-02')  # Different date

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 24  # All 24 hours
        assert all(result['hourly_kwh'] == 0)  # All zeros

    def test_missing_hours_filled_with_zero(self):
        """Test that missing hours are filled with zero."""
        # Only data for hour 12
        periods_df = pd.DataFrame({
            'period_start': pd.date_range('2023-01-01 12:00:00', periods=4, freq='15min'),  # Hour 12 only
            'period_end': pd.date_range('2023-01-01 12:15:00', periods=4, freq='15min'),
            'energy_kwh': [0.1] * 4  # 0.4 kWh for hour 12
        })

        result = prepare_day_column_data(periods_df, '2023-01-01')

        # Hour 12 should have value
        assert result[result['hour'] == 12]['hourly_kwh'].iloc[0] == 0.4

        # Other hours should be 0
        assert all(result[result['hour'] != 12]['hourly_kwh'] == 0)


class TestPrepareDay15minHeatmapData:
    """Test prepare_day_15min_heatmap_data function."""

    def test_basic_15min_matrix(self):
        """Test basic day/time matrix creation."""
        # Create data for 2 days with 15min intervals
        periods_df = pd.DataFrame({
            'period_start': pd.date_range('2023-01-01', periods=8, freq='15min'),  # 2 hours on day 1
            'period_end': pd.date_range('2023-01-01 00:15:00', periods=8, freq='15min'),
            'energy_kwh': [0.1] * 8
        })
        # Add day 2 data
        day2_data = pd.DataFrame({
            'period_start': pd.date_range('2023-01-02', periods=8, freq='15min'),  # 2 hours on day 2
            'period_end': pd.date_range('2023-01-02 00:15:00', periods=8, freq='15min'),
            'energy_kwh': [0.2] * 8
        })
        periods_df = pd.concat([periods_df, day2_data], ignore_index=True)

        result = prepare_day_15min_heatmap_data(periods_df)

        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # 2 days
        assert len(result.columns) == 8  # 8 time slots

        # Check date order (newest first)
        assert result.index.tolist() == ['2023-01-02', '2023-01-01']

        # Check time slots
        expected_times = ['00:00', '00:15', '00:30', '00:45', '01:00', '01:15', '01:30', '01:45']
        assert result.columns.tolist() == expected_times

        # Check values
        assert all(result.loc['2023-01-01'] == 0.1)
        assert all(result.loc['2023-01-02'] == 0.2)

    def test_time_range_filtering(self):
        """Test filtering by start_hour and end_hour."""
        # Create full day data
        periods_df = pd.DataFrame({
            'period_start': pd.date_range('2023-01-01', periods=96, freq='15min'),  # Full day
            'period_end': pd.date_range('2023-01-01 00:15:00', periods=96, freq='15min'),
            'energy_kwh': [0.1] * 96
        })

        # Filter to daylight hours only (6 AM to 6 PM)
        result = prepare_day_15min_heatmap_data(periods_df, start_hour=6, end_hour=18)

        # Should have 12 hours * 4 slots = 48 time slots
        assert len(result.columns) == 48

        # First time slot should be 06:00
        assert result.columns[0] == '06:00'

        # Last time slot should be 17:45 (since end_hour=18 means up to but not including 18:00)
        assert result.columns[-1] == '17:45'

    def test_missing_periods_filled_with_zero(self):
        """Test that missing time periods are filled with zero."""
        # Only data for 00:00-00:15
        periods_df = pd.DataFrame({
            'period_start': pd.to_datetime(['2023-01-01 00:00:00']),
            'period_end': pd.to_datetime(['2023-01-01 00:15:00']),
            'energy_kwh': [0.1]
        })

        result = prepare_day_15min_heatmap_data(periods_df)

        # 00:00 should have value
        assert result.loc['2023-01-01', '00:00'] == 0.1

        # Other time slots should be 0 (filled by pivot_table with fill_value=0)
        assert all(result.loc['2023-01-01', result.columns != '00:00'] == 0)


class TestPrepare15minBarData:
    """Test prepare_15min_bar_data function."""

    def test_basic_15min_bar_data(self):
        """Test basic 15min bar data preparation."""
        periods_df = pd.DataFrame({
            'period_start': pd.date_range('2023-01-01', periods=4, freq='15min'),  # 1 hour
            'period_end': pd.date_range('2023-01-01 00:15:00', periods=4, freq='15min'),
            'energy_kwh': [0.1, 0.2, 0.3, 0.4]  # Total: 1.0 kWh
        })

        result = prepare_15min_bar_data(periods_df, '2023-01-01')

        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # 4 periods
        assert list(result.columns) == ['time_label', 'energy_wh']

        # Check time labels
        expected_labels = ['00:00', '00:15', '00:30', '00:45']
        assert result['time_label'].tolist() == expected_labels

        # Check Wh conversion (kWh * 1000)
        expected_wh = [100, 200, 300, 400]  # 0.1, 0.2, 0.3, 0.4 kWh * 1000
        assert result['energy_wh'].tolist() == expected_wh

    def test_time_range_filtering(self):
        """Test filtering by time range."""
        # Create full day data
        periods_df = pd.DataFrame({
            'period_start': pd.date_range('2023-01-01', periods=96, freq='15min'),  # Full day
            'period_end': pd.date_range('2023-01-01 00:15:00', periods=96, freq='15min'),
            'energy_kwh': [0.1] * 96
        })

        # Filter to morning hours only
        result = prepare_15min_bar_data(periods_df, '2023-01-01', start_hour=6, end_hour=12)

        # Should have 6 hours * 4 slots = 24 periods
        assert len(result) == 24

        # First time should be 06:00
        assert result['time_label'].iloc[0] == '06:00'

        # Last time should be 11:45
        assert result['time_label'].iloc[-1] == '11:45'

    def test_empty_result_for_missing_date(self):
        """Test empty result when date has no data."""
        periods_df = pd.DataFrame({
            'period_start': pd.date_range('2023-01-01', periods=96, freq='15min'),
            'period_end': pd.date_range('2023-01-01 00:15:00', periods=96, freq='15min'),
            'energy_kwh': [0.1] * 96
        })

        result = prepare_15min_bar_data(periods_df, '2023-01-02')  # Different date

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_sorting_by_time(self):
        """Test that results are sorted by time."""
        # Create out-of-order data
        periods_df = pd.DataFrame({
            'period_start': [
                pd.Timestamp('2023-01-01 00:30:00'),
                pd.Timestamp('2023-01-01 00:00:00'),
                pd.Timestamp('2023-01-01 00:15:00'),
            ],
            'period_end': [
                pd.Timestamp('2023-01-01 00:45:00'),
                pd.Timestamp('2023-01-01 00:15:00'),
                pd.Timestamp('2023-01-01 00:30:00'),
            ],
            'energy_kwh': [0.3, 0.1, 0.2]
        })

        result = prepare_15min_bar_data(periods_df, '2023-01-01')

        # Should be sorted by time
        expected_labels = ['00:00', '00:15', '00:30']
        assert result['time_label'].tolist() == expected_labels

        expected_wh = [100, 200, 300]  # Sorted by time
        assert result['energy_wh'].tolist() == expected_wh


class TestHelperFunctions:
    """Test internal helper functions."""

    def test_aggregate_to_daily(self):
        """Test daily aggregation helper."""
        periods_df = pd.DataFrame({
            'period_start': pd.date_range('2023-01-01', periods=96, freq='15min'),  # Full day
            'period_end': pd.date_range('2023-01-01 00:15:00', periods=96, freq='15min'),
            'energy_kwh': [0.1] * 96  # Total: 9.6 kWh
        })

        result = _aggregate_to_daily(periods_df)

        assert len(result) == 1
        assert result.iloc[0]['date'] == '2023-01-01'
        assert abs(result.iloc[0]['daily_kwh'] - 9.6) < 1e-10

    def test_aggregate_to_hourly(self):
        """Test hourly aggregation helper."""
        # Create data for hour 12 only
        periods_df = pd.DataFrame({
            'period_start': pd.date_range('2023-01-01 12:00:00', periods=4, freq='15min'),
            'period_end': pd.date_range('2023-01-01 12:15:00', periods=4, freq='15min'),
            'energy_kwh': [0.1] * 4  # Total: 0.4 kWh
        })

        result = _aggregate_to_hourly(periods_df, '2023-01-01')

        # Should have all 24 hours
        assert len(result) == 24

        # Hour 12 should have value
        hour_12_row = result[result['hour'] == 12]
        assert hour_12_row['hourly_kwh'].iloc[0] == 0.4

        # Other hours should be 0
        other_hours = result[result['hour'] != 12]
        # Other hours should be 0
        other_hours = result[result['hour'] != 12]
        assert all(other_hours['hourly_kwh'] == 0)

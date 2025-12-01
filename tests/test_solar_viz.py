"""
test_solar_viz.py
Unit tests for solar visualization functions.

Tests cover chart creation, data processing, and edge cases including
DST transitions and data gaps as specified in requirements.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import lookout.core.solar_viz as solar_viz
from lookout.core.solar_analysis import LOCATION


class TestSolarTimeSeriesChart:
    """Test solar time series chart creation."""

    def test_create_solar_time_series_chart_with_valid_data(self):
        """Test chart creation with valid solar data."""
        # Create test data
        dates = pd.date_range('2024-01-01', periods=24, freq='H', tz='UTC')
        df = pd.DataFrame({
            'datetime': dates,
            'solarradiation': np.random.uniform(0, 1000, 24),
            'daylight_period': ['day'] * 24
        })

        fig = solar_viz.create_solar_time_series_chart(df)

        # Check that figure was created
        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].name == 'Solar Radiation'

    def test_create_solar_time_series_chart_empty_data(self):
        """Test chart creation with empty data."""
        df = pd.DataFrame()

        fig = solar_viz.create_solar_time_series_chart(df)

        # Should return empty figure
        assert fig is not None
        assert len(fig.data) == 0

    def test_create_solar_time_series_chart_no_daytime_data(self):
        """Test chart creation with no daytime data."""
        dates = pd.date_range('2024-01-01', periods=24, freq='H', tz='UTC')
        df = pd.DataFrame({
            'datetime': dates,
            'solarradiation': np.random.uniform(0, 1000, 24),
            'daylight_period': ['night'] * 24  # All night
        })

        fig = solar_viz.create_solar_time_series_chart(df)

        # Should return empty figure
        assert fig is not None
        assert len(fig.data) == 0

    def test_create_solar_time_series_chart_with_nan_values(self):
        """Test chart creation with NaN values."""
        dates = pd.date_range('2024-01-01', periods=24, freq='H', tz='UTC')
        solar_data = np.random.uniform(0, 1000, 24)
        solar_data[10:15] = np.nan  # Some NaN values

        df = pd.DataFrame({
            'datetime': dates,
            'solarradiation': solar_data,
            'daylight_period': ['day'] * 24
        })

        fig = solar_viz.create_solar_time_series_chart(df)

        # Should create chart, filtering out NaN values
        assert fig is not None
        assert len(fig.data) == 1


class TestDailyEnergyChart:
    """Test daily energy chart creation."""

    def test_create_daily_energy_chart_with_valid_data(self):
        """Test chart creation with valid daily energy data."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        energy_values = np.random.uniform(1, 8, 10)

        daily_energy = pd.Series(energy_values, index=dates)

        fig = solar_viz.create_daily_energy_chart(daily_energy)

        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].name == 'Daily Energy'

    def test_create_daily_energy_chart_empty_data(self):
        """Test chart creation with empty energy data."""
        daily_energy = pd.Series(dtype=float)

        fig = solar_viz.create_daily_energy_chart(daily_energy)

        assert fig is not None
        assert len(fig.data) == 0


class TestSeasonalComparisonChart:
    """Test seasonal comparison chart creation."""

    def test_create_seasonal_comparison_chart_with_valid_data(self):
        """Test chart creation with valid seasonal data."""
        seasonal_data = {
            'winter': {'mean_kwh_per_m2': 2.5, 'max_kwh_per_m2': 5.0, 'days': 90},
            'spring': {'mean_kwh_per_m2': 5.0, 'max_kwh_per_m2': 8.0, 'days': 91},
            'summer': {'mean_kwh_per_m2': 7.0, 'max_kwh_per_m2': 10.0, 'days': 92},
            'fall': {'mean_kwh_per_m2': 3.5, 'max_kwh_per_m2': 6.0, 'days': 90},
            'seasonal_variation_ratio': 2.8
        }

        fig = solar_viz.create_seasonal_comparison_chart(seasonal_data)

        assert fig is not None
        assert len(fig.data) == 2  # Mean bars and max markers

    def test_create_seasonal_comparison_chart_empty_data(self):
        """Test chart creation with empty seasonal data."""
        seasonal_data = {}

        fig = solar_viz.create_seasonal_comparison_chart(seasonal_data)

        assert fig is not None
        assert len(fig.data) == 0

    def test_create_seasonal_comparison_chart_partial_data(self):
        """Test chart creation with partial seasonal data."""
        seasonal_data = {
            'summer': {'mean_kwh_per_m2': 7.0, 'max_kwh_per_m2': 10.0, 'days': 92},
            'winter': {'mean_kwh_per_m2': 2.5, 'max_kwh_per_m2': 5.0, 'days': 90}
        }

        fig = solar_viz.create_seasonal_comparison_chart(seasonal_data)

        assert fig is not None
        assert len(fig.data) == 2


class TestHourlyPatternChart:
    """Test hourly pattern chart creation."""

    def test_create_hourly_pattern_chart_with_valid_data(self):
        """Test chart creation with valid hourly pattern data."""
        # Create hourly data for a full day
        hours = range(24)
        radiation_values = np.zeros(24)

        # Simulate solar pattern peaking at noon
        for hour in hours:
            if 6 <= hour <= 18:  # Daylight hours
                radiation_values[hour] = 800 * np.sin(np.pi * (hour - 6) / 12)

        hourly_patterns = pd.Series(radiation_values, index=hours)

        fig = solar_viz.create_hourly_pattern_chart(hourly_patterns)

        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].name == 'Average Radiation'

    def test_create_hourly_pattern_chart_empty_data(self):
        """Test chart creation with empty hourly data."""
        hourly_patterns = pd.Series(dtype=float)

        fig = solar_viz.create_hourly_pattern_chart(hourly_patterns)

        assert fig is not None
        assert len(fig.data) == 0


class TestSolarHeatmap:
    """Test solar heatmap creation."""

    def test_prepare_solar_heatmap_data_day_mode(self):
        """Test heatmap data preparation in day mode."""
        dates = pd.date_range('2024-01-01', periods=48, freq='H', tz='UTC')
        df = pd.DataFrame({
            'datetime': dates,
            'solarradiation': np.random.uniform(0, 1000, 48),
            'daylight_period': ['day'] * 48
        })

        heatmap_df = solar_viz.prepare_solar_heatmap_data(df, row_mode='day')

        assert not heatmap_df.empty
        assert 'date' in heatmap_df.columns
        assert 'hour' in heatmap_df.columns
        assert 'radiation' in heatmap_df.columns

    def test_prepare_solar_heatmap_data_week_mode(self):
        """Test heatmap data preparation in week mode."""
        # Create a week's worth of data
        dates = pd.date_range('2024-01-01', periods=168, freq='H', tz='UTC')
        df = pd.DataFrame({
            'datetime': dates,
            'solarradiation': np.random.uniform(0, 1000, 168),
            'daylight_period': ['day'] * 168
        })

        heatmap_df = solar_viz.prepare_solar_heatmap_data(df, row_mode='week')

        assert not heatmap_df.empty
        assert 'date' in heatmap_df.columns
        assert 'hour' in heatmap_df.columns
        assert 'radiation' in heatmap_df.columns

    def test_prepare_solar_heatmap_data_empty(self):
        """Test heatmap data preparation with empty data."""
        df = pd.DataFrame()

        heatmap_df = solar_viz.prepare_solar_heatmap_data(df)

        assert heatmap_df.empty

    def test_create_solar_heatmap_with_valid_data(self):
        """Test heatmap creation with valid data."""
        # Create test heatmap data
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        hours = range(24)
        data = []

        for date in dates:
            for hour in hours:
                radiation = np.random.uniform(0, 1000) if 6 <= hour <= 18 else 0
                data.append({'date': date.date(), 'hour': hour, 'radiation': radiation})

        heatmap_df = pd.DataFrame(data)

        fig = solar_viz.create_solar_heatmap(heatmap_df)

        assert fig is not None
        assert len(fig.data) == 1

    def test_create_solar_heatmap_empty_data(self):
        """Test heatmap creation with empty data."""
        heatmap_df = pd.DataFrame()

        fig = solar_viz.create_solar_heatmap(heatmap_df)

        assert fig is not None
        assert len(fig.data) == 0


class TestDSTHandling:
    """Test DST transition handling."""

    def test_solar_data_dst_spring_forward(self):
        """Test handling of spring forward DST transition."""
        # Create data around March 10, 2024 (DST starts)
        dates = pd.date_range('2024-03-09 20:00', '2024-03-11 04:00', freq='H', tz='UTC')
        df = pd.DataFrame({
            'datetime': dates,
            'solarradiation': np.random.uniform(0, 500, len(dates)),
            'daylight_period': ['day' if 12 <= d.hour <= 20 else 'night' for d in dates]
        })

        # This should not crash even with DST transition
        fig = solar_viz.create_solar_time_series_chart(df)
        assert fig is not None

    def test_solar_data_dst_fall_back(self):
        """Test handling of fall back DST transition."""
        # Create data around November 3, 2024 (DST ends)
        dates = pd.date_range('2024-11-02 20:00', '2024-11-04 04:00', freq='H', tz='UTC')
        df = pd.DataFrame({
            'datetime': dates,
            'solarradiation': np.random.uniform(0, 500, len(dates)),
            'daylight_period': ['day' if 12 <= d.hour <= 20 else 'night' for d in dates]
        })

        # This should not crash even with DST transition
        fig = solar_viz.create_solar_time_series_chart(df)
        assert fig is not None


class TestDataGaps:
    """Test data gap handling."""

    def test_solar_data_with_gaps(self):
        """Test handling of data gaps in solar readings."""
        dates = pd.date_range('2024-01-01', periods=100, freq='H', tz='UTC')

        # Create data with gaps (some hours missing)
        solar_data = np.random.uniform(0, 1000, 100)
        # Simulate gaps by setting some values to NaN
        solar_data[20:30] = np.nan  # 10-hour gap
        solar_data[50:55] = np.nan  # 5-hour gap

        df = pd.DataFrame({
            'datetime': dates,
            'solarradiation': solar_data,
            'daylight_period': ['day'] * 100
        })

        fig = solar_viz.create_solar_time_series_chart(df)

        # Should handle gaps gracefully
        assert fig is not None
        assert len(fig.data) == 1

    def test_solar_data_extended_gaps(self):
        """Test handling of extended data gaps (days missing)."""
        # Create data with a full day gap
        dates1 = pd.date_range('2024-01-01', periods=48, freq='H', tz='UTC')
        dates2 = pd.date_range('2024-01-03', periods=48, freq='H', tz='UTC')  # Skip Jan 2
        dates = dates1.append(dates2)

        solar_data = np.random.uniform(0, 1000, len(dates))

        df = pd.DataFrame({
            'datetime': dates,
            'solarradiation': solar_data,
            'daylight_period': ['day'] * len(dates)
        })

        fig = solar_viz.create_solar_time_series_chart(df)

        # Should handle extended gaps
        assert fig is not None
        assert len(fig.data) == 1


class TestColorConfiguration:
    """Test solar color configuration."""

    def test_solar_colors_exist_in_palette(self):
        """Test that solar colors are properly defined in chart config."""
        from lookout.core.chart_config import get_standard_colors

        colors = get_standard_colors()

        # Check that solar colors exist
        assert 'solar_line' in colors
        assert 'solar_fill' in colors
        assert 'solar_high' in colors
        assert 'solar_medium' in colors
        assert 'solar_low' in colors

        # Check that colors are valid hex codes or rgba strings
        for color_name in ['solar_line', 'solar_fill', 'solar_high', 'solar_medium', 'solar_low']:
            color_value = colors[color_name]
            assert isinstance(color_value, str)
            assert len(color_value) > 0

    def test_solar_color_values_are_valid(self):
        """Test that solar color values are valid CSS color strings."""
        from lookout.core.chart_config import get_standard_colors

        colors = get_standard_colors()

        # Basic validation - should start with # or rgba
        solar_colors = ['solar_line', 'solar_high', 'solar_medium', 'solar_low']
        for color_name in solar_colors:
            color_value = colors[color_name]
            assert color_value.startswith('#'), f"{color_name} should be hex color"

        # Fill should be rgba
        assert colors['solar_fill'].startswith('rgba('), "solar_fill should be rgba color"
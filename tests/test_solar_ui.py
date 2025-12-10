"""
test_solar_ui.py
Unit tests for solar UI components.

Tests cover UI rendering, caching, and data processing with mocked Streamlit components.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import plotly.graph_objects as go

import lookout.core.solar_analysis as solar_analysis
import lookout.core.solar_viz as solar_viz
# Temporarily commented out during solar UI rebuild
# from lookout.ui.solar import (
#     _get_cached_solar_statistics,
#     _get_cached_daily_energy,
#     _get_cached_seasonal_breakdown,
#     _get_cached_hourly_patterns,
#     _get_cached_solar_heatmap_data,
# )


# Temporarily commented out during solar UI rebuild
# class TestSolarUICaching:
#     """Test UI caching functions."""
#
#     def test_get_cached_solar_statistics(self):
#         """Test solar statistics caching."""
#         # Create test data
#         dates = pd.date_range('2024-01-01', periods=24, freq='h', tz='UTC')
#         df = pd.DataFrame({
#             'datetime': dates,
#             'solarradiation': [500, 600, 700] + [0] * 21,  # Some daytime values
#             'daylight_period': ['day'] * 3 + ['night'] * 21
#         })
#         daily_energy = pd.Series([2.5, 3.0, 3.5], index=pd.date_range('2024-01-01', periods=3))
#
#         # Call cached function
#         result = _get_cached_solar_statistics(df, daily_energy)
#
#         # Should return a dictionary with statistics
#         assert isinstance(result, dict)
#         assert 'peak_radiation_w_per_m2' in result
#
#     def test_get_cached_daily_energy(self):
#         """Test daily energy caching."""
#         dates = pd.date_range('2024-01-01', periods=48, freq='h', tz='UTC')
#         df = pd.DataFrame({
#             'datetime': dates,
#             'date': [d.date() for d in dates],  # Add date column
#             'solarradiation': np.random.uniform(0, 1000, 48),
#             'daylight_period': ['day'] * 48
#         })
#
#         result = _get_cached_daily_energy(df)
#
#         # Should return a Series
#         assert isinstance(result, pd.Series)
#         assert not result.empty
#
#     def test_get_cached_seasonal_breakdown(self):
#         """Test seasonal breakdown caching."""
#         # Create a longer series with proper date index
#         dates = pd.date_range('2024-01-01', periods=365, freq='D')
#         daily_energy = pd.Series(
#             np.random.uniform(1, 8, 365),
#             index=dates,
#             name='daily_kwh_per_m2'
#         )
#         # Set index name to match expected format
#         daily_energy.index.name = 'date'
#
#         result = _get_cached_seasonal_breakdown(daily_energy)
#
#         # Should return a dictionary
#         assert isinstance(result, dict)
#
#     def test_get_cached_hourly_patterns(self):
#         """Test hourly patterns caching."""
#         df = pd.DataFrame({
#             'datetime': pd.date_range('2024-01-01', periods=48, freq='h', tz='UTC'),
#             'solarradiation': np.random.uniform(0, 1000, 48),
#             'daylight_period': ['day'] * 48
#         })
#
#         result = _get_cached_hourly_patterns(df)
#
#         # Should return a Series
#         assert isinstance(result, pd.Series)
#
#     def test_get_cached_solar_heatmap_data(self):
#         """Test heatmap data caching."""
#         df = pd.DataFrame({
#             'datetime': pd.date_range('2024-01-01', periods=48, freq='h', tz='UTC'),
#             'solarradiation': np.random.uniform(0, 1000, 48),
#             'daylight_period': ['day'] * 48
#         })
#
#         result = _get_cached_solar_heatmap_data(df, 'day')
#
#         # Should return a DataFrame
#         assert isinstance(result, pd.DataFrame)


# Temporarily commented out during solar UI rebuild
# class TestSolarUIRendering:
#     """Test UI rendering functions with mocked Streamlit."""
#
#     @patch('lookout.ui.solar.st')
#     @patch('lookout.ui.solar.st.session_state', {'history_df': None})
#     def test_render_no_data(self, mock_st):
#         """Test render function with no data."""
#         from lookout.ui.solar import render
#
#         render()
#
#         # Should show warning about no data
#         mock_st.warning.assert_called_once()
#
#     @patch('lookout.ui.solar.st')
#     def test_render_no_solar_data(self, mock_st):
#         """Test render function with data but no solar readings."""
#         # Mock session state with data but no solar
#         mock_df = pd.DataFrame({
#             'solarradiation': [None, None, None],
#             'dateutc': [1, 2, 3]
#         })
#
#         with patch('lookout.ui.solar.st.session_state', {'history_df': mock_df}):
#             from lookout.ui.solar import render
#             render()
#
#         # Should show warning about no solar data
#         mock_st.warning.assert_called()
#
#     @patch('lookout.ui.solar.st')
#     @patch('lookout.ui.solar._display_solar_statistics')
#     @patch('lookout.ui.solar._display_solar_data_quality')
#     @patch('lookout.ui.solar._render_time_series_tab')
#     @patch('lookout.ui.solar._render_seasonal_analysis_tab')
#     @patch('lookout.ui.solar._render_patterns_heatmap_tab')
#     def test_render_with_data(self, mock_patterns, mock_seasonal, mock_time_series,
#                              mock_data_quality, mock_stats, mock_st):
#         # Mock st.tabs to return 3 mock tab objects (updated structure)
#         mock_tab1 = MagicMock()
#         mock_tab2 = MagicMock()
#         mock_tab3 = MagicMock()
#         mock_st.tabs.return_value = [mock_tab1, mock_tab2, mock_tab3]
#         """Test render function with valid solar data."""
#         # Create mock data with solar readings
#         mock_df = pd.DataFrame({
#             'datetime': pd.date_range('2024-01-01', periods=24, freq='h', tz='UTC'),
#             'solarradiation': np.random.uniform(0, 1000, 24),
#             'daylight_period': ['day'] * 24,
#             'dateutc': range(24)
#         })
#
#         with patch('lookout.ui.solar.st.session_state', {'history_df': mock_df}):
#             from lookout.ui.solar import render
#             render()
#
#         # Should call all the rendering functions
#         mock_stats.assert_called_once()
#         mock_time_series.assert_called_once()
#         mock_seasonal.assert_called_once()
#         mock_patterns.assert_called_once()
#         mock_data_quality.assert_called_once()


# Temporarily commented out during solar UI rebuild
# class TestSolarUIStatistics:
#     """Test statistics display function."""
#
#     @patch('lookout.ui.solar.st')
#     def test_display_solar_statistics_no_data(self, mock_st):
#         """Test statistics display with no data."""
#         from lookout.ui.solar import _display_solar_statistics
#
#         # Create empty DataFrame with required columns to avoid KeyError
#         empty_df = pd.DataFrame({'datetime': [], 'solarradiation': [], 'daylight_period': []})
#         _display_solar_statistics(empty_df, pd.Series(dtype=float))
#
#         # Should not create any metrics
#         mock_st.columns.assert_not_called()
#
#     @patch('lookout.ui.solar.st')
#     @patch('lookout.ui.solar._get_cached_solar_statistics')
#     def test_display_solar_statistics_with_data(self, mock_get_stats, mock_st):
#         """Test statistics display with valid data."""
#         # Mock the statistics function to return valid data
#         mock_get_stats.return_value = {
#             'peak_radiation_w_per_m2': 800.0,
#             'avg_daily_energy_kwh_per_m2': 4.5,
#             'annual_energy_kwh_per_m2': 1642.5,
#             'peak_hours_utc': [12, 13],
#             'peak_hours_local': [12, 13]
#         }
#
#         df = pd.DataFrame({
#             'datetime': pd.date_range('2024-01-01', periods=24, freq='h', tz='UTC'),
#             'solarradiation': [500, 600, 700] + [0] * 21,
#             'daylight_period': ['day'] * 3 + ['night'] * 21
#         })
#         daily_energy = pd.Series([2.5, 3.0, 3.5], index=pd.date_range('2024-01-01', periods=3))
#
#         # Mock columns to return context managers
#         mock_columns = [MagicMock() for _ in range(4)]
#         mock_st.columns.return_value = mock_columns
#
#         from lookout.ui.solar import _display_solar_statistics
#         _display_solar_statistics(df, daily_energy)
#
#         # Should create 4 columns
#         mock_st.columns.assert_called_once_with(4)
#         # Should call metric 4 times (once in each column context)
#         assert mock_st.metric.call_count == 4


# Temporarily commented out during solar UI rebuild
# class TestSolarUITabs:
#     """Test individual tab rendering functions."""
#
#     @patch('lookout.ui.solar.st')
#     @patch('lookout.ui.solar.solar_viz.create_solar_time_series_chart')
#     def test_render_time_series_tab(self, mock_chart, mock_st):
#         """Test time series tab rendering."""
#         df = pd.DataFrame({
#             'datetime': pd.date_range('2024-01-01', periods=24, freq='h', tz='UTC'),
#             'solarradiation': np.random.uniform(0, 1000, 24),
#             'daylight_period': ['day'] * 24
#         })
#
#         # Create daily energy for the test
#         daily_energy = pd.Series([2.5, 3.0, 3.5], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
#
#         from lookout.ui.solar import _render_time_series_tab
#         with patch('lookout.ui.components.create_date_range_slider') as mock_date_slider, \
#              patch('lookout.ui.solar.solar_viz.create_solar_time_series_chart') as mock_time_chart, \
#              patch('lookout.ui.solar.solar_viz.create_daily_energy_chart') as mock_energy_chart, \
#              patch('lookout.ui.solar.st') as mock_st:
#             mock_date_slider.return_value = (None, None)  # No date filtering
#             # Mock st.columns for the energy metrics
#             mock_col1 = MagicMock()
#             mock_col2 = MagicMock()
#             mock_col3 = MagicMock()
#             mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]
#             _render_time_series_tab(df, daily_energy)
#
#             # Should call time series chart creation
#             mock_time_chart.assert_called_once()
#
#     @patch('lookout.ui.solar.st')
#     @patch('lookout.ui.solar.solar_viz.create_daily_energy_chart')
#     @patch('lookout.ui.solar.solar_viz.create_solar_time_series_chart')
#     @patch('lookout.ui.components.create_date_range_slider')
#     def test_render_combined_time_series_tab(self, mock_date_slider, mock_time_series_chart, mock_energy_chart, mock_st):
#         """Test combined time series and daily energy tab rendering."""
#         # Create test data
#         dates = pd.date_range('2024-01-01', periods=24, freq='h', tz='UTC')
#         df = pd.DataFrame({
#             'datetime': dates,
#             'solarradiation': [500, 600, 700] + [0] * 21,
#             'daylight_period': ['day'] * 3 + ['night'] * 21
#         })
#         daily_energy = pd.Series([2.5, 3.0, 3.5], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
#
#         # Mock date slider to return timestamps
#         mock_date_slider.return_value = (
#             pd.Timestamp('2024-01-01').tz_localize('UTC'),
#             pd.Timestamp('2024-01-03').tz_localize('UTC')
#         )
#
#         # Mock st.columns to return 3 mock column objects for energy metrics
#         mock_col1 = MagicMock()
#         mock_col2 = MagicMock()
#         mock_col3 = MagicMock()
#         mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]
#
#         from lookout.ui.solar import _render_time_series_tab
#         _render_time_series_tab(df, daily_energy)
#
#         # Should call both chart creation functions
#         mock_time_series_chart.assert_called_once()
#         mock_energy_chart.assert_called_once()
#
#     @patch('lookout.ui.solar.st')
#     @patch('lookout.ui.solar.solar_viz.create_seasonal_comparison_chart')
#     @patch('lookout.ui.solar._get_cached_seasonal_breakdown')
#     def test_render_seasonal_analysis_tab(self, mock_get_seasonal, mock_chart, mock_st):
#         """Test seasonal analysis tab rendering."""
#         # Mock seasonal data
#         mock_get_seasonal.return_value = {
#             'winter': {'mean_kwh_per_m2': 2.5, 'max_kwh_per_m2': 5.0, 'days': 90},
#             'summer': {'mean_kwh_per_m2': 7.0, 'max_kwh_per_m2': 10.0, 'days': 92}
#         }
#
#         # Create a series for seasonal analysis
#         dates = pd.date_range('2024-01-01', periods=365, freq='D')
#         daily_energy = pd.Series(np.random.uniform(1, 8, 365), index=dates)
#
#         from lookout.ui.solar import _render_seasonal_analysis_tab
#         _render_seasonal_analysis_tab(daily_energy)
#
#         # Should call chart creation
#         mock_chart.assert_called_once()
#
#     @patch('lookout.ui.solar.st')
#     @patch('lookout.ui.solar.solar_viz.create_hourly_pattern_chart')
#     @patch('lookout.ui.solar.solar_viz.create_solar_heatmap')
#     def test_render_patterns_heatmap_tab(self, mock_heatmap, mock_pattern, mock_st):
#         """Test patterns and heatmap tab rendering."""
#         df = pd.DataFrame({
#             'datetime': pd.date_range('2024-01-01', periods=48, freq='h', tz='UTC'),
#             'solarradiation': np.random.uniform(0, 1000, 48),
#             'daylight_period': ['day'] * 48
#         })
#
#         # Mock st.columns to return 2 mock column objects
#         mock_col1 = MagicMock()
#         mock_col2 = MagicMock()
#         mock_st.columns.return_value = [mock_col1, mock_col2]
#
#         from lookout.ui.solar import _render_patterns_heatmap_tab
#         _render_patterns_heatmap_tab(df)
#
#         # Should call both chart creation functions
#         mock_pattern.assert_called_once()
#         mock_heatmap.assert_called_once()


# Temporarily commented out during solar UI rebuild
# class TestSolarUIEdgeCases:
#     """Test edge cases and error handling."""
#
#     @patch('lookout.ui.solar.st')
#     @patch('lookout.ui.solar.solar_viz.create_daily_energy_chart')
#     @patch('lookout.ui.solar.solar_viz.create_solar_time_series_chart')
#     @patch('lookout.ui.components.create_date_range_slider')
#     def test_render_empty_daily_energy(self, mock_date_slider, mock_time_chart, mock_energy_chart, mock_st):
#         """Test rendering with empty daily energy data."""
#         # Create test data
#         df = pd.DataFrame({
#             'datetime': pd.date_range('2024-01-01', periods=24, freq='h', tz='UTC'),
#             'solarradiation': np.random.uniform(0, 1000, 24),
#             'daylight_period': ['day'] * 24
#         })
#
#         # Mock date slider
#         mock_date_slider.return_value = (None, None)
#
#         # Mock st.columns to return 3 mock column objects
#         mock_col1 = MagicMock()
#         mock_col2 = MagicMock()
#         mock_col3 = MagicMock()
#         mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]
#
#         # Mock the chart functions
#         mock_time_chart.return_value = go.Figure()
#         mock_energy_chart.return_value = go.Figure()
#
#         from lookout.ui.solar import _render_time_series_tab
#         _render_time_series_tab(df, pd.Series(dtype=float))
#
#         # Should call time series chart creation (energy chart not called when empty)
#         mock_time_chart.assert_called_once()
#         # Energy chart is not called when daily_energy is empty
#         mock_energy_chart.assert_not_called()
#
#     @patch('lookout.ui.solar.st')
#     def test_render_empty_seasonal_data(self, mock_st):
#         """Test handling of empty seasonal data."""
#         from lookout.ui.solar import _render_seasonal_analysis_tab
#
#         _render_seasonal_analysis_tab(pd.Series(dtype=float))
#
#         # Should show warning
#         mock_st.warning.assert_called_once()
#
#     @patch('lookout.ui.solar.st')
#     @patch('lookout.ui.components.create_date_range_slider')
#     def test_time_series_no_data_in_range(self, mock_date_slider, mock_st):
#         """Test time series with no data in selected range."""
#         df = pd.DataFrame({
#             'datetime': pd.date_range('2024-01-01', periods=24, freq='h', tz='UTC'),
#             'solarradiation': np.random.uniform(0, 1000, 24),
#             'daylight_period': ['day'] * 24
#         })
#
#         # Create daily energy for the test
#         daily_energy = pd.Series([2.5, 3.0, 3.5], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
#
#         # Mock date slider to return range with no data
#         mock_date_slider.return_value = (
#             pd.Timestamp('2024-02-01').tz_localize('UTC'),
#             pd.Timestamp('2024-02-02').tz_localize('UTC')
#         )
#
#         from lookout.ui.solar import _render_time_series_tab
#         _render_time_series_tab(df, daily_energy)
#
#         # Should show warning for no data in range
#         mock_st.warning.assert_called_once()
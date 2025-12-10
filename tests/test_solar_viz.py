"""
Tests for solar_viz module
"""

import pandas as pd
import pytest
import plotly.graph_objects as go
from datetime import datetime
from unittest.mock import patch, MagicMock

from lookout.core.solar_viz import create_month_day_heatmap


class TestCreateMonthDayHeatmap:
    """Test the create_month_day_heatmap function."""

    def test_basic_functionality(self):
        """Test basic heatmap creation with sample data."""
        # Create sample periods data for a few days
        periods_df = pd.DataFrame({
            'period_start': pd.date_range('2023-01-01', periods=96, freq='15min'),  # 1 day
            'period_end': pd.date_range('2023-01-01 00:15:00', periods=96, freq='15min'),
            'energy_kwh': [0.1] * 96  # 1 kWh total for the day
        })

        fig = create_month_day_heatmap(periods_df)

        # Check return type
        assert isinstance(fig, go.Figure)

        # Check basic layout properties
        assert fig.layout.height == 500
        assert fig.layout.xaxis.title.text == "Day of Month"
        assert fig.layout.yaxis.title.text == "Month"
        assert fig.layout.yaxis.autorange == "reversed"  # Newest months at top

        # Check heatmap properties
        heatmap = fig.data[0]
        assert isinstance(heatmap, go.Heatmap)
        assert heatmap.colorscale == (
            (0.0, "#FFF9E6"), (0.3, "#FFE680"), (0.6, "#FFB732"), (1.0, "#FF8C00")
        )
        assert heatmap.zmin == 0
        assert heatmap.showscale is True
        assert heatmap.colorbar.title.text == "kWh/day"
        assert heatmap.colorbar.ticksuffix == " kWh"

    def test_empty_dataframe(self):
        """Test handling of empty input data."""
        empty_df = pd.DataFrame({
            'period_start': pd.Series(dtype='datetime64[ns]'),
            'period_end': pd.Series(dtype='datetime64[ns]'),
            'energy_kwh': pd.Series(dtype=float)
        })

        fig = create_month_day_heatmap(empty_df)

        # Should return a figure with "No Solar Data Available" title
        assert isinstance(fig, go.Figure)
        assert fig.layout.height == 500

    def test_single_day_data(self):
        """Test with data for a single day."""
        periods_df = pd.DataFrame({
            'period_start': pd.date_range('2023-06-15', periods=96, freq='15min'),
            'period_end': pd.date_range('2023-06-15 00:15:00', periods=96, freq='15min'),
            'energy_kwh': [0.01] * 96  # 0.96 kWh total
        })

        fig = create_month_day_heatmap(periods_df)
        assert isinstance(fig, go.Figure)

        heatmap = fig.data[0]
        # Should have 31 days as x-labels
        assert len(heatmap.x) == 31
        assert list(heatmap.x) == [str(i) for i in range(1, 32)]

        # Should have one month (2023-06)
        assert len(heatmap.y) == 1
        assert heatmap.y[0] == "2023-06"

    def test_multiple_months(self):
        """Test with data spanning multiple months."""
        # Create data for January and February 2023
        dates_jan = pd.date_range('2023-01-01', '2023-01-31', freq='D')
        dates_feb = pd.date_range('2023-02-01', '2023-02-15', freq='D')

        periods_data = []
        for date in dates_jan[:5]:  # First 5 days of Jan
            periods_data.extend([{
                'period_start': pd.Timestamp(f"{date.date()} 06:00:00"),
                'period_end': pd.Timestamp(f"{date.date()} 18:00:00"),
                'energy_kwh': 2.0
            }])

        for date in dates_feb[:3]:  # First 3 days of Feb
            periods_data.extend([{
                'period_start': pd.Timestamp(f"{date.date()} 06:00:00"),
                'period_end': pd.Timestamp(f"{date.date()} 18:00:00"),
                'energy_kwh': 1.5
            }])

        periods_df = pd.DataFrame(periods_data)
        fig = create_month_day_heatmap(periods_df)

        heatmap = fig.data[0]
        # Should have months sorted newest first
        expected_months = ["2023-02", "2023-01"]
        assert list(heatmap.y) == expected_months

    def test_zero_energy_days(self):
        """Test that zero energy days are handled correctly (valid cloudy days)."""
        periods_df = pd.DataFrame({
            'period_start': pd.date_range('2023-01-01', periods=96, freq='15min'),
            'period_end': pd.date_range('2023-01-01 00:15:00', periods=96, freq='15min'),
            'energy_kwh': [0.0] * 96  # Zero energy (cloudy day)
        })

        fig = create_month_day_heatmap(periods_df)
        heatmap = fig.data[0]

        # Should have zmin=0 to include zero values in scale
        assert heatmap.zmin == 0

        # Check that zero values are preserved (not treated as missing)
        z_values = heatmap.z
        assert z_values[0][0] == 0.0  # January 1st should be 0.0

    def test_hover_text_format(self):
        """Test that hover text is formatted correctly."""
        periods_df = pd.DataFrame({
            'period_start': pd.date_range('2023-01-01', periods=96, freq='15min'),
            'period_end': pd.date_range('2023-01-01 00:15:00', periods=96, freq='15min'),
            'energy_kwh': [0.01] * 96  # 0.96 kWh total
        })

        fig = create_month_day_heatmap(periods_df)
        heatmap = fig.data[0]

        # Check hover text format
        hover_text = heatmap.text
        assert "1/2023-01" in hover_text[0][0]
        assert "0.96 kWh" in hover_text[0][0]

    def test_missing_days_handling(self):
        """Test that missing days (no data) are handled as NaN."""
        # Create data with gaps - only day 1 and day 3
        periods_data = []

        # Day 1: full data
        for hour in range(6, 18):  # 6 AM to 6 PM
            periods_data.append({
                'period_start': pd.Timestamp(f"2023-01-01 {hour:02d}:00:00"),
                'period_end': pd.Timestamp(f"2023-01-01 {hour:02d}:15:00"),
                'energy_kwh': 0.1
            })

        # Day 3: full data
        for hour in range(6, 18):  # 6 AM to 6 PM
            periods_data.append({
                'period_start': pd.Timestamp(f"2023-01-03 {hour:02d}:00:00"),
                'period_end': pd.Timestamp(f"2023-01-03 {hour:02d}:15:00"),
                'energy_kwh': 0.1
            })

        periods_df = pd.DataFrame(periods_data)
        fig = create_month_day_heatmap(periods_df)
        heatmap = fig.data[0]

        z_values = heatmap.z
        # Day 1 should have value (12 * 0.1 = 1.2)
        assert abs(z_values[0][0] - 1.2) < 1e-10  # January 1st
        # Day 2 should be NaN (missing)
        assert pd.isna(z_values[0][1])  # January 2nd
        # Day 3 should have value
        assert abs(z_values[0][2] - 1.2) < 1e-10  # January 3rd


class TestSolarUIRender:
    """Test the solar UI render function."""

    def test_no_history_df(self):
        """Test handling when history_df is not in session state."""
        from lookout.ui.solar import render

        # Create a mock session state that doesn't contain 'history_df'
        mock_session_state = MagicMock()
        mock_session_state.__contains__.return_value = False  # 'history_df' not in session_state

        with patch('lookout.ui.solar.st.session_state', mock_session_state), \
             patch('lookout.ui.solar.st.warning') as mock_warning, \
             patch('lookout.ui.solar.st.header'), \
             patch('lookout.ui.solar.st.caption'):

            render()

            mock_warning.assert_called_once_with("No weather data loaded. Please load data from main app.")

    @patch('lookout.ui.solar.st.session_state')
    def test_missing_date_column(self, mock_session_state):
        """Test handling when date column is missing."""
        from lookout.ui.solar import render

        # Mock session state with data missing date column
        mock_session_state.__getitem__.return_value = pd.DataFrame({
            'dateutc': [1640995200000, 1640995260000],
            'solarradiation': [100.0, 150.0]  # No date column
        })

        with patch('lookout.ui.solar.st.warning'), \
             patch('lookout.ui.solar.st.header'), \
             patch('lookout.ui.solar.st.caption'), \
             patch('lookout.ui.solar.st.error') as mock_error:

            render()

            mock_error.assert_called_once_with("Solar data not available in dataset.")

    @patch('lookout.ui.solar.st.session_state')
    def test_missing_solarradiation_column(self, mock_session_state):
        """Test handling when solarradiation column is missing."""
        from lookout.ui.solar import render

        # Mock session state with data missing solarradiation
        mock_session_state.__getitem__.return_value = pd.DataFrame({
            'dateutc': [1640995200000, 1640995260000],
            'date': pd.to_datetime([1640995200000, 1640995260000], unit='ms', utc=True).tz_convert('America/Los_Angeles')
        })

        with patch('lookout.ui.solar.st.warning'), \
             patch('lookout.ui.solar.st.header'), \
             patch('lookout.ui.solar.st.caption'), \
             patch('lookout.ui.solar.st.error') as mock_error:

            render()

            mock_error.assert_called_once_with("Solar data not available in dataset.")

    @patch('lookout.ui.solar.st.session_state')
    def test_successful_rendering(self, mock_session_state):
        """Test successful rendering with valid data."""
        from lookout.ui.solar import render

        # Mock session state with valid solar data
        mock_session_state.__getitem__.return_value = pd.DataFrame({
            'dateutc': [1640995200000, 1640995260000, 1640995320000],  # Jan 1, 2022
            'date': pd.to_datetime([1640995200000, 1640995260000, 1640995320000],
                                 unit='ms', utc=True).tz_convert('America/Los_Angeles'),
            'solarradiation': [100.0, 150.0, 200.0]
        })

        with patch('lookout.ui.solar.st.header'), \
             patch('lookout.ui.solar.st.caption'), \
             patch('lookout.ui.solar.st.spinner'), \
             patch('lookout.ui.solar.st.success') as mock_success, \
             patch('lookout.ui.solar.st.expander'), \
             patch('lookout.ui.solar.st.dataframe'), \
             patch('lookout.ui.solar.st.subheader'), \
             patch('lookout.ui.solar.st.plotly_chart') as mock_plotly_chart:

            render()

            # Should show success messages for calculation and rendering
            assert mock_success.call_count >= 2

            # Should render plotly chart
            mock_plotly_chart.assert_called_once()

    @patch('lookout.ui.solar.st.session_state')
    def test_heatmap_rendering_error(self, mock_session_state):
        """Test handling of errors during heatmap rendering."""
        from lookout.ui.solar import render

        # Mock session state with valid data
        mock_session_state.__getitem__.return_value = pd.DataFrame({
            'dateutc': [1640995200000, 1640995260000],
            'date': pd.to_datetime([1640995200000, 1640995260000],
                                 unit='ms', utc=True).tz_convert('America/Los_Angeles'),
            'solarradiation': [100.0, 150.0]
        })

        with patch('lookout.ui.solar.st.header'), \
             patch('lookout.ui.solar.st.caption'), \
             patch('lookout.ui.solar.st.spinner'), \
             patch('lookout.ui.solar.st.success'), \
             patch('lookout.ui.solar.st.expander'), \
             patch('lookout.ui.solar.st.dataframe'), \
             patch('lookout.ui.solar.st.subheader'), \
             patch('lookout.ui.solar.st.error') as mock_error, \
             patch('lookout.ui.solar.st.exception') as mock_exception, \
             patch('lookout.core.solar_viz.create_month_day_heatmap', side_effect=Exception("Test error")):

            render()

            # Should show error message and exception details
            mock_error.assert_called_once_with("Error rendering heatmap: Test error")
            mock_exception.assert_called_once()


class TestSolarUIRender:
    """Test the solar UI render function."""

    def test_no_history_df(self):
        """Test handling when history_df is not in session state."""
        import streamlit as st
        from lookout.ui.solar import render

        # Mock streamlit functions
        with patch('streamlit.warning') as mock_warning, \
             patch('streamlit.header'), \
             patch('streamlit.caption'):

            render()

            # Should show warning about no data loaded
            mock_warning.assert_called_once_with("No weather data loaded. Please load data from main app.")

    def test_missing_solarradiation_column(self):
        """Test handling when solarradiation column is missing."""
        from lookout.ui.solar import render

        # Mock session state with data missing solarradiation
        mock_session_state = MagicMock()
        mock_session_state.__contains__.return_value = True
        mock_session_state.__getitem__.return_value = pd.DataFrame({
            'dateutc': [1640995200000, 1640995260000],
            'date': pd.to_datetime([1640995200000, 1640995260000], unit='ms', utc=True).tz_convert('America/Los_Angeles')
        })

        with patch('lookout.ui.solar.st.session_state', mock_session_state), \
             patch('lookout.ui.solar.st.warning'), \
             patch('lookout.ui.solar.st.header'), \
             patch('lookout.ui.solar.st.caption'), \
             patch('lookout.ui.solar.st.error') as mock_error:

            render()

            mock_error.assert_called_once_with("Solar data not available in dataset.")

    def test_missing_dateutc_column(self):
        """Test handling when dateutc column is missing."""
        from lookout.ui.solar import render

        # Mock session state with data missing dateutc
        mock_session_state = MagicMock()
        mock_session_state.__contains__.return_value = True
        mock_session_state.__getitem__.return_value = pd.DataFrame({
            'solarradiation': [100.0, 150.0],
            'date': pd.to_datetime([1640995200000, 1640995260000], unit='ms', utc=True).tz_convert('America/Los_Angeles')
        })

        with patch('lookout.ui.solar.st.session_state', mock_session_state), \
             patch('lookout.ui.solar.st.warning'), \
             patch('lookout.ui.solar.st.header'), \
             patch('lookout.ui.solar.st.caption'), \
             patch('lookout.ui.solar.st.error') as mock_error:

            render()

            mock_error.assert_called_once_with("Solar data not available in dataset.")

    def test_successful_rendering(self):
        """Test successful rendering with valid data."""
        from lookout.ui.solar import render

        # Mock session state with valid solar data
        mock_session_state = MagicMock()
        mock_session_state.__contains__.return_value = True
        mock_session_state.__getitem__.return_value = pd.DataFrame({
            'dateutc': [1640995200000, 1640995260000, 1640995320000],  # Jan 1, 2022
            'date': pd.to_datetime([1640995200000, 1640995260000, 1640995320000],
                                 unit='ms', utc=True).tz_convert('America/Los_Angeles'),
            'solarradiation': [100.0, 150.0, 200.0]
        })

        with patch('lookout.ui.solar.st.session_state', mock_session_state), \
             patch('lookout.ui.solar.st.header'), \
             patch('lookout.ui.solar.st.caption'), \
             patch('lookout.ui.solar.st.spinner'), \
             patch('lookout.ui.solar.st.success') as mock_success, \
             patch('lookout.ui.solar.st.expander'), \
             patch('lookout.ui.solar.st.dataframe'), \
             patch('lookout.ui.solar.st.subheader'), \
             patch('lookout.ui.solar.st.plotly_chart') as mock_plotly_chart:

            render()

            # Should show success messages for calculation and rendering
            assert mock_success.call_count >= 2

            # Should render plotly chart
            mock_plotly_chart.assert_called_once()

    def test_heatmap_rendering_error(self):
        """Test handling of errors during heatmap rendering."""
        from lookout.ui.solar import render

        # Mock session state with valid data
        mock_session_state = MagicMock()
        mock_session_state.__contains__.return_value = True
        mock_session_state.__getitem__.return_value = pd.DataFrame({
            'dateutc': [1640995200000, 1640995260000],
            'date': pd.to_datetime([1640995200000, 1640995260000],
                                 unit='ms', utc=True).tz_convert('America/Los_Angeles'),
            'solarradiation': [100.0, 150.0]
        })

        with patch('lookout.ui.solar.st.session_state', mock_session_state), \
             patch('lookout.ui.solar.st.header'), \
             patch('lookout.ui.solar.st.caption'), \
             patch('lookout.ui.solar.st.spinner'), \
             patch('lookout.ui.solar.st.success'), \
             patch('lookout.ui.solar.st.expander'), \
             patch('lookout.ui.solar.st.dataframe'), \
             patch('lookout.ui.solar.st.subheader'), \
             patch('lookout.ui.solar.st.error') as mock_error, \
             patch('lookout.ui.solar.st.exception') as mock_exception, \
             patch('lookout.ui.solar.create_month_day_heatmap', side_effect=Exception("Test error")):

            render()

            # Should show error message and exception details
            mock_error.assert_called_once_with("Error rendering heatmap: Test error")
            mock_exception.assert_called_once()

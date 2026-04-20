"""
Tests for solar_viz module
"""

import pandas as pd
import pytest
import plotly.graph_objects as go
from datetime import datetime
from unittest.mock import patch, MagicMock

from lookout.core.solar_viz import (
    create_month_day_heatmap,
    create_day_column_chart,
    create_15min_bar_chart,
)


class TestCreateMonthDayHeatmap:
    """Test the create_month_day_heatmap function."""

    def test_basic_functionality(self):
        """Test basic heatmap creation with sample data."""
        # Create sample periods data for a few days
        periods_df = pd.DataFrame(
            {
                "period_start": pd.date_range(
                    "2023-01-01", periods=96, freq="15min"
                ),  # 1 day
                "period_end": pd.date_range(
                    "2023-01-01 00:15:00", periods=96, freq="15min"
                ),
                "energy_kwh": [0.1] * 96,  # 1 kWh total for the day
            }
        )

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
            (0.0, "#FFF9E6"),
            (0.3, "#FFE680"),
            (0.6, "#FFB732"),
            (1.0, "#FF8C00"),
        )
        assert heatmap.zmin == 0
        assert heatmap.showscale is True
        assert heatmap.colorbar.title.text == "kWh/day"
        assert heatmap.colorbar.ticksuffix == " kWh"

    def test_empty_dataframe(self):
        """Test handling of empty input data."""
        empty_df = pd.DataFrame(
            {
                "period_start": pd.Series(dtype="datetime64[ns]"),
                "period_end": pd.Series(dtype="datetime64[ns]"),
                "energy_kwh": pd.Series(dtype=float),
            }
        )

        fig = create_month_day_heatmap(empty_df)

        # Should return a figure with "No Solar Data Available" title
        assert isinstance(fig, go.Figure)
        assert fig.layout.height == 500

    def test_single_day_data(self):
        """Test with data for a single day."""
        periods_df = pd.DataFrame(
            {
                "period_start": pd.date_range("2023-06-15", periods=96, freq="15min"),
                "period_end": pd.date_range(
                    "2023-06-15 00:15:00", periods=96, freq="15min"
                ),
                "energy_kwh": [0.01] * 96,  # 0.96 kWh total
            }
        )

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
        dates_jan = pd.date_range("2023-01-01", "2023-01-31", freq="D")
        dates_feb = pd.date_range("2023-02-01", "2023-02-15", freq="D")

        periods_data = []
        for date in dates_jan[:5]:  # First 5 days of Jan
            periods_data.extend(
                [
                    {
                        "period_start": pd.Timestamp(f"{date.date()} 06:00:00"),
                        "period_end": pd.Timestamp(f"{date.date()} 18:00:00"),
                        "energy_kwh": 2.0,
                    }
                ]
            )

        for date in dates_feb[:3]:  # First 3 days of Feb
            periods_data.extend(
                [
                    {
                        "period_start": pd.Timestamp(f"{date.date()} 06:00:00"),
                        "period_end": pd.Timestamp(f"{date.date()} 18:00:00"),
                        "energy_kwh": 1.5,
                    }
                ]
            )

        periods_df = pd.DataFrame(periods_data)
        fig = create_month_day_heatmap(periods_df)

        heatmap = fig.data[0]
        # Should have months sorted newest first
        expected_months = ["2023-02", "2023-01"]
        assert list(heatmap.y) == expected_months

    def test_zero_energy_days(self):
        """Test that zero energy days are handled correctly (valid cloudy days)."""
        periods_df = pd.DataFrame(
            {
                "period_start": pd.date_range("2023-01-01", periods=96, freq="15min"),
                "period_end": pd.date_range(
                    "2023-01-01 00:15:00", periods=96, freq="15min"
                ),
                "energy_kwh": [0.0] * 96,  # Zero energy (cloudy day)
            }
        )

        fig = create_month_day_heatmap(periods_df)
        heatmap = fig.data[0]

        # Should have zmin=0 to include zero values in scale
        assert heatmap.zmin == 0

        # Check that zero values are preserved (not treated as missing)
        z_values = heatmap.z
        assert z_values[0][0] == 0.0  # January 1st should be 0.0

    def test_hover_text_format(self):
        """Test that hover text is formatted correctly."""
        periods_df = pd.DataFrame(
            {
                "period_start": pd.date_range("2023-01-01", periods=96, freq="15min"),
                "period_end": pd.date_range(
                    "2023-01-01 00:15:00", periods=96, freq="15min"
                ),
                "energy_kwh": [0.01] * 96,  # 0.96 kWh total
            }
        )

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
            periods_data.append(
                {
                    "period_start": pd.Timestamp(f"2023-01-01 {hour:02d}:00:00"),
                    "period_end": pd.Timestamp(f"2023-01-01 {hour:02d}:15:00"),
                    "energy_kwh": 0.1,
                }
            )

        # Day 3: full data
        for hour in range(6, 18):  # 6 AM to 6 PM
            periods_data.append(
                {
                    "period_start": pd.Timestamp(f"2023-01-03 {hour:02d}:00:00"),
                    "period_end": pd.Timestamp(f"2023-01-03 {hour:02d}:15:00"),
                    "energy_kwh": 0.1,
                }
            )

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


class TestCreateDayColumnChart:
    """Test the create_day_column_chart function."""

    def test_basic_functionality(self):
        """Test basic bar chart creation with sample data."""
        # Create sample periods data for one day with hourly production
        periods_data = []
        for hour in range(6, 18):  # 6 AM to 6 PM
            periods_data.append(
                {
                    "period_start": pd.Timestamp(f"2023-01-01 {hour:02d}:00:00"),
                    "period_end": pd.Timestamp(f"2023-01-01 {hour:02d}:15:00"),
                    "energy_kwh": 0.1,
                }
            )

        periods_df = pd.DataFrame(periods_data)
        fig = create_day_column_chart(periods_df, "2023-01-01")

        # Check return type
        assert isinstance(fig, go.Figure)

        # Check basic layout properties
        assert fig.layout.height == 400
        assert fig.layout.xaxis.title.text == "Hour of Day"
        assert fig.layout.yaxis.title.text == "Energy (kWh)"
        assert fig.layout.title.text == "Hourly Production - 2023-01-01"

        # Check bar chart properties
        bar = fig.data[0]
        assert isinstance(bar, go.Bar)
        assert bar.marker.color == "#FFB732"
        assert bar.hovertemplate == "<b>%{x}:00</b><br>%{y:.2f} kWh<extra></extra>"

        # Check x-axis range (should show all hours 0-23)
        assert fig.layout.xaxis.range == (-0.5, 23.5)

    def test_empty_dataframe(self):
        """Test handling of empty input data."""
        empty_df = pd.DataFrame()

        fig = create_day_column_chart(empty_df, "2023-01-01")

        # Should return a figure with "No Solar Data Available" title
        assert isinstance(fig, go.Figure)
        assert "No Solar Data Available" in fig.layout.title.text
        assert fig.layout.height == 400

    def test_no_data_for_date(self):
        """Test handling when no data exists for the selected date."""
        # Create data for a different date
        periods_data = [
            {
                "period_start": pd.Timestamp("2023-01-02 12:00:00"),
                "period_end": pd.Timestamp("2023-01-02 12:15:00"),
                "energy_kwh": 0.1,
            }
        ]

        periods_df = pd.DataFrame(periods_data)
        fig = create_day_column_chart(periods_df, "2023-01-01")  # Different date

        # Should still create a normal chart with all zeros
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Hourly Production - 2023-01-01"
        assert fig.layout.height == 400

        # Check that all values are zero
        bar = fig.data[0]
        assert all(y == 0.0 for y in bar.y)


class TestCreateDay15minHeatmap:
    """Test the create_day_15min_heatmap function."""

    def test_basic_functionality(self):
        """Test basic heatmap creation with sample data."""
        from lookout.core.solar_viz import create_day_15min_heatmap

        # Create sample periods data for a full 24-hour day
        periods_df = pd.DataFrame(
            {
                "period_start": pd.date_range(
                    "2023-01-01 00:00:00",
                    periods=96,
                    freq="15min",
                    tz="America/Los_Angeles",
                ),
                "period_end": pd.date_range(
                    "2023-01-01 00:15:00",
                    periods=96,
                    freq="15min",
                    tz="America/Los_Angeles",
                ),
                "energy_kwh": [
                    0.01 if 6 <= i // 4 <= 18 else 0.0 for i in range(96)
                ],  # Production only during daylight hours
            }
        )

        fig = create_day_15min_heatmap(periods_df)

        # Check return type
        assert isinstance(fig, go.Figure)

        # Check basic layout properties
        assert fig.layout.height == 500
        assert fig.layout.xaxis.title.text == "Time of Day"
        assert fig.layout.yaxis.title.text == "Date"
        assert fig.layout.yaxis.autorange == "reversed"  # Newest dates at top

        # Check heatmap properties
        heatmap = fig.data[0]
        assert isinstance(heatmap, go.Heatmap)
        assert heatmap.colorscale == (
            (0.0, "#FFF9E6"),
            (0.3, "#FFE680"),
            (0.6, "#FFB732"),
            (1.0, "#FF8C00"),
        )
        assert heatmap.zmin == 0
        assert heatmap.showscale is True
        assert heatmap.colorbar.title.text == "Wh/15min"
        assert heatmap.colorbar.ticksuffix == " Wh"

        # Default start_hour=0, end_hour=23 inclusive → full 24h in 15min intervals
        assert heatmap.z.shape == (1, 96)  # 24 hours * 4 slots
        assert len(heatmap.x) == 96
        assert len(heatmap.y) == 1  # 1 day

    def test_empty_dataframe(self):
        """Test handling of empty input data."""
        from lookout.core.solar_viz import create_day_15min_heatmap

        empty_df = pd.DataFrame(
            {
                "period_start": pd.Series(dtype="datetime64[ns]"),
                "period_end": pd.Series(dtype="datetime64[ns]"),
                "energy_kwh": pd.Series(dtype=float),
            }
        )

        fig = create_day_15min_heatmap(empty_df)

        # Should return a figure with "No Solar Production Data Available" title
        assert isinstance(fig, go.Figure)
        assert fig.layout.height == 500

    def test_zero_energy_data(self):
        """Test handling when all energy values are zero (but still shows heatmap)."""
        from lookout.core.solar_viz import create_day_15min_heatmap

        periods_df = pd.DataFrame(
            {
                "period_start": pd.date_range(
                    "2023-01-01 06:00:00",
                    periods=10,
                    freq="15min",
                    tz="America/Los_Angeles",
                ),
                "period_end": pd.date_range(
                    "2023-01-01 06:15:00",
                    periods=10,
                    freq="15min",
                    tz="America/Los_Angeles",
                ),
                "energy_kwh": [0.0] * 10,  # All zero energy
            }
        )

        fig = create_day_15min_heatmap(periods_df)

        # Should return heatmap with zero values (not empty figure)
        assert isinstance(fig, go.Figure)
        assert fig.layout.height == 500
        assert fig.layout.title.text == "15-Minute Energy Periods"

        heatmap = fig.data[0]
        assert isinstance(heatmap, go.Heatmap)
        # Should have all zero values
        assert all(val == 0.0 for val in heatmap.z.flatten())

    def test_partial_day_data(self):
        """Test handling when data covers only part of the day."""
        from lookout.core.solar_viz import create_day_15min_heatmap

        periods_df = pd.DataFrame(
            {
                "period_start": pd.date_range(
                    "2023-01-01 22:00:00",
                    periods=8,
                    freq="15min",
                    tz="America/Los_Angeles",
                ),
                "period_end": pd.date_range(
                    "2023-01-01 22:15:00",
                    periods=8,
                    freq="15min",
                    tz="America/Los_Angeles",
                ),
                "energy_kwh": [0.1] * 8,  # Production during night
            }
        )

        fig = create_day_15min_heatmap(periods_df)

        # Should return heatmap with data only in nighttime hours
        assert isinstance(fig, go.Figure)
        assert fig.layout.height == 500
        assert fig.layout.title.text == "15-Minute Energy Periods"

        heatmap = fig.data[0]
        assert isinstance(heatmap, go.Heatmap)
        # Should have 8 time slots with data, rest filled with zeros
        assert heatmap.z.shape == (1, 8)

    def test_custom_time_range(self):
        """Test with custom start_hour and end_hour parameters."""
        from lookout.core.solar_viz import create_day_15min_heatmap

        # Create full day data
        periods_df = pd.DataFrame(
            {
                "period_start": pd.date_range(
                    "2023-01-01 00:00:00",
                    periods=96,
                    freq="15min",
                    tz="America/Los_Angeles",
                ),
                "period_end": pd.date_range(
                    "2023-01-01 00:15:00",
                    periods=96,
                    freq="15min",
                    tz="America/Los_Angeles",
                ),
                "energy_kwh": [
                    0.01 if 6 <= i // 4 <= 18 else 0.0 for i in range(96)
                ],  # Production only during daylight
            }
        )

        fig = create_day_15min_heatmap(periods_df, start_hour=6, end_hour=18)

        # Check return type
        assert isinstance(fig, go.Figure)

        # Check title includes time range
        assert "(06:00-18:00)" in fig.layout.title.text

        # Check basic layout properties
        assert fig.layout.height == 500
        assert fig.layout.xaxis.title.text == "Time of Day"
        assert fig.layout.yaxis.title.text == "Date"
        assert fig.layout.yaxis.autorange == "reversed"

        # Check heatmap properties
        heatmap = fig.data[0]
        assert isinstance(heatmap, go.Heatmap)
        assert heatmap.colorscale == (
            (0.0, "#FFF9E6"),
            (0.3, "#FFE680"),
            (0.6, "#FFB732"),
            (1.0, "#FF8C00"),
        )
        assert heatmap.zmin == 0
        assert heatmap.showscale is True
        assert heatmap.colorbar.title.text == "Wh/15min"

        # Should have 1 day and 52 time slots (13 hours * 4 periods per hour)
        assert heatmap.z.shape == (1, 52)
        assert len(heatmap.x) == 52
        assert len(heatmap.y) == 1

        # Check axis labels are within the specified range
        tick_labels = fig.layout.xaxis.ticktext
        assert tick_labels[0] == "06:00"
        assert tick_labels[-1] == "18:00"

    def test_multiple_days(self):
        """Test with data spanning multiple days."""
        from lookout.core.solar_viz import create_day_15min_heatmap

        # Create data for 3 days with production only during daylight hours
        periods_data = []
        for day in range(3):
            # Only daylight hours (8:00-16:00)
            start_time = pd.Timestamp(
                f"2023-01-0{day+1} 08:00:00", tz="America/Los_Angeles"
            )
            periods = pd.date_range(
                start_time, periods=32, freq="15min"
            )  # 8 hours = 32 periods
            periods_data.extend(
                [
                    {
                        "period_start": period,
                        "period_end": period + pd.Timedelta(minutes=15),
                        "energy_kwh": 0.02,  # 20 Wh per 15min period
                    }
                    for period in periods
                ]
            )

        periods_df = pd.DataFrame(periods_data)

        fig = create_day_15min_heatmap(periods_df)
        assert isinstance(fig, go.Figure)

        heatmap = fig.data[0]
        # Should have 3 days and 32 time slots per day
        assert heatmap.z.shape == (3, 32)
        assert len(heatmap.y) == 3  # 3 days
        assert len(heatmap.x) == 32  # 32 time slots (08:00 to 15:45)


class TestCreate15minBarChart:
    """Test the create_15min_bar_chart function."""

    def test_basic_functionality(self):
        """Test basic bar chart creation with sample data."""
        # Create sample periods data for one day with 15-minute intervals
        periods_df = pd.DataFrame(
            {
                "period_start": pd.date_range(
                    "2023-01-01 06:00:00", periods=48, freq="15min"
                ),
                "period_end": pd.date_range(
                    "2023-01-01 06:15:00", periods=48, freq="15min"
                ),
                "energy_kwh": [0.01] * 48,  # 10 Wh per 15min period
            }
        )
        fig = create_15min_bar_chart(periods_df, "2023-01-01")

        # Check return type
        assert isinstance(fig, go.Figure)

        # Check basic layout properties
        assert fig.layout.height == 400
        assert fig.layout.xaxis.title.text == "Time"
        assert fig.layout.yaxis.title.text == "Energy (Wh)"
        assert fig.layout.title.text == "15-Minute Periods - 2023-01-01"

        # Check bar chart properties
        bar = fig.data[0]
        assert isinstance(bar, go.Bar)
        assert bar.marker.color == "#FFB732"
        assert bar.hovertemplate == "<b>%{x}</b><br>%{y:.1f} Wh<extra></extra>"

        # Should have 48 periods (12 hours × 4 periods per hour)
        assert len(bar.x) == 48
        assert len(bar.y) == 48

        # Check that all values are converted to Wh (10 Wh per period)
        assert all(y == 10.0 for y in bar.y)

        # Check time labels are in HH:MM format
        assert bar.x[0] == "06:00"
        assert bar.x[-1] == "17:45"

    def test_empty_dataframe(self):
        """Test handling of empty input data."""
        empty_df = pd.DataFrame()

        fig = create_15min_bar_chart(empty_df, "2023-01-01")

        # Should return a figure with "No Solar Data Available" title
        assert isinstance(fig, go.Figure)
        assert "No Solar Data Available" in fig.layout.title.text
        assert fig.layout.height == 400

    def test_no_data_for_date(self):
        """Test handling when no data exists for the selected date."""
        # Create data for a different date
        periods_data = [
            {
                "period_start": pd.Timestamp("2023-01-02 12:00:00"),
                "period_end": pd.Timestamp("2023-01-02 12:15:00"),
                "energy_kwh": 0.1,
            }
        ]

        periods_df = pd.DataFrame(periods_data)
        fig = create_15min_bar_chart(periods_df, "2023-01-01")  # Different date

        # Should still create a normal chart with no-data message
        assert isinstance(fig, go.Figure)
        assert "No Solar Data Available" in fig.layout.title.text
        assert fig.layout.height == 400

    def test_zero_energy_periods(self):
        """Test that zero energy periods are included in the chart."""
        # Create data with some zero energy periods
        periods_data = [
            {
                "period_start": pd.Timestamp("2023-01-01 06:00:00"),
                "period_end": pd.Timestamp("2023-01-01 06:15:00"),
                "energy_kwh": 0.1,  # 100 Wh
            },
            {
                "period_start": pd.Timestamp("2023-01-01 06:15:00"),
                "period_end": pd.Timestamp("2023-01-01 06:30:00"),
                "energy_kwh": 0.0,  # Zero energy
            },
            {
                "period_start": pd.Timestamp("2023-01-01 06:30:00"),
                "period_end": pd.Timestamp("2023-01-01 06:45:00"),
                "energy_kwh": 0.05,  # 50 Wh
            },
        ]

        periods_df = pd.DataFrame(periods_data)
        fig = create_15min_bar_chart(periods_df, "2023-01-01")

        bar = fig.data[0]
        # Should have all 3 periods including the zero
        assert len(bar.y) == 3
        assert bar.y[0] == 100.0  # 0.1 kWh = 100 Wh
        assert bar.y[1] == 0.0  # Zero energy
        assert bar.y[2] == 50.0  # 0.05 kWh = 50 Wh

    def test_custom_time_range(self):
        """Test with custom start_hour and end_hour parameters."""
        # Create full day data
        periods_data = []
        for hour in range(24):  # Full day
            periods_data.append(
                {
                    "period_start": pd.Timestamp(f"2023-01-01 {hour:02d}:00:00"),
                    "period_end": pd.Timestamp(f"2023-01-01 {hour:02d}:15:00"),
                    "energy_kwh": 0.01,
                }
            )

        periods_df = pd.DataFrame(periods_data)
        fig = create_15min_bar_chart(
            periods_df, "2023-01-01", start_hour=6, end_hour=18
        )

        # Check title includes time range
        assert "(06:00-18:00)" in fig.layout.title.text

        bar = fig.data[0]
        # Should have 13 periods (hours 6-18 inclusive = 13 hours)
        assert len(bar.y) == 13

        # Check time labels are within the specified range
        assert bar.x[0] == "06:00"
        assert bar.x[-1] == "18:00"

    def test_invalid_date_format(self):
        """Test handling of invalid date format."""
        periods_df = pd.DataFrame(
            {
                "period_start": [pd.Timestamp("2023-01-01 12:00:00")],
                "period_end": [pd.Timestamp("2023-01-01 12:15:00")],
                "energy_kwh": [0.1],
            }
        )

        # Should raise ValueError for invalid date format
        with pytest.raises(ValueError, match="Invalid date format"):
            create_15min_bar_chart(periods_df, "invalid-date")




class TestSolarUIRender:
    """Test the solar UI render function against the current energy_catalog data model."""

    def test_no_energy_catalog_in_session(self):
        """If 'energy_catalog' isn't in session_state, render() must st.error and return."""
        from lookout.ui.solar import render

        mock_session_state = MagicMock()
        mock_session_state.__contains__.return_value = False

        with patch("lookout.ui.solar.st.session_state", mock_session_state), patch(
            "lookout.ui.solar.st.header"
        ), patch("lookout.ui.solar.st.caption"), patch(
            "lookout.ui.solar.st.error"
        ) as mock_error:
            render()

        mock_error.assert_called_once_with(
            "Solar energy catalog not available. Please refresh the page."
        )

    def test_empty_energy_catalog_warns(self):
        """If energy_catalog exists but is empty, render() must warn and return."""
        from lookout.ui.solar import render

        mock_session_state = MagicMock()
        mock_session_state.__contains__.return_value = True
        mock_session_state.get.side_effect = lambda key, default=None: (
            pd.DataFrame() if key == "energy_catalog" else default
        )

        with patch("lookout.ui.solar.st.session_state", mock_session_state), patch(
            "lookout.ui.solar.st.header"
        ), patch("lookout.ui.solar.st.caption"), patch(
            "lookout.ui.solar.st.warning"
        ) as mock_warning:
            render()

        mock_warning.assert_called_once_with("No solar data available")

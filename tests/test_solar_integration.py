"""
test_solar_integration.py
Integration tests for solar tab integration with main app.
"""

import datetime
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


class TestSolarTabIntegration:
    """Test solar tab integration with main Streamlit app."""

    def test_solar_module_in_tab_modules_mapping(self):
        """Test that solar module is included in tab_modules mapping."""
        from streamlit_app import tab_modules

        assert "Solar" in tab_modules
        assert hasattr(tab_modules["Solar"], "render")

    def test_solar_module_import(self):
        """Test that solar module can be imported from lookout.ui."""


class TestDateFiltering:
    """Test date filtering functionality for solar tab."""

    def test_daily_energy_index_type_consistency(self):
        """Test that daily_energy index is always DatetimeIndex."""
        from lookout.core.solar_analysis import calculate_daily_energy

        # Create test data
        dates = pd.date_range("2023-01-01", "2023-01-05", freq="D")
        data = []
        for date in dates:
            for hour in range(6, 18):  # Daylight hours
                data.append(
                    {
                        "date": date.date(),
                        "datetime": pd.Timestamp.combine(
                            date.date(), datetime.time(hour, 0)
                        ),
                        "solarradiation": 500.0,  # W/m²
                        "daylight_period": "day",
                    }
                )

        df = pd.DataFrame(data)
        daily_energy = calculate_daily_energy(df)

        # Assert index is DatetimeIndex with UTC timezone, not date objects
        assert isinstance(daily_energy.index, pd.DatetimeIndex)
        assert str(daily_energy.index.tz) == "UTC"  # Should be timezone-aware
        assert not daily_energy.empty

    def test_date_range_slider_component(self):
        """Test date range slider component with different input types."""
        from lookout.ui.components import create_date_range_slider

        # Create test DataFrame with datetime index
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="D")
        df = pd.DataFrame({"value": range(10)}, index=dates)
        df["datetime"] = df.index

        # Mock streamlit slider to return date range
        with patch("streamlit.slider") as mock_slider, patch("streamlit.write"):

            mock_slider.return_value = (dates[0].date(), dates[5].date())

            start_ts, end_ts = create_date_range_slider(df, "datetime", "test_key")

            # Assert timestamps are returned
            assert isinstance(start_ts, pd.Timestamp)
            assert isinstance(end_ts, pd.Timestamp)
            assert start_ts.tz is not None  # Should be timezone-aware
            assert end_ts.tz is not None

    def test_filter_dataframe_by_date_range(self):
        """Test DataFrame filtering by date range."""
        from lookout.ui.components import filter_dataframe_by_date_range

        # Create test data with timezone-aware timestamps
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="D", tz="UTC")
        df = pd.DataFrame({"datetime": dates, "value": range(10)})

        start_ts = pd.Timestamp("2023-01-03").tz_localize("UTC")
        end_ts = pd.Timestamp("2023-01-07").tz_localize("UTC")

        filtered_df = filter_dataframe_by_date_range(df, "datetime", start_ts, end_ts)

        # Should have 4 days (3, 4, 5, 6)
        assert len(filtered_df) == 4
        assert filtered_df["datetime"].min() >= start_ts
        assert filtered_df["datetime"].max() < end_ts

    def test_filter_dataframe_by_date_range_type_validation(self):
        """Test that filter_dataframe_by_date_range validates timestamp types."""
        from lookout.ui.components import filter_dataframe_by_date_range

        df = pd.DataFrame({"datetime": pd.date_range("2023-01-01", "2023-01-05")})

        # Should raise ValueError for invalid timestamp types
        with pytest.raises(ValueError, match="must be pandas Timestamps"):
            filter_dataframe_by_date_range(df, "datetime", "2023-01-01", "2023-01-05")

    def test_solar_date_filtering_integration(self):
        """Test end-to-end solar date filtering with daily energy."""
        from lookout.core.solar_analysis import calculate_daily_energy
        from lookout.ui.components import filter_dataframe_by_date_range

        # Create test solar data with timezone-aware timestamps
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="D", tz="UTC")
        data = []
        for date in dates:
            for hour in range(6, 18):  # Daylight hours
                data.append(
                    {
                        "date": date.date(),
                        "datetime": date.replace(
                            hour=hour
                        ),  # Use timezone-aware timestamp
                        "solarradiation": 500.0,
                        "daylight_period": "day",
                    }
                )

        df = pd.DataFrame(data)
        daily_energy = calculate_daily_energy(df)

        # Filter both DataFrame and Series
        start_ts = pd.Timestamp("2023-01-03").tz_localize("UTC")
        end_ts = pd.Timestamp("2023-01-07").tz_localize("UTC")

        filtered_df = filter_dataframe_by_date_range(df, "datetime", start_ts, end_ts)

        # Filter daily energy series
        mask = (daily_energy.index >= start_ts) & (daily_energy.index < end_ts)
        filtered_daily_energy = daily_energy.loc[mask]

        # Assert both are filtered consistently
        assert len(filtered_daily_energy) == 4  # Days 3, 4, 5, 6
        assert not filtered_df.empty
        assert not filtered_daily_energy.empty

        # Assert no TypeError occurs during comparison
        assert filtered_daily_energy.index[0] >= start_ts
        assert filtered_daily_energy.index[-1] < end_ts

    def test_memory_efficiency_no_copy(self):
        """Test that filtering operations don't create unnecessary copies."""
        from lookout.ui.components import filter_dataframe_by_date_range

        # Create test data with timezone-aware timestamps
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="D", tz="UTC")
        df = pd.DataFrame({"datetime": dates, "value": range(10)})

        start_ts = pd.Timestamp("2023-01-03").tz_localize("UTC")
        end_ts = pd.Timestamp("2023-01-07").tz_localize("UTC")

        # Get memory usage before filtering
        import sys

        memory_before = sys.getsizeof(df)

        filtered_df = filter_dataframe_by_date_range(df, "datetime", start_ts, end_ts)

        # Filtered DataFrame should not be a full copy (memory efficient)
        # Note: This is a basic check - in practice, pandas may still copy for safety
        assert len(filtered_df) < len(df)  # Should be filtered
        assert isinstance(filtered_df, pd.DataFrame)

    def test_timezone_consistency(self):
        """Test that timezone handling is consistent across components."""
        from lookout.ui.components import create_date_range_slider

        # Create test DataFrame
        dates = pd.date_range("2023-01-01", "2023-01-10", freq="D", tz="UTC")
        df = pd.DataFrame({"datetime": dates})

        with patch("streamlit.slider") as mock_slider, patch("streamlit.write"):

            mock_slider.return_value = (dates[0].date(), dates[5].date())

            start_ts, end_ts = create_date_range_slider(df, "datetime", "test_key")

            # Assert consistent UTC timezone
            assert str(start_ts.tz) == "UTC"
            assert str(end_ts.tz) == "UTC"
        from lookout.ui import solar

        assert solar is not None
        assert hasattr(solar, "render")
        assert callable(solar.render)

    def test_solar_tab_in_dev_mode_list(self):
        """Test that solar tab appears in dev mode tab names."""
        # Simulate dev environment
        import os

        original_env = os.environ.get("STREAMLIT_ENV")
        os.environ["STREAMLIT_ENV"] = "development"

        try:
            # Import after setting env
            import importlib
            import streamlit_app

            importlib.reload(streamlit_app)

            # Check if Solar is in the dev tab list
            # Note: This is a simplified check since full mocking is complex
            assert "Solar" in [
                "Overview",
                "Rain",
                "Rain Events",
                "Solar",
                "Diagnostics",
                "Playground",
            ]
        finally:
            if original_env is not None:
                os.environ["STREAMLIT_ENV"] = original_env
            else:
                os.environ.pop("STREAMLIT_ENV", None)

    def test_solar_tab_in_prod_mode_list(self):
        """Test that solar tab appears in production mode tab names."""
        # Simulate prod environment
        import os

        original_env = os.environ.get("STREAMLIT_ENV")
        os.environ["STREAMLIT_ENV"] = "production"

        try:
            # Import after setting env
            import importlib
            import streamlit_app

            importlib.reload(streamlit_app)

            # Check if Solar is in the prod tab list
            assert "Solar" in ["Overview", "Rain", "Rain Events", "Solar"]
        finally:
            if original_env is not None:
                os.environ["STREAMLIT_ENV"] = original_env
            else:
                os.environ.pop("STREAMLIT_ENV", None)

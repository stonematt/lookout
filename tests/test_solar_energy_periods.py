"""
Tests for solar_energy_periods module
"""

import pandas as pd
import pytest
from datetime import datetime
import pytz

from lookout.core.solar_energy_periods import calculate_15min_energy_periods


class TestCalculate15MinEnergyPeriods:
    """Test the calculate_15min_energy_periods function."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=["datetime", "solarradiation"])
        result = calculate_15min_energy_periods(df)

        assert result.empty
        assert list(result.columns) == ["period_start", "period_end", "energy_kwh"]

    def test_missing_required_columns(self):
        """Test error handling for missing required columns."""
        df = pd.DataFrame({"datetime": [], "other_col": []})

        with pytest.raises(ValueError, match="Missing required columns"):
            calculate_15min_energy_periods(df)

    def test_basic_functionality(self):
        """Test basic functionality with sample solar data."""
        # Create test data with multiple observations per 15-minute period
        pacific = pytz.timezone("America/Los_Angeles")
        base_time = pacific.localize(datetime(2023, 6, 15, 6, 0, 0))  # Summer morning

        data = []
        for period_idx in range(6):  # 6 periods over 1.5 hours
            period_start = base_time + pd.Timedelta(minutes=period_idx * 15)
            # Add 2 observations per period
            for obs_idx in range(2):
                obs_time = period_start + pd.Timedelta(
                    minutes=obs_idx * 7
                )  # 0 and 7 minutes into period
                solar_rad = max(
                    0, 500 * (1 - abs(period_idx - 2) / 3)
                )  # Peak in middle periods
                data.append({"datetime": obs_time, "solarradiation": solar_rad})

        df = pd.DataFrame(data)
        result = calculate_15min_energy_periods(df)

        # Should have 6 periods
        assert len(result) == 6
        assert list(result.columns) == ["period_start", "period_end", "energy_kwh"]

        # Check period alignment (should be aligned to 15-minute boundaries)
        for i, row in result.iterrows():
            expected_start = base_time + pd.Timedelta(minutes=i * 15)
            expected_end = expected_start + pd.Timedelta(minutes=15)
            assert row["period_start"] == expected_start
            assert row["period_end"] == expected_end

        # Some periods should have positive energy (middle periods with solar radiation)
        daylight_periods = result[result["energy_kwh"] > 0]
        assert len(daylight_periods) > 0

    def test_nighttime_periods_included(self):
        """Test that nighttime periods with zero energy are included."""
        pacific = pytz.timezone("America/Los_Angeles")
        base_time = pacific.localize(datetime(2023, 6, 15, 0, 0, 0))  # Midnight

        # Create data for 24 hours with no solar radiation (nighttime)
        data = []
        for i in range(96):  # 96 observations over 24 hours
            data.append(
                {
                    "datetime": base_time + pd.Timedelta(minutes=i * 15),
                    "solarradiation": 0.0,
                }
            )

        df = pd.DataFrame(data)
        result = calculate_15min_energy_periods(df)

        # Should have 96 periods (24 hours * 4 periods per hour)
        assert len(result) == 96

        # All periods should have zero energy
        assert (result["energy_kwh"] == 0.0).all()

    def test_partial_periods(self):
        """Test handling of partial periods with limited data."""
        pacific = pytz.timezone("America/Los_Angeles")
        base_time = pacific.localize(datetime(2023, 6, 15, 12, 0, 0))

        # Create data with only 2 observations in a period
        data = [
            {"datetime": base_time, "solarradiation": 500.0},
            {"datetime": base_time + pd.Timedelta(minutes=5), "solarradiation": 600.0},
        ]

        df = pd.DataFrame(data)
        result = calculate_15min_energy_periods(df)

        # Should have 1 period
        assert len(result) == 1
        assert result["energy_kwh"].iloc[0] > 0  # Should have calculated energy

    def test_single_observation_period(self):
        """Test periods with only single observation (should have zero energy)."""
        pacific = pytz.timezone("America/Los_Angeles")
        base_time = pacific.localize(datetime(2023, 6, 15, 12, 0, 0))

        # Create data with only 1 observation in a period
        data = [{"datetime": base_time, "solarradiation": 500.0}]

        df = pd.DataFrame(data)
        result = calculate_15min_energy_periods(df)

        # Should have 1 period with zero energy
        assert len(result) == 1
        assert result["energy_kwh"].iloc[0] == 0.0

    def test_timezone_preservation(self):
        """Test that timezone information is preserved in output."""
        pacific = pytz.timezone("America/Los_Angeles")
        base_time = pacific.localize(datetime(2023, 6, 15, 12, 0, 0))

        data = [
            {"datetime": base_time, "solarradiation": 500.0},
            {"datetime": base_time + pd.Timedelta(minutes=15), "solarradiation": 600.0},
        ]

        df = pd.DataFrame(data)
        result = calculate_15min_energy_periods(df)

        # Check timezone preservation
        assert result["period_start"].dt.tz is not None
        assert result["period_end"].dt.tz is not None
        assert str(result["period_start"].dt.tz) == "America/Los_Angeles"

    def test_energy_calculation_accuracy(self):
        """Test that energy calculation matches expected trapezoidal integration."""
        pacific = pytz.timezone("America/Los_Angeles")
        base_time = pacific.localize(datetime(2023, 6, 15, 12, 0, 0))

        # Simple case: constant radiation over 15 minutes
        # Two observations in the same 15-minute period: 500 W/m² at t=0 and t=10min
        # Expected energy: (500 + 500) / 2 * (10/60) hours = 500 * (10/60) = 500 * 0.1667 ≈ 83.33 Wh/m² = 0.0833 kWh/m²
        data = [
            {"datetime": base_time, "solarradiation": 500.0},
            {"datetime": base_time + pd.Timedelta(minutes=10), "solarradiation": 500.0},
        ]

        df = pd.DataFrame(data)
        result = calculate_15min_energy_periods(df)

        expected_energy = 500 * 0.1667 / 1000  # kWh/m²
        assert len(result) == 1  # Should have one period
        assert (
            abs(result["energy_kwh"].iloc[0] - expected_energy) < 1e-3
        )  # Allow small tolerance

    def test_multiple_periods_with_gaps(self):
        """Test handling of multiple periods with gaps between them."""
        pacific = pytz.timezone("America/Los_Angeles")
        base_time = pacific.localize(datetime(2023, 6, 15, 12, 0, 0))

        # Create data for four separate periods (each observation starts its own 15-min period)
        data = [
            # Period 1: 12:00-12:15 (observation at 12:00)
            {"datetime": base_time, "solarradiation": 500.0},
            # Period 2: 12:15-12:30 (observation at 12:15)
            {"datetime": base_time + pd.Timedelta(minutes=15), "solarradiation": 500.0},
            # Period 3: 13:00-13:15 (observation at 13:00)
            {"datetime": base_time + pd.Timedelta(hours=1), "solarradiation": 600.0},
            # Period 4: 13:15-13:30 (observation at 13:15)
            {
                "datetime": base_time + pd.Timedelta(hours=1, minutes=15),
                "solarradiation": 600.0,
            },
        ]

        df = pd.DataFrame(data)
        result = calculate_15min_energy_periods(df)

        # Should have 4 periods (each single observation creates a period with 0 energy)
        assert len(result) == 4

        # All periods should have zero energy (single observations)
        assert (result["energy_kwh"] == 0.0).all()

        # Check period timing
        assert result["period_start"].iloc[0] == base_time
        assert result["period_start"].iloc[1] == base_time + pd.Timedelta(minutes=15)
        assert result["period_start"].iloc[2] == base_time + pd.Timedelta(hours=1)
        assert result["period_start"].iloc[3] == base_time + pd.Timedelta(
            hours=1, minutes=15
        )

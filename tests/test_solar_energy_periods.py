"""
Tests for solar_energy_periods module
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
import pytz

from lookout.core.solar_energy_periods import (
    calculate_15min_energy_periods,
)
from tests.fixtures.archive import build_archive


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
        positive_energy_periods = result[result["energy_kwh"] > 0]
        assert len(positive_energy_periods) > 0

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

        # Create data for four periods (each observation starts its own 15-min period)
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

        # Should have 4 periods, one per observation
        assert len(result) == 4

        # 12:00 -> 12:15 is one period of 500 W/m² (0.125 kWh/m²) and
        # 13:00 -> 13:15 one of 600 (0.15). The 45 minutes between 12:15 and
        # 13:00 are a gap, and nothing follows 13:15, so those read 0.
        assert list(result["energy_kwh"].round(6)) == [0.125, 0.0, 0.15, 0.0]

        # Check period timing
        assert result["period_start"].iloc[0] == base_time
        assert result["period_start"].iloc[1] == base_time + pd.Timedelta(minutes=15)
        assert result["period_start"].iloc[2] == base_time + pd.Timedelta(hours=1)
        assert result["period_start"].iloc[3] == base_time + pd.Timedelta(
            hours=1, minutes=15
        )


class TestEnergyAcrossPeriodBoundaries:
    """
    A day's periods add up to the integral of the whole curve: the interval
    between the last row of one period and the first row of the next is
    counted, split at the boundary between them.
    """

    @staticmethod
    def rows(times, watts) -> pd.DataFrame:
        """Rows at local station times with the given W/m²."""
        return pd.DataFrame(
            {
                "datetime": pd.DatetimeIndex(times).tz_localize("America/Los_Angeles"),
                "solarradiation": watts,
            }
        )

    def test_a_day_of_five_minute_rows_sums_to_the_whole_curve(self):
        # 0 W/m² at 06:00 rising straight to 1000 at 12:00 and back to 0 at
        # 18:00: a triangle of 12 h x 1000 W/m² / 2 = 6.0 kWh/m², which the
        # trapezoid rule reproduces exactly for a piecewise-linear curve.
        times = pd.date_range("2026-06-01 00:00", "2026-06-01 23:55", freq="5min")
        hours = times.hour + times.minute / 60
        watts = [max(0.0, 1000 * (1 - abs(h - 12) / 6)) for h in hours]

        result = calculate_15min_energy_periods(self.rows(times, watts))

        assert len(result) == 96
        assert result["energy_kwh"].sum() == pytest.approx(6.0)

    def test_a_constant_sun_reads_its_exact_energy(self):
        # 400 W/m² from 06:00 to 18:00 is 400 x 12 Wh/m² = 4.8 kWh/m².
        times = pd.date_range("2026-06-01 06:00", "2026-06-01 18:00", freq="5min")

        result = calculate_15min_energy_periods(self.rows(times, 400.0))

        assert result["energy_kwh"].sum() == pytest.approx(4.8)
        # Every whole period holds its 15 minutes; 18:00 holds no interval
        assert list(result["energy_kwh"].round(6).unique()) == [0.1, 0.0]

    def test_a_day_with_an_extra_hour_reads_it(self):
        # 2026-11-01 falls back: 25 local hours, 300 five-minute rows and 100
        # periods.
        # 100 W/m² for 25 hours is 2.5 kWh/m².
        times = pd.date_range(
            "2026-11-01 00:00",
            "2026-11-02 00:00",
            freq="5min",
            tz="America/Los_Angeles",
        )
        rows = pd.DataFrame({"datetime": times, "solarradiation": 100.0})
        end = times[-1]

        result = calculate_15min_energy_periods(rows)
        day = result[result["period_start"] < end]

        assert len(day) == 100
        assert day["energy_kwh"].sum() == pytest.approx(2.5)

    def test_the_synthetic_station_day_reads_its_sun_curve(self):
        # The shared fixture's sun is 800 W/m² x sin over 06:00-18:00, which
        # integrates to 800 x 12 x 2 / pi Wh/m² = 6.11 kWh/m².
        archive = build_archive("2026-06-01 07:00", "2026-06-02 07:00")
        day = archive.assign(datetime=archive["date"])

        result = calculate_15min_energy_periods(day)

        assert round(result["energy_kwh"].sum(), 2) == 6.11
        # ... and equals the trapezoid of the whole day's curve
        curve = day.sort_values("datetime")
        hours = (curve["datetime"] - curve["datetime"].iloc[0]).dt.total_seconds()
        whole_kwh = np.trapezoid(curve["solarradiation"], hours / 3600) / 1000
        assert result["energy_kwh"].sum() == pytest.approx(whole_kwh)

    def test_the_boundary_interval_is_split_at_the_boundary(self):
        # 600 W/m² at 12:10 rising to 1200 at 12:20: 750 over 12:10-12:15
        # (0.0625 kWh/m²) and 1050 over 12:15-12:20 (0.0875 kWh/m²).
        result = calculate_15min_energy_periods(
            self.rows(["2026-06-01 12:10", "2026-06-01 12:20"], [600.0, 1200.0])
        )

        assert list(result["period_start"].dt.strftime("%H:%M")) == ["12:00", "12:15"]
        assert list(result["energy_kwh"].round(6)) == [0.0625, 0.0875]

    def test_a_row_on_the_boundary_closes_the_period_before_it(self):
        # 12:10 -> 12:15 lies wholly in 12:00-12:15: 600 W/m² x 5 min.
        result = calculate_15min_energy_periods(
            self.rows(["2026-06-01 12:10", "2026-06-01 12:15"], [600.0, 600.0])
        )

        assert list(result["energy_kwh"].round(6)) == [0.05, 0.0]

    def test_an_interval_longer_than_a_period_is_not_integrated(self):
        # Rows 20 minutes apart: what the sun did between them is not known.
        result = calculate_15min_energy_periods(
            self.rows(["2026-06-01 12:10", "2026-06-01 12:30"], [600.0, 600.0])
        )

        assert list(result["energy_kwh"]) == [0.0, 0.0]

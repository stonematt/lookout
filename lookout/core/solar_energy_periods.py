"""
Solar Energy Period Calculations
Converts raw solar radiation measurements into 15-minute energy periods.
"""

import pandas as pd
from typing import Dict
from lookout.core.solar_analysis import LOCATION
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def calculate_15min_energy_periods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate energy production for 15-minute periods using trapezoidal integration.

    :param df: DataFrame with dateutc, datetime, date, and solarradiation columns
               - datetime: TZ-aware datetime column in US/Pacific (already converted)
               - solarradiation: Solar radiation in W/m²
    :return: DataFrame with period_start, period_end, energy_kwh columns
             - period_start: TZ-aware datetime in US/Pacific (start of 15min period)
             - period_end: TZ-aware datetime in US/Pacific (end of 15min period)
             - energy_kwh: Energy accumulated in kWh (can be 0.0 for nighttime periods)
    """
    # Input validation
    required_cols = ["datetime", "solarradiation"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        empty_df = pd.DataFrame({
            "period_start": pd.Series(dtype="datetime64[ns, America/Los_Angeles]"),
            "period_end": pd.Series(dtype="datetime64[ns, America/Los_Angeles]"),
            "energy_kwh": pd.Series(dtype=float)
        })
        return empty_df

    # Sort by datetime ascending (archive is reverse-sorted)
    df_sorted = df.sort_values("datetime").copy()

    if df_sorted.empty:
        empty_df = pd.DataFrame({
            "period_start": pd.Series(dtype="datetime64[ns, America/Los_Angeles]"),
            "period_end": pd.Series(dtype="datetime64[ns, America/Los_Angeles]"),
            "energy_kwh": pd.Series(dtype=float)
        })
        return empty_df

    # Create 15-minute period buckets aligned to clock times
    df_sorted["period_start"] = df_sorted["datetime"].dt.floor("15min")
    df_sorted["period_end"] = df_sorted["period_start"] + pd.Timedelta(minutes=15)

    # Calculate energy for each period using trapezoidal integration
    # Group by period_start and calculate energy for each group
    period_energy_dict = {}

    grouped = df_sorted.groupby("period_start")
    for period_start_val, group in grouped:
        # Convert group to DataFrame and sort
        period_df = pd.DataFrame(group)
        period_df = period_df.sort_values("datetime")

        if len(period_df) < 2:
            period_energy_dict[period_start_val] = 0.0
            continue

        # Time differences in hours
        time_diffs = period_df["datetime"].diff().dt.total_seconds() / 3600

        # Trapezoidal rule: (y1 + y2) / 2 * dx
        # Convert W/m² to Wh/m² by multiplying by time in hours
        energy_wh = (
            (period_df["solarradiation"].shift(1) + period_df["solarradiation"]) / 2 * time_diffs
        ).sum()

        # Convert Wh/m² to kWh/m²
        period_energy_dict[period_start_val] = energy_wh / 1000

    # Create result DataFrame
    energy_series = pd.Series(period_energy_dict)
    result_df = energy_series.reset_index()
    result_df.columns = ["period_start", "energy_kwh"]
    result_df["period_end"] = result_df["period_start"] + pd.Timedelta(minutes=15)

    # Reorder columns to match expected output
    result_df = result_df[["period_start", "period_end", "energy_kwh"]]

    # Sort by period_start
    result_df = result_df.sort_values("period_start").reset_index(drop=True)

    return result_df


def aggregate_to_daily(periods_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 15-minute periods to daily totals.

    TODO: Implement in Phase 1 - Epic 1.2
    """
    raise NotImplementedError("Phase 1 - Epic 1.2 in progress")


def aggregate_to_hourly(periods_df: pd.DataFrame, date: str) -> pd.DataFrame:
    """
    Aggregate 15-minute periods to hourly for a specific date.

    TODO: Implement in Phase 1 - Epic 1.2
    """
    raise NotImplementedError("Phase 1 - Epic 1.2 in progress")


def get_period_stats(periods_df: pd.DataFrame) -> Dict:
    """
    Calculate summary statistics for energy periods.

    Returns:
        Dict with keys: total_kwh, days_with_production, avg_daily_kwh, peak_day

    TODO: Implement in Phase 1 - Epic 1.2
    """
    raise NotImplementedError("Phase 1 - Epic 1.2 in progress")

"""
Solar Energy Period Calculations
Converts raw solar radiation measurements into 15-minute energy periods.
"""

import pandas as pd
import pytz
import streamlit as st
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
        logger.info("No solar data provided for period calculation")
        empty_df = pd.DataFrame(
            {
                "period_start": pd.Series(dtype="datetime64[ns, America/Los_Angeles]"),
                "period_end": pd.Series(dtype="datetime64[ns, America/Los_Angeles]"),
                "energy_kwh": pd.Series(dtype=float),
            }
        )
        return empty_df

    # Sort by datetime ascending (archive is reverse-sorted)
    df_sorted = df.sort_values("datetime").copy()

    if df_sorted.empty:
        logger.info("No solar data after sorting for period calculation")
        empty_df = pd.DataFrame(
            {
                "period_start": pd.Series(dtype="datetime64[ns, America/Los_Angeles]"),
                "period_end": pd.Series(dtype="datetime64[ns, America/Los_Angeles]"),
                "energy_kwh": pd.Series(dtype=float),
            }
        )
        return empty_df

    logger.debug(
        f"Processing {len(df_sorted)} solar radiation records for 15-minute periods"
    )

    # Create 15-minute period buckets aligned to clock times
    # Convert to UTC to avoid DST issues, floor, then convert back to Pacific
    utc_dt = df_sorted["datetime"].dt.tz_convert("UTC")
    utc_floored = utc_dt.dt.floor("15min")
    df_sorted["period_start"] = utc_floored.dt.tz_convert("America/Los_Angeles")

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
            (period_df["solarradiation"].shift(1) + period_df["solarradiation"])
            / 2
            * time_diffs
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

    logger.info(f"Generated {len(result_df)} 15-minute solar energy periods")
    return result_df


@st.cache_data(show_spinner=False, max_entries=20, ttl=7200)
def calculate_15min_energy_periods_cached(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized version of calculate_15min_energy_periods with vectorized operations and caching.

    Uses pandas resample() for O(n) vectorized processing instead of O(n²) groupby iteration.
    Includes Streamlit caching to avoid recalculation on every page load.

    :param df: DataFrame with datetime and solarradiation columns
    :return: DataFrame with period_start, period_end, energy_kwh columns
    """
    # Input validation
    required_cols = ["datetime", "solarradiation"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        logger.info("No solar data provided for cached period calculation")
        empty_df = pd.DataFrame(
            {
                "period_start": pd.Series(dtype="datetime64[ns, America/Los_Angeles]"),
                "period_end": pd.Series(dtype="datetime64[ns, America/Los_Angeles]"),
                "energy_kwh": pd.Series(dtype=float),
            }
        )
        return empty_df

    # Sort by datetime ascending (archive is reverse-sorted)
    df_sorted = df.sort_values("datetime").copy()

    if df_sorted.empty:
        logger.info("No solar data after sorting for cached period calculation")
        empty_df = pd.DataFrame(
            {
                "period_start": pd.Series(dtype="datetime64[ns, America/Los_Angeles]"),
                "period_end": pd.Series(dtype="datetime64[ns, America/Los_Angeles]"),
                "energy_kwh": pd.Series(dtype=float),
            }
        )
        return empty_df

    logger.debug(
        f"Processing {len(df_sorted)} solar radiation records for cached 15-minute periods"
    )

    # Set datetime as index for resampling
    df_indexed = df_sorted.set_index("datetime")

    # Resample to 15-minute periods and calculate energy using vectorized trapezoidal integration
    def vectorized_trapezoidal_integration(group):
        """Vectorized trapezoidal integration for a 15-minute period"""
        if len(group) < 2:
            return 0.0

        # Time differences in hours (vectorized)
        time_diffs = group.index.to_series().diff().dt.total_seconds() / 3600

        # Trapezoidal rule: (y1 + y2) / 2 * dx (vectorized)
        energy_wh = (
            (group["solarradiation"].shift(1) + group["solarradiation"])
            / 2
            * time_diffs
        ).sum()

        # Convert Wh/m² to kWh/m²
        return energy_wh / 1000

    # Resample and apply vectorized integration
    resampled = df_indexed.resample("15min", origin="start")
    energy_series = resampled.apply(vectorized_trapezoidal_integration)

    # Convert to DataFrame with proper timezone handling
    result_df = energy_series.reset_index()
    result_df.columns = ["period_start", "energy_kwh"]

    # Ensure period_start is in Pacific timezone
    if result_df["period_start"].dt.tz is None:
        # If naive, assume UTC and convert to Pacific
        result_df["period_start"] = (
            result_df["period_start"]
            .dt.tz_localize("UTC")
            .dt.tz_convert("America/Los_Angeles")
        )
    else:
        # If already timezone-aware, convert to Pacific
        result_df["period_start"] = result_df["period_start"].dt.tz_convert(
            "America/Los_Angeles"
        )

    # Add period_end
    result_df["period_end"] = result_df["period_start"] + pd.Timedelta(minutes=15)

    # Reorder columns and sort
    result_df = result_df[["period_start", "period_end", "energy_kwh"]]
    result_df = result_df.sort_values("period_start").reset_index(drop=True)

    logger.info(f"Generated {len(result_df)} cached 15-minute solar energy periods")
    return result_df


def aggregate_to_daily(periods_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 15-minute periods to daily totals.

    :param periods_df: DataFrame from calculate_15min_energy_periods with columns [period_start, period_end, energy_kwh]
    :return: DataFrame with columns [date, daily_kwh]
    """
    if periods_df.empty:
        return pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "daily_kwh": pd.Series(dtype=float),
            }
        )

    # Extract date from period_start (already TZ-aware US/Pacific)
    periods_df = periods_df.copy()
    periods_df["date"] = periods_df["period_start"].dt.date

    # Group by date and sum energy_kwh, keeping all dates including zeros
    daily_df = periods_df.groupby("date", as_index=False)["energy_kwh"].sum()
    daily_df.columns = ["date", "daily_kwh"]

    # Sort by date
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    return daily_df


def aggregate_to_hourly(periods_df: pd.DataFrame, date: str) -> pd.DataFrame:
    """
    Aggregate 15-minute periods to hourly for a specific date.

    :param periods_df: DataFrame from calculate_15min_energy_periods
    :param date: String in format 'YYYY-MM-DD' (US/Pacific timezone)
    :return: DataFrame with columns [hour, hourly_kwh]
    """
    if periods_df.empty:
        return pd.DataFrame(
            {"hour": pd.Series(dtype=int), "hourly_kwh": pd.Series(dtype=float)}
        )

    try:
        target_date = pd.to_datetime(date).date()
    except ValueError:
        raise ValueError(f"Invalid date format: {date}. Expected 'YYYY-MM-DD'")

    # Filter to specified date
    date_filter = periods_df["period_start"].dt.date == target_date
    filtered_df = periods_df[date_filter].copy()

    if filtered_df.empty:
        # Return all hours with zero energy
        hourly_dict = {hour: 0.0 for hour in range(24)}
    else:
        # Extract hour from period_start
        filtered_df["hour"] = filtered_df["period_start"].dt.hour

        # Group by hour and sum energy_kwh
        hourly_dict = filtered_df.groupby("hour")["energy_kwh"].sum().to_dict()

        # Ensure all hours 0-23 are present (including zeros)
        for hour in range(24):
            if hour not in hourly_dict:
                hourly_dict[hour] = 0.0

    # Create result DataFrame
    result_df = pd.DataFrame(
        {"hour": list(hourly_dict.keys()), "hourly_kwh": list(hourly_dict.values())}
    )

    # Sort by hour
    result_df = result_df.sort_values("hour").reset_index(drop=True)

    return result_df


def get_period_stats(periods_df: pd.DataFrame) -> Dict:
    """
    Calculate summary statistics for energy periods.

    :param periods_df: DataFrame from calculate_15min_energy_periods
    :return: Dict with keys: total_kwh, days_with_production, avg_daily_kwh, peak_day
    """
    if periods_df.empty:
        return {
            "total_kwh": 0.0,
            "days_with_production": 0,
            "avg_daily_kwh": 0.0,
            "peak_day": {"date": "", "kwh": 0.0},
        }

    # Calculate total_kwh
    total_kwh = periods_df["energy_kwh"].sum()

    # Get daily aggregation
    daily_df = aggregate_to_daily(periods_df)

    # Calculate days_with_production
    days_with_production = (daily_df["daily_kwh"] > 0).sum()

    # Calculate avg_daily_kwh
    avg_daily_kwh = (
        total_kwh / days_with_production if days_with_production > 0 else 0.0
    )

    # Find peak_day
    if not daily_df.empty and not daily_df["daily_kwh"].empty:
        peak_idx = daily_df["daily_kwh"].idxmax()
        peak_day = {
            "date": daily_df.loc[peak_idx, "date"].strftime("%Y-%m-%d"),
            "kwh": daily_df.loc[peak_idx, "daily_kwh"],
        }
    else:
        peak_day = {"date": "", "kwh": 0.0}

    return {
        "total_kwh": total_kwh,
        "days_with_production": days_with_production,
        "avg_daily_kwh": avg_daily_kwh,
        "peak_day": peak_day,
    }

"""
Solar Energy Period Calculations
Converts raw solar radiation measurements into 15-minute energy periods.
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, Optional
from lookout.core.solar_analysis import LOCATION
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

PERIOD = "15min"

# Longest gap between two rows that is integrated: one period. Across a longer
# gap the radiation between the rows is not known, so nothing is credited.
MAX_INTERVAL_HOURS = 0.25


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
    utc_dt = df_sorted["datetime"].dt.tz_convert("UTC").reset_index(drop=True)
    utc_floored = utc_dt.dt.floor(PERIOD)
    watts = df_sorted["solarradiation"].astype(float).reset_index(drop=True)

    # Trapezoid over each pair of consecutive rows: (y0 + y1) / 2 * dx. A pair
    # that crosses a period boundary is split there, at the radiation read off
    # the straight line between the two rows, and each part is credited to the
    # period it lies in. Splitting a trapezoid on its own line keeps its area,
    # so the periods add up to the trapezoid of the whole curve.
    t0, t1 = utc_dt.iloc[:-1].values, utc_dt.iloc[1:].values
    y0, y1 = watts.iloc[:-1].values, watts.iloc[1:].values
    p0 = utc_floored.iloc[:-1].values
    boundary = utc_floored.iloc[1:].values  # start of the later row's period
    span = (t1 - t0) / np.timedelta64(1, "h")
    counted = (span > 0) & (span <= MAX_INTERVAL_HOURS)
    crosses = counted & (boundary > t0) & (boundary < t1)

    before = np.where(crosses, (boundary - t0) / np.timedelta64(1, "h"), span)
    after = span - before
    with np.errstate(divide="ignore", invalid="ignore"):
        y_boundary = np.where(crosses, y0 + (y1 - y0) * before / span, y1)
    first_wh = np.where(counted, (y0 + y_boundary) / 2 * before, 0.0)
    second_wh = np.where(crosses, (y_boundary + y1) / 2 * after, 0.0)

    credited = pd.concat(
        [
            pd.Series(first_wh, index=pd.DatetimeIndex(p0, tz="UTC")),
            pd.Series(second_wh, index=pd.DatetimeIndex(boundary, tz="UTC")),
        ]
    )
    # Missing radiation (NaN) contributes nothing, as pandas sum skips it
    energy_wh = credited.groupby(level=0).sum()

    # Every period holding a row is reported, with 0.0 when nothing is credited
    periods = pd.DatetimeIndex(utc_floored.unique())
    energy_wh = energy_wh.reindex(periods, fill_value=0.0).fillna(0.0)

    # Convert Wh/m² to kWh/m²
    result_df = pd.DataFrame(
        {
            "period_start": energy_wh.index.tz_convert("America/Los_Angeles"),
            "energy_kwh": (energy_wh / 1000).values,
        }
    )
    result_df["period_end"] = result_df["period_start"] + pd.Timedelta(minutes=15)

    # Reorder columns to match expected output
    result_df = result_df[["period_start", "period_end", "energy_kwh"]]

    # Sort by period_start
    result_df = result_df.sort_values("period_start").reset_index(drop=True)

    logger.info(f"Generated {len(result_df)} 15-minute solar energy periods")
    return result_df


_cached_calculation: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None


def calculate_15min_energy_periods_cached(df: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlit-cached `calculate_15min_energy_periods`.

    Streamlit is imported on first call, so importing this module (and the
    refresh command that depends on it) never loads Streamlit.

    :param df: DataFrame with datetime and solarradiation columns
    :return: DataFrame with period_start, period_end, energy_kwh columns
    """
    global _cached_calculation
    if _cached_calculation is None:
        import streamlit as st

        _cached_calculation = st.cache_data(
            show_spinner=False, max_entries=20, ttl=7200
        )(calculate_15min_energy_periods)
    return _cached_calculation(df)


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

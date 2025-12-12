"""
solar_data_transformer.py

Pure data processing functions for solar energy visualizations.

This module contains all data transformation and aggregation logic
for solar visualizations, separated from presentation concerns.
All functions are pure and testable without Plotly dependencies.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def prepare_month_day_heatmap_data(periods_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform energy periods data into month/day pivot table for heatmap visualization.

    :param periods_df: DataFrame with period_start, period_end, energy_kwh columns
    :return: Pivot table with months as index, days as columns, kWh values
    """
    if periods_df.empty:
        logger.warning("Empty periods_df provided to prepare_month_day_heatmap_data")
        return pd.DataFrame()

    # Aggregate to daily totals (keeps ALL days including zeros)
    daily_df = _aggregate_to_daily(periods_df)

    if daily_df.empty:
        logger.warning("No daily data available after aggregation")
        return pd.DataFrame()

    # Extract month and day for pivot table
    daily_df = daily_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df["month"] = daily_df["date"].dt.strftime("%Y-%m")  # YYYY-MM format
    daily_df["day"] = daily_df["date"].dt.day

    # Create pivot table: months x days (1-31)
    pivot_df = daily_df.pivot_table(
        values="daily_kwh",
        index="month",
        columns="day",
        aggfunc="sum",  # Should be single value per month/day
    )

    # Ensure all days 1-31 are present as columns (fill missing with NaN)
    all_days = list(range(1, 32))
    for day in all_days:
        if day not in pivot_df.columns:
            pivot_df[day] = np.nan

    # Sort columns by day and index by month (newest first)
    pivot_df = pivot_df[all_days]
    pivot_df = pivot_df.sort_index(ascending=False)

    # Convert column names to strings for consistency
    pivot_df.columns = pivot_df.columns.astype(str)

    logger.debug(f"Prepared month/day heatmap data: {len(pivot_df)} months × {len(pivot_df.columns)} days")
    return pivot_df


def _aggregate_to_daily(periods_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate energy periods to daily totals.

    :param periods_df: DataFrame with period_start, period_end, energy_kwh columns
    :return: DataFrame with date and daily_kwh columns
    """
    periods_df = periods_df.copy()
    periods_df["period_start"] = pd.to_datetime(periods_df["period_start"])
    periods_df["date"] = periods_df["period_start"].dt.date.astype(str)

    daily_df = periods_df.groupby("date")["energy_kwh"].sum().reset_index()
    daily_df = daily_df.rename(columns={"energy_kwh": "daily_kwh"})

    return daily_df


def prepare_day_column_data(periods_df: pd.DataFrame, date: str) -> pd.DataFrame:
    """
    Transform energy periods data into hourly column data for a specific date.

    :param periods_df: DataFrame with period_start, period_end, energy_kwh columns
    :param date: Date string in 'YYYY-MM-DD' format
    :return: DataFrame with hour and hourly_kwh columns
    """
    if periods_df.empty:
        logger.warning("Empty periods_df provided to prepare_day_column_data")
        return pd.DataFrame()

    # Filter to specific date and aggregate to hourly
    hourly_df = _aggregate_to_hourly(periods_df, date)

    if hourly_df.empty:
        logger.warning(f"No hourly data available for date {date}")
        return pd.DataFrame()

    # Ensure all 24 hours are present (fill missing with 0)
    all_hours = pd.DataFrame({'hour': range(24)})
    result = all_hours.merge(hourly_df, on='hour', how='left').fillna(0)

    logger.debug(f"Prepared day column data for {date}: {len(result)} hours")
    return result


def prepare_day_15min_heatmap_data(periods_df: pd.DataFrame, start_hour: Optional[int] = None, end_hour: Optional[int] = None) -> pd.DataFrame:
    """
    Transform energy periods data into day/time pivot table for 15min heatmap visualization.

    :param periods_df: DataFrame with period_start, period_end, energy_kwh columns
    :param start_hour: Optional start hour filter (0-23)
    :param end_hour: Optional end hour filter (0-23, exclusive)
    :return: Pivot table with dates as index, time slots as columns, kWh values
    """
    if periods_df.empty:
        logger.warning("Empty periods_df provided to prepare_day_15min_heatmap_data")
        return pd.DataFrame()

    periods_df = periods_df.copy()
    periods_df["period_start"] = pd.to_datetime(periods_df["period_start"])
    periods_df["date"] = periods_df["period_start"].dt.date.astype(str)
    periods_df["time_slot"] = periods_df["period_start"].dt.strftime("%H:%M")

    # Apply time filtering if specified
    if start_hour is not None or end_hour is not None:
        hour = periods_df["period_start"].dt.hour
        if start_hour is not None:
            periods_df = periods_df[hour >= start_hour]
        if end_hour is not None:
            periods_df = periods_df[hour < end_hour]

    if periods_df.empty:
        logger.warning("No data available after time filtering")
        return pd.DataFrame()

    # Create pivot table: dates x time_slots
    pivot_df = periods_df.pivot_table(
        values="energy_kwh",
        index="date",
        columns="time_slot",
        aggfunc="sum",
        fill_value=0,  # Fill missing periods with 0
    )

    # Sort by date (newest first)
    pivot_df = pivot_df.sort_index(ascending=False)

    logger.debug(f"Prepared day/15min heatmap data: {len(pivot_df)} days × {len(pivot_df.columns)} time slots")
    return pivot_df


def prepare_15min_bar_data(periods_df: pd.DataFrame, date: str, start_hour: Optional[int] = None, end_hour: Optional[int] = None) -> pd.DataFrame:
    """
    Transform energy periods data into 15min bar chart data for a specific date.

    :param periods_df: DataFrame with period_start, period_end, energy_kwh columns
    :param date: Date string in 'YYYY-MM-DD' format
    :param start_hour: Optional start hour filter (0-23)
    :param end_hour: Optional end hour filter (0-23, exclusive)
    :return: DataFrame with time_label and energy_wh columns
    """
    if periods_df.empty:
        logger.warning("Empty periods_df provided to prepare_15min_bar_data")
        return pd.DataFrame()

    periods_df = periods_df.copy()
    periods_df["period_start"] = pd.to_datetime(periods_df["period_start"])

    # Filter to specific date
    periods_df = periods_df[periods_df["period_start"].dt.date.astype(str) == date]

    # Apply time filtering if specified
    if start_hour is not None or end_hour is not None:
        hour = periods_df["period_start"].dt.hour
        if start_hour is not None:
            periods_df = periods_df[hour >= start_hour]
        if end_hour is not None:
            periods_df = periods_df[hour < end_hour]

    if periods_df.empty:
        logger.warning(f"No data available for date {date} after filtering")
        return pd.DataFrame()

    # Create time labels and convert to Wh
    result = pd.DataFrame({
        'time_label': periods_df["period_start"].dt.strftime("%H:%M"),
        'energy_wh': periods_df["energy_kwh"] * 1000  # Convert kWh to Wh
    })

    # Sort by time
    result = result.sort_values('time_label').reset_index(drop=True)

    logger.debug(f"Prepared 15min bar data for {date}: {len(result)} periods")
    return result


def _aggregate_to_hourly(periods_df: pd.DataFrame, date: str) -> pd.DataFrame:
    """
    Aggregate energy periods to hourly totals for a specific date.

    :param periods_df: DataFrame with period_start, period_end, energy_kwh columns
    :param date: Date string in 'YYYY-MM-DD' format
    :return: DataFrame with hour and hourly_kwh columns
    """
    periods_df = periods_df.copy()
    periods_df["period_start"] = pd.to_datetime(periods_df["period_start"])

    # Filter to specific date
    periods_df = periods_df[periods_df["period_start"].dt.date.astype(str) == date]

    # Extract hour and aggregate
    periods_df["hour"] = periods_df["period_start"].dt.hour
    hourly_df = periods_df.groupby("hour")["energy_kwh"].sum().reset_index()
    hourly_df = hourly_df.rename(columns={"energy_kwh": "hourly_kwh"})

    # Ensure all 24 hours are present (fill missing with 0)
    all_hours = pd.DataFrame({'hour': range(24)})
    result = all_hours.merge(hourly_df, on='hour', how='left').fillna(0)

    return result

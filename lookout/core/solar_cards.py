"""
Solar Radiation Metric Cards - Core Data Processing

Provides data processing functions for the solar radiation metric grid component.
Calculates period metrics, aggregates data for sparklines, and handles unit formatting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import date, datetime
import pytz

from lookout.core.solar_energy_periods import aggregate_to_daily
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def aggregate_to_hourly(periods_df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """
    Aggregate 15-minute periods to hourly totals for a specific date.

    Used for today's sparkline showing 24 hourly bars.

    :param periods_df: DataFrame from calculate_15min_energy_periods with period_start, energy_kwh
    :param target_date: String in format 'YYYY-MM-DD'
    :return: DataFrame with columns [hour, hourly_kwh] for all 24 hours (NaN for missing data)
    """
    if periods_df.empty:
        # Return all hours with NaN (will show as gray bars)
        return pd.DataFrame({
            "hour": list(range(24)),
            "hourly_kwh": [np.nan] * 24
        })

    try:
        target_date_obj = pd.to_datetime(target_date).date()
    except ValueError:
        raise ValueError(f"Invalid date format: {target_date}. Expected 'YYYY-MM-DD'")

    # Filter to specified date
    periods_df = periods_df.copy()
    periods_df["date"] = periods_df["period_start"].dt.date
    date_filter = periods_df["date"] == target_date_obj
    filtered_df = periods_df[date_filter].copy()

    if filtered_df.empty:
        # Return all hours with NaN
        return pd.DataFrame({
            "hour": list(range(24)),
            "hourly_kwh": [np.nan] * 24
        })

    # Extract hour from period_start (already TZ-aware)
    filtered_df["hour"] = filtered_df["period_start"].dt.hour

    # Group by hour and sum energy_kwh
    hourly_dict = filtered_df.groupby("hour")["energy_kwh"].sum().to_dict()

    # Create result DataFrame with all 24 hours (NaN for missing)
    result_data = []
    for hour in range(24):
        value = hourly_dict.get(hour, np.nan)
        result_data.append({"hour": hour, "hourly_kwh": value})

    return pd.DataFrame(result_data)


def get_daily_range(periods_df: pd.DataFrame, end_date: date, days: int) -> List[float]:
    """
    Get N days of daily values ending at end_date (inclusive).

    Used for 7d, 30d, 365d sparklines showing daily bars.

    :param periods_df: DataFrame from calculate_15min_energy_periods
    :param end_date: End date (inclusive) in America/Los_Angeles timezone
    :param days: Number of days to include
    :return: List of daily kWh values (NaN for missing days)
    """
    if periods_df.empty:
        return [np.nan] * days

    # Get daily aggregation
    daily_df = aggregate_to_daily(periods_df)

    if daily_df.empty:
        return [np.nan] * days

    # Calculate start date (inclusive)
    start_date = pd.Timestamp(end_date) - pd.Timedelta(days=days-1)

    # Create date range for all expected days
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create dictionary of existing daily values
    daily_values = dict(zip(daily_df["date"], daily_df["daily_kwh"]))

    # Build result list (NaN for missing days)
    result = []
    for expected_date in date_range:
        expected_date_only = expected_date.date()
        value = daily_values.get(expected_date_only, np.nan)
        result.append(value)

    return result


def format_metric_value(value: float) -> Tuple[float, str]:
    """
    Format metric value with auto-scaling units.

    Auto-scales to Wh/m² for values < 1.0 kWh/m² (Oregon winter values).

    :param value: Value in kWh/m²
    :return: Tuple of (formatted_value, unit_string)
    """
    if pd.isna(value):
        return 0.0, "kWh/m²"

    if value < 1.0:
        # Convert to Wh/m² for better readability of small values
        return value * 1000, "Wh/m²"
    else:
        return value, "kWh/m²"


def get_global_axis_range(all_periods_data: List[List[float]]) -> Tuple[float, float]:
    """
    Calculate consistent y-axis range across all sparklines for intuitive comparison.

    :param all_periods_data: List of lists containing values for each period [7d, 30d, 365d]
    :return: Tuple of (min_value, max_value) for fixed axis
    """
    # Flatten all values and filter out NaN
    all_values = []
    for period_data in all_periods_data:
        all_values.extend([v for v in period_data if not pd.isna(v)])

    if not all_values:
        return (0, 1.0)  # Default range if no data

    # Find max value
    max_value = max(all_values)

    # Round up to nice number for axis
    if max_value <= 0.5:
        axis_max = 0.5
    elif max_value <= 1.0:
        axis_max = 1.0
    elif max_value <= 2.0:
        axis_max = 2.0
    elif max_value <= 5.0:
        axis_max = 5.0
    elif max_value <= 10.0:
        axis_max = 10.0
    elif max_value <= 20.0:
        axis_max = 20.0
    elif max_value <= 50.0:
        axis_max = 50.0
    elif max_value <= 100.0:
        axis_max = 100.0
    else:
        # Round up to nearest 100
        axis_max = np.ceil(max_value / 100) * 100

    return (0, axis_max)


def calculate_period_metrics(periods_df: pd.DataFrame) -> Dict:
    """
    Calculate metrics for all standard periods: Today, 7d, 30d, 365d.

    Includes values, units, sparkline data, and delta comparisons.

    :param periods_df: DataFrame from calculate_15min_energy_periods
    :return: Dict with keys 'today', 'last_7d', 'last_30d', 'last_365d'
             Each value is dict with: value, unit, sparkline_data, delta
    """
    # Get current date in Pacific timezone
    pacific_tz = pytz.timezone("America/Los_Angeles")
    today = pd.Timestamp.now(tz=pacific_tz).date()
    today_str = today.strftime("%Y-%m-%d")

    result = {}

    # Calculate metrics for each period
    periods_config = [
        ("today", today, 1, "hourly"),
        ("last_7d", today, 7, "daily"),
        ("last_30d", today, 30, "daily"),
        ("last_365d", today, 365, "daily"),
    ]

    for period_key, end_date_obj, days, data_type in periods_config:
        if data_type == "hourly":
            # Today: hourly data for 24-hour sparkline
            sparkline_df = aggregate_to_hourly(periods_df, today_str)
            sparkline_data = sparkline_df["hourly_kwh"].tolist()

            # Today's total value
            value = sum(v for v in sparkline_data if not pd.isna(v))

            # Delta vs yesterday
            yesterday_ts = pd.Timestamp(today) - pd.Timedelta(days=1)
            yesterday_str = yesterday_ts.strftime("%Y-%m-%d")
            yesterday_df = aggregate_to_hourly(periods_df, yesterday_str)
            yesterday_value = sum(v for v in yesterday_df["hourly_kwh"].tolist() if not pd.isna(v))
            delta_value = value - yesterday_value if not pd.isna(yesterday_value) else None

        else:
            # Daily periods: get N days of daily data
            sparkline_data = get_daily_range(periods_df, end_date_obj, days)
            value = sum(v for v in sparkline_data if not pd.isna(v))

            # Delta vs previous period
            prev_end_ts = pd.Timestamp(end_date_obj) - pd.Timedelta(days=days)
            prev_sparkline_data = get_daily_range(periods_df, prev_end_ts.date(), days)
            prev_value = sum(v for v in prev_sparkline_data if not pd.isna(v))
            delta_value = value - prev_value if not pd.isna(prev_value) else None

        # Format value with auto-scaling units
        formatted_value, unit = format_metric_value(value)

        # Calculate delta info
        delta = None
        if delta_value is not None:
            delta = {
                "value": abs(delta_value),
                "direction": "up" if delta_value > 0 else "down"
            }

        result[period_key] = {
            "value": formatted_value,
            "unit": unit,
            "sparkline_data": sparkline_data,
            "delta": delta
        }

    # Calculate global axis range for 7d, 30d, 365d sparklines (exclude today)
    daily_periods_data = [
        result["last_7d"]["sparkline_data"],
        result["last_30d"]["sparkline_data"],
        result["last_365d"]["sparkline_data"]
    ]
    global_axis = get_global_axis_range(daily_periods_data)

    # Add axis info to daily periods
    for period_key in ["last_7d", "last_30d", "last_365d"]:
        result[period_key]["axis_range"] = global_axis

    # Today uses its own axis (0 to max of its hourly values)
    today_values = [v for v in result["today"]["sparkline_data"] if not pd.isna(v)]
    today_max = max(today_values) if today_values else 1.0
    result["today"]["axis_range"] = (0, today_max)

    return result
"""
Solar Radiation 2x2 Tile System - Core Data Processing

Provides aggregation functions for additive tile system.
Calculates metrics for Last 24h, 7d, 30d, and 365d periods.
"""

from typing import Dict

import numpy as np
import pandas as pd
import pytz

from lookout.core.solar_energy_periods import aggregate_to_daily
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def calculate_tile_metrics(periods_df: pd.DataFrame) -> Dict:
    """
    Calculate metrics for all 4 tiles using existing aggregation functions.

    Returns dict with keys: 'last_24h', 'last_7d', 'last_30d', 'last_365d'
    Each contains parameters for render_solar_tile() function.
    """
    if periods_df.empty:
        logger.warning("No energy periods available for tile metrics")
        return _get_empty_tile_metrics()

    try:
        # Calculate metrics for each period
        last_24h = aggregate_to_last_24h(periods_df)
        last_7d = aggregate_to_daily_periods(periods_df, 7)
        last_30d = aggregate_to_daily_periods(periods_df, 30)
        last_365d = aggregate_to_daily_periods(periods_df, 365)

        # Calculate global y-axis range for daily tiles (7d, 30d, 365d)
        daily_max = max(
            [
                max(last_7d["sparkline_data"]) if last_7d["sparkline_data"] else 0,
                max(last_30d["sparkline_data"]) if last_30d["sparkline_data"] else 0,
                max(last_365d["sparkline_data"]) if last_365d["sparkline_data"] else 0,
            ]
        )
        daily_axis_range = (0, daily_max if daily_max > 0 else 1.0)

        # Set axis ranges
        last_7d["y_axis_range"] = daily_axis_range
        last_30d["y_axis_range"] = daily_axis_range
        last_365d["y_axis_range"] = daily_axis_range

        # 24h tile uses its own axis range
        if last_24h["sparkline_data"]:
            last_24h_max = max(last_24h["sparkline_data"])
            last_24h["y_axis_range"] = (0, last_24h_max if last_24h_max > 0 else 1.0)
        else:
            last_24h["y_axis_range"] = (0, 1.0)

        return {
            "last_24h": last_24h,
            "last_7d": last_7d,
            "last_30d": last_30d,
            "last_365d": last_365d,
        }

    except Exception as e:
        logger.error(f"Error calculating tile metrics: {e}")
        return _get_empty_tile_metrics()


def aggregate_to_last_24h(periods_df: pd.DataFrame) -> Dict:
    """
    Option C: Yesterday 18:00 → Current time (partial current hour).

    When it's 18:14 today:
    - Time window: Yesterday 18:00 → Today 18:14
    - 24 hours data including current partial hour
    - Allows comparison of yesterday at this time vs today so far

    :param periods_df: DataFrame from calculate_15min_energy_periods
    :return: Dict with total_kwh, sparkline_data, hover_labels, delta_value, current_index
    """
    if periods_df.empty:
        return _get_empty_tile_data("last_24h")

    try:
        now = pd.Timestamp.now(tz="America/Los_Angeles")
        current_hour = now.hour

        # Time window: Yesterday current hour boundary → Now
        start_time = now.replace(minute=0, second=0, microsecond=0) - pd.Timedelta(
            days=1
        )
        end_time = now

        # Filter 15-min periods in window
        filtered = periods_df[
            (periods_df["period_start"] >= start_time)
            & (periods_df["period_start"] < end_time)
        ].copy()

        if filtered.empty:
            return _get_empty_tile_data("last_24h")

        # Extract hour from period_start safely - ensure it's a Series
        filtered_copy = filtered.copy()
        period_starts = pd.Series(filtered_copy["period_start"])
        filtered_copy["hour"] = period_starts.apply(lambda x: x.hour)

        # Group by hour and sum energy
        hourly_dict = filtered_copy.groupby("hour")["energy_kwh"].sum().to_dict()

        # Create rolling 25-hour sequence: yesterday current hour → current partial hour
        rolling_hours = [(current_hour + offset) % 24 for offset in range(25)]

        sparkline_data = []
        hover_labels = []

        for i, hour in enumerate(rolling_hours):
            # Get data for this hour (from catalog or current session)
            value = hourly_dict.get(hour, np.nan)
            sparkline_data.append(value)

            # Determine if this is the current partial hour (last position)
            if i == 24:
                hover_labels.append(f"Hour {hour:02d} (Current)<br>{value:.2f} kWh/m²")
            else:
                is_yesterday = i < (24 - current_hour)
                time_desc = "Yesterday" if is_yesterday else "Today"
                if not pd.isna(value):
                    hover_labels.append(
                        f"Hour {hour:02d} ({time_desc})<br>{value:.2f} kWh/m²"
                    )
                else:
                    hover_labels.append(f"Hour {hour:02d} ({time_desc})<br>No data")

        # Calculate total - only count each hour once, not the repeated current hour
        total_kwh = filtered["energy_kwh"].sum()

        # Delta vs previous 24h period (yesterday 18:00 → yesterday same time)
        prev_start = start_time - pd.Timedelta(days=1)
        prev_end = end_time - pd.Timedelta(days=1)

        prev_filtered = periods_df[
            (periods_df["period_start"] >= prev_start)
            & (periods_df["period_start"] < prev_end)
        ].copy()

        prev_total = prev_filtered["energy_kwh"].sum() if not prev_filtered.empty else 0
        delta_value = total_kwh - prev_total

        # Current period index for highlighting - always the last (current partial hour)
        current_period_index = 24

        return {
            "title": "Last 24h",
            "total_kwh": total_kwh,
            "sparkline_data": sparkline_data,
            "hover_labels": hover_labels,
            "current_period_index": current_period_index,
            "delta_value": delta_value,
        }

    except Exception as e:
        logger.error(f"Error in aggregate_to_last_24h: {e}")
        return _get_empty_tile_data("last_24h")


def aggregate_to_daily_periods(periods_df: pd.DataFrame, days: int) -> Dict:
    """
    Daily aggregation for 7d, 30d, 365d tiles.

    :param periods_df: DataFrame from calculate_15min_energy_periods
    :param days: Number of days to include (7, 30, or 365)
    :return: Dict with total_kwh, sparkline_data, hover_labels, delta, current_index
    """
    if periods_df.empty:
        return _get_empty_tile_data(f"last_{days}d")

    try:
        # Get current date in Pacific timezone
        pacific_tz = pytz.timezone("America/Los_Angeles")
        today = pd.Timestamp.now(tz=pacific_tz).date()

        # Get daily aggregation using existing function
        daily_df = aggregate_to_daily(periods_df)

        if daily_df.empty:
            return _get_empty_tile_data(f"last_{days}d")

        # Calculate date range
        start_date = pd.Timestamp(today) - pd.Timedelta(days=days - 1)
        date_range = pd.date_range(start=start_date, end=today, freq="D")

        # Build daily data
        daily_values = dict(zip(daily_df["date"], daily_df["daily_kwh"]))
        sparkline_data = []
        hover_labels = []
        current_period_index = None

        for i, expected_date in enumerate(date_range):
            expected_date_only = expected_date.date()
            value = daily_values.get(expected_date_only, np.nan)
            sparkline_data.append(value)

            # Hover labels: "Sun 2025-01-04<br>2.3 kWh/m²"
            if not pd.isna(value):
                hover_labels.append(
                    f"{expected_date.strftime('%a %Y-%m-%d')}<br>{value:.2f} kWh/m²"
                )
            else:
                hover_labels.append(
                    f"{expected_date.strftime('%a %Y-%m-%d')}<br>No data"
                )

            # Find current period index
            if expected_date_only == today:
                current_period_index = i

        # Calculate total
        total_kwh = sum(v for v in sparkline_data if not pd.isna(v))

        # Delta vs previous period
        prev_start = start_date - pd.Timedelta(days=days)
        prev_end = start_date - pd.Timedelta(days=1)
        prev_date_range = pd.date_range(start=prev_start, end=prev_end, freq="D")

        prev_total = 0
        for prev_date in prev_date_range:
            prev_date_only = prev_date.date()
            prev_total += daily_values.get(prev_date_only, 0)

        delta_value = total_kwh - prev_total

        return {
            "title": f"Last {days}d",
            "total_kwh": total_kwh,
            "sparkline_data": sparkline_data,
            "hover_labels": hover_labels,
            "current_period_index": current_period_index,
            "delta_value": delta_value,
        }

    except Exception as e:
        logger.error(f"Error in aggregate_to_daily_periods: {e}")
        return _get_empty_tile_data(f"last_{days}d")


def _get_empty_tile_metrics() -> Dict:
    """Return empty metrics for all tiles when no data available."""
    empty_24h = _get_empty_tile_data("last_24h")
    empty_24h["y_axis_range"] = (0, 1.0)

    empty_7d = _get_empty_tile_data("last_7d")
    empty_7d["y_axis_range"] = (0, 1.0)

    empty_30d = _get_empty_tile_data("last_30d")
    empty_30d["y_axis_range"] = (0, 1.0)

    empty_365d = _get_empty_tile_data("last_365d")
    empty_365d["y_axis_range"] = (0, 1.0)

    return {
        "last_24h": empty_24h,
        "last_7d": empty_7d,
        "last_30d": empty_30d,
        "last_365d": empty_365d,
    }


def _get_empty_tile_data(title: str) -> Dict:
    """Return empty tile data structure."""
    return {
        "title": title,
        "total_kwh": 0.0,
        "sparkline_data": [],
        "hover_labels": [],
        "current_period_index": None,
        "delta_value": None,
        "y_axis_range": (0, 1.0),
    }

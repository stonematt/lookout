"""
Trend calculation utilities for weather data.

This module provides functions to calculate temperature and barometer trends
based on historical data comparisons.
"""

import pandas as pd
from typing import Optional

from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def get_historical_value(
    df: pd.DataFrame, target_time: int, field: str
) -> Optional[float]:
    """
    Get value at specific timestamp from historical data.

    :param df: DataFrame with historical weather data (reverse sorted by date)
    :param target_time: Target timestamp in milliseconds (UTC)
    :param field: Field name to retrieve (e.g., 'tempf', 'baromrelin')
    :return: Value at target time or None if not found/too far away
    """
    if df.empty or field not in df.columns:
        return None

    # Calculate time differences from target
    time_diffs = abs(df["dateutc"] - target_time)

    # Find closest match
    closest_idx = time_diffs.idxmin()
    time_diff = time_diffs.iloc[closest_idx]

    # Only use if within reasonable time window (±30 minutes)
    max_time_diff = 30 * 60 * 1000  # 30 minutes in milliseconds
    if time_diff > max_time_diff:
        logger.debug(
            f"No historical data found within ±30min for {field} at target time"
        )
        return None

    return df.iloc[closest_idx][field]


def calculate_temperature_trend(df: pd.DataFrame, current_temp: float) -> str:
    """
    Calculate temperature trend by comparing current to same time yesterday.

    :param df: DataFrame with historical weather data (reverse sorted by date)
    :param current_temp: Current temperature in Fahrenheit
    :return: Trend arrow: "↑" (rising), "↓" (falling), or "→" (steady)
    """
    if df.empty:
        logger.debug("Empty DataFrame for temperature trend calculation")
        return "→"

    # Get current timestamp (latest data point)
    current_time = df.iloc[0]["dateutc"]

    # Calculate target time (24 hours ago)
    target_time = current_time - (24 * 60 * 60 * 1000)  # 24h ago in ms

    # Get temperature 24 hours ago
    temp_24h_ago = get_historical_value(df, target_time, "tempf")
    if temp_24h_ago is None:
        logger.debug("No temperature data available 24h ago")
        return "→"

    # Calculate difference and determine trend
    temp_diff = current_temp - temp_24h_ago

    if temp_diff > 1.0:
        trend = "↑"
        logger.debug(
            f"Temperature rising: {current_temp:.1f}°F vs {temp_24h_ago:.1f}°F (diff: {temp_diff:+.1f})"
        )
    elif temp_diff < -1.0:
        trend = "↓"
        logger.debug(
            f"Temperature falling: {current_temp:.1f}°F vs {temp_24h_ago:.1f}°F (diff: {temp_diff:+.1f})"
        )
    else:
        trend = "→"
        logger.debug(
            f"Temperature steady: {current_temp:.1f}°F vs {temp_24h_ago:.1f}°F (diff: {temp_diff:+.1f})"
        )

    return trend


def calculate_barometer_trend(df: pd.DataFrame, current_pressure: float) -> str:
    """
    Calculate barometer trend based on 3-hour rate of change.

    :param df: DataFrame with historical weather data (reverse sorted by date)
    :param current_pressure: Current barometric pressure in inches
    :return: Trend arrow: "↑" (rising), "↓" (falling), or "→" (steady)
    """
    if df.empty:
        logger.debug("Empty DataFrame for barometer trend calculation")
        return "→"

    # Get current timestamp (latest data point)
    current_time = df.iloc[0]["dateutc"]

    # Calculate target time (3 hours ago)
    target_time = current_time - (3 * 60 * 60 * 1000)  # 3h ago in ms

    # Get pressure 3 hours ago
    pressure_3h_ago = get_historical_value(df, target_time, "baromrelin")
    if pressure_3h_ago is None:
        logger.debug("No barometer data available 3h ago")
        return "→"

    # Calculate rate of change per hour
    pressure_change = (current_pressure - pressure_3h_ago) / 3.0

    # Determine trend based on threshold
    threshold = 0.05  # inches per hour

    if pressure_change > threshold:
        trend = "↑"
        logger.debug(
            f'Pressure rising: {current_pressure:.2f}" vs {pressure_3h_ago:.2f}" (rate: {pressure_change:+.3f}"/hr)'
        )
    elif pressure_change < -threshold:
        trend = "↓"
        logger.debug(
            f'Pressure falling: {current_pressure:.2f}" vs {pressure_3h_ago:.2f}" (rate: {pressure_change:+.3f}"/hr)'
        )
    else:
        trend = "→"
        logger.debug(
            f'Pressure steady: {current_pressure:.2f}" vs {pressure_3h_ago:.2f}" (rate: {pressure_change:+.3f}"/hr)'
        )

    return trend

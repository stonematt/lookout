"""
Weather utility functions for formatting and calculations.

This module provides reusable weather-related calculations and formatting
functions that can be used across the application.
"""

from typing import Union
import pandas as pd


def format_wind_direction(degrees: float) -> str:
    """
    Convert wind direction in degrees to cardinal direction.

    :param degrees: Wind direction in degrees (0-360)
    :return: Cardinal direction string (N, NE, E, SE, S, SW, W, NW)
    """
    if 337.5 <= degrees or degrees < 22.5:
        return "N"
    elif 22.5 <= degrees < 67.5:
        return "NE"
    elif 67.5 <= degrees < 112.5:
        return "E"
    elif 112.5 <= degrees < 157.5:
        return "SE"
    elif 157.5 <= degrees < 202.5:
        return "S"
    elif 202.5 <= degrees < 247.5:
        return "SW"
    elif 247.5 <= degrees < 292.5:
        return "W"
    elif 292.5 <= degrees < 337.5:
        return "NW"
    else:
        return "N"


def classify_uv_level(uv_index: float) -> str:
    """
    Classify UV index into human-readable levels.

    :param uv_index: UV index value
    :return: UV level string (Low, Moderate, High, Very High)
    """
    if uv_index >= 8:
        return "Very High"
    elif uv_index >= 6:
        return "High"
    elif uv_index >= 3:
        return "Moderate"
    else:
        return "Low"


def format_time_since(
    last_rain_timestamp: Union[str, None], current_timestamp: int
) -> tuple[str, str]:
    """
    Format time since last rain event.

    :param last_rain_timestamp: Last rain timestamp from weather data
    :param current_timestamp: Current timestamp in milliseconds
    :return: Tuple of (rain_status, time_since_rain)
    """
    try:
        if not last_rain_timestamp:
            return "Dry", ""

        last_rain = pd.to_datetime(last_rain_timestamp)
        current_time = pd.to_datetime(current_timestamp, unit="ms", utc=True)
        time_since = current_time - last_rain
        hours_since = time_since.total_seconds() / 3600

        if hours_since < 1:
            time_since_rain = f"{time_since.total_seconds()/60:.0f}min"
        else:
            time_since_rain = f"{hours_since:.1f}h"

        if hours_since < 24:
            rain_status = f"Dry {time_since_rain}"
        else:
            rain_status = f"Dry {hours_since/24:.1f}d"

        return rain_status, time_since_rain
    except Exception:
        return "Dry", ""


def calculate_rain_rate(total_rain: float, duration_hours: float) -> float:
    """
    Calculate rain rate in inches per hour.

    :param total_rain: Total rainfall in inches
    :param duration_hours: Duration in hours
    :return: Rain rate in inches per hour
    """
    if duration_hours > 0:
        return total_rain / duration_hours
    return 0.0


def format_duration_hours(duration_hours: float) -> str:
    """
    Format duration in hours to human-readable string.

    :param duration_hours: Duration in hours
    :return: Formatted duration string
    """
    if duration_hours >= 24:
        return f"{duration_hours/24:.1f}d"
    else:
        return f"{duration_hours:.1f}h"

"""
Weather event detection and analysis.

This module provides functionality for detecting and analyzing weather events,
particularly rain events, from weather data.
"""

from typing import Any, Dict, Optional

import pandas as pd

from lookout.core.rain_events import is_event_ongoing
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def find_active_rain_event(
    events_df: Optional[pd.DataFrame], current_eventrainin: Optional[float] = 0
) -> Optional[pd.Series]:
    """
    Pick the Event Catalog's active rain event, as the catalog row itself.

    An event is active when the catalog marks it ongoing and the live reading
    still reports event rain. Of several, the one that started last wins.

    :param events_df: The Event Catalog; None or empty means no events.
    :param current_eventrainin: The live `eventrainin`; None or 0 means dry.
    :return: The active event's catalog row, or None.
    """
    if events_df is None or events_df.empty:
        return None
    if not current_eventrainin or current_eventrainin <= 0:
        return None

    ongoing = [event for _, event in events_df.iterrows() if is_event_ongoing(event)]
    if not ongoing:
        return None
    return max(ongoing, key=lambda event: event["start_time"])


def get_active_rain_event(
    events_df: Optional[pd.DataFrame], current_eventrainin: Optional[float] = 0
) -> Optional[Dict[str, Any]]:
    """
    Detect the active rain event and format it for the header.

    :param events_df: The Event Catalog, passed in by the caller.
    :param current_eventrainin: Current event rain amount for validation
    :return: Active event information dict or None if no active event
    """
    try:
        event = find_active_rain_event(events_df, current_eventrainin)
        if event is not None:
            return _format_active_event_info([event])
    except Exception as e:
        logger.warning(f"Error detecting active rain events: {e}")

    return None


def _format_active_event_info(ongoing_events: list) -> Dict[str, Any]:
    """
    Format active event information from ongoing events list.

    :param ongoing_events: List of ongoing event dictionaries
    :return: Formatted active event information
    """
    from lookout.utils.weather_utils import calculate_rain_rate, format_duration_hours

    # Get most recent ongoing event
    latest_event = max(ongoing_events, key=lambda x: x["start_time"])

    # Calculate duration and format
    duration_h = latest_event["duration_minutes"] / 60
    total_rain = latest_event["total_rainfall"]

    # Calculate rain rate
    rain_rate = calculate_rain_rate(total_rain, duration_h)

    # Format duration string
    duration_str = format_duration_hours(duration_h)

    # Format start time
    start_time_str = "Unknown"
    try:
        start_time = pd.to_datetime(latest_event["start_time"], unit="ms")
        start_time_str = start_time.strftime("%b %d %-I:%M %p")
    except Exception:
        pass

    return {
        "duration": duration_str,
        "total_rain": total_rain,
        "rain_rate": rain_rate,
        "start_time": start_time_str,
        "duration_h": duration_h,
    }

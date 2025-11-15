"""
Weather event detection and analysis.

This module provides functionality for detecting and analyzing weather events,
particularly rain events, from weather data.
"""

import json
from typing import Optional, Dict, Any
import pandas as pd

from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def get_active_rain_event(
    device_mac: str, file_type: str, current_eventrainin: float = 0
) -> Optional[Dict[str, Any]]:
    """
    Detect and return active rain event information.

    :param device_mac: Device MAC address
    :param file_type: File type for data storage
    :param current_eventrainin: Current event rain amount for validation
    :return: Active event information dict or None if no active event
    """
    try:
        from lookout.core.rain_events import RainEventCatalog

        catalog = RainEventCatalog(device_mac, file_type)

        # Try to get events from session or storage
        events_df = None
        import streamlit as st

        if "rain_events_catalog" in st.session_state:
            events_df = st.session_state["rain_events_catalog"]
        elif catalog.catalog_exists():
            events_df = catalog.load_catalog()

        if events_df is not None and not events_df.empty:
            # Check for ongoing events
            ongoing_events = []
            for _, event in events_df.iterrows():
                is_ongoing = _is_event_ongoing(event)
                if is_ongoing:
                    ongoing_events.append(event)

            # Validate ongoing events with current data
            validated_ongoing_events = []
            for event in ongoing_events:
                if current_eventrainin > 0:
                    validated_ongoing_events.append(event)
                else:
                    logger.info(
                        f"Event {event.get('event_id', 'unknown')[:8]} marked ongoing in catalog but current eventrainin=0, skipping"
                    )

            if validated_ongoing_events:
                return _format_active_event_info(validated_ongoing_events)

    except ImportError:
        # Rain events not available
        logger.debug("Rain events module not available")
    except Exception as e:
        logger.warning(f"Error detecting active rain events: {e}")

    return None


def _is_event_ongoing(event: Dict[str, Any]) -> bool:
    """
    Check if an event is marked as ongoing.

    :param event: Event data dictionary
    :return: True if event is ongoing
    """
    is_ongoing = False
    if "ongoing" in event and event["ongoing"]:
        is_ongoing = True
    elif "flags" in event and event["flags"]:
        flags = event["flags"]
        if isinstance(flags, str):
            flags = json.loads(flags)
        is_ongoing = flags.get("ongoing", False) is True

    return is_ongoing


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

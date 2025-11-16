"""
Header rendering module for weather dashboard.

This module provides header-specific rendering functions including
current conditions display and weather station information.
"""

import streamlit as st
import pandas as pd

from lookout.models.weather import CurrentConditions, ActiveEvent, WeatherData
from lookout.core.weather_events import get_active_rain_event
from lookout.utils.weather_utils import (
    format_wind_direction,
    classify_uv_level,
    format_time_since,
)
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def render_weather_header(device_name: str) -> None:
    """
    Render the main weather header with station name and current conditions.

    :param device_name: Weather station name
    """
    # Two-column header layout
    header_col1, header_col2 = st.columns([1, 1])
    with header_col1:
        st.header(f"Station: {device_name}")
    with header_col2:
        render_current_conditions()


def render_current_conditions() -> None:
    """Render current conditions summary in header column."""
    try:
        weather_data = get_weather_data()

        if weather_data.active_event:
            _render_active_event_display(weather_data)
        else:
            _render_no_event_display(weather_data)

    except Exception as e:
        logger.error(f"Error rendering current conditions: {e}")
        st.caption("Weather conditions temporarily unavailable")


def get_weather_data() -> WeatherData:
    """
    Gather and process current weather data.

    :return: Complete weather data structure
    """
    last_data = st.session_state.get("last_data", {})

    # Get current weather values
    temp_f = last_data.get("tempf", 0)
    humidity = last_data.get("humidity", 0)
    barom_relin = last_data.get("baromrelin", 0)
    wind_speed = last_data.get("windspeedmph", 0)
    wind_dir = last_data.get("winddir", 0)
    uv = last_data.get("uv", 0)
    event_rain = last_data.get("eventrainin", 0) or 0

    # Process weather data
    wind_dir_cardinal = format_wind_direction(wind_dir)
    uv_level = classify_uv_level(uv)
    rain_status, time_since_rain = format_time_since(
        last_data.get("lastRain"), last_data.get("dateutc", 0)
    )

    # Create current conditions
    current = CurrentConditions(
        temperature=temp_f,
        humidity=humidity,
        barometer=barom_relin,
        wind_speed=wind_speed,
        wind_direction=wind_dir_cardinal,
        uv_index=uv,
        uv_level=uv_level,
        rain_status=rain_status,
        time_since_rain=time_since_rain,
    )

    # Get active event if available
    active_event = None
    try:
        if "device" in st.session_state:
            device = st.session_state["device"]
            device_mac = device["macAddress"]
            file_type = "parquet"

            active_event_data = get_active_rain_event(device_mac, file_type, event_rain)
            if active_event_data:
                active_event = ActiveEvent(
                    duration=active_event_data["duration"],
                    total_rain=active_event_data["total_rain"],
                    rain_rate=active_event_data["rain_rate"],
                    start_time=active_event_data["start_time"],
                    duration_hours=active_event_data["duration_h"],
                )
    except Exception as e:
        logger.warning(f"Error getting active event: {e}")

    return WeatherData(current=current, active_event=active_event)


def _render_active_event_display(weather_data: WeatherData) -> None:
    """Render current conditions when active rain event is present."""
    current = weather_data.current
    event = weather_data.active_event

    # Main weather line with active event
    st.caption(
        f"""
    🌡️ {current.temperature:.0f}°F {current.temp_trend} • 💨 {current.wind_speed:.0f}mph {current.wind_direction} • 🌧️ {current.rain_status}    
    **🌧️ ACTIVE EVENT ({event.duration} running)**
    Total: {event.total_rain:.2f}" • Rate: {event.rain_rate:.2f}"/hr • Last rain: {current.time_since_rain} ago
    Started: {event.start_time} • Duration ongoing
    """
    )

    # Secondary weather metrics
    st.caption(
        f"""
    🌊 {current.barometer:.2f}" {current.barom_trend} • 💧 {current.humidity:.0f}% • ☀️ {current.uv_level}
    """
    )


def _render_no_event_display(weather_data: WeatherData) -> None:
    """Render current conditions when no active rain event."""
    current = weather_data.current

    # Main weather line
    st.caption(
        f"""
    🌡️ {current.temperature:.0f}°F {current.temp_trend} • 💨 {current.wind_speed:.0f}mph {current.wind_direction} • 🌧️ {current.rain_status}
    """
    )

    # Secondary weather metrics
    st.caption(
        f"""
    🌊 {current.barometer:.2f}" {current.barom_trend} • 💧 {current.humidity:.0f}% • ☀️ {current.uv_level}
    """
    )

"""
Weather data models and type definitions.

This module provides type-safe data structures for weather information
used throughout the application.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CurrentConditions:
    """Current weather conditions data structure."""

    temperature: float
    humidity: float
    barometer: float
    wind_speed: float
    wind_direction: str
    uv_index: float
    uv_level: str
    rain_status: str
    time_since_rain: str
    temp_trend: str = "→"
    barom_trend: str = "→"


@dataclass
class ActiveEvent:
    """Active rain event information."""

    duration: str
    total_rain: float
    rain_rate: float
    start_time: str
    duration_hours: float


@dataclass
class WeatherData:
    """Complete weather data for current conditions."""

    current: CurrentConditions
    active_event: Optional[ActiveEvent] = None

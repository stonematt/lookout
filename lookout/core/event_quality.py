"""
Event Quality Analysis Module

This module provides comprehensive analysis of rain event data quality and completeness.
It evaluates event data against expected standards for data gaps, completeness, and
rainfall rate calculations.

Key concepts:
- Data completeness: Ratio of actual vs expected 5-minute readings
- Gap analysis: Identifies significant data gaps (>10 minutes) within events
- Rainfall rates: Calculates hourly rates from interval rainfall data
- Quality classification: Rates events as excellent/good/fair/poor based on metrics

Quality thresholds:
- Excellent: ≥95% complete, no gaps
- Good: ≥90% complete, gaps ≤30min
- Fair: ≥75% complete, gaps ≤60min
- Poor: Below fair thresholds

Usage:
    from lookout.core.event_quality import classify_event_quality

    quality_metrics = classify_event_quality(event_dict, archive_df)
"""

from typing import Dict

import numpy as np
import pandas as pd

from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def classify_event_quality(event: Dict, archive_df: pd.DataFrame) -> Dict:
    """
    Analyze event data quality and completeness.

    :param event: Event dictionary from detect_rain_events
    :param archive_df: Complete archive DataFrame
    :return: Quality metrics dictionary
    """
    # Extract event data using timestamps
    archive_copy = archive_df.copy()
    archive_copy["timestamp"] = pd.to_datetime(
        archive_copy["dateutc"], unit="ms", utc=True
    )
    start_time = pd.to_datetime(event["start_time"], utc=True)
    end_time = pd.to_datetime(event["end_time"], utc=True)

    mask = (archive_copy["timestamp"] >= start_time) & (
        archive_copy["timestamp"] <= end_time
    )
    event_data = archive_copy[mask].copy()

    if event_data.empty:
        return {"quality_rating": "invalid", "error": "No data in time range"}

    # Calculate expected vs actual readings based on time span
    event_data = event_data.sort_values("timestamp")
    start_ts = event_data["timestamp"].min()
    end_ts = event_data["timestamp"].max()
    time_span_minutes = (end_ts - start_ts).total_seconds() / 60

    # Expected readings: time_span / 5 (number of 5-min intervals)
    # For a 10-minute span with readings at 0, 5, 10 min, we have 3 readings
    # Time span = 10 min, intervals = 10/5 = 2, but we have 3 readings (start + 2 intervals)
    # So expected = (time_span / 5) + 1 if we count endpoints
    # BUT: if actual readings = 3 and expected = 2, we get >100%
    #
    # Correct approach: expected = number of 5-min intervals that SHOULD exist
    # In 10 minutes: 0min, 5min, 10min = 3 timestamps = (10/5) + 1
    # In 60 minutes: 0, 5, 10, ..., 60 = 13 timestamps = (60/5) + 1
    #
    # Let's use actual interval count instead:
    expected_intervals = time_span_minutes / 5
    expected_readings = expected_intervals + 1  # Include starting point
    actual_readings = len(event_data)

    # Alternative: count actual intervals and compare
    actual_intervals = actual_readings - 1 if actual_readings > 0 else 0
    completeness = (
        actual_intervals / expected_intervals if expected_intervals > 0 else 0
    )

    logger.debug(
        f"Event quality for {event.get('event_id', 'unknown')[:8]}: "
        f"span={time_span_minutes:.1f}min, expected_intervals={expected_intervals:.1f}, "
        f"actual_intervals={actual_intervals}, completeness={completeness:.3f}"
    )

    # Analyze gaps within event (only count gaps > 10 minutes as abnormal)
    time_gaps = event_data["timestamp"].diff().dt.total_seconds() / 60
    abnormal_gaps = time_gaps[time_gaps > 10]  # Only gaps >10min are "gaps"
    max_gap_minutes = abnormal_gaps.max() if len(abnormal_gaps) > 0 else 0
    significant_gaps = len(abnormal_gaps)

    # Calculate rainfall statistics using time-aware rate calculation
    if "dailyrainin" in event_data.columns:
        event_data["time_diff_min"] = (
            event_data["timestamp"].diff().dt.total_seconds() / 60
        )
        event_data.loc[event_data.index[0], "time_diff_min"] = 5

        event_data["interval_rain"] = event_data["dailyrainin"].diff().clip(lower=0)
        if len(event_data) > 0 and pd.isna(event_data["interval_rain"].iloc[0]):
            event_data.loc[event_data.index[0], "interval_rain"] = 0

        event_data["rate_in_per_hr"] = event_data["interval_rain"] / (
            event_data["time_diff_min"] / 60
        )

        normal_readings = event_data[event_data["time_diff_min"] <= 10]

        max_hourly_rate = (
            float(normal_readings["rate_in_per_hr"].max())
            if len(normal_readings) > 0
            else 0.0
        )
        avg_hourly_rate = (
            float(normal_readings["rate_in_per_hr"].mean())
            if len(normal_readings) > 0
            else 0.0
        )

        non_zero_rates = normal_readings[normal_readings["rate_in_per_hr"] > 0]

        logger.debug(
            f"Event {event.get('event_id', 'unknown')[:8]}: "
            f"interval readings={len(event_data)}, "
            f"normal readings={len(normal_readings)}, "
            f"non-zero rates={len(non_zero_rates)}, "
            f"max_rate={max_hourly_rate:.3f}, avg_rate={avg_hourly_rate:.3f}"
        )
    else:
        max_hourly_rate = 0.0
        avg_hourly_rate = 0.0
        logger.warning(
            f"Event {event.get('event_id', 'unknown')[:8]}: dailyrainin column not found"
        )

    # Quality classification based on actual gaps and completeness
    if completeness >= 0.95 and max_gap_minutes == 0:
        quality = "excellent"
    elif completeness >= 0.90 and max_gap_minutes <= 30:
        quality = "good"
    elif completeness >= 0.75 and max_gap_minutes <= 60:
        quality = "fair"
    else:
        quality = "poor"

    return {
        "data_completeness": round(completeness, 3),
        "max_gap_minutes": round(max_gap_minutes, 1),
        "significant_gaps": int(significant_gaps),
        "quality_rating": quality,
        "max_hourly_rate": round(max_hourly_rate, 3),
        "avg_hourly_rate": round(avg_hourly_rate, 3),
        "usable_for_analysis": quality in ["excellent", "good", "fair"],
        "flags": {
            "has_gaps": significant_gaps > 0,
            "low_completeness": completeness < 0.90,
            "interrupted": max_gap_minutes >= 60,
            "ongoing": event.get("ongoing", False),
        },
    }

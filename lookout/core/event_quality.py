"""
Event Quality Analysis Module

This module provides comprehensive analysis of rain event data quality and completeness.
It evaluates event data against expected standards for data gaps, completeness, and
rainfall rate calculations.

Key concepts:
- Data completeness: Ratio of actual vs expected 5-minute readings
- Gap analysis: Identifies significant data gaps (>10 minutes) within events
- Rainfall rates: Calculates hourly rates from interval rainfall data, the
  rain each reading adds to the daily counter (`rain_counter`)
- Quality classification: Rates events as excellent/good/fair/poor based on metrics

Quality thresholds:
- Excellent: ≥95% complete, no gaps
- Good: ≥90% complete, gaps ≤30min
- Fair: ≥75% complete, gaps ≤60min
- Poor: Below fair thresholds

Functions:
- classify_event_quality(event, archive_df): One event's quality metrics.
- recheck_events(catalog, archive_df): Stored events whose recomputed peak
  rate, quality or category differ from the stored values (read-only).

Usage:
    from lookout.core.event_quality import classify_event_quality

    quality_metrics = classify_event_quality(event_dict, archive_df)
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from lookout.core.event_category import event_category
from lookout.core.rain_counter import rain_increments
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
    # Time span = 10 min, intervals = 10/5 = 2, but we have 3 readings
    # (start + 2 intervals)
    # So expected = (time_span / 5) + 1 if we count endpoints
    # BUT: if actual readings = 3 and expected = 2, we get >100%
    #
    # Correct approach: expected = number of 5-min intervals that SHOULD exist
    # In 10 minutes: 0min, 5min, 10min = 3 timestamps = (10/5) + 1
    # In 60 minutes: 0, 5, 10, ..., 60 = 13 timestamps = (60/5) + 1
    #
    # Let's use actual interval count instead:
    expected_intervals = time_span_minutes / 5
    actual_readings = len(event_data)

    # Alternative: count actual intervals and compare
    actual_intervals = actual_readings - 1 if actual_readings > 0 else 0
    completeness = (
        actual_intervals / expected_intervals if expected_intervals > 0 else 0
    )

    logger.debug(
        f"Event quality for {event.get('event_id', 'unknown')[:8]}: "
        f"span={time_span_minutes:.1f}min, "
        f"expected_intervals={expected_intervals:.1f}, "
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

        # A counter drop is the reset: the reading after it keeps its rain
        # (#654). The first reading has nothing before it: 0.
        event_data["interval_rain"] = rain_increments(event_data["dailyrainin"]).fillna(
            0
        )

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
            f"Event {event.get('event_id', 'unknown')[:8]}: "
            "dailyrainin column not found"
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


RECHECK_COLUMNS = (
    "start_ms",
    "stored_peak",
    "peak",
    "stored_quality",
    "quality",
    "stored_category",
    "category",
)


def _peak(value: object) -> Optional[float]:
    """A stored or computed peak rate, rounded; None when absent or NaN."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if np.isnan(number) else round(number, 3)


def recheck_events(catalog: pd.DataFrame, archive_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stored events whose peak rate, quality or category the archive no longer gives.

    Recomputes each event's quality and peak rate from the archive with
    `classify_event_quality`, and its category from its stored total and the
    peak (`event_category`), then compares them with the stored values. Reads
    only: it neither changes `catalog` nor writes anywhere (#654).

    :param catalog: The stored Event Catalog: `start_time`, `end_time`,
        `total_rainfall`, `max_hourly_rate`, `quality_rating`.
    :param archive_df: The archive, rain corrected as the refresh run reads it.
    :return: One row per event that differs, oldest first, with
        `RECHECK_COLUMNS`; `start_ms` is the event's start, epoch ms.
    """
    rows = []
    for event in catalog.sort_values("start_time").to_dict("records"):
        fresh = classify_event_quality(event, archive_df)
        total = float(event["total_rainfall"])
        stored_peak = _peak(event.get("max_hourly_rate"))
        peak = _peak(fresh.get("max_hourly_rate"))
        stored = (stored_peak, event.get("quality_rating"))
        stored_category = event_category(total, stored_peak)
        category = event_category(total, peak)
        if stored == (peak, fresh["quality_rating"]) and stored_category == category:
            continue
        start = pd.Timestamp(pd.to_datetime(event["start_time"], utc=True))
        rows.append(
            (
                int(start.value // 10**6),
                stored_peak,
                peak,
                stored[1],
                fresh["quality_rating"],
                stored_category,
                category,
            )
        )
    return pd.DataFrame(rows, columns=list(RECHECK_COLUMNS))

"""
Rain Event Catalog Management

This module provides functionality to detect, catalog, and manage discrete rain events
using Ambient Weather's built-in event detection via the 'eventrainin' field.

Key concepts:
- Rain events are detected by monitoring eventrainin resets (decreases)
- Each event has start/end times, total rainfall, and quality metrics
- Events are stored in Storj for persistence and incremental updates
- Power outages during events are handled gracefully (data continuity maintained)

Usage:
    catalog = RainEventCatalog("98:CD:AC:22:0D:E5")
    events = catalog.detect_events(archive_df)
    catalog.save_catalog(events)
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lookout.storage.storj import get_s3_client
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def detect_rain_events(
    archive_df: pd.DataFrame,
    min_event_duration_min: int = 15,
    min_total_rainfall: float = 0.01,
    min_gap_detection_min: int = 5,
) -> List[Dict]:
    """
    Detect rain events using Ambient's eventrainin field resets.

    Algorithm:
    1. Find where eventrainin > 0 (event active)
    2. Find where eventrainin decreases (event reset)
    3. Group continuous periods into events
    4. Filter by minimum duration and rainfall

    :param archive_df: DataFrame with weather data including eventrainin
    :param min_event_duration_min: Minimum event duration in minutes
    :param min_total_rainfall: Minimum total rainfall in inches
    :param min_gap_detection_min: Minimum gap to detect separate events
    :return: List of event dictionaries
    """
    if archive_df.empty or "eventrainin" not in archive_df.columns:
        logger.warning("No eventrainin data available for event detection")
        return []

    df = archive_df.copy().sort_values("dateutc").reset_index()
    df["timestamp"] = pd.to_datetime(df["dateutc"], unit="ms", utc=True)
    df["eventrainin_diff"] = df["eventrainin"].diff()

    events = []
    current_event = None

    logger.info(f"Analyzing {len(df)} records for rain events...")

    for idx, row in df.iterrows():
        eventrainin = row.get("eventrainin", 0) or 0  # Handle None values
        timestamp = row["timestamp"]
        eventrainin_diff = row.get("eventrainin_diff", 0) or 0

        # Event start: eventrainin becomes > 0 for first time
        if eventrainin > 0 and current_event is None:
            current_event = {
                "start_idx": idx,
                "start_time": timestamp,
                "start_eventrainin": eventrainin,
                "data_points": [idx],
            }
            logger.debug(f"Event started at {timestamp}")

        # Event continuation: eventrainin still > 0
        elif eventrainin > 0 and current_event is not None:
            current_event["data_points"].append(idx)

        # Event end: eventrainin resets (decreases significantly)
        elif current_event is not None and (
            eventrainin == 0 or eventrainin_diff < -0.001
        ):

            # Get the last positive reading before reset
            if current_event["data_points"]:
                last_idx = current_event["data_points"][-1]
                last_row = df.iloc[last_idx]
            else:
                last_idx = idx
                last_row = row

            # Complete the event
            event = {
                "event_id": str(uuid.uuid4()),
                "start_time": current_event["start_time"],
                "end_time": last_row["timestamp"],
                "start_idx": current_event["start_idx"],
                "end_idx": last_idx,
                "total_rainfall": last_row.get("eventrainin", 0),
                "duration_minutes": (
                    last_row["timestamp"] - current_event["start_time"]
                ).total_seconds()
                / 60,
                "data_point_count": len(current_event["data_points"]),
                "created_at": datetime.now(timezone.utc),
            }

            # Apply filters
            if (
                event["duration_minutes"] >= min_event_duration_min
                and event["total_rainfall"] >= min_total_rainfall
            ):
                events.append(event)
                logger.debug(
                    f"Event completed: {event['duration_minutes']:.1f}min, {event['total_rainfall']:.3f}in"
                )
            else:
                logger.debug(
                    f"Event filtered out: {event['duration_minutes']:.1f}min, {event['total_rainfall']:.3f}in"
                )

            current_event = None

    # Handle ongoing event at end of data
    if current_event is not None:
        last_row = df.iloc[-1]
        event = {
            "event_id": str(uuid.uuid4()),
            "start_time": current_event["start_time"],
            "end_time": last_row["timestamp"],
            "start_idx": current_event["start_idx"],
            "end_idx": len(df) - 1,
            "total_rainfall": last_row.get("eventrainin", 0),
            "duration_minutes": (
                last_row["timestamp"] - current_event["start_time"]
            ).total_seconds()
            / 60,
            "data_point_count": len(current_event["data_points"]),
            "ongoing": True,  # Flag for ongoing events
            "created_at": datetime.now(timezone.utc),
        }

        if (
            event["duration_minutes"] >= min_event_duration_min
            and event["total_rainfall"] >= min_total_rainfall
        ):
            events.append(event)
            logger.info(
                f"Ongoing event detected: {event['duration_minutes']:.1f}min, {event['total_rainfall']:.3f}in"
            )

    logger.info(f"Detected {len(events)} rain events")
    return events


def classify_event_quality(event: Dict, archive_df: pd.DataFrame) -> Dict:
    """
    Analyze event data quality and completeness.

    :param event: Event dictionary from detect_rain_events
    :param archive_df: Complete archive DataFrame
    :return: Quality metrics dictionary
    """
    if event["start_idx"] >= len(archive_df) or event["end_idx"] >= len(archive_df):
        return {"quality_rating": "invalid", "error": "Invalid indices"}

    # Extract event data
    event_data = archive_df.iloc[event["start_idx"] : event["end_idx"] + 1].copy()
    event_data["timestamp"] = pd.to_datetime(event_data["dateutc"], unit="ms", utc=True)

    # Calculate expected vs actual readings
    duration_hours = event["duration_minutes"] / 60
    expected_readings = duration_hours * 12  # 5-minute intervals
    actual_readings = len(event_data)
    completeness = actual_readings / expected_readings if expected_readings > 0 else 0

    # Analyze gaps within event
    event_data = event_data.sort_values("timestamp")
    time_gaps = event_data["timestamp"].diff().dt.total_seconds() / 60
    max_gap_minutes = time_gaps.max() if len(time_gaps) > 0 else 0
    significant_gaps = (time_gaps > 10).sum()  # Gaps > 10 minutes

    # Calculate rainfall statistics
    hourly_rates = event_data.get("hourlyrainin", pd.Series([0]))
    max_hourly_rate = hourly_rates.max() if len(hourly_rates) > 0 else 0
    avg_hourly_rate = hourly_rates.mean() if len(hourly_rates) > 0 else 0

    # Quality classification
    if completeness >= 0.95 and max_gap_minutes <= 15:
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
            "has_gaps": max_gap_minutes > 10,
            "low_completeness": completeness < 0.90,
            "interrupted": max_gap_minutes > 60,
            "ongoing": event.get("ongoing", False),
        },
    }


class RainEventCatalog:
    """
    Manages rain event detection, storage, and retrieval.

    Provides CRUD operations for rain event catalog stored in Storj.
    Handles incremental updates and data quality validation.
    """

    def __init__(self, mac_address: str, bucket: str = "lookout"):
        self.mac_address = mac_address
        self.bucket = bucket
        self.catalog_path = f"events/{mac_address}/rain_events.parquet"
        self.backup_path_prefix = f"events/{mac_address}/backups/rain_events"

    def _get_storage_client(self):
        """Get S3 client for Storj operations"""
        return get_s3_client()

    def catalog_exists(self) -> bool:
        """Check if event catalog exists in storage"""
        try:
            client = self._get_storage_client()
            client.head_object(Bucket=self.bucket, Key=self.catalog_path)
            return True
        except Exception:
            return False

    def load_catalog(self) -> pd.DataFrame:
        """
        Load existing event catalog from Storj.

        :return: DataFrame with event catalog, empty if doesn't exist
        """
        if not self.catalog_exists():
            logger.info("No existing catalog found, returning empty DataFrame")
            return pd.DataFrame()

        try:
            client = self._get_storage_client()
            response = client.get_object(Bucket=self.bucket, Key=self.catalog_path)
            catalog_df = pd.read_parquet(response["Body"])
            logger.info(f"Loaded catalog with {len(catalog_df)} events")
            return catalog_df
        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
            return pd.DataFrame()

    def save_catalog(self, events_df: pd.DataFrame) -> bool:
        """
        Save event catalog to Storj.

        :param events_df: DataFrame with event data
        :return: Success boolean
        """
        try:
            if events_df.empty:
                logger.warning("Attempting to save empty catalog")
                return False

            # Convert timestamps to ensure proper serialization
            events_df = events_df.copy()
            timestamp_cols = ["start_time", "end_time", "created_at", "updated_at"]
            for col in timestamp_cols:
                if col in events_df.columns:
                    events_df[col] = pd.to_datetime(events_df[col], utc=True)

            # Save to parquet in memory
            import io

            buffer = io.BytesIO()
            events_df.to_parquet(buffer, index=False)
            buffer.seek(0)

            # Upload to Storj
            client = self._get_storage_client()
            client.put_object(
                Bucket=self.bucket,
                Key=self.catalog_path,
                Body=buffer.getvalue(),
                ContentType="application/octet-stream",
            )

            logger.info(
                f"Saved catalog with {len(events_df)} events to {self.catalog_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save catalog: {e}")
            return False

    def backup_catalog(self) -> Optional[str]:
        """
        Create timestamped backup of current catalog.

        :return: Backup path if successful, None if failed
        """
        if not self.catalog_exists():
            logger.info("No catalog to backup")
            return None

        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.backup_path_prefix}_{timestamp}.parquet"

            client = self._get_storage_client()

            # Copy current catalog to backup location
            copy_source = {"Bucket": self.bucket, "Key": self.catalog_path}
            client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket,
                Key=backup_path,
            )

            logger.info(f"Catalog backed up to {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Failed to backup catalog: {e}")
            return None

    def detect_and_catalog_events(
        self,
        archive_df: pd.DataFrame,
        backup_existing: bool = True,
    ) -> pd.DataFrame:
        """
        Detect events from archive data and create/update catalog.

        :param archive_df: Complete weather archive DataFrame
        :param backup_existing: Whether to backup existing catalog first
        :return: Complete event catalog DataFrame
        """
        logger.info("Starting event detection and cataloging...")

        # Backup existing catalog if requested
        if backup_existing and self.catalog_exists():
            self.backup_catalog()

        # Detect events
        events_list = detect_rain_events(archive_df)

        if not events_list:
            logger.warning("No events detected")
            return pd.DataFrame()

        # Convert to DataFrame
        events_df = pd.DataFrame(events_list)

        # Add quality metrics
        quality_metrics = []
        logger.info("Analyzing event data quality...")

        for idx, event in events_df.iterrows():
            quality = classify_event_quality(event.to_dict(), archive_df)
            quality_metrics.append(quality)

        # Add quality metrics to DataFrame
        quality_df = pd.DataFrame(quality_metrics)
        events_df = pd.concat([events_df, quality_df], axis=1)

        # Add metadata
        events_df["updated_at"] = datetime.now(timezone.utc)
        events_df["catalog_version"] = "1.0"

        # Save catalog
        if self.save_catalog(events_df):
            logger.info(f"Successfully cataloged {len(events_df)} rain events")
        else:
            logger.error("Failed to save event catalog")

        return events_df

    def get_events_in_period(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Retrieve events within specified date range.

        :param start_date: Start of date range (timezone-aware)
        :param end_date: End of date range (timezone-aware)
        :return: Filtered event DataFrame
        """
        catalog_df = self.load_catalog()

        if catalog_df.empty:
            return pd.DataFrame()

        # Ensure timestamps are datetime objects
        catalog_df["start_time"] = pd.to_datetime(catalog_df["start_time"], utc=True)
        catalog_df["end_time"] = pd.to_datetime(catalog_df["end_time"], utc=True)

        # Filter by date range
        mask = (catalog_df["start_time"] >= start_date) & (
            catalog_df["start_time"] <= end_date
        )

        filtered_df = catalog_df[mask]
        return (
            filtered_df.copy()
            if isinstance(filtered_df, pd.DataFrame)
            else pd.DataFrame()
        )

    def get_event_by_id(self, event_id: str) -> Optional[Dict]:
        """
        Retrieve specific event by ID.

        :param event_id: Event UUID string
        :return: Event dictionary or None if not found
        """
        catalog_df = self.load_catalog()

        if catalog_df.empty:
            return None

        event_rows = catalog_df[catalog_df["event_id"] == event_id]

        if event_rows.empty:
            return None

        return event_rows.iloc[0].to_dict()

    def get_event_data(self, event_id: str, archive_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract raw weather data for specific event.

        :param event_id: Event UUID string
        :param archive_df: Complete archive DataFrame
        :return: Weather data for event period
        """
        event = self.get_event_by_id(event_id)

        if not event:
            return pd.DataFrame()

        start_idx = event.get("start_idx", 0)
        end_idx = event.get("end_idx", 0)

        if start_idx >= len(archive_df) or end_idx >= len(archive_df):
            logger.error(f"Invalid event indices for event {event_id}")
            return pd.DataFrame()

        event_slice = archive_df.iloc[start_idx : end_idx + 1]
        return (
            event_slice.copy()
            if isinstance(event_slice, pd.DataFrame)
            else pd.DataFrame()
        )

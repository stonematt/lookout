"""
Energy Catalog Management

This module provides functionality to calculate, catalog, and manage 15-minute solar energy periods.
Energy periods are calculated using trapezoidal integration of solar radiation measurements.

Key concepts:
- Energy periods are 15-minute aggregations of solar radiation data
- Each period has start/end times and total energy production in kWh
- Periods are stored in Storj for persistence and incremental updates
- Missing data periods are handled gracefully (zero energy for nighttime/missing data)

Usage:
    catalog = EnergyCatalog("98:CD:AC:22:0D:E5")
    periods = catalog.detect_and_calculate_periods(archive_df)
    catalog.save_catalog(periods)
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd

from lookout.core.solar_energy_periods import calculate_15min_energy_periods
from lookout.storage.storj import get_s3_client
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


class EnergyCatalog:
    """
    Manages solar energy period calculation, storage, and retrieval.

    Provides CRUD operations for energy catalog stored in Storj.
    Handles incremental updates and data quality validation.
    """

    def __init__(
        self, mac_address: str, file_type: str = "parquet", bucket: str = "lookout"
    ):
        self.mac_address = mac_address
        self.file_type = file_type
        self.bucket = bucket
        self.catalog_path = f"{mac_address}.energy_catalog.{file_type}"
        self.backup_path_prefix = f"backups/{mac_address}.energy_catalog"

    def _get_storage_client(self):
        """Get S3 client for Storj operations"""
        return get_s3_client()

    def catalog_exists(self) -> bool:
        """Check if energy catalog exists in storage"""
        try:
            client = self._get_storage_client()
            client.head_object(Bucket=self.bucket, Key=self.catalog_path)
            return True
        except Exception:
            return False

    def load_catalog(self) -> pd.DataFrame:
        """
        Load existing energy catalog from Storj.

        :return: DataFrame with energy catalog, empty if doesn't exist
        """
        if not self.catalog_exists():
            logger.info("No existing energy catalog found, returning empty DataFrame")
            return pd.DataFrame()

        try:
            import io

            client = self._get_storage_client()
            response = client.get_object(Bucket=self.bucket, Key=self.catalog_path)
            body = response["Body"].read()
            catalog_df = pd.read_parquet(io.BytesIO(body))

            # Ensure datetime columns are properly parsed and converted to Pacific timezone
            datetime_cols = ["period_start", "period_end"]
            for col in datetime_cols:
                if col in catalog_df.columns:
                    # Convert to datetime first, then to Pacific timezone
                    catalog_df[col] = pd.to_datetime(catalog_df[col], utc=True).dt.tz_convert("America/Los_Angeles")

            logger.info(f"Loaded energy catalog with {len(catalog_df)} periods")
            return catalog_df
        except Exception as e:
            logger.error(f"Failed to load energy catalog: {e}")
            return pd.DataFrame()

    def save_catalog(self, periods_df: pd.DataFrame) -> bool:
        """
        Save energy catalog to Storj.

        :param periods_df: DataFrame with energy period data
        :return: Success boolean
        """
        try:
            if periods_df.empty:
                logger.warning("Attempting to save empty energy catalog")
                return False

            # Convert timestamps to ensure proper serialization
            periods_df = periods_df.copy()
            timestamp_cols = ["period_start", "period_end"]
            for col in timestamp_cols:
                if col in periods_df.columns:
                    periods_df[col] = pd.to_datetime(periods_df[col], utc=True)

            # Save to parquet in memory
            import io

            buffer = io.BytesIO()
            periods_df.to_parquet(buffer, compression='snappy', index=False)
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
                f"Saved energy catalog with {len(periods_df)} periods to {self.catalog_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save energy catalog: {e}")
            return False

    def _create_backup(self, periods_df: pd.DataFrame) -> Optional[str]:
        """
        Create timestamped backup of current catalog.

        :param periods_df: Current catalog DataFrame
        :return: Backup path if successful, None otherwise
        """
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.backup_path_prefix}_{timestamp}.{self.file_type}"

            # Save backup directly
            import io

            buffer = io.BytesIO()
            periods_df.to_parquet(buffer, compression='snappy', index=False)
            buffer.seek(0)

            client = self._get_storage_client()
            client.put_object(
                Bucket=self.bucket,
                Key=backup_path,
                Body=buffer.getvalue(),
                ContentType="application/octet-stream",
            )

            logger.info(f"Energy catalog backed up to {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Failed to backup energy catalog: {e}")
            return None

    def detect_and_calculate_periods(
        self,
        archive_df: pd.DataFrame,
        backup_existing: bool = True,
        auto_save: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate energy periods from archive data and create/update catalog.

        :param archive_df: Complete weather archive DataFrame
        :param backup_existing: Whether to backup existing catalog first
        :param auto_save: Whether to automatically save to storage
        :return: Complete energy catalog DataFrame
        """
        logger.info("Starting energy period calculation and cataloging...")

        # Backup existing catalog if requested and auto-saving
        if auto_save and backup_existing and self.catalog_exists():
            existing_catalog = self.load_catalog()
            if not existing_catalog.empty:
                self._create_backup(existing_catalog)

        # Prepare data: ensure datetime column exists (use 'date' column which is already TZ-aware datetime)
        processed_df = archive_df.copy()
        if 'datetime' not in processed_df.columns:
            # The 'date' column is already a TZ-aware datetime, so use it as 'datetime'
            processed_df['datetime'] = processed_df['date']

        # Calculate energy periods
        periods_df = calculate_15min_energy_periods(processed_df)

        if periods_df.empty:
            logger.warning("No energy periods calculated")
            return pd.DataFrame()

        # Add metadata
        periods_df["updated_at"] = datetime.now(timezone.utc)
        periods_df["catalog_version"] = "1.0"

        # Save catalog if requested
        if auto_save:
            if self.save_catalog(periods_df):
                logger.info(f"Successfully cataloged {len(periods_df)} energy periods")
            else:
                logger.error("Failed to save energy catalog")
        else:
            logger.info(
                f"Energy catalog generated in memory: {len(periods_df)} periods (not saved)"
            )

        return periods_df

    def update_catalog_with_new_data(
        self,
        archive_df: pd.DataFrame,
        existing_catalog: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Update existing catalog with new energy periods from recent archive data.

        For energy catalogs, this is simpler than rain events since we just need to
        calculate periods for any new data that extends beyond the last period.

        :param archive_df: Complete weather archive DataFrame
        :param existing_catalog: Existing catalog DataFrame (will load if None)
        :return: Updated catalog DataFrame
        """
        if existing_catalog is None:
            existing_catalog = self.load_catalog()

        if existing_catalog.empty:
            logger.info("No existing energy catalog, performing full calculation")
            return self.detect_and_calculate_periods(archive_df, auto_save=False)

        # Find the latest period in existing catalog
        existing_catalog["period_start"] = pd.to_datetime(
            existing_catalog["period_start"], utc=True
        ).dt.tz_convert("America/Los_Angeles")
        existing_catalog["period_end"] = pd.to_datetime(
            existing_catalog["period_end"], utc=True
        ).dt.tz_convert("America/Los_Angeles")
        existing_catalog = existing_catalog.sort_values("period_start").reset_index(
            drop=True
        )
        last_period = existing_catalog.iloc[-1]
        last_period_end = last_period["period_end"]

        logger.info(f"Last energy period ends at {last_period_end}")

        # Filter archive data to only what's after the last period
        archive_df = archive_df.copy()
        # Use 'date' column which is already TZ-aware datetime
        archive_df["datetime"] = archive_df["date"]

        # Get data that starts after the last period end
        new_data = archive_df[archive_df["datetime"] > last_period_end].copy()

        if new_data.empty:
            logger.debug("No new data to process for energy periods")
            return existing_catalog

        logger.debug(f"Processing {len(new_data)} new records after {last_period_end}")

        # Calculate new periods from the new data
        new_periods_df = calculate_15min_energy_periods(new_data)

        if new_periods_df.empty:
            logger.debug("No new energy periods calculated")
            return existing_catalog

        # Add metadata to new periods
        new_periods_df["updated_at"] = datetime.now(timezone.utc)
        new_periods_df["catalog_version"] = "1.0"

        # Merge with existing catalog
        updated_catalog = pd.concat(
            [existing_catalog, new_periods_df], ignore_index=True
        )
        updated_catalog = updated_catalog.sort_values("period_start").reset_index(
            drop=True
        )

        logger.info(
            f"Added {len(new_periods_df)} new energy periods to catalog (total: {len(updated_catalog)})"
        )

        return updated_catalog

    def get_catalog_age(self):
        """
        Get age of existing catalog.

        :return: Timedelta since last update, or very large timedelta if no catalog
        """
        try:
            client = self._get_storage_client()
            response = client.head_object(Bucket=self.bucket, Key=self.catalog_path)
            last_modified = response['LastModified']

            age = pd.Timestamp.now(tz='UTC') - pd.Timestamp(last_modified, tz='UTC')
            return age

        except Exception:
            # Return a very large timedelta if catalog doesn't exist or can't be accessed
            return pd.Timedelta(days=999)
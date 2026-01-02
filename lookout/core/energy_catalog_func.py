"""
Functional Energy Catalog Management

Ultra-simplified pure functions for managing energy period catalogs.
Reuses existing storage patterns from storj.py - no reinvention needed.
MAC address automatically determined from session state context.
"""

import pandas as pd

from lookout.config import DEFAULT_BUCKET
from lookout.storage.storj import backup_and_save_catalog
from lookout.core.catalog_utils import load_catalog, catalog_exists
from lookout.core.solar_energy_periods import calculate_15min_energy_periods
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

# Configuration constants
CATALOG_TYPE = "energy"


def detect_and_calculate_periods(archive_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Calculate energy periods - MAC from context, not parameters."""
    bucket = kwargs.get("bucket", DEFAULT_BUCKET)
    auto_save = kwargs.get("auto_save", True)
    backup_existing = kwargs.get("backup_existing", True)
    
    logger.info("Starting energy period calculation and cataloging...")
    
    # Backup existing catalog if requested and auto-saving
    if auto_save and backup_existing and catalog_exists(CATALOG_TYPE, bucket):
        existing_catalog = load_catalog(CATALOG_TYPE, bucket)
        if not existing_catalog.empty:
            try:
                backup_and_save_catalog(existing_catalog, CATALOG_TYPE, bucket)
                logger.info("Existing energy catalog backed up")
            except Exception as e:
                logger.warning(f"Failed to backup existing catalog: {e}")
    
    # Prepare archive data for solar_energy_periods - convert 'date' to 'datetime'
    archive_df = archive_df.copy()
    if 'date' in archive_df.columns and 'datetime' not in archive_df.columns:
        archive_df['datetime'] = archive_df['date']
    
    # Calculate energy periods
    periods_df = calculate_15min_energy_periods(archive_df)

    if periods_df.empty:
        logger.warning("No energy periods calculated")
        return pd.DataFrame()

    # Save catalog if requested
    if auto_save:
        try:
            backup_and_save_catalog(periods_df, CATALOG_TYPE, bucket)
            logger.info(f"Successfully cataloged {len(periods_df)} energy periods")
        except Exception as e:
            logger.error(f"Failed to save energy catalog: {e}")
            # Return periods even if save failed
    else:
        logger.info(
            f"Energy catalog generated in memory: {len(periods_df)} periods (not saved)"
        )

    return periods_df


def load_energy_catalog(bucket: str = DEFAULT_BUCKET) -> pd.DataFrame:
    """Load energy catalog from storage."""
    df = load_catalog(CATALOG_TYPE, bucket)
    if not df.empty:
        logger.info(f"Loaded energy catalog: {len(df)} periods")
    return df


def save_energy_catalog(df: pd.DataFrame, bucket: str = DEFAULT_BUCKET) -> bool:
    """Save energy catalog to storage with backup."""
    if df.empty:
        logger.warning("Attempting to save empty energy catalog")
        return False

    try:
        backup_and_save_catalog(df, CATALOG_TYPE, bucket)
        logger.info(f"Saved energy catalog: {len(df)} periods")
        return True
    except Exception as e:
        logger.error(f"Failed to save energy catalog: {e}")
        return False


def update_energy_catalog(
    archive_df: pd.DataFrame, bucket: str = DEFAULT_BUCKET
) -> pd.DataFrame:
    """Update energy catalog with new data from archive."""
    existing_catalog = load_energy_catalog(bucket)

    if existing_catalog.empty:
        logger.info("No existing energy catalog, performing full calculation")
        return detect_and_calculate_periods(archive_df, auto_save=False, bucket=bucket)

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
    mask = archive_df["datetime"] > last_period_end
    new_data = archive_df[mask].copy()

    # Ensure we have a DataFrame
    if new_data.empty:
        logger.debug("No new data to process for energy periods")
        return existing_catalog

    logger.debug(f"Processing {len(new_data)} new records after {last_period_end}")

    # Calculate new periods from the new data
    if isinstance(new_data, pd.DataFrame):
        new_periods_df = calculate_15min_energy_periods(new_data)
    else:
        logger.error("new_data is not a DataFrame, cannot calculate periods")
        return existing_catalog

    if new_periods_df.empty:
        logger.debug("No new energy periods calculated")
        return existing_catalog

    # Merge with existing catalog
    updated_catalog = pd.concat([existing_catalog, new_periods_df], ignore_index=True)
    updated_catalog = updated_catalog.sort_values("period_start").reset_index(drop=True)

    logger.info(
        f"Added {len(new_periods_df)} new energy periods to catalog (total: {len(updated_catalog)})"
    )

    return updated_catalog


def energy_catalog_exists(bucket: str = DEFAULT_BUCKET) -> bool:
    """Check if energy catalog exists in storage."""
    return catalog_exists(CATALOG_TYPE, bucket)

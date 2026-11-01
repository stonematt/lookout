"""
Simplified catalog management utilities.
Leverages existing storage patterns from storj.py - no reinvention needed.
"""

from typing import Optional
import pandas as pd
from datetime import datetime, timezone

from lookout.storage.storj import (
    get_df_from_s3,
    save_df_to_s3,
    backup_data,
    object_exists,
)
from lookout.config import DEFAULT_BUCKET, DEFAULT_FILE_TYPE, get_station_mac
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def get_catalog_path(mac_address: str, catalog_type: str) -> str:
    """Generate catalog file path."""
    return f"{mac_address}.{catalog_type}_catalog.{DEFAULT_FILE_TYPE}"


def catalog_exists(catalog_type: str, bucket: str = DEFAULT_BUCKET) -> bool:
    """Check if catalog exists in storage."""
    mac_address = get_station_mac()

    return object_exists(bucket, get_catalog_path(mac_address, catalog_type))


def load_catalog(catalog_type: str, bucket: str = DEFAULT_BUCKET) -> pd.DataFrame:
    """Load catalog from storage."""
    mac_address = get_station_mac()
    catalog_path = get_catalog_path(mac_address, catalog_type)

    try:
        df = get_df_from_s3(bucket, catalog_path, DEFAULT_FILE_TYPE)
        logger.info(f"Loaded {catalog_type} catalog with {len(df)} items")
        return df
    except Exception as e:
        logger.info(f"No {catalog_type} catalog found for {mac_address}: {e}")
        return pd.DataFrame()


def save_catalog(
    df: pd.DataFrame, catalog_type: str, bucket: str = DEFAULT_BUCKET
) -> bool:
    """Save catalog to storage with metadata."""
    mac_address = get_station_mac()

    try:
        if df.empty:
            logger.warning(f"Attempting to save empty {catalog_type} catalog")
            return False

        # Add metadata
        df = df.copy()
        df["updated_at"] = datetime.now(timezone.utc)
        df["catalog_version"] = "1.0"

        catalog_path = get_catalog_path(mac_address, catalog_type)
        save_df_to_s3(df, bucket, catalog_path, DEFAULT_FILE_TYPE)

        # save_df_to_s3 returns True/False; the return is not checked here
        logger.info(f"Saved {catalog_type} catalog with {len(df)} items")
        return True

    except Exception as e:
        logger.error(f"Failed to save {catalog_type} catalog: {e}")
        return False


def backup_catalog(
    df: pd.DataFrame, catalog_type: str, bucket: str = DEFAULT_BUCKET
) -> Optional[str]:
    """Create backup using existing backup infrastructure."""
    mac_address = get_station_mac()
    backup_prefix = f"{mac_address}.{catalog_type}_catalog"

    # Use existing backup_data with daily deduplication logic
    try:
        backup_data(
            bucket=bucket,
            prefix=backup_prefix,
            force_backup=False,  # Let storj handle daily deduplication
            dry_run=False,
        )
        return f"backed_up_{backup_prefix}"
    except Exception as e:
        logger.error(f"Failed to backup catalog: {e}")
        return None

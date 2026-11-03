"""
Functional Energy Catalog Management

Ultra-simplified pure functions for managing energy period catalogs.
Reuses existing storage patterns from storj.py - no reinvention needed.
The station is always named by the caller (`mac_address`); there is no
pinned station (#271).
"""

from typing import Optional

import pandas as pd

from lookout.config import DEFAULT_BUCKET
from lookout.storage.storj import backup_and_save_catalog
from lookout.core.catalog_utils import load_catalog
from lookout.core.solar_energy_periods import (
    calculate_15min_energy_periods,
    recompute_from_period,
)
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

# Configuration constants
CATALOG_TYPE = "energy"


def detect_and_calculate_periods(archive_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Calculate energy periods from an archive.

    With `auto_save`, the save goes through `backup_and_save_catalog`, which
    backs up the stored catalog first and saves nothing when that fails.

    :param archive_df: The archive.
    :param kwargs: `bucket` (default `DEFAULT_BUCKET`), `auto_save` (default
        True) and `mac_address`, the station to save under (needed with
        `auto_save`).
    :return: The periods; empty when none could be calculated.
    :raises ValueError: If `auto_save` is on and no `mac_address` is given.
    """
    bucket = kwargs.get("bucket", DEFAULT_BUCKET)
    auto_save = kwargs.get("auto_save", True)
    mac_address = kwargs.get("mac_address")
    if auto_save and not mac_address:
        raise ValueError("auto_save needs the station's mac_address")

    logger.info("Starting energy period calculation and cataloging...")

    # Prepare archive data for solar_energy_periods - convert 'date' to 'datetime'
    archive_df = archive_df.copy()
    if "date" in archive_df.columns and "datetime" not in archive_df.columns:
        archive_df["datetime"] = archive_df["date"]

    # Calculate energy periods
    periods_df = calculate_15min_energy_periods(archive_df)

    if periods_df.empty:
        logger.warning("No energy periods calculated")
        return pd.DataFrame()

    # Save catalog if requested
    if auto_save:
        try:
            backup_and_save_catalog(periods_df, CATALOG_TYPE, bucket, mac_address)
            logger.info(f"Successfully cataloged {len(periods_df)} energy periods")
        except Exception as e:
            logger.error(f"Failed to save energy catalog: {e}")
            # Return periods even if save failed
    else:
        logger.info(
            f"Energy catalog generated in memory: {len(periods_df)} periods (not saved)"
        )

    return periods_df


def load_energy_catalog(mac_address: str, bucket: str = DEFAULT_BUCKET) -> pd.DataFrame:
    """
    Load one station's energy catalog from storage.

    :param mac_address: The station.
    :param bucket: The bucket.
    :return: The catalog; empty when none is stored.
    """
    df = load_catalog(CATALOG_TYPE, mac_address, bucket)
    if not df.empty:
        logger.info(f"Loaded energy catalog: {len(df)} periods")
    return df


def save_energy_catalog(
    df: pd.DataFrame, mac_address: str, bucket: str = DEFAULT_BUCKET
) -> bool:
    """
    Save one station's energy catalog, backing up the stored one first.

    :param df: The catalog.
    :param mac_address: The station.
    :param bucket: The bucket.
    :return: True when saved; False when empty or the backup or save failed.
    """
    if df.empty:
        logger.warning("Attempting to save empty energy catalog")
        return False

    try:
        backup_and_save_catalog(df, CATALOG_TYPE, bucket, mac_address)
        logger.info(f"Saved energy catalog: {len(df)} periods")
        return True
    except Exception as e:
        logger.error(f"Failed to save energy catalog: {e}")
        return False


def update_energy_catalog(
    archive_df: pd.DataFrame,
    existing_catalog: pd.DataFrame = None,
    bucket: str = DEFAULT_BUCKET,
    mac_address: Optional[str] = None,
) -> pd.DataFrame:
    """
    Update energy catalog with new data from archive.

    The newest stored period and every later one are recomputed (see
    `recompute_from_period`) and replace what was stored.

    :param archive_df: Complete weather archive DataFrame
    :param existing_catalog: Existing catalog DataFrame (will load if None)
    :param bucket: S3 bucket name
    :param mac_address: The station, to load its catalog when
        `existing_catalog` is None.
    :return: Updated catalog DataFrame
    :raises ValueError: If the catalog must be loaded and no station is given.
    """
    if existing_catalog is None:
        if not mac_address:
            raise ValueError("Loading the stored catalog needs the station's MAC")
        existing_catalog = load_energy_catalog(mac_address, bucket)

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
    # The newest period may still have been filling when it was stored, so it
    # is recomputed with what came after and replaces its stored value.
    last_period_start = existing_catalog.iloc[-1]["period_start"]

    logger.info(f"Recomputing energy periods from {last_period_start}")

    archive_df = archive_df.copy()
    # Use 'date' column which is already TZ-aware datetime
    archive_df["datetime"] = archive_df["date"]

    new_periods_df = recompute_from_period(archive_df, last_period_start)

    if new_periods_df.empty:
        logger.debug("No new energy periods calculated")
        return existing_catalog

    kept = existing_catalog[existing_catalog["period_start"] < last_period_start]
    updated_catalog = pd.concat([kept, new_periods_df], ignore_index=True)
    updated_catalog = updated_catalog.sort_values("period_start").reset_index(drop=True)

    logger.info(
        f"Recomputed {len(new_periods_df)} energy periods "
        f"(total: {len(updated_catalog)})"
    )

    return updated_catalog

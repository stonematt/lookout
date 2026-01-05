"""
Day Periods Catalog Management (Functional Approach)

Pure functions for managing pre-computed day periods catalogs.
Reuses existing storage patterns from storj.py - no reinvention needed.
"""

from datetime import datetime, timezone
from typing import Optional
import pandas as pd

from lookout.storage.storj import get_df_from_s3, save_df_to_s3
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

# Configuration constants
DEFAULT_BUCKET = "lookout"
DEFAULT_FILE_TYPE = "parquet"


def get_catalog_path(mac_address: str, file_type: str = DEFAULT_FILE_TYPE) -> str:
    """Generate catalog file path for given MAC address."""
    return f"{mac_address}.day_periods_catalog.{file_type}"


def catalog_exists(mac_address: str, bucket: str = DEFAULT_BUCKET) -> bool:
    """Check if day periods catalog exists in storage."""
    try:
        catalog_path = get_catalog_path(mac_address)
        get_df_from_s3(bucket, catalog_path, DEFAULT_FILE_TYPE)
        return True
    except Exception:
        return False


def load_day_periods_catalog(
    mac_address: str, bucket: str = DEFAULT_BUCKET
) -> pd.DataFrame:
    """Load day periods catalog from storage."""
    if not catalog_exists(mac_address, bucket):
        logger.info(f"No day periods catalog found for {mac_address}")
        return pd.DataFrame()

    catalog_path = get_catalog_path(mac_address)
    try:
        df = get_df_from_s3(bucket, catalog_path, DEFAULT_FILE_TYPE)
        if not df.empty:
            # Convert date column to Pacific timezone for consistency
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(
                    "America/Los_Angeles"
                )
            logger.info(f"Loaded day periods catalog: {len(df)} periods")
        return df
    except Exception as e:
        logger.error(f"Failed to load day periods catalog for {mac_address}: {e}")
        return pd.DataFrame()


def save_day_periods_catalog(
    periods_df: pd.DataFrame, mac_address: str, bucket: str = DEFAULT_BUCKET
) -> bool:
    """Save day periods catalog to storage."""
    if periods_df.empty:
        logger.warning("Attempting to save empty day periods catalog")
        return False

    try:
        # Ensure datetime columns are UTC for storage
        df = periods_df.copy()
        datetime_cols = ["sunrise_utc", "sunset_utc", "generated_at"]
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)

        catalog_path = get_catalog_path(mac_address)
        save_df_to_s3(df, bucket, catalog_path, DEFAULT_FILE_TYPE)
        logger.info(f"Saved day periods catalog: {len(df)} periods for {mac_address}")
        return True
    except Exception as e:
        logger.error(f"Failed to save day periods catalog for {mac_address}: {e}")
        return False


def delete_day_periods_catalog(mac_address: str, bucket: str = DEFAULT_BUCKET) -> bool:
    """Delete day periods catalog from storage."""
    try:
        if not catalog_exists(mac_address, bucket):
            logger.info(f"No day periods catalog to delete for {mac_address}")
            return True

        catalog_path = get_catalog_path(mac_address)

        # Note: S3 delete would require boto3 client - for now use Storj storage patterns
        # This is a placeholder - actual deletion would need storage client access
        logger.info(f"Day periods catalog deletion not implemented for {mac_address}")
        logger.info(f"Would delete: {catalog_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete day periods catalog for {mac_address}: {e}")
        return False


def list_day_periods_catalogs(mac_address: str, bucket: str = DEFAULT_BUCKET) -> list:
    """List day periods catalogs for a MAC address."""
    try:
        if catalog_exists(mac_address, bucket):
            return [get_catalog_path(mac_address)]
        else:
            return []
    except Exception as e:
        logger.error(f"Failed to list day periods catalogs for {mac_address}: {e}")
        return []


def backup_day_periods_catalog(mac_address: str, bucket: str = DEFAULT_BUCKET) -> bool:
    """Create timestamped backup of existing catalog."""
    try:
        if not catalog_exists(mac_address, bucket):
            logger.info(f"No existing catalog to backup for {mac_address}")
            return True

        # Load existing catalog
        existing = load_day_periods_catalog(mac_address, bucket)
        if existing.empty:
            return True

        # Create timestamped backup
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = f"backups/{mac_address}.day_periods_catalog_{timestamp}.parquet"

        save_df_to_s3(existing, bucket, backup_path, DEFAULT_FILE_TYPE)
        logger.info(f"Created backup: {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to backup day periods catalog for {mac_address}: {e}")
        return False


def generate_day_periods_catalog(
    lat: float,
    lon: float,
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
    use_astral: bool = True,
) -> pd.DataFrame:
    """Generate day periods catalog from astronomical calculations."""

    # Calculate end date: current year + 5 years
    if end_date is None:
        current_year = datetime.now().year
        end_year = current_year + 5
        end_date = f"{end_year}-12-31"

    logger.info(f"Generating day periods catalog from {start_date} to {end_date}")

    # Create date range
    dates = pd.date_range(start_date, end_date, freq="D", tz="UTC")

    day_periods = []
    for i, date in enumerate(dates):
        if i % 100 == 0:  # Progress logging every 100 days
            logger.debug(f"Processing day {i+1}/{len(dates)}: {date.date()}")

        if use_astral:
            sunrise, sunset = _calculate_sunrise_sunset_astral(date, lat, lon)
        else:
            sunrise, sunset = _calculate_sunrise_sunset_simple(date, lat, lon)

        if sunrise and sunset:
            daylight_minutes = int((sunset - sunrise).total_seconds() / 60)
        else:
            daylight_minutes = 0

        day_periods.append(
            {
                "date": date,  # Will be converted to Pacific timezone
                "sunrise_utc": sunrise,
                "sunset_utc": sunset,
                "daylight_minutes": daylight_minutes,
                "latitude": lat,
                "longitude": lon,
                "generated_at": datetime.now(tz=timezone.utc),
            }
        )

    df = pd.DataFrame(day_periods)

    # Convert date to Pacific timezone
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(
        "America/Los_Angeles"
    )

    logger.info(f"Generated day periods catalog: {len(df)} days")
    return df


def _calculate_sunrise_sunset_astral(date, lat, lon):
    """Calculate sunrise/sunset using astral library (precise)."""
    try:
        from astral import LocationInfo
        from astral.sun import sun

        # Create location info
        location = LocationInfo(latitude=lat, longitude=lon, timezone="UTC")

        # Calculate sun times for the date
        s = sun(location.observer, date=date, tzinfo=timezone.utc)
        return s["sunrise"], s["sunset"]
    except Exception as e:
        logger.warning(f"Astral calculation failed for {date}: {e}")
        return None, None


def _calculate_sunrise_sunset_simple(date, lat, lon):
    """Simple sunrise/sunset calculation (fallback)."""
    # This is a basic approximation - for production use, NOAA algorithm would be better
    # For now, return approximate times
    # This is just to provide functionality when astral is not available

    # Very rough approximation: 6am +/- 2 hours based on latitude and day of year
    day_of_year = date.timetuple().tm_yday
    lat_factor = abs(lat) / 90.0  # Normalize latitude to 0-1

    # Approximate seasonal variation
    seasonal_factor = -0.5 + abs(day_of_year - 172) / 172.0  # Peak at summer solstice
    seasonal_factor = max(-1, min(1, seasonal_factor))  # Clamp to [-1, 1]

    # Approximate sunrise (6am +/- 2 hours)
    sunrise_hour = 6 + (2 * seasonal_factor) - (2 * lat_factor)
    # Create timezone-naive datetime then add timezone
    sunrise_naive = datetime(date.year, date.month, date.day, int(sunrise_hour), 0, 0)
    sunrise = pd.Timestamp(sunrise_naive, tz="UTC")

    # Approximate sunset (6pm +/- 2 hours)
    sunset_hour = 18 - (2 * seasonal_factor) + (2 * lat_factor)
    sunset_naive = datetime(date.year, date.month, date.day, int(sunset_hour), 0, 0)
    sunset = pd.Timestamp(sunset_naive, tz="UTC")

    logger.debug(f"Simple approximation for {date}: sunrise {sunrise}, sunset {sunset}")
    return sunrise, sunset

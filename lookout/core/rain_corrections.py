"""
Rain Carryover Detection and Correction

Detects and corrects rain accumulation errors caused by equipment outages
spanning midnight. When the station is offline at midnight, dailyrainin
fails to reset, carrying forward the previous day's total.

Detection Algorithm:
- Identify day transitions where first record is NOT between 00:00-00:05
- Flag as carryover if first_dailyrainin ≈ prev_day_last_dailyrainin (±0.05")
- Requires prev_day value > 0.01" (something to carry over)

Sensitivity:
- Only catches outages spanning midnight
- 0.05" tolerance for sensor noise
- 0.01" floor ignores trace amounts
- Expected: ~1-2 carryovers per year based on historical data
"""

from typing import Dict, List
import pandas as pd

from lookout.storage.storj import get_json_from_s3, save_json_to_s3, object_exists
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def _get_catalog_path(mac_address: str, bucket: str = "lookout") -> str:
    """Return S3 path for corrections catalog."""
    return f"s3://{bucket}/{mac_address}.rain_corrections.json"


def catalog_exists(mac_address: str, bucket: str = "lookout") -> bool:
    """Check if corrections catalog exists in Storj."""
    return object_exists(bucket, f"{mac_address}.rain_corrections.json")


def load_catalog(mac_address: str, bucket: str = "lookout") -> Dict:
    """Load corrections catalog from Storj."""
    return get_json_from_s3(bucket, f"{mac_address}.rain_corrections.json")


def save_catalog(catalog: Dict, mac_address: str, bucket: str = "lookout") -> bool:
    """Save corrections catalog to Storj."""
    return save_json_to_s3(catalog, bucket, f"{mac_address}.rain_corrections.json")


def detect_carryovers(archive_df: pd.DataFrame) -> List[Dict]:
    """
    Detect midnight gaps with carryover.

    Scans archive for day transitions where:
    1. First record of day missed midnight window (not 00:00-00:05)
    2. First dailyrainin ≈ previous day's last dailyrainin
    3. Previous day's value > 0.01"

    :param archive_df: Full archive DataFrame (reverse sorted by dateutc)
    :return: List of carryover dicts with affected_date, carryover_amount, etc.
    """
    if archive_df.empty or "dailyrainin" not in archive_df.columns:
        return []

    df = archive_df.copy()

    # Archive is reverse sorted by dateutc, sort ascending for analysis
    df = df.sort_values("dateutc", ascending=True)

    # Extract calendar day from date column
    df["calendar_day"] = pd.to_datetime(df["date"]).dt.date

    days = sorted(df["calendar_day"].unique())
    carryovers = []

    for i, day in enumerate(days[1:], 1):
        day_data = df[df["calendar_day"] == day]
        first_ts = pd.to_datetime(day_data["date"].iloc[0])

        # Check if first record missed midnight window
        if first_ts.hour != 0 or first_ts.minute > 5:
            prev_day = days[i - 1]
            prev_data = df[df["calendar_day"] == prev_day]

            first_val = day_data["dailyrainin"].iloc[0]
            last_val = prev_data["dailyrainin"].iloc[-1]

            # Carryover if values match (didn't reset) and non-trivial
            if abs(first_val - last_val) < 0.05 and last_val > 0.01:
                carryovers.append(
                    {
                        "affected_date": str(day),
                        "carryover_amount": round(float(last_val), 3),
                        "source_date": str(prev_day),
                        "gap_start": str(prev_data["date"].iloc[-1]),
                        "gap_end": str(day_data["date"].iloc[0]),
                    }
                )
                logger.info(
                    f'Detected carryover: {day} = {last_val:.3f}" '
                    f"(gap: {prev_data['date'].iloc[-1]} → {day_data['date'].iloc[0]})"
                )

    return carryovers


def apply_corrections(
    df: pd.DataFrame, mac_address: str, bucket: str = "lookout"
) -> pd.DataFrame:
    """
    Apply the stored carryover corrections to rain accumulation fields.

    Loads `<MAC>.rain_corrections.json` and hands its corrections to
    `correct_counters`, which holds the correction rules.

    :param df: Archive DataFrame (reverse sorted by dateutc)
    :param mac_address: Device MAC address
    :param bucket: S3 bucket name
    :return: Corrected DataFrame (preserves original sort order); `df` itself
        when no catalog or no correction is stored.
    """
    catalog = load_catalog(mac_address, bucket)
    return correct_counters(df, (catalog or {}).get("corrections") or [])


def correct_counters(df: pd.DataFrame, corrections: List[Dict]) -> pd.DataFrame:
    """
    Subtract each carryover from the rain counters within its reset period.

    - dailyrainin: affected_date only
    - weeklyrainin: affected_date through Saturday (week resets Sunday)
    - monthlyrainin: affected_date through end of month
    - yearlyrainin: affected_date through end of year

    A row's day is its `date` column's calendar day (station-local in the
    archive). Counters are clipped at zero. Vectorized; the one place the
    correction rules live, for the archive and the live reading alike.

    :param df: Rows with `date` and the counter columns, any order.
    :param corrections: Catalog entries with `affected_date` and
        `carryover_amount`.
    :return: A corrected copy, same order and index; `df` itself when there
        are no corrections.
    """
    if not corrections:
        return df

    df = df.copy()
    dates = pd.to_datetime(df["date"]).dt.date

    for c in corrections:
        amt = c["carryover_amount"]
        affected = pd.Timestamp(c["affected_date"]).date()
        affected_ts = pd.Timestamp(c["affected_date"])

        # Daily: just affected date
        mask = dates == affected
        df.loc[mask, "dailyrainin"] -= amt
        logger.info(f"Applied daily correction: {c['affected_date']} = -{amt:.3f}\"")

        # Weekly: through Saturday (week resets Sunday)
        week_end = (affected_ts + pd.offsets.Week(weekday=5)).date()
        mask = (dates >= affected) & (dates <= week_end)
        df.loc[mask, "weeklyrainin"] -= amt
        logger.info(
            f"Applied weekly correction: {c['affected_date']} through {week_end}"
            f' = -{amt:.3f}"'
        )

        # Monthly: through end of month
        month_end = (affected_ts + pd.offsets.MonthEnd(0)).date()
        mask = (dates >= affected) & (dates <= month_end)
        df.loc[mask, "monthlyrainin"] -= amt
        logger.info(
            f"Applied monthly correction: {c['affected_date']} through {month_end}"
            f' = -{amt:.3f}"'
        )

        # Yearly: through end of year
        year_end = pd.Timestamp(f"{affected_ts.year}-12-31").date()
        mask = (dates >= affected) & (dates <= year_end)
        df.loc[mask, "yearlyrainin"] -= amt
        logger.info(
            f"Applied yearly correction: {c['affected_date']} through {year_end}"
            f' = -{amt:.3f}"'
        )

    # Clip negative values
    for col in ["dailyrainin", "weeklyrainin", "monthlyrainin", "yearlyrainin"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    return df

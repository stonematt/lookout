import json
import logging
import time
from datetime import datetime, timedelta, timezone

from lookout.api import ambient_client

# --- Config ---
MAC = "98:CD:AC:22:0D:E5"
LIMIT = 288  # max per request
SLEEP_SECONDS = 1

# --- Logger setup ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_data_at(end_dt):
    """Return DataFrame for records at/before this UTC datetime."""
    ts_ms = int(end_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    logger.debug(
        f"Fetching history: limit={LIMIT}, end_date={ts_ms} ({end_dt.isoformat()})"
    )
    return ambient_client.get_device_history(MAC, limit=LIMIT, end_date=ts_ms)


def has_data_at(end_dt):
    df = get_data_at(end_dt)
    logger.info(f"Data at {end_dt.isoformat()} → {len(df)} rows")
    return not df.empty


def find_earliest_record_minute():
    logger.info(f"🔎 Finding earliest record for device {MAC} ...")
    now = datetime.now(timezone.utc)

    # Step 1: Exponential search backwards
    step = 30
    earliest_with_data = now
    logger.info("Starting exponential search...")
    while has_data_at(earliest_with_data):
        logger.info(
            f"✅ Data found {earliest_with_data.date()}, stepping back {step} days..."
        )
        earliest_with_data -= timedelta(days=step)
        time.sleep(SLEEP_SECONDS)
        step *= 2
        if step > 365:
            break

    # Step 2: Binary search to the minute
    logger.info("Starting binary search...")
    low = earliest_with_data
    high = earliest_with_data + timedelta(days=step // 2)
    while (high - low) > timedelta(minutes=1):
        mid = low + (high - low) / 2
        logger.debug(
            f"Checking {mid.isoformat()} between {low.isoformat()} and {high.isoformat()}"
        )
        if has_data_at(mid):
            high = mid
        else:
            low = mid
        time.sleep(SLEEP_SECONDS)

    # Step 3: Fetch and display earliest row
    df = get_data_at(high)
    if not df.empty:
        earliest_row = df.sort_values("dateutc").iloc[0].to_dict()
        logger.info("📅 Earliest record found:")
        print(json.dumps(earliest_row, indent=2, default=str))
    else:
        logger.warning("⚠ No data found.")


if __name__ == "__main__":
    find_earliest_record_minute()

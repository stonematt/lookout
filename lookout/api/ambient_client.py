"""
ambient_client.py: Lightweight interface for interacting with Ambient Weather API
using direct requests. Designed to replace ambient_api.ambientapi with clearer,
testable, and debuggable calls.

Functions:
- get_devices()
- get_device_by_mac(mac)
- get_device_history(mac, limit, end_date)

Requires:
- Secrets AMBIENT_ENDPOINT, AMBIENT_API_KEY, AMBIENT_APPLICATION_KEY, read via
  lookout.secrets_loader.get_secret on each request, never at import.
"""

import time
from datetime import datetime, timezone
from pprint import pprint
from typing import Dict, List, Optional

import pandas as pd
import requests

from lookout.secrets_loader import get_secret
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def _base_url() -> str:
    """Return the Ambient endpoint (e.g. https://rt.ambientweather.net/v1)."""
    return get_secret("AMBIENT_ENDPOINT").rstrip("/")


def _auth_params() -> Dict[str, str]:
    """Return the Ambient API and application keys as request params."""
    return {
        "apiKey": get_secret("AMBIENT_API_KEY"),
        "applicationKey": get_secret("AMBIENT_APPLICATION_KEY"),
    }


def format_ts_utc(ms_ts: Optional[int]) -> str:
    """Convert a UNIX timestamp in ms to a human-readable UTC string."""
    if ms_ts is None:
        return "latest"
    try:
        return datetime.fromtimestamp(ms_ts / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    except Exception:
        return "Invalid timestamp"


def get_devices() -> List[Dict]:
    """
    Retrieve the list of devices associated with the API keys.

    :return: List of device dicts.
    """
    url = f"{_base_url()}/devices"
    params = _auth_params()
    time.sleep(1)
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        logger.error(f"Device list failed: {resp.status_code} {resp.text}")
        return []
    return resp.json()


def get_device_by_mac(mac: str) -> Optional[Dict]:
    """
    Fetch a specific device by MAC address.

    :param mac: MAC address of the device.
    :return: Device dict or None.
    """
    devices = get_devices()
    for d in devices:
        if d.get("macAddress") == mac:
            return d
    return None


def get_device_history(
    mac: str, limit: int = 288, end_date: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch historical weather data for a specific device.

    Uses `get_device_history_raw()` to retrieve the data and normalizes it into a DataFrame.

    :param mac: MAC address of the device to fetch data for.
    :param limit: Max number of records to return (default: 288).
    :param end_date: Optional UNIX timestamp in milliseconds for the end of the time range.
    :return: A DataFrame of weather data or empty DataFrame on failure.
    """
    try:
        # human = format_ts_utc(end_date) if end_date else "latest"
        # logger.info(f"Fetching history: {mac}, limit={limit}, end_date={end_date} ({human})")

        raw = get_device_history_raw(mac, limit=limit, end_date=end_date)
        if not raw:
            logger.debug("No data returned from device history.")
            return pd.DataFrame()

        df = pd.json_normalize(raw)
        if df.empty:
            logger.warning("Normalized DataFrame is empty.")
            return pd.DataFrame()

        df.sort_values(by="dateutc", inplace=True)
        return df

    except Exception as e:
        logger.exception(f"Error during device history fetch: {e}")
        return pd.DataFrame()


def get_device_history_raw(
    mac: str, limit: int = 288, end_date: Optional[int] = None
) -> list[dict]:
    """
    Fetch raw device history from the Ambient Weather Network.

    :param mac: Device MAC address.
    :param limit: Number of records to retrieve.
    :param end_date: Optional end timestamp (ms).
    :return: List of raw data records (dict).
    """
    url = f"{_base_url()}/devices/{mac}"
    params = {**_auth_params(), "limit": limit}
    if end_date is not None:
        params["endDate"] = end_date

    human = format_ts_utc(end_date) if end_date else "latest"
    logger.info(
        f"Fetching history: {mac}, limit={limit}, end_date={end_date} ({human})"
    )

    for attempt in range(3):
        try:
            time.sleep(1)  # Ambient API rate limit: 1 request/second
            resp = requests.get(url, params=params)
            if resp.status_code == 429:
                logger.warning(
                    f"Rate limited (429) on history fetch. Retrying in 1s... attempt {attempt + 1}/3"
                )
                time.sleep(1)
                continue
            if resp.status_code != 200:
                logger.error(f"History fetch failed: {resp.status_code} {resp.text}")
                return []
            return resp.json()
        except Exception as e:
            logger.error(f"Request error while fetching raw history: {e}")
            return []

    logger.error("Exceeded retry attempts after 429 errors")
    return []


def main():
    devices = get_devices()
    if not devices:
        print("❌ No devices found.")
        return

    device = devices[0]
    print("✅ First device:")
    pprint(device)

    mac = device.get("macAddress")
    if not mac:
        print("❌ No MAC address in device data.")
        return

    print(f"\n📡 Pulling latest data for {mac}")

    time.sleep(1)
    raw = get_device_history_raw(mac, limit=1)
    if raw:
        print("✅ Sample Record:")
        pprint(raw[0])
    else:
        print("❌ No data returned.")


if __name__ == "__main__":
    main()

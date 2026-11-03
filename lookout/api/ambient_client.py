"""
ambient_client.py: Lightweight interface for interacting with Ambient Weather API
using direct requests. Designed to replace ambient_api.ambientapi with clearer,
testable, and debuggable calls.

Functions:
- get_devices()
- find_station(devices, mac): one station out of a device list, by MAC and
  never by list position, or StationNotFoundError
- get_station_device(bucket, mac): the device list's entry for `mac`, or for
  the default station (the station inventory's first, ADR-0006) when no MAC
  is given; the one path every single-station CLI picks its station by
- get_device_by_mac(mac)
- get_device_history(mac, limit, end_date, strict): one page, as a frame
- get_device_history_raw(mac, limit, end_date, strict): one page, raw

Every request waits a second first (Ambient's 1 request per second per API
key), so this module is the one throttled path to Ambient. History fetchers
read a failed request as an empty page by default; with `strict=True` they
raise `AmbientRequestError` instead, for a caller that must not mistake a
failure for "no older readings" (the first archive's backfill).

Requires:
- Secrets AMBIENT_ENDPOINT, AMBIENT_API_KEY, AMBIENT_APPLICATION_KEY, read via
  lookout.secrets_loader.get_secret on each request, never at import.
"""

import sys
import time
from datetime import datetime, timezone
from pprint import pprint
from typing import Dict, List, Optional

import pandas as pd
import requests

from lookout.config import DEFAULT_BUCKET
from lookout.secrets_loader import get_secret
from lookout.storage.station_inventory import default_station
from lookout.storage.storj import StorageReadError
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


class AmbientRequestError(RuntimeError):
    """A strict history request failed: an HTTP error, retries ran out, or
    the body could not be read as history."""


class StationNotFoundError(RuntimeError):
    """Raised when a station is not in the account's device list."""


def find_station(devices: List[Dict], mac: str) -> Dict:
    """
    Pick one station out of a device list, never by list position.

    :param devices: The `/devices` body, as `get_devices()` returns it.
    :param mac: The station's MAC address, any case.
    :return: That station's device dict.
    :raises StationNotFoundError: If the list is empty or lacks the station.
    """
    mac = mac.strip().upper()
    if not devices:
        raise StationNotFoundError(
            f"Ambient returned no devices, so station {mac} cannot be found "
            "(API error, wrong keys, or no station on the account)"
        )
    for device in devices:
        if str(device.get("macAddress", "")).upper() == mac:
            return device
    listed = ", ".join(str(d.get("macAddress")) for d in devices)
    raise StationNotFoundError(
        f"Station {mac} is not on this Ambient account (it lists: {listed})"
    )


def get_station_device(bucket: str, mac: Optional[str] = None) -> Dict:
    """
    Fetch the device list and return one station from it.

    :param bucket: The bucket holding the station inventory.
    :param mac: The station's MAC address; the default station (the first
        in the station inventory) when None.
    :return: That station's device dict, including `lastData`.
    :raises StorageReadError: If no MAC is given and the station inventory
        is missing, unreadable or corrupt.
    :raises StationNotFoundError: If the account does not list the station.
    """
    # The station first: with no inventory, Ambient is never called.
    mac = mac or default_station(bucket)
    return find_station(get_devices(), mac)


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
    mac: str,
    limit: int = 288,
    end_date: Optional[int] = None,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Fetch historical weather data for a specific device.

    Uses `get_device_history_raw()` to retrieve the data and normalizes it into a DataFrame.

    :param mac: MAC address of the device to fetch data for.
    :param limit: Max number of records to return (default: 288).
    :param end_date: Optional UNIX timestamp in milliseconds for the end of the time range.
    :param strict: Raise on a failed request instead of returning empty.
    :return: A DataFrame of weather data; empty when Ambient has none, or
        (unless `strict`) on failure.
    :raises AmbientRequestError: If `strict` and the request failed or its
        body could not be read.
    """
    try:
        # human = format_ts_utc(end_date) if end_date else "latest"
        # logger.info(f"Fetching history: {mac}, limit={limit}, end_date={end_date} ({human})")

        raw = get_device_history_raw(mac, limit=limit, end_date=end_date, strict=strict)
        if not raw:
            logger.debug("No data returned from device history.")
            return pd.DataFrame()

        df = pd.json_normalize(raw)
        if df.empty:
            logger.warning("Normalized DataFrame is empty.")
            return pd.DataFrame()

        df.sort_values(by="dateutc", inplace=True)
        return df

    except AmbientRequestError:
        raise
    except Exception as e:
        logger.exception(f"Error during device history fetch: {e}")
        if strict:
            raise AmbientRequestError(f"history for {mac} unreadable: {e}") from e
        return pd.DataFrame()


def get_device_history_raw(
    mac: str,
    limit: int = 288,
    end_date: Optional[int] = None,
    strict: bool = False,
) -> list[dict]:
    """
    Fetch raw device history from the Ambient Weather Network.

    :param mac: Device MAC address.
    :param limit: Number of records to retrieve.
    :param end_date: Optional end timestamp (ms).
    :param strict: Raise on a failed request instead of returning `[]`.
    :return: List of raw data records (dict); `[]` when Ambient has none,
        or (unless `strict`) on failure.
    :raises AmbientRequestError: If `strict` and the request failed or the
        429 retries ran out.
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
                return _history_failed(
                    strict, f"history fetch for {mac} failed: {resp.status_code}"
                )
            return resp.json()
        except AmbientRequestError:
            raise
        except Exception as e:
            logger.error(f"Request error while fetching raw history: {e}")
            return _history_failed(strict, f"history fetch for {mac} failed: {e}")

    logger.error("Exceeded retry attempts after 429 errors")
    return _history_failed(strict, f"history fetch for {mac} rate limited 3 times")


def _history_failed(strict: bool, reason: str) -> list[dict]:
    """
    End a failed history request: an empty page, or a raise when strict.

    :param strict: Whether the caller asked failures to raise.
    :param reason: What failed, for the exception.
    :return: `[]` when not strict.
    :raises AmbientRequestError: If strict.
    """
    if strict:
        raise AmbientRequestError(reason)
    return []


def main():
    try:
        device = get_station_device(DEFAULT_BUCKET)
    except (StationNotFoundError, StorageReadError) as e:
        logger.error(f"❌ {e}")
        sys.exit(1)

    print("✅ Default station:")
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

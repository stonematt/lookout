"""
ambient_client.py: Lightweight interface for interacting with Ambient Weather API
using direct requests. Designed to replace ambient_api.ambientapi with clearer,
testable, and debuggable calls.

Functions:
- get_devices()
- get_device_by_mac(mac)
- get_device_history(mac, limit, end_date)

Requires:
- Streamlit secrets: AMBIENT_API_KEY, AMBIENT_APPLICATION_KEY
"""

import time
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

from log_util import app_logger

logger = app_logger(__name__)

# Secrets
# BASE_URL = "https://rt.ambientweather.net/v1"
BASE_URL = st.secrets["AMBIENT_ENDPOINT"].rstrip("/")
API_KEY = st.secrets["AMBIENT_API_KEY"]
APP_KEY = st.secrets["AMBIENT_APPLICATION_KEY"]


def get_devices() -> List[Dict]:
    """
    Retrieve the list of devices associated with the API keys.

    :return: List of device dicts.
    """
    url = f"{BASE_URL}/devices"
    params = {
        "apiKey": API_KEY,
        "applicationKey": APP_KEY,
    }
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
        logger.info(f"Fetching history: {mac}, limit={limit}, end_date={end_date}")
        time.sleep(1)  # Ambient API rate limit: 1 request/second

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
    url = f"{BASE_URL}/devices/{mac}"
    params = {
        "apiKey": API_KEY,
        "applicationKey": APP_KEY,
        "limit": limit,
    }
    if end_date is not None:
        params["endDate"] = end_date

    logger.info(f"Fetching history: {mac}, limit={limit}, end_date={end_date}")
    try:
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            logger.error(f"History fetch failed: {resp.status_code} {resp.text}")
            return []
        return resp.json()
    except Exception as e:
        logger.error(f"Request error while fetching raw history: {e}")
        return []

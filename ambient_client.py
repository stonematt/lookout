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
    Fetch historical weather data for a specific device using Ambient's real-time endpoint.

    :param mac: MAC address of the device to fetch data for.
    :param limit: Max number of records to return (default: 288).
    :param end_date: Optional UNIX timestamp in milliseconds for the end of the time range.
                     If omitted, the latest data will be returned.
    :return: A DataFrame of weather data or empty DataFrame on failure.
    """
    url = f"{BASE_URL}/devices/{mac}"
    params = {
        "apiKey": API_KEY,
        "applicationKey": APP_KEY,
        "limit": limit,
    }
    if end_date:
        params["endDate"] = end_date  # match ambient_api parameter exactly

    logger.info(f"Fetching history: {mac}, limit={limit}, end_date={end_date}")
    time.sleep(1)  # Global Ambient API rate limit: 1 request/second

    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        logger.error(f"History fetch failed: {resp.status_code} {resp.text}")
        return pd.DataFrame()

    try:
        data = resp.json()
    except Exception as e:
        logger.error(f"JSON decode error: {e}")
        return pd.DataFrame()

    if not data:
        logger.debug("No data returned from device history.")
        return pd.DataFrame()

    try:
        df = pd.json_normalize(data)
        df.sort_values(by="dateutc", inplace=True)
        return df
    except Exception as e:
        logger.error(f"Data normalization error: {e}")
        return pd.DataFrame()

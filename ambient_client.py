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

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

from log_util import app_logger

logger = app_logger(__name__)

# Secrets
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
    Fetch historical weather data for a device.

    :param mac: MAC address of the device.
    :param limit: Number of records to retrieve.
    :param end_date: Optional end timestamp in milliseconds.
    :return: DataFrame of weather data.
    """
    url = f"{BASE_URL}/devices/{mac}/historic"
    params = {
        "apiKey": API_KEY,
        "applicationKey": APP_KEY,
        "limit": limit,
    }
    if end_date:
        params["end_date"] = end_date

    logger.info(f"Fetching history: {mac}, Params: {params}")
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        logger.error(f"History fetch failed: {resp.status_code} {resp.text}")
        return pd.DataFrame()

    data = resp.json()
    if not data:
        logger.debug("No data returned from device history.")
        return pd.DataFrame()

    df = pd.json_normalize(data)
    df.sort_values(by="dateutc", inplace=True)
    return df

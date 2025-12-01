#!/usr/bin/env python3
"""
pull_archive.py: Pull weather data archive from S3 to local storage for offline analysis.

This script downloads the parquet archive from S3 and saves it locally in the data/ directory,
enabling fast offline analysis without cloud round-trips.

Usage:
    python pull_archive.py [--bucket lookout] [--device-mac 98:CD:AC:22:0D:E5]
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import streamlit as st

import lookout.api.awn_controller as awn
import lookout.storage.storj as sj
from lookout.api.ambient_client import get_devices
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

# Ensure data directory exists (repo root)
DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def get_device_info(mac_address: str = None) -> tuple:
    """Get device info (mac, name) from Ambient API."""
    devices = get_devices()
    if not devices:
        raise ValueError("No devices found")
    
    if mac_address:
        device = next((d for d in devices if d.get("macAddress") == mac_address), None)
        if not device:
            raise ValueError(f"Device {mac_address} not found")
    else:
        device = devices[0]
    
    mac = device.get("macAddress")
    name = device.get("info", {}).get("name", "Unnamed Device")
    return mac, name, device


def pull_archive_to_local(bucket_name: str, mac_address: str = None) -> str:
    """
    Pull archive from S3 and save locally.
    
    Returns path to local parquet file.
    """
    mac, name, device = get_device_info(mac_address)
    logger.info(f"Pulling archive for: {name} ({mac})")
    
    # Load archive from S3
    archive_df = awn.load_archive_for_device(device, bucket=bucket_name)
    
    if archive_df is None or archive_df.empty:
        logger.warning("No archive data found")
        return ""
    
    logger.info(f"Archive contains {len(archive_df)} records")
    
    # Save locally
    local_path = DATA_DIR / f"{mac.replace(':', '-')}.parquet"
    archive_df.to_parquet(local_path, index=False)
    logger.info(f"Archive saved to: {local_path}")
    
    # Show date range
    if "dateutc" in archive_df.columns:
        min_ms = archive_df["dateutc"].min()
        max_ms = archive_df["dateutc"].max()
        min_date = pd.to_datetime(min_ms, unit="ms", utc=True)
        max_date = pd.to_datetime(max_ms, unit="ms", utc=True)
        logger.info(f"Date range: {min_date} to {max_date}")
    
    return str(local_path)


def main():
    parser = argparse.ArgumentParser(description="Pull weather archive from S3 to local storage")
    parser.add_argument("--bucket", type=str, default="lookout", help="S3 bucket name")
    parser.add_argument("--device-mac", type=str, help="Device MAC address (optional)")
    args = parser.parse_args()
    
    try:
        local_path = pull_archive_to_local(args.bucket, args.device_mac)
        if local_path:
            print(f"✅ Archive pulled to: {local_path}")
        else:
            print("❌ No archive data found")
    except Exception as e:
        logger.exception(f"Failed to pull archive: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
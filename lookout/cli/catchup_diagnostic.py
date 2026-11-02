#!/usr/bin/env python3
"""
catchup_diagnostic.py: Diagnostic script to identify issues with the catchup process.

This script performs detailed analysis of:
1. Archive data status
2. API connectivity and data availability
3. Date range gaps
4. Validation logic testing

Usage:
    python catchup_diagnostic.py --bucket lookout
"""

import argparse
import sys
from datetime import datetime, timedelta

import pandas as pd

import lookout.api.awn_controller as awn
from lookout.api.ambient_client import (
    StationNotFoundError,
    get_device_history_raw,
    get_station_device,
)
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def diagnose_archive(device, bucket_name):
    """Analyze the current archive state."""
    print("\n" + "=" * 50)
    print("ARCHIVE DIAGNOSIS")
    print("=" * 50)

    archive_df = awn.load_archive_for_device(device, bucket_name)

    if archive_df.empty:
        print("❌ Archive is empty!")
        return None

    print(f"✅ Archive loaded: {len(archive_df)} records")

    if "dateutc" in archive_df.columns:
        archive_min = pd.to_datetime(archive_df["dateutc"].min(), unit="ms")
        archive_max = pd.to_datetime(archive_df["dateutc"].max(), unit="ms")
        print(f"📅 Date range: {archive_min} to {archive_max}")

        # Calculate gap to current time
        current_time = datetime.now()
        gap = current_time - archive_max.to_pydatetime()
        print(f"⏰ Gap to current time: {gap}")

        if gap.days > 0:
            print(f"⚠️  Archive is {gap.days} days behind!")
        elif gap.total_seconds() > 3600:
            hours = gap.total_seconds() / 3600
            print(f"⚠️  Archive is {hours:.1f} hours behind")
        else:
            print("✅ Archive is recent")

        # Show last few records
        print("\n📋 Last 3 records in archive:")
        last_records = archive_df.tail(3)
        for idx, row in last_records.iterrows():
            ts = pd.to_datetime(row["dateutc"], unit="ms")
            print(f"  {ts}")

    return archive_df


def diagnose_api_data(device, archive_df):
    """Test API data retrieval and compare with archive."""
    print("\n" + "=" * 50)
    print("API DATA DIAGNOSIS")
    print("=" * 50)

    mac = device.get("macAddress")

    # Test basic API connectivity
    print("🔌 Testing API connectivity...")
    try:
        raw_data = get_device_history_raw(mac, limit=5)
        if not raw_data:
            print("❌ No data returned from API")
            return
        print(f"✅ API returned {len(raw_data)} records")

        # Show latest API record
        latest = raw_data[0]  # Most recent record
        latest_ts = pd.to_datetime(latest["dateutc"], unit="ms")
        print(f"📅 Latest API record: {latest_ts}")

    except Exception as e:
        print(f"❌ API error: {e}")
        return

    if archive_df is None or archive_df.empty:
        print("⚠️  Cannot compare with archive (empty)")
        return

    # Compare archive max with API latest
    archive_max = pd.to_datetime(archive_df["dateutc"].max(), unit="ms")
    api_latest = pd.to_datetime(latest["dateutc"], unit="ms")

    print("\n🔍 Comparison:")
    print(f"  Archive max: {archive_max}")
    print(f"  API latest:  {api_latest}")

    if api_latest > archive_max:
        gap = api_latest - archive_max
        print(f"✅ New data available! Gap: {gap}")
    elif api_latest == archive_max:
        print("⚠️  Archive is up to date (no new data)")
    else:
        print("❌ API data is older than archive?!")


def test_fetch_logic(device, archive_df):
    """Test the specific fetch logic that's failing."""
    print("\n" + "=" * 50)
    print("FETCH LOGIC TESTING")
    print("=" * 50)

    if archive_df is None or archive_df.empty:
        print("❌ Cannot test fetch logic without archive data")
        return

    # Get the last date from archive
    last_date = pd.to_datetime(archive_df["dateutc"].max(), unit="ms").to_pydatetime()
    print(f"📅 Starting fetch from: {last_date}")

    # Test the actual fetch function
    print("🔄 Testing get_device_history_from_date...")
    try:
        new_data = awn.get_device_history_from_date(device, last_date, limit=250)

        if new_data.empty:
            print("❌ get_device_history_from_date returned empty DataFrame")

            # Try fetching from a slightly earlier date
            earlier_date = last_date - timedelta(hours=1)
            print(f"🔄 Trying from 1 hour earlier: {earlier_date}")
            new_data = awn.get_device_history_from_date(device, earlier_date, limit=250)

            if new_data.empty:
                print("❌ Still empty with earlier date")
            else:
                print(f"✅ Got {len(new_data)} records with earlier date")

        else:
            print(f"✅ Fetched {len(new_data)} records")

            # Analyze the fetched data
            if "dateutc" in new_data.columns:
                new_min = pd.to_datetime(new_data["dateutc"].min(), unit="ms")
                new_max = pd.to_datetime(new_data["dateutc"].max(), unit="ms")
                print(f"📅 Fetched range: {new_min} to {new_max}")

                # Check if it's actually new
                archive_max = pd.to_datetime(archive_df["dateutc"].max(), unit="ms")
                if new_max > archive_max:
                    print("✅ Fetched data is newer than archive")
                else:
                    print("⚠️  Fetched data is not newer than archive")
                    print(f"   Archive max: {archive_max}")
                    print(f"   Fetched max: {new_max}")

    except Exception as e:
        print(f"❌ Error in fetch logic: {e}")


def test_date_calculation(archive_df):
    """Test the date calculation logic."""
    print("\n" + "=" * 50)
    print("DATE CALCULATION TESTING")
    print("=" * 50)

    if archive_df is None or archive_df.empty:
        print("❌ Cannot test without archive data")
        return

    last_date = pd.to_datetime(archive_df["dateutc"].max(), unit="ms").to_pydatetime()
    print(f"📅 Archive max date: {last_date}")

    current_time = datetime.now()
    print(f"📅 Current time: {current_time}")

    # Test the end_date calculation from get_device_history_from_date
    limit = 250
    end_date = last_date + timedelta(minutes=(limit - 3) * 5)

    if end_date > current_time:
        end_date = current_time

    print(f"📅 Calculated end_date: {end_date}")
    print(f"📅 Time span being requested: {end_date - last_date}")

    # Check if this makes sense
    if last_date > current_time:
        print("❌ Last date is in the future!")
    elif end_date <= last_date:
        print("❌ End date is not after start date!")
    elif (end_date - last_date).total_seconds() < 300:  # Less than 5 minutes
        print("⚠️  Very small time window being requested")
    else:
        print("✅ Date calculation looks reasonable")


def main():
    parser = argparse.ArgumentParser(description="Diagnose catchup issues")
    parser.add_argument("--bucket", type=str, default="lookout", help="S3 bucket name")
    args = parser.parse_args()

    print("🔍 CATCHUP DIAGNOSTIC TOOL")
    print("=" * 50)

    # Get device
    try:
        device = get_station_device()
    except StationNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    mac = device.get("macAddress")
    name = device.get("info", {}).get("name", "Unnamed Device")
    print(f"📡 Device: {name} ({mac})")

    # Run diagnostics
    archive_df = diagnose_archive(device, args.bucket)
    diagnose_api_data(device, archive_df)
    test_fetch_logic(device, archive_df)
    test_date_calculation(archive_df)

    print("\n" + "=" * 50)
    print("DIAGNOSIS COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Investigation script for rain event gaps during power outages.

This was a one-time analysis to validate our event catalog approach.
Key findings:
- Short power outages don't break Ambient's eventrainin tracking
- Weather station stores data locally during outages
- Event continuity is maintained across brief connectivity losses

Usage: PYTHONPATH=. python lookout/cli/outage_investigation.py
"""

import sys
import pandas as pd
from datetime import datetime, timezone, timedelta

# Add lookout to path
sys.path.append(".")

import streamlit as st
import lookout.core.data_processing as lo_dp
import lookout.api.ambient_client as ambient_client


# Mock streamlit secrets for standalone operation
class MockSecrets:
    def __getitem__(self, key):
        # These are mock values - real secrets are in .streamlit/secrets.toml
        secrets = {
            "AMBIENT_ENDPOINT": "https://api.ambientweather.net/v1",
            "AMBIENT_API_KEY": "mock",
            "AMBIENT_APPLICATION_KEY": "mock",
            "lookout_storage_options": {
                "ACCESS_KEY_ID": "mock",
                "SECRET_ACCESS_KEY": "mock",
                "ENDPOINT_URL": "https://gateway.storjshare.io",
            },
        }
        return secrets[key]


# Mock streamlit session state
class MockSessionState:
    def __init__(self):
        self.data = {}

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value


def analyze_outage_gap(df, outage_date_str, outage_time_str, window_hours=4):
    """Analyze data around a specific outage"""

    print(f"\n{'='*60}")
    print(f"ANALYZING OUTAGE: {outage_date_str} {outage_time_str}")
    print(f"{'='*60}")

    # Convert to timezone-aware datetime (Pacific Time)
    outage_dt = pd.to_datetime(f"{outage_date_str} {outage_time_str}").tz_localize(
        "America/Los_Angeles"
    )
    print(f"Outage datetime: {outage_dt}")

    # Get data around the outage (±window_hours)
    start_window = outage_dt - timedelta(hours=window_hours)
    end_window = outage_dt + timedelta(hours=window_hours)

    # Filter data around outage - use existing 'date' column which is already timezone-aware
    if "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(
            df["dateutc"], unit="ms", utc=True
        ).dt.tz_convert("America/Los_Angeles")

    window_data = df[
        (df["datetime"] >= start_window) & (df["datetime"] <= end_window)
    ].copy()

    if len(window_data) == 0:
        print("❌ No data found in window")
        return

    print(f"\nData points in ±{window_hours}h window: {len(window_data)}")
    print(f"Window: {start_window} to {end_window}")

    # Analyze gaps
    window_data = window_data.sort_values("datetime")
    window_data["time_gap_min"] = window_data["datetime"].diff().dt.total_seconds() / 60

    # Find the main gap around outage time
    significant_gaps = window_data[window_data["time_gap_min"] > 10]  # >10min gaps

    print(f"\nSignificant gaps (>10min) found: {len(significant_gaps)}")
    for idx, row in significant_gaps.iterrows():
        gap_start = (
            window_data.loc[window_data.index < idx, "datetime"].iloc[-1]
            if len(window_data.loc[window_data.index < idx]) > 0
            else "Start"
        )
        print(f"  Gap: {gap_start} → {row['datetime']} ({row['time_gap_min']:.0f} min)")

    # Analyze rain data before/during/after
    pre_outage = window_data[window_data["datetime"] < outage_dt]
    post_outage = window_data[window_data["datetime"] > outage_dt]

    print(f"\n📊 RAIN DATA ANALYSIS")
    print(f"Before outage ({len(pre_outage)} readings):")
    last_before = None
    if len(pre_outage) > 0:
        last_before = pre_outage.iloc[-1]
        print(f"  Last reading: {last_before['datetime']}")
        print(f"  eventrainin: {last_before.get('eventrainin', 'N/A')}")
        print(f"  hourlyrainin: {last_before.get('hourlyrainin', 'N/A')}")
        print(f"  dailyrainin: {last_before.get('dailyrainin', 'N/A')}")

    print(f"\nAfter outage ({len(post_outage)} readings):")
    if len(post_outage) > 0:
        first_after = post_outage.iloc[0]
        print(f"  First reading: {first_after['datetime']}")
        print(f"  eventrainin: {first_after.get('eventrainin', 'N/A')}")
        print(f"  hourlyrainin: {first_after.get('hourlyrainin', 'N/A')}")
        print(f"  dailyrainin: {first_after.get('dailyrainin', 'N/A')}")

        if last_before is not None:
            pre_event = last_before.get("eventrainin", 0)
            post_event = first_after.get("eventrainin", 0)

            print(f"\n🔍 EVENT ANALYSIS:")
            print(f"  eventrainin before: {pre_event}")
            print(f"  eventrainin after: {post_event}")

            if pre_event > 0 and post_event == 0:
                print(
                    f"  ❗ LIKELY EVENT LOST: Event was active before outage, reset after"
                )
                print(f"  💧 Estimated lost rainfall: {pre_event} inches")
            elif pre_event > 0 and post_event > pre_event:
                print(f"  ✅ EVENT CONTINUED: Event persisted through outage")
                print(f"  💧 Rainfall during gap: {post_event - pre_event} inches")
            elif pre_event == 0 and post_event == 0:
                print(f"  ℹ️  NO EVENT: No rain activity detected")
            else:
                print(f"  ❓ UNCLEAR: Unusual pattern detected")

    # Show detailed readings around gap
    print(f"\n📋 DETAILED READINGS (5 before/after outage):")
    if len(pre_outage) >= 5:
        print("Before outage:")
        for _, row in pre_outage.tail(5).iterrows():
            print(
                f"  {row['datetime']} | event:{row.get('eventrainin', 'N/A'):>6} | hourly:{row.get('hourlyrainin', 'N/A'):>6}"
            )

    if len(post_outage) >= 5:
        print("After outage:")
        for _, row in post_outage.head(5).iterrows():
            print(
                f"  {row['datetime']} | event:{row.get('eventrainin', 'N/A'):>6} | hourly:{row.get('hourlyrainin', 'N/A'):>6}"
            )


def main():
    print("🌧️  RAIN EVENT OUTAGE INVESTIGATION")
    print("Analyzing power outages during potential rain events")

    # Setup mocks
    st.secrets = MockSecrets()
    st.session_state = MockSessionState()

    try:
        # Get device info
        devices = ambient_client.get_devices()
        if not devices:
            print("❌ No devices found")
            return

        device = devices[0]  # Use first device
        print(f"📡 Using device: {device.get('info', {}).get('name', 'Unknown')}")

        # Load data using the same method as streamlit app
        lo_dp.load_or_update_data(
            device=device,
            bucket="lookout",
            file_type="parquet",
            auto_update=False,
            short_minutes=6,
            long_minutes=3 * 24 * 60,
        )

        if "history_df" not in st.session_state:
            print("❌ Failed to load history data")
            return

        df = st.session_state["history_df"]
        print(f"✅ Loaded {len(df)} records")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"🏷️  Columns: {list(df.columns)}")

        # Analyze specific outages
        analyze_outage_gap(df, "2024-10-25", "20:15", window_hours=6)
        analyze_outage_gap(df, "2024-10-31", "07:00", window_hours=4)

    except Exception as e:
        print(f"❌ Error during investigation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

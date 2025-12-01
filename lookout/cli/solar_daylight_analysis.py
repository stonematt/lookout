#!/usr/bin/env python3
"""
solar_daylight_analysis.py: Annotate solar data with actual sunrise/sunset times.

This script calculates true sunrise/sunset times for your location and annotates
the solar radiation data with daylight periods (night, dawn, day, dusk) for
accurate filtering and analysis.

Usage:
    python solar_daylight_analysis.py [--data-file 98-CD-AC-22-0D-E5.parquet]
"""

import argparse
from pathlib import Path
from datetime import datetime, timezone
import math

import pandas as pd
import numpy as np
from astral import LocationInfo
from astral.sun import sun

from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

# Default data path (repo root)
DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Location coordinates (you'll need to set these for your station)
# These are approximate for San Francisco - you should update with your actual location
LATITUDE = 37.7749  # TODO: Update with your actual latitude
LONGITUDE = -122.4194  # TODO: Update with your actual longitude


def calculate_sunrise_sunset(date, lat, lon):
    """
    Calculate sunrise and sunset times for a given date and location.
    Uses the astral library for accurate calculations.
    
    Returns: (sunrise_utc, sunset_utc) as datetime objects
    """
    # Create location info
    location = LocationInfo(
        latitude=lat,
        longitude=lon,
        timezone="UTC"  # We'll work in UTC
    )
    
    # Calculate sun times for the date
    s = sun(location.observer, date=date, tzinfo=timezone.utc)
    
    return s["sunrise"], s["sunset"]


def classify_daylight_period(dt, sunrise, sunset):
    """
    Classify a datetime as night, dawn, day, or dusk.
    
    Dawn: 30 min before sunrise to sunrise
    Day: sunrise to sunset
    Dusk: sunset to 30 min after sunset
    Night: all other times
    
    Note: Handles case where sunset is after midnight UTC (next day)
    """
    dawn_start = sunrise - pd.Timedelta(minutes=30)
    dusk_end = sunset + pd.Timedelta(minutes=30)
    
    # Handle case where sunset is after midnight UTC (next day)
    # If sunset < sunrise, it means sunset is on the next day in UTC
    if sunset < sunrise:
        # For times before sunrise, check if they're after sunset (previous day's sunset)
        if dt.time() < sunrise.time():
            # This could be after previous day's sunset
            if dt.time() >= sunset.time():
                return "dusk" if dt < dusk_end else "night"
        # For times after sunrise, normal logic applies
        if dawn_start <= dt < sunrise:
            return "dawn"
        elif sunrise <= dt:
            return "day"
    else:
        # Normal case: sunrise and sunset on same UTC day
        if dawn_start <= dt < sunrise:
            return "dawn"
        elif sunrise <= dt < sunset:
            return "day"
        elif sunset <= dt < dusk_end:
            return "dusk"
    
    return "night"


def annotate_daylight_periods(df, lat, lon):
    """
    Add daylight period annotations to the dataframe.
    """
    logger.info(f"Annotating daylight periods for location {lat}, {lon}")
    
    # Convert to datetime
    df["datetime"] = pd.to_datetime(df["dateutc"], unit="ms", utc=True)
    df["date"] = df["datetime"].dt.date
    
    # Calculate sunrise/sunset for each unique date
    unique_dates = sorted(df["date"].unique())
    sunrise_sunset_data = {}
    
    for date in unique_dates:
        sunrise, sunset = calculate_sunrise_sunset(date, lat, lon)
        sunrise_sunset_data[date] = (sunrise, sunset)
    
    # Add sunrise/sunset columns
    df["sunrise_utc"] = df["date"].map(lambda d: sunrise_sunset_data[d][0])
    df["sunset_utc"] = df["date"].map(lambda d: sunrise_sunset_data[d][1])
    
    # Classify daylight periods
    df["daylight_period"] = df.apply(
        lambda row: classify_daylight_period(
            row["datetime"], 
            row["sunrise_utc"], 
            row["sunset_utc"]
        ), 
        axis=1
    )
    
    return df


def analyze_solar_by_daylight_period(df):
    """
    Analyze solar radiation patterns by daylight period.
    """
    print("\n🌅 Solar Radiation by Daylight Period")
    print("=" * 50)
    
    # Filter valid solar data
    solar_df = df[df["solarradiation"].notna()].copy()
    
    # Statistics by period
    periods = ["night", "dawn", "day", "dusk"]
    
    for period in periods:
        period_data = solar_df[solar_df["daylight_period"] == period]
        if len(period_data) > 0:
            avg_solar = period_data["solarradiation"].mean()
            max_solar = period_data["solarradiation"].max()
            zero_count = (period_data["solarradiation"] == 0).sum()
            total_count = len(period_data)
            
            print(f"{period.title():6s}: {avg_solar:6.1f} W/m² avg, {max_solar:6.1f} W/m² max, "
                  f"{zero_count}/{total_count} zeros ({zero_count/total_count*100:.1f}%)")
    
    # Cross-reference: check zeros during day
    day_data = solar_df[solar_df["daylight_period"] == "day"]
    day_zeros = (day_data["solarradiation"] == 0).sum()
    day_total = len(day_data)
    
    print(f"\n🔍 Daytime Zero Analysis:")
    print(f"  Daytime records: {day_total:,}")
    print(f"  Zeros during day: {day_zeros:,} ({day_zeros/day_total*100:.1f}%)")
    print(f"  Valid daytime solar: {day_total - day_zeros:,} ({(day_total-day_zeros)/day_total*100:.1f}%)")
    
    if day_zeros > 0:
        print(f"  ⚠️  Unexpected zeros during daytime - possible sensor issues or clouds")


def validate_daylight_classification(df):
    """
    Validate the daylight classification by checking solar patterns.
    """
    print("\n✅ Daylight Classification Validation")
    print("=" * 50)
    
    # Sample a few days to show patterns
    df["hour_utc"] = df["datetime"].dt.hour
    sample_dates = sorted(df["date"].unique())[-5:]  # Last 5 days
    
    for date in sample_dates:
        date_data = df[df["date"] == date].copy()
        date_data = date_data.sort_values("hour_utc")
        
        print(f"\n📅 {date} (UTC):")
        
        for _, row in date_data.iterrows():
            period = row["daylight_period"]
            solar = row["solarradiation"] if pd.notna(row["solarradiation"]) else 0
            hour = int(row["hour_utc"])
            
            # Show sunrise/sunset times
            if hour == int(row["sunrise_utc"].hour):
                sunrise_str = row["sunrise_utc"].strftime("%H:%M")
                print(f"  🌅 Sunrise: {sunrise_str}")
            if hour == int(row["sunset_utc"].hour):
                sunset_str = row["sunset_utc"].strftime("%H:%M")
                print(f"  🌇 Sunset: {sunset_str}")
            
            # Show solar data
            if solar > 0 or period != "night":
                print(f"  {hour:02d}:00 - {period:6s} - {solar:6.1f} W/m²")


def save_annotated_data(df, original_file):
    """
    Save the annotated dataframe to a new file.
    """
    # Create output filename
    input_path = Path(original_file)
    output_path = input_path.parent / f"{input_path.stem}_daylight_annotated.parquet"
    
    # Save annotated data
    df.to_parquet(output_path, index=False)
    logger.info(f"Annotated data saved to: {output_path}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Annotate solar data with daylight periods")
    parser.add_argument("--data-file", type=str, help="Local parquet file (optional)")
    parser.add_argument("--lat", type=float, default=LATITUDE, help="Latitude (default: SF)")
    parser.add_argument("--lon", type=float, default=LONGITUDE, help="Longitude (default: SF)")
    parser.add_argument("--save", action="store_true", help="Save annotated data")
    args = parser.parse_args()
    
    try:
        # Load data
        if args.data_file is None:
            parquet_files = list(DATA_DIR.glob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in {DATA_DIR}")
            data_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
        else:
            data_file = Path(args.data_file)
            if not data_file.is_absolute():
                data_file = DATA_DIR / data_file
        
        logger.info(f"Loading data from: {data_file}")
        df = pd.read_parquet(data_file)
        
        print(f"📍 Location: {args.lat}°, {args.lon}°")
        print(f"📊 Analyzing {len(df)} records from {df['dateutc'].min()} to {df['dateutc'].max()}")
        
        # Annotate daylight periods
        df_annotated = annotate_daylight_periods(df, args.lat, args.lon)
        
        # Analyze solar by daylight period
        analyze_solar_by_daylight_period(df_annotated)
        
        # Validate classification
        validate_daylight_classification(df_annotated)
        
        # Save if requested
        if args.save:
            output_path = save_annotated_data(df_annotated, data_file)
            print(f"\n💾 Annotated data saved to: {output_path}")
        
        print(f"\n🎯 Key Finding: Use 'daylight_period == \"day\"' filter for accurate solar analysis!")
        
    except Exception as e:
        logger.exception(f"Error in daylight analysis: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
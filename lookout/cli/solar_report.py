#!/usr/bin/env python3
"""
solar_report.py: Simplified solar radiation report for Salem, OR.

This script provides a clean, single report using authoritative dateutc data.
Uses weekly analysis for better system design insights.

Usage:
    python solar_report.py [--data-file annotated.parquet]
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from lookout.core.solar_analysis import (
    LOCATION,
    calculate_daily_energy,
    get_solar_statistics,
    get_seasonal_breakdown,
    get_weekly_analysis,
)
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

# Default data path (repo root)
DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Location constants imported from core module


def load_data(data_file: str = None) -> pd.DataFrame:
    """Load daylight-annotated data."""
    if data_file is None:
        annotated_files = list(DATA_DIR.glob("*_daylight_annotated.parquet"))
        if not annotated_files:
            raise FileNotFoundError(f"No annotated files found in {DATA_DIR}")
        data_file = max(annotated_files, key=lambda p: p.stat().st_mtime)

    data_path = Path(data_file)
    df = pd.read_parquet(data_path)

    # Convert to datetime (dateutc is authoritative)
    df["datetime"] = pd.to_datetime(df["dateutc"], unit="ms", utc=True)
    df["date"] = df["datetime"].dt.date
    df["week"] = df["datetime"].dt.isocalendar().week
    df["year"] = df["datetime"].dt.year
    df["year_week"] = (
        df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2)
    )

    return df


# calculate_daily_energy function moved to core.solar_analysis module


def analyze_by_week(df: pd.DataFrame, daily_energy: pd.Series) -> None:
    """Analyze solar patterns by week."""
    print("📅 WEEKLY SOLAR ANALYSIS")
    print("=" * 40)

    weekly_data = get_weekly_analysis(daily_energy)

    print(f"Weeks with data: {weekly_data['total_weeks']}")
    print(
        f"Weekly average energy: {weekly_data['overall_avg_kwh_per_m2']:.2f} kWh/m²/day"
    )
    print(
        f"Weekly peak energy: {weekly_data['overall_peak_kwh_per_m2']:.2f} kWh/m²/day"
    )
    print()

    # Best and worst weeks
    best = weekly_data["best_week"]
    worst = weekly_data["worst_week"]

    print(f"Best week: {best['week']} - {best['avg_kwh_per_m2']:.2f} kWh/m²/day avg")
    print(f"Worst week: {worst['week']} - {worst['avg_kwh_per_m2']:.2f} kWh/m²/day avg")
    print()

    # All weeks
    print("All weeks:")
    for week, stats in weekly_data["weekly_data"].items():
        print(f"  {week}: {stats['mean']:.2f} kWh/m²/day avg ({stats['count']} days)")
    print()


def analyze_by_season(df: pd.DataFrame, daily_energy: pd.Series) -> None:
    """Analyze solar patterns by season."""
    print("🌤️ SEASONAL ANALYSIS")
    print("=" * 30)

    seasonal_data = get_seasonal_breakdown(daily_energy)

    for season in ["winter", "spring", "summer", "fall"]:
        if season in seasonal_data:
            data = seasonal_data[season]
            season_name = season.capitalize()
            print(
                f"{season_name:6s}: {data['mean_kwh_per_m2']:.2f} kWh/m²/day avg, {data['max_kwh_per_m2']:.2f} kWh/m²/day max ({data['days']} days)"
            )

    # Seasonal variation
    if "seasonal_variation_ratio" in seasonal_data:
        variation = seasonal_data["seasonal_variation_ratio"]
        print(f"\nSeasonal variation: {variation:.1f}x (summer vs winter)")
    print()


def print_system_design_specs(df: pd.DataFrame, daily_energy: pd.Series) -> None:
    """Print system design specifications."""
    print("⚙️ SYSTEM DESIGN SPECIFICATIONS")
    print("=" * 40)

    stats = get_solar_statistics(df, daily_energy)

    print(f"Solar Resource:")
    print(f"  Location: {LOCATION['name']}")
    print(f"  Peak radiation: {stats['peak_radiation_w_per_m2']:.0f} W/m²")
    print(f"  Daily energy: {stats['avg_daily_energy_kwh_per_m2']:.2f} kWh/m²/day")
    print(f"  Annual energy: {stats['annual_energy_kwh_per_m2']:.0f} kWh/m²/year")
    print()

    print(f"System Sizing:")
    print(f"  • Design for {stats['peak_radiation_w_per_m2']:.0f} W/m² peak input")
    print(
        f"  • Expect {stats['avg_daily_energy_kwh_per_m2']:.2f} kWh/m² per day average"
    )
    print(f"  • Annual yield: {stats['annual_energy_kwh_per_m2']:.0f} kWh/m²")
    print()

    print(f"Peak Production Hours:")
    print(f"  • UTC: {stats['peak_hours_utc']}")
    print(f"  • Local: {[f'{h:02d}:00' for h in stats['peak_hours_local']]}")
    print()


def print_data_quality(df: pd.DataFrame, daily_energy: pd.Series) -> None:
    """Print data quality metrics."""
    print("📊 DATA QUALITY")
    print("=" * 25)

    # Filter for daytime data
    daytime_df = df[
        (df["daylight_period"] == "day") & (df["solarradiation"].notna())
    ].copy()

    print(f"Data period: {df['date'].min()} to {df['date'].max()}")
    print(f"Total records: {len(df):,}")
    print(f"Daytime solar records: {len(daytime_df):,}")
    print(f"Days with energy data: {len(daily_energy):,}")
    print(
        f"Data coverage: {len(daily_energy)/((df['date'].max() - df['date'].min()).days + 1)*100:.1f}%"
    )
    print()

    # Data intervals
    if len(daytime_df) > 1:
        intervals = daytime_df["datetime"].diff().dt.total_seconds() / 60  # minutes
        print(f"Data intervals:")
        print(f"  Median: {intervals.median():.1f} minutes")
        print(f"  Mean: {intervals.mean():.1f} minutes")
        print(
            f"  Gaps > 10min: {(intervals > 10).sum()} ({(intervals > 10).sum()/len(intervals)*100:.1f}%)"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Simplified solar radiation report")
    parser.add_argument("--data-file", type=str, help="Annotated parquet file")
    args = parser.parse_args()

    try:
        # Print header
        print("🌍 SOLAR RADIATION REPORT")
        print("=" * 50)
        print(f"📍 Location: {LOCATION['name']}")
        print(f"🏠 Address: {LOCATION['address']}")
        print(
            f"🌐 Coordinates: {LOCATION['latitude']:.3f}°N, {abs(LOCATION['longitude']):.3f}°W"
        )
        print(f"⛰️  Elevation: {LOCATION['elevation']:.0f} meters")
        print()

        # Load data
        df = load_data(args.data_file)

        # Calculate energy
        daily_energy = calculate_daily_energy(df)

        if daily_energy.empty:
            print("❌ No valid solar data found")
            return

        # Print all sections
        print_data_quality(df, daily_energy)
        analyze_by_week(df, daily_energy)
        analyze_by_season(df, daily_energy)
        print_system_design_specs(df, daily_energy)

        print("🎯 REPORT SUMMARY")
        print("=" * 20)
        print(f"• Uses authoritative dateutc (reverse-sorted archive)")
        print(f"• Trapezoidal integration for accurate energy calculation")
        print(f"• Weekly analysis for system design insights")
        print(f"• Location-specific for Salem, OR")
        print()

    except Exception as e:
        logger.exception(f"Error in solar report: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()

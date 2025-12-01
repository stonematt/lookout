"""
solar_analysis.py
Core solar data processing and calculation functions for Lookout weather station.

This module contains reusable functions for solar radiation analysis,
extracted from CLI scripts to enable both command-line and Streamlit usage.
"""

import pandas as pd
from typing import Dict, Any, Tuple

# Location constants for Salem, OR weather station
LOCATION = {
    'name': 'Salem, OR',
    'address': '3818 Homestead Rd S, Salem, OR 97302, USA',
    'latitude': 44.90142112432992,
    'longitude': -123.08386129847136,
    'elevation': 137.6079864501953,
    'timezone': 'America/Los_Angeles'
}


def filter_daytime_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame for daytime solar data with valid radiation readings.

    :param df: DataFrame with daylight_period and solarradiation columns
    :return: Filtered DataFrame containing only daytime valid solar data
    """
    return df[(df['daylight_period'] == 'day') &
              (df['solarradiation'].notna())].copy()


def calculate_daily_energy(df: pd.DataFrame) -> pd.Series:
    """
    Calculate daily energy using trapezoidal integration.

    :param df: DataFrame with date, datetime, and solarradiation columns
    :return: Series with daily kWh/m² values indexed by date
    """
    # Filter for daytime data with valid solar
    daytime_df = filter_daytime_data(df)

    if daytime_df.empty:
        return pd.Series(dtype=float)

    # Sort by datetime (dateutc is authoritative and reverse-sorted in archive)
    daytime_df = daytime_df.sort_values('datetime')

    # Future-proof implementation: calculate daily energy without groupby.apply
    # This implements the behavior pandas will have in the future
    daily_energy_dict = {}

    for date_val, group in daytime_df.groupby('date'):
        # Only pass the data columns (not the grouping column) to the calculation
        data = group[['datetime', 'solarradiation']].sort_values('datetime')
        if len(data) < 2:
            daily_energy_dict[date_val] = 0
            continue

        # Time differences in hours
        time_diffs = data['datetime'].diff().dt.total_seconds() / 3600

        # Trapezoidal rule: (y1 + y2) / 2 * dx
        energy_wh = ((data['solarradiation'].shift(1) + data['solarradiation']) / 2 * time_diffs).sum()

        daily_energy_dict[date_val] = energy_wh / 1000  # Convert to kWh/m²

    daily_energy = pd.Series(daily_energy_dict)
    daily_energy.name = 'daily_kwh_per_m2'
    daily_energy.index.name = 'date'

    return daily_energy


def get_solar_statistics(df: pd.DataFrame, daily_energy: pd.Series = None) -> Dict[str, Any]:
    """
    Calculate key solar radiation statistics.

    :param df: DataFrame with solarradiation and daylight_period columns
    :param daily_energy: Optional pre-calculated daily energy series
    :return: Dictionary with solar statistics
    """
    # Get daytime data for peak radiation
    daytime_df = filter_daytime_data(df)

    if daytime_df.empty:
        return {}

    max_solar = daytime_df['solarradiation'].max()

    # Calculate daily energy if not provided
    if daily_energy is None:
        daily_energy = calculate_daily_energy(df)

    avg_daily_energy = daily_energy.mean() if not daily_energy.empty else 0

    # Peak production hours (UTC)
    hourly_avg = daytime_df.groupby(daytime_df['datetime'].dt.hour)['solarradiation'].mean()
    peak_hours_utc = hourly_avg[hourly_avg >= hourly_avg.max() * 0.8].index.tolist()

    # Convert to local time (Pacific)
    peak_hours_local = [(h - 8) % 24 for h in peak_hours_utc]

    return {
        'peak_radiation_w_per_m2': max_solar,
        'avg_daily_energy_kwh_per_m2': avg_daily_energy,
        'annual_energy_kwh_per_m2': avg_daily_energy * 365,
        'peak_hours_utc': peak_hours_utc,
        'peak_hours_local': peak_hours_local,
        'data_days': len(daily_energy) if daily_energy is not None else 0
    }


def calculate_hourly_patterns(df: pd.DataFrame) -> pd.Series:
    """
    Calculate average solar radiation by hour of day.

    :param df: DataFrame with datetime and solarradiation columns
    :return: Series with average radiation by hour (0-23)
    """
    daytime_df = filter_daytime_data(df)

    if daytime_df.empty:
        return pd.Series(dtype=float)

    return daytime_df.groupby(daytime_df['datetime'].dt.hour)['solarradiation'].mean()


def get_seasonal_breakdown(daily_energy: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Calculate seasonal statistics from daily energy data.

    :param daily_energy: Series with daily kWh/m² values indexed by date
    :return: Dictionary with seasonal statistics
    """
    if daily_energy.empty:
        return {}

    # Create energy DataFrame with date components
    energy_df = daily_energy.reset_index()
    energy_df['date'] = pd.to_datetime(energy_df['date'])
    energy_df['month'] = energy_df['date'].dt.month
    energy_df['season'] = energy_df['month'].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall"
    })

    # Calculate seasonal statistics
    seasonal_stats = energy_df.groupby('season')['daily_kwh_per_m2'].agg(['mean', 'max', 'count'])

    result = {}
    for season in ["Winter", "Spring", "Summer", "Fall"]:
        if season in seasonal_stats.index:
            row = seasonal_stats.loc[season]
            result[season.lower()] = {
                'mean_kwh_per_m2': row['mean'],
                'max_kwh_per_m2': row['max'],
                'days': int(row['count'])
            }

    # Calculate seasonal variation if both summer and winter exist
    if 'summer' in result and 'winter' in result:
        result['seasonal_variation_ratio'] = result['summer']['mean_kwh_per_m2'] / result['winter']['mean_kwh_per_m2']

    return result


def get_weekly_analysis(daily_energy: pd.Series) -> Dict[str, Any]:
    """
    Analyze solar patterns by week.

    :param daily_energy: Series with daily kWh/m² values indexed by date
    :return: Dictionary with weekly analysis results
    """
    if daily_energy.empty:
        return {}

    # Create energy DataFrame with week info
    energy_df = daily_energy.reset_index()
    energy_df['date'] = pd.to_datetime(energy_df['date'])
    energy_df['week'] = energy_df['date'].dt.isocalendar().week
    energy_df['year'] = energy_df['date'].dt.year
    energy_df['year_week'] = energy_df['year'].astype(str) + '-W' + energy_df['week'].astype(str).str.zfill(2)

    # Weekly statistics
    weekly_stats = energy_df.groupby('year_week')['daily_kwh_per_m2'].agg(['mean', 'max', 'count'])

    # Find best and worst weeks
    best_week = weekly_stats['mean'].idxmax()
    worst_week = weekly_stats['mean'].idxmin()

    return {
        'total_weeks': len(weekly_stats),
        'overall_avg_kwh_per_m2': weekly_stats['mean'].mean(),
        'overall_peak_kwh_per_m2': weekly_stats['max'].max(),
        'best_week': {
            'week': best_week,
            'avg_kwh_per_m2': weekly_stats.loc[best_week, 'mean']
        },
        'worst_week': {
            'week': worst_week,
            'avg_kwh_per_m2': weekly_stats.loc[worst_week, 'mean']
        },
        'weekly_data': weekly_stats.to_dict('index')
    }
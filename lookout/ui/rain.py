"""
Precipitation analysis module for Lookout weather station dashboard.

This module provides comprehensive rainfall data analysis including current accumulations,
historical statistics, and data processing functions for precipitation visualizations.
Handles daily rainfall extraction from accumulating fields and dry spell calculations.
"""

import streamlit as st
import pandas as pd
import lookout.core.visualization as lo_viz


def extract_daily_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract daily rainfall totals from dailyrainin field.
    :param df: pd.DataFrame - Weather data with 'dateutc' and 'dailyrainin' columns.
    :return: pd.DataFrame - With 'date' and 'rainfall' columns containing daily totals.
    """
    df_local = df.copy()

    # Convert to local time and extract dates
    df_local["local_datetime"] = pd.to_datetime(
        df_local["dateutc"], unit="ms", utc=True
    ).dt.tz_convert("America/Los_Angeles")
    df_local["local_date"] = df_local["local_datetime"].dt.date

    # Group by date and take max dailyrainin (handles accumulation)
    daily_max = df_local.groupby("local_date")["dailyrainin"].max()

    # Calculate daily totals (handle midnight resets)
    daily_totals = daily_max.diff().fillna(daily_max.iloc[0])
    daily_totals = daily_totals.clip(lower=0)

    # Handle edge case where field doesn't reset
    mask = (daily_totals == 0) & (daily_max > 0)
    daily_totals[mask] = daily_max[mask]

    return pd.DataFrame({"date": daily_totals.index, "rainfall": daily_totals.values})


def calculate_dry_spell_stats(df: pd.DataFrame) -> dict:
    """
    Calculate current dry spell duration using lastRain timestamp.
    :param df: pd.DataFrame - Weather data with 'lastRain' and 'dateutc' columns.
    :return: dict - With 'current_dry_days' and 'current_dry_hours' keys.
    """
    if "lastRain" not in df.columns:
        return {"current_dry_days": 0, "current_dry_hours": 0}

    latest_record = df.iloc[-1]
    last_rain_str = latest_record["lastRain"]
    current_time = pd.to_datetime(latest_record["dateutc"], unit="ms", utc=True)

    try:
        last_rain = pd.to_datetime(last_rain_str)
        dry_period = current_time - last_rain
        return {
            "current_dry_days": dry_period.days,
            "current_dry_hours": dry_period.total_seconds() / 3600,
        }
    except Exception:
        return {"current_dry_days": 0, "current_dry_hours": 0}


def calculate_rainfall_accumulations(
    daily_rain_df: pd.DataFrame, df: pd.DataFrame
) -> dict:
    """
    Calculate rainfall accumulations over different time periods.
    :param daily_rain_df: pd.DataFrame - Daily rainfall data with 'date' and 'rainfall' columns.
    :param df: pd.DataFrame - Raw weather data for current hourly rate.
    :return: dict - Accumulation totals for different periods with current rate.
    """
    if len(daily_rain_df) == 0:
        return {}

    # Convert date to datetime for easier filtering
    df_calc = daily_rain_df.copy()
    df_calc["date"] = pd.to_datetime(df_calc["date"])

    # Get current date (use last date in data)
    current_date = df_calc["date"].max()

    # Get current hourly rain rate (same for all periods)
    current_rate = df["hourlyrainin"].iloc[-1] if len(df) > 0 else 0

    # Define periods
    periods = {
        "today": 1,
        "last 7d": 7,
        "last 30d": 30,
        "last 90d": 90,
        "last 365d": 365,
    }

    results = {}

    for period_name, days in periods.items():
        # Calculate start date
        start_date = current_date - pd.Timedelta(days=days - 1)

        # Filter data for period
        period_data = df_calc[
            (df_calc["date"] >= start_date) & (df_calc["date"] <= current_date)
        ]

        # Sum rainfall for period
        total_rainfall = period_data["rainfall"].sum()

        # Structure like get_history_min_max
        results[period_name] = {
            "min": 0,  # Rainfall can't be negative
            "max": total_rainfall,
            "current": current_rate,  # Current hourly rate for all periods
        }

    return results


def calculate_rainfall_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate comprehensive rainfall statistics and metrics.
    :param df: pd.DataFrame - Weather data with rain accumulation fields.
    :return: dict - Current accumulations, historical averages, and dry spell info.
    """
    daily_rain = extract_daily_rainfall(df)
    dry_stats = calculate_dry_spell_stats(df)

    # Current accumulations from latest record
    latest = df.iloc[-1]

    # Historical analysis
    rain_days = (daily_rain["rainfall"] > 0).sum()
    max_daily = daily_rain["rainfall"].max()
    total_days = len(daily_rain)
    avg_annual = (
        daily_rain["rainfall"].sum() / (total_days / 365.25) if total_days > 365 else 0
    )

    return {
        "current_ytd": latest.get("yearlyrainin", 0),
        "current_monthly": latest.get("monthlyrainin", 0),
        "current_weekly": latest.get("weeklyrainin", 0),
        "current_daily": latest.get("dailyrainin", 0),
        "avg_annual": avg_annual,
        "total_rain_days": rain_days,
        "max_daily_this_year": max_daily,
        "current_dry_days": dry_stats["current_dry_days"],
        "last_rain": latest.get("lastRain", "Unknown"),
        "total_days": total_days,
    }


def render():
    """Render the precipitation analysis tab."""
    st.header("Precipitation Analysis")
    st.write("Comprehensive rainfall data analysis and visualization")

    # Get data from session state
    if "history_df" not in st.session_state:
        st.error("No weather data available in session")
        return

    df = st.session_state["history_df"]

    # Use most recent data from last_data if available, otherwise latest record
    if "last_data" in st.session_state:
        latest = st.session_state["last_data"]
        # Calculate dry spell from current data
        try:
            last_rain = pd.to_datetime(latest["lastRain"])
            current_time = pd.to_datetime(latest["dateutc"], unit="ms", utc=True)
            dry_period = current_time - last_rain
            current_dry_days = dry_period.days
            current_dry_hours = (dry_period.total_seconds() % (24 * 3600)) / 3600

            # Format time since rain
            if current_dry_days > 0:
                time_since_rain = f"{current_dry_days}d {current_dry_hours:.1f}h"
            else:
                time_since_rain = f"{current_dry_hours:.1f}h"
        except Exception:
            current_dry_days = 0
            time_since_rain = "0h"
    else:
        latest = df.iloc[-1].to_dict()
        current_dry_days = 0
        time_since_rain = "0h"

    # Calculate statistics
    with st.spinner("Calculating rainfall statistics..."):
        stats = calculate_rainfall_statistics(df)
        # Override current values with last_data if available
        if "last_data" in st.session_state:
            stats.update(
                {
                    "current_ytd": latest.get("yearlyrainin", 0),
                    "current_monthly": latest.get("monthlyrainin", 0),
                    "current_weekly": latest.get("weeklyrainin", 0),
                    "current_daily": latest.get("dailyrainin", 0),
                    "last_rain": latest.get("lastRain", "Unknown"),
                    "current_dry_days": current_dry_days,
                    "time_since_rain": time_since_rain,
                }
            )

    # Summary Section
    st.subheader("Rainfall Summary")

    # Current accumulations
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Year to Date", f"{stats['current_ytd']:.2f}\"")
    with col2:
        st.metric("This Month", f"{stats['current_monthly']:.2f}\"")
    with col3:
        st.metric("This Week", f"{stats['current_weekly']:.2f}\"")
    with col4:
        st.metric("Today", f"{stats['current_daily']:.2f}\"")

    # Historical context
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Historical Context:**")
        st.write(f"• Average Annual: {stats['avg_annual']:.1f}\"")
        st.write(f"• Total Data Days: {stats['total_days']:,}")
        st.write(f"• Days with Rain: {stats['total_rain_days']:,}")
        rain_percentage = (
            (stats["total_rain_days"] / stats["total_days"] * 100)
            if stats["total_days"] > 0
            else 0
        )
        st.write(f"• Rain Frequency: {rain_percentage:.1f}%")

    with col2:
        st.write("**Recent Activity:**")
        st.write(f"• Max Daily Rain: {stats['max_daily_this_year']:.2f}\"")
        if "time_since_rain" in stats:
            st.write(f"• Time Since Rain: {stats['time_since_rain']}")
        else:
            st.write(f"• Days Since Rain: {stats['current_dry_days']}")

        # Format last rain date properly
        try:
            last_rain_dt = pd.to_datetime(stats["last_rain"])
            # Convert to local timezone for display
            last_rain_local = last_rain_dt.tz_convert("America/Los_Angeles")
            formatted_date = last_rain_local.strftime("%m/%d/%Y %H:%M")
            st.write(f"• Last Rain: {formatted_date}")
        except Exception:
            st.write(f"• Last Rain: {stats['last_rain']}")

    st.divider()

    # Daily Rainfall Chart
    st.subheader("Daily Rainfall Chart")

    # Extract daily rainfall data and calculate accumulations
    daily_rain_df = extract_daily_rainfall(df)

    if len(daily_rain_df) > 0:
        rain_accumulations = calculate_rainfall_accumulations(daily_rain_df, df)

        if rain_accumulations:
            # Data is already in the correct format for draw_horizontal_bars
            lo_viz.draw_horizontal_bars(
                rain_accumulations, label="Rainfall Accumulation (inches)"
            )
        else:
            st.error("Could not calculate rainfall accumulations")
    else:
        st.error("No daily rainfall data available")

    st.divider()

    # Placeholder sections for upcoming visualizations
    st.subheader("Year-over-Year Accumulation")
    st.info(
        "Coming next: Line chart showing cumulative rainfall by day of year, with separate lines for each year"
    )

    st.subheader("Rain Intensity Heatmap")
    st.info(
        "Coming next: Heatmap of hourly rain rates with configurable time granularity (daily avg, max, etc.)"
    )

    st.subheader("Dry Spell & Event Analysis")
    st.info(
        "Coming next: Analysis of dry periods, rain events, and precipitation patterns"
    )

    # Data health check (condensed)
    with st.expander("Data Health Check"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            date_range = (df["date"].max() - df["date"].min()).days
            st.metric("Days of Data", date_range)
        with col3:
            rain_fields = [col for col in df.columns if "rain" in col.lower()]
            st.metric("Rain Fields", len(rain_fields))

        # Show sample daily rainfall data
        if st.checkbox("Show daily rainfall sample"):
            daily_data = extract_daily_rainfall(df)
            st.write("**Recent daily totals:**")
            st.dataframe(daily_data.tail(10))

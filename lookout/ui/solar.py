"""
Solar Radiation UI Module - Phase 3.2: Summary Metrics

UI structure with data loading, date filtering, and 4 summary metric cards for solar radiation measurements.
Today metric compares current day to yesterday's production.
"""

import streamlit as st
import pandas as pd
import datetime
import lookout.ui.components as ui_components
from lookout.utils.log_util import app_logger
from lookout.core.solar_energy_periods import get_period_stats
from lookout.core.solar_viz import (
    create_month_day_heatmap,
    create_day_column_chart,
    create_day_15min_heatmap,
    create_15min_bar_chart,
)

logger = app_logger(__name__)


def render():
    """Main entry point for solar radiation tab."""
    st.header("Solar Radiation")
    st.caption("Weather station solar radiation measurements (kWh/m²)")

    # Validate data availability
    if "energy_catalog" not in st.session_state:
        st.error("Solar energy catalog not available. Please refresh the page.")
        return

    # Load and cache periods data first (needed for date slider)
    periods_df = _load_and_cache_data(None, None)

    if periods_df.empty:
        st.warning("No solar data available")
        return

    # Date range slider - use ui_components.create_date_range_slider
    # Pattern from rain_events.py
    # Default to last 365 days
    start_ts, end_ts = ui_components.create_date_range_slider(
        periods_df, date_column="period_start", key_prefix="solar", default_days=365
    )

    # Load and filter data by date range
    periods_df = _load_and_cache_data(start_ts, end_ts)

    if periods_df.empty:
        st.warning("No solar data in selected date range")
        return

    # Summary metrics
    _render_summary_metrics(periods_df)

    # Heatmap view selector
    heatmap_view = st.selectbox(
        "Select Heatmap View:",
        options=["Month/Day View", "Day/15min View"],
        index=0,  # Default to Month/Day View
        key="solar_heatmap_view",
        help="Choose between daily monthly view or granular 15-minute view"
    )

    # Render selected heatmap
    if heatmap_view == "Month/Day View":
        _render_month_day_heatmap(periods_df)
    else:
        _render_day_15min_heatmap(periods_df)


def _load_and_cache_data(start_ts, end_ts) -> pd.DataFrame:
    """
    Load energy periods from session state (pre-calculated during app startup).
    """
    # Get catalog from session state (loaded early in data flow)
    periods_df = st.session_state.get("energy_catalog", pd.DataFrame())

    if periods_df.empty:
        logger.warning("No energy catalog available in session state")
        return pd.DataFrame()

    logger.debug(f"Using energy catalog from session: {len(periods_df)} periods")

    # Filter to date range if specified
    if start_ts and end_ts:
        mask = (periods_df["period_start"] >= start_ts) & (
            periods_df["period_start"] < end_ts
        )
        filtered_df = periods_df.loc[mask].copy()
        logger.info(f"Filtered to date range: {len(filtered_df)} periods")
        return filtered_df

    return periods_df


def _render_summary_metrics(periods_df):
    """
    Display 4 summary metric cards using st.columns(4).

    Metrics:
    1. Period Total (kWh/m²) - Total radiation in selected date range
    2. Today (kWh/m²) - Today's radiation (if in range) with yesterday comparison
    3. Peak Day (kWh/m²) - Highest radiation day with date
    4. Daily Average (kWh/m²/day) - Average daily radiation

    Format: XX.X with 1 decimal place, units in title
    """
    stats = get_period_stats(periods_df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Period Total (kWh/m²)", f"{stats['total_kwh']:.1f}")

    with col2:
        # Calculate today's radiation from periods_df
        today = pd.Timestamp.now(tz="America/Los_Angeles").date()
        periods_today = periods_df[periods_df["period_start"].dt.date == today]
        today_kwh = periods_today["energy_kwh"].sum()

        # Calculate yesterday's radiation for comparison
        yesterday = (pd.Timestamp(today) - pd.Timedelta(days=1)).date()
        periods_yesterday = periods_df[periods_df["period_start"].dt.date == yesterday]
        yesterday_kwh = periods_yesterday["energy_kwh"].sum()

        st.metric(
            "Today (kWh/m²)",
            f"{today_kwh:.1f}",
            delta=f"{today_kwh - yesterday_kwh:.1f}",
        )

    with col3:
        peak = stats["peak_day"]
        if peak["date"]:
            st.metric("Peak Day (kWh/m²)", f"{peak['kwh']:.1f}", delta=peak["date"])
        else:
            st.metric("Peak Day", "—")

    with col4:
        st.metric("Daily Average (kWh/m²/day)", f"{stats['avg_daily_kwh']:.1f}")


def _render_month_day_heatmap(filtered_df):
    """Render Month/Day heatmap with drill-down."""
    # Initialize today's date as default if not set
    if "selected_solar_date" not in st.session_state:
        today = pd.Timestamp.now(tz="America/Los_Angeles").date()
        st.session_state["selected_solar_date"] = today.strftime("%Y-%m-%d")

    selected_date = st.session_state.get("selected_solar_date")

    # ALWAYS SHOW: Month/Day heatmap
    st.subheader("Monthly Solar Radiation")

    try:
        fig = create_month_day_heatmap(filtered_df, height=500)
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.error(f"Error creating heatmap: {e}")
        logger.exception("Month/day heatmap error")

    # Calendar control - synchronized with session state
    st.caption("Select a date to see hourly details")
    default_date = pd.to_datetime(selected_date).date() if selected_date else None

    test_date = st.date_input(
        "Select date:", value=default_date, key="drill_down_calendar"
    )
    if test_date:
        st.session_state["selected_solar_date"] = test_date.strftime("%Y-%m-%d")

    # CONDITIONAL: Drill-down details (appears below heatmap when date selected)
    if selected_date:
        # Daily total and hourly chart in 1:4 column ratio
        col1, col2 = st.columns([1, 4])

        # Calculate daily total for metric
        daily_periods = filtered_df[
            filtered_df["period_start"].dt.date.astype(str) == selected_date
        ]
        daily_total = daily_periods["energy_kwh"].sum()

        with col1:
            st.metric("Daily Total (kWh/m²)", f"{daily_total:.2f}")

        with col2:
            # Hourly chart (title shows date)
            try:
                fig = create_day_column_chart(filtered_df, selected_date)
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.error(f"Error creating hourly chart: {e}")
                logger.exception("Hourly chart error")


def _render_day_15min_heatmap(filtered_df):
    """Render Day/15min heatmap with drill-down."""
    # Initialize today's date as default if not set
    if "selected_solar_date" not in st.session_state:
        today = pd.Timestamp.now(tz="America/Los_Angeles").date()
        st.session_state["selected_solar_date"] = today.strftime("%Y-%m-%d")

    selected_date = st.session_state.get("selected_solar_date")

    # ALWAYS SHOW: Day/15min heatmap
    st.subheader("15-Minute Solar Radiation")

    try:
        fig = create_day_15min_heatmap(filtered_df, start_hour=4, end_hour=21, height=500, dense_view=True)
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.error(f"Error creating heatmap: {e}")
        logger.exception("Day/15min heatmap error")

    # Calendar control - synchronized with session state
    st.caption("Select a date to see 15-minute details")
    default_date = pd.to_datetime(selected_date).date() if selected_date else None

    test_date = st.date_input(
        "Select date:", value=default_date, key="drill_down_calendar_15min"
    )
    if test_date:
        st.session_state["selected_solar_date"] = test_date.strftime("%Y-%m-%d")

    # CONDITIONAL: Drill-down details (appears below heatmap when date selected)
    if selected_date:
        # Daily total and 15min chart in 1:4 column ratio
        col1, col2 = st.columns([1, 4])

        # Calculate daily total for metric
        daily_periods = filtered_df[
            filtered_df["period_start"].dt.date.astype(str) == selected_date
        ]
        daily_total = daily_periods["energy_kwh"].sum()

        with col1:
            st.metric("Daily Total (kWh/m²)", f"{daily_total:.2f}")

        with col2:
            # 15min chart (title shows date)
            try:
                fig = create_15min_bar_chart(filtered_df, selected_date)
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.error(f"Error creating 15min chart: {e}")
                logger.exception("15min chart error")

        # Manual clear button for testing (Phase 4 will remove this)
        if st.button("← Clear Selection (Test)", key="clear_15min"):
            del st.session_state["selected_solar_date"]

    st.caption("Phase 4 will add click-to-drill interaction")

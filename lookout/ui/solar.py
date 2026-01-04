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

    # Load and cache unfiltered periods data (for metric grid)
    full_periods_df = _load_and_cache_data(None, None)

    if full_periods_df.empty:
        st.warning("No solar data available")
        return

    # NEW: Render 2x2 metric grid (unfiltered, fixed periods)
    _render_metric_grid(full_periods_df)

    # Date range slider - use ui_components.create_date_range_slider
    # Pattern from rain_events.py
    # Default to last 365 days
    start_ts, end_ts = ui_components.create_date_range_slider(
        full_periods_df, date_column="period_start", key_prefix="solar", default_days=365
    )

    # Load and filter data by date range for heatmap
    periods_df = _load_and_cache_data(start_ts, end_ts)

    if periods_df.empty:
        st.warning("No solar data in selected date range")
        return

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


def _render_metric_grid(periods_df):
    """
    Render 2x2 metric grid with solar radiation cards.

    Uses unfiltered full catalog data (not date-range filtered).
    Shows Today, Last 7 Days, Last 30 Days, Last 365 Days.
    Each card has value, unit, step-chart sparkline, and optional delta.
    """
    from lookout.core.solar_cards import calculate_period_metrics
    from lookout.ui.components import render_solar_metric_card

    # Calculate metrics for all periods
    metrics = calculate_period_metrics(periods_df)

    # Create 2x2 grid layout
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # Top row: Today and Last 7 Days
    with col1:
        today_data = metrics["today"]
        render_solar_metric_card(
            "Today",
            today_data["value"],
            today_data["unit"],
            today_data["sparkline_data"],
            "today",
            today_data["axis_range"],
            today_data["delta"]
        )

    with col2:
        week_data = metrics["last_7d"]
        render_solar_metric_card(
            "Last 7 Days",
            week_data["value"],
            week_data["unit"],
            week_data["sparkline_data"],
            "last_7d",
            week_data["axis_range"],
            week_data["delta"]
        )

    # Bottom row: Last 30 Days and Last 365 Days
    with col3:
        month_data = metrics["last_30d"]
        render_solar_metric_card(
            "Last 30 Days",
            month_data["value"],
            month_data["unit"],
            month_data["sparkline_data"],
            "last_30d",
            month_data["axis_range"],
            month_data["delta"]
        )

    with col4:
        year_data = metrics["last_365d"]
        render_solar_metric_card(
            "Last 365 Days",
            year_data["value"],
            year_data["unit"],
            year_data["sparkline_data"],
            "last_365d",
            year_data["axis_range"],
            year_data["delta"]
        )


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

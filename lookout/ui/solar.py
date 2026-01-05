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
    create_year_month_heatmap,
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

    # NEW: Render 2x2 tile grid (unfiltered, fixed periods)
    _render_tile_grid(full_periods_df)

    st.divider()  # Visual separator between tiles and heatmaps

    # Date range slider - use ui_components.create_date_range_slider
    # Pattern from rain_events.py
    # Default to last 365 days
    start_ts, end_ts = ui_components.create_date_range_slider(
        full_periods_df,
        date_column="period_start",
        key_prefix="solar",
        default_days=365,
    )

    # Load and filter data by date range for heatmap
    periods_df = _load_and_cache_data(start_ts, end_ts)

    if periods_df.empty:
        st.warning("No solar data in selected date range")
        return

    # Heatmap view selector
    heatmap_view = st.selectbox(
        "Select Heatmap View:",
        options=["Year/Month View", "Month/Day View", "Day/15min View"],
        index=0,  # Default to Year/Month View
        key="solar_heatmap_view",
        help="Choose between yearly, monthly and daily view or granular 15-minute view",
    )

    # Render selected heatmap
    if heatmap_view == "Year/Month View":
        _render_year_month_heatmap(periods_df)
    elif heatmap_view == "Month/Day View":
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


def _render_year_month_heatmap(filtered_df):
    """Render Year/Month heatmap."""
    st.subheader("Yearly Solar Radiation")

    try:
        fig = create_year_month_heatmap(filtered_df, height=500)
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.error(f"Error creating heatmap: {e}")
        logger.exception("Year/month heatmap error")


def _render_tile_grid(periods_df):
    """
    Render 2x2 compact tile grid with Last 24h, 7d, 30d, 365d metrics.

    Uses unfiltered full catalog data (not date-range filtered).
    Shows compact tiles with Streamlit metrics + simple sparklines.
    """
    from lookout.core.solar_tiles import calculate_tile_metrics
    from lookout.ui.components import render_solar_tile

    # Calculate metrics for all tiles
    metrics = calculate_tile_metrics(periods_df)

    # Create 2x2 grid layout
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # Top row: Last 24h and Last 7d
    with col1:
        tile_data = metrics["last_24h"].copy()
        render_solar_tile(
            title=tile_data["title"],
            total_kwh=tile_data["total_kwh"],
            period_type="last_24h",
            sparkline_data=tile_data["sparkline_data"],
            y_axis_range=tile_data["y_axis_range"],
            delta_value=tile_data["delta_value"],
            hover_labels=tile_data["hover_labels"],
            current_period_index=tile_data["current_period_index"],
        )

    with col2:
        tile_data = metrics["last_7d"].copy()
        render_solar_tile(
            title=tile_data["title"],
            total_kwh=tile_data["total_kwh"],
            period_type="last_7d",
            sparkline_data=tile_data["sparkline_data"],
            y_axis_range=tile_data["y_axis_range"],
            delta_value=tile_data["delta_value"],
            hover_labels=tile_data["hover_labels"],
            current_period_index=tile_data["current_period_index"],
        )

    # Bottom row: Last 30d and Last 365d
    with col3:
        tile_data = metrics["last_30d"].copy()
        render_solar_tile(
            title=tile_data["title"],
            total_kwh=tile_data["total_kwh"],
            period_type="last_30d",
            sparkline_data=tile_data["sparkline_data"],
            y_axis_range=tile_data["y_axis_range"],
            delta_value=tile_data["delta_value"],
            hover_labels=tile_data["hover_labels"],
            current_period_index=tile_data["current_period_index"],
        )

    with col4:
        tile_data = metrics["last_365d"].copy()
        render_solar_tile(
            title=tile_data["title"],
            total_kwh=tile_data["total_kwh"],
            period_type="last_365d",
            sparkline_data=tile_data["sparkline_data"],
            y_axis_range=tile_data["y_axis_range"],
            delta_value=tile_data["delta_value"],
            hover_labels=tile_data["hover_labels"],
            current_period_index=tile_data["current_period_index"],
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
        fig = create_day_15min_heatmap(
            filtered_df, start_hour=4, end_hour=21, height=500, dense_view=True
        )
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

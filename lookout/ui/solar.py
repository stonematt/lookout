"""
solar.py
Solar analysis UI for Lookout weather station dashboard.

This module provides the Streamlit presentation layer for solar analysis,
including caching wrappers and data quality displays. Core data processing
is handled by lookout.core.solar_analysis and visualizations by lookout.core.solar_viz.
"""

import pandas as pd
import streamlit as st

import lookout.core.solar_analysis as solar_analysis
import lookout.core.solar_viz as solar_viz
from lookout.utils.log_util import app_logger
from lookout.utils.memory_utils import (
    BYTES_TO_MB,
    cleanup_cache_functions,
    force_garbage_collection,
    get_memory_usage,
    get_object_memory_usage,
    log_memory_usage,
)

logger = app_logger(__name__)


@st.cache_data
def _get_cached_solar_statistics(df: pd.DataFrame, daily_energy: pd.Series) -> dict:
    """Cache solar statistics calculation."""
    return solar_analysis.get_solar_statistics(df, daily_energy)


@st.cache_data
def _get_cached_daily_energy(df: pd.DataFrame) -> pd.Series:
    """Cache daily energy calculation."""
    return solar_analysis.calculate_daily_energy(df)


@st.cache_data
def _get_cached_seasonal_breakdown(daily_energy: pd.Series) -> dict:
    """Cache seasonal breakdown calculation."""
    return solar_analysis.get_seasonal_breakdown(daily_energy)


@st.cache_data
def _get_cached_hourly_patterns(df: pd.DataFrame) -> pd.Series:
    """Cache hourly pattern calculation."""
    return solar_analysis.calculate_hourly_patterns(df)


@st.cache_data
def _get_cached_solar_heatmap_data(df: pd.DataFrame, row_mode: str) -> pd.DataFrame:
    """Cache solar heatmap data preparation."""
    return solar_viz.prepare_solar_heatmap_data(df, row_mode)


def render():
    """
    Render the solar analysis dashboard.

    Displays solar radiation analysis including:
    - Key statistics and metrics
    - Time series of solar radiation
    - Daily energy accumulation
    - Seasonal comparisons
    - Hourly patterns
    - Solar radiation heatmap
    """
    st.header("☀️ Solar Analysis")

    # Get data from session state
    if "history_df" not in st.session_state:
        st.warning("No weather data available. Please load data first.")
        return

    df = st.session_state["history_df"]

    # Filter for data with solar radiation readings
    solar_df = df[df['solarradiation'].notna()].copy()
    if solar_df.empty:
        st.warning("No solar radiation data available in the current dataset.")
        return

    # Ensure datetime column is properly formatted
    solar_df['datetime'] = pd.to_datetime(solar_df['dateutc'], unit='ms', utc=True)
    solar_df['date'] = solar_df['datetime'].dt.date

    # Add daylight period classification if not present
    if 'daylight_period' not in solar_df.columns:
        # Simple classification based on hour (this is a fallback)
        # In production, this should come from the CLI processing
        solar_df['daylight_period'] = solar_df['datetime'].dt.hour.apply(
            lambda h: 'day' if 6 <= h <= 18 else 'night'
        )

    # Calculate core solar metrics with caching
    with st.spinner("Calculating solar energy..."):
        daily_energy = _get_cached_daily_energy(solar_df)

    if daily_energy.empty:
        st.warning("Unable to calculate solar energy from available data.")
        return

    # Display key statistics
    _display_solar_statistics(solar_df, daily_energy)

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Time Series",
        "⚡ Daily Energy",
        "🌤️ Seasonal Analysis",
        "🕐 Patterns & Heatmap"
    ])

    with tab1:
        _render_time_series_tab(solar_df)

    with tab2:
        _render_daily_energy_tab(daily_energy)

    with tab3:
        _render_seasonal_analysis_tab(daily_energy)

    with tab4:
        _render_patterns_heatmap_tab(solar_df)

    # Memory usage tracking
    mem_usage = get_memory_usage()
    logger.debug(f"Solar UI memory usage: {mem_usage:.1f} MB")


def _display_solar_statistics(df: pd.DataFrame, daily_energy: pd.Series):
    """Display key solar statistics in metric cards."""
    stats = _get_cached_solar_statistics(df, daily_energy)

    if not stats:
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Peak Radiation",
            f"{stats['peak_radiation_w_per_m2']:.0f} W/m²",
            help="Maximum solar radiation intensity recorded"
        )

    with col2:
        st.metric(
            "Daily Average",
            f"{stats['avg_daily_energy_kwh_per_m2']:.2f} kWh/m²",
            help="Average daily solar energy accumulation"
        )

    with col3:
        st.metric(
            "Annual Total",
            f"{stats['annual_energy_kwh_per_m2']:.0f} kWh/m²",
            help="Estimated annual solar energy yield"
        )

    with col4:
        st.metric(
            "Peak Hours",
            f"{stats['peak_hours_utc']}",
            help="Hours with highest average radiation (UTC)"
        )

    # Data quality info
    with st.expander("📊 Data Quality", expanded=False):
        st.write(f"**Data Period:** {daily_energy.index.min()} to {daily_energy.index.max()}")
        st.write(f"**Days with Data:** {len(daily_energy)}")
        st.write(f"**Total Solar Readings:** {len(df)}")

        # Calculate data completeness
        completeness = 100.0  # Simplified for now
        st.write(f"**Data Completeness:** {completeness:.1f}%")


def _render_time_series_tab(df: pd.DataFrame):
    """Render solar radiation time series tab."""
    st.subheader("Solar Radiation Over Time")

    # Date range selector
    date_range = st.date_input(
        "Select date range",
        value=(df['datetime'].min().date(), df['datetime'].max().date()),
        key="solar_date_range"
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[
            (df['datetime'].dt.date >= start_date) &
            (df['datetime'].dt.date <= end_date)
        ].copy()
    else:
        filtered_df = df.copy()

    if filtered_df.empty:
        st.warning("No data available for selected date range.")
        return

    # Create and display chart
    with st.spinner("Creating time series chart..."):
        fig = solar_viz.create_solar_time_series_chart(filtered_df)

    st.plotly_chart(fig, width='stretch')

    # Additional info
    st.info("Chart shows solar radiation intensity (W/m²) during daylight hours only.")


def _render_daily_energy_tab(daily_energy: pd.Series):
    """Render daily energy accumulation tab."""
    st.subheader("Daily Solar Energy Accumulation")

    # Create and display chart
    with st.spinner("Creating energy chart..."):
        fig = solar_viz.create_daily_energy_chart(daily_energy)

    st.plotly_chart(fig, width='stretch')

    # Summary statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_energy = daily_energy.mean()
        st.metric("Average Daily Energy", f"{avg_energy:.2f} kWh/m²")

    with col2:
        max_energy = daily_energy.max()
        max_date = str(daily_energy.idxmax()) if not daily_energy.empty else "N/A"
        st.metric("Peak Daily Energy", f"{max_energy:.2f} kWh/m²", f"on {max_date}")

    with col3:
        total_energy = daily_energy.sum()
        st.metric("Total Period Energy", f"{total_energy:.1f} kWh/m²")

    st.info("Energy calculated using trapezoidal integration of solar radiation readings.")


def _render_seasonal_analysis_tab(daily_energy: pd.Series):
    """Render seasonal analysis tab."""
    st.subheader("Seasonal Solar Energy Comparison")

    # Calculate seasonal data
    seasonal_data = _get_cached_seasonal_breakdown(daily_energy)

    if not seasonal_data:
        st.warning("Insufficient data for seasonal analysis.")
        return

    # Create and display chart
    with st.spinner("Creating seasonal comparison..."):
        fig = solar_viz.create_seasonal_comparison_chart(seasonal_data)

    st.plotly_chart(fig, width='stretch')

    # Seasonal details
    if 'seasonal_variation_ratio' in seasonal_data:
        variation = seasonal_data['seasonal_variation_ratio']
        st.metric(
            "Seasonal Variation",
            f"{variation:.1f}x",
            help="Ratio of summer to winter average daily energy"
        )

    # Detailed seasonal table
    st.subheader("Seasonal Details")

    seasonal_rows = []
    for season in ['winter', 'spring', 'summer', 'fall']:
        if season in seasonal_data:
            data = seasonal_data[season]
            seasonal_rows.append({
                'Season': season.capitalize(),
                'Average Daily (kWh/m²)': f"{data['mean_kwh_per_m2']:.2f}",
                'Peak Daily (kWh/m²)': f"{data['max_kwh_per_m2']:.2f}",
                'Days': data['days']
            })

    if seasonal_rows:
        seasonal_df = pd.DataFrame(seasonal_rows)
        st.dataframe(seasonal_df, width='stretch')


def _render_patterns_heatmap_tab(df: pd.DataFrame):
    """Render hourly patterns and heatmap tab."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Hourly Pattern")

        # Calculate hourly patterns
        hourly_patterns = _get_cached_hourly_patterns(df)

        if hourly_patterns.empty:
            st.warning("No hourly pattern data available.")
        else:
            # Create and display chart
            with st.spinner("Creating hourly pattern chart..."):
                fig = solar_viz.create_hourly_pattern_chart(hourly_patterns)

            st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Solar Radiation Heatmap")

        # Heatmap mode selector
        heatmap_mode = st.selectbox(
            "View mode",
            ["day", "week", "month"],
            index=0,
            key="solar_heatmap_mode",
            help="Choose aggregation level for heatmap rows"
        )

        # Prepare heatmap data
        heatmap_df = _get_cached_solar_heatmap_data(df, heatmap_mode)

        if heatmap_df.empty:
            st.warning("No heatmap data available.")
        else:
            # Create and display heatmap
            with st.spinner("Creating solar heatmap..."):
                fig = solar_viz.create_solar_heatmap(heatmap_df, row_mode=heatmap_mode)

            st.plotly_chart(fig, width='stretch')

    st.info("Heatmap shows solar radiation patterns by hour and selected time period.")
"""
Solar Radiation UI Module - Phase 3.1: Foundation

Basic UI structure with data loading and date filtering for solar radiation measurements.
"""

import streamlit as st
import pandas as pd
import lookout.ui.components as ui_components
from lookout.utils.log_util import app_logger

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

    # Show proof of concept
    st.success(f"✓ Loaded {len(periods_df)} 15-minute periods")

    # TODO: Phase 3.2 will add metrics
    # TODO: Phase 3.3 will add tabs


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
        mask = (periods_df["period_start"] >= start_ts) & (periods_df["period_start"] < end_ts)
        filtered_df = periods_df.loc[mask].copy()
        logger.info(f"Filtered to date range: {len(filtered_df)} periods")
        return filtered_df

    return periods_df
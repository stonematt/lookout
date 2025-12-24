"""
Solar UI Service
Provides data preparation and business logic for solar energy visualizations.
Follows SOLID principles by separating business logic from presentation.
"""

import streamlit as st
import pandas as pd
from typing import List, Optional, Tuple
from lookout.core.solar_energy_periods import aggregate_to_daily
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


@st.cache_data(show_spinner=False, max_entries=20, ttl=7200)
def get_cached_daily_aggregation(periods_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cached version of daily aggregation to prevent recalculation on every page run.

    :param periods_df: DataFrame with energy period data
    :return: DataFrame with daily aggregations
    """
    return aggregate_to_daily(periods_df)


class SolarUIService:
    """
    Service layer for solar energy UI data preparation.
    Handles business logic separate from presentation layer.
    """

    def get_available_dates(self, periods_df: pd.DataFrame) -> List[str]:
        """
        Get list of dates that have solar production data.

        :param periods_df: DataFrame with energy period data
        :return: List of date strings in YYYY-MM-DD format
        """
        daily_df = get_cached_daily_aggregation(periods_df)
        if daily_df.empty:
            return []

        # Get dates that have production (> 0 kWh)
        dates_with_data = daily_df[daily_df["daily_kwh"] > 0]["date"]
        return [d.strftime("%Y-%m-%d") for d in dates_with_data]

    def validate_energy_catalog(self, periods_df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate that energy catalog data is properly processed.

        :param periods_df: DataFrame to validate
        :return: Tuple of (is_valid, error_message)
        """
        if periods_df.empty:
            return False, "No energy periods data available in cache"

        if "period_start" not in periods_df.columns:
            return (
                False,
                "Energy catalog data is not properly processed. Please regenerate the energy catalog.",
            )

        return True, ""

    def get_selected_date(self, available_dates: List[str], session_state: dict) -> str:
        """
        Get the currently selected date from session state, with fallback to first available date.

        :param available_dates: List of available date strings
        :param session_state: Session state dictionary
        :return: Selected date string
        """
        if not available_dates:
            return ""

        # Initialize session state if not set
        if "selected_solar_date" not in session_state:
            session_state["selected_solar_date"] = available_dates[0]

        # Ensure stored date is still valid
        stored_date = session_state["selected_solar_date"]
        if stored_date in available_dates:
            return stored_date
        else:
            # Fallback to first available date
            session_state["selected_solar_date"] = available_dates[0]
            return available_dates[0]

    def update_selected_date(self, selected_date: str, session_state: dict) -> None:
        """
        Update the selected date in session state.

        :param selected_date: Date string to store
        :param session_state: Session state dictionary
        """
        session_state["selected_solar_date"] = selected_date

    def get_selectbox_index(
        self, available_dates: List[str], selected_date: str
    ) -> int:
        """
        Calculate the correct index for selectbox based on selected date.

        :param available_dates: List of available date strings
        :param selected_date: Currently selected date string
        :return: Index for selectbox
        """
        try:
            return available_dates.index(selected_date)
        except ValueError:
            return 0  # Fallback to first option

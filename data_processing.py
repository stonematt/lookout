# data_processing.py
from dateutil import parser
import pandas as pd

# import streamlit as st
import awn_controller as awn  # Assuming awn_controller is your custom module
from log_util import app_logger  # Ensure you have this module set up for logging

logger = app_logger(__name__)


def get_history_min_max(df, date_column="date", data_column="tempf", data_label="temp"):
    """
    Calculate the minimum and maximum values of a data column for specific time periods.

    :param df: pd.DataFrame - Input dataframe with weather data.
    :param date_column: str - The column representing dates that are tz-aware.
    :param data_column: str - The column representing data.
    :param data_label: str - The label for the data column.
    :return: dict - Dictionary with min and max values for specific time periods.
    """
    # Ensure the date column is datetime and timezone-aware
    df[date_column] = pd.to_datetime(df[date_column])
    tz = df[date_column].dt.tz

    # Get the current timestamp in the same timezone
    now = pd.Timestamp.now(tz=tz)
    today_start = now.normalize()  # Start of the current day

    # Define date ranges
    date_ranges = {
        "today": (today_start, now),
        "yesterday": (today_start - pd.Timedelta(days=1), today_start),
        "last 7d": (now - pd.Timedelta(days=7), now),
        "last 30d": (now - pd.Timedelta(days=30), now),
        "last 90d": (now - pd.Timedelta(days=90), now),
        "last 365d": (now - pd.Timedelta(days=365), now),
    }

    results = {}

    # Current value is the data_column for the max(date_column)
    current_data = df.loc[df[date_column].idxmax(), data_column]

    for label, (start, end) in date_ranges.items():
        # Filter data for the specific time period
        period_data = df[(df[date_column] >= start) & (df[date_column] < end)]

        # Calculate min and max values
        results[label] = {
            "min": period_data[data_column].min(),
            "max": period_data[data_column].max(),
            "current": current_data,
        }

    return results

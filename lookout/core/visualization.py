"""
visualizations.py
Collection of functions to create charts and visualizations for streamlit
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def display_current_data(data):
    """
    Display current device data in the Streamlit app.

    :param data: Dictionary with device's last known data.
    """
    st.subheader("Current Data")
    for key, value in data.items():
        st.text(f"{key}: {value}")


def display_heatmap(df, metric, interval="15T"):
    """
    Generate and display a heatmap for a given metric over specified intervals.

    :param df: DataFrame with the data.
    :param metric: Metric to visualize.
    :param interval: Interval for aggregation, default 15 minutes.
    """
    df["date"] = pd.to_datetime(df["date"])
    df["interval"] = df["date"].dt.floor(interval).dt.strftime("%H:%M")
    fig = px.density_heatmap(df, x="date", y="interval", z=metric, histfunc="avg")
    st.plotly_chart(fig, width="stretch")


def display_line_chart(df, metrics):
    """
    Display a line chart for selected metrics from the data.

    :param df: DataFrame containing the data.
    :param metrics: List of metric names to include in the chart.
    """
    fig = px.line(df, x="date", y=metrics, title="Historical Data")
    st.plotly_chart(fig, width="stretch")


def better_heatmap_table(df, metric, aggfunc="max", interval=1800):
    """
    Create a pivot table of aggregate values for a given metric, with the row index
    as a time stamp for every `interval`-second interval and the column index as
    the unique dates in the "date" column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame with a "date" column and a column with the desired `metric`.
    metric : str
        The name of the column in `df` containing the desired metric.
    aggfunc : str or function
        The aggregation function to use when computing the pivot table. Can be a string
        of a built-in function (e.g., "mean", "sum", "count"), or a custom function.
    interval : int
        The number of seconds for each interval. For example, `interval=15` would
        create an interval of 15 seconds.

    Returns
    -------
    pandas.DataFrame
        A pivot table where the row index is a time stamp for every `interval`-second
        interval, and the column index is the unique dates in the "date" column.
        The values are the aggregate value of the `metric` column for each interval
        and each date.
    """

    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.date
    df["interval"] = df["date"].dt.floor(f"{interval}s").dt.strftime("%H:%M:%S")
    table = df.pivot_table(
        index=["interval"],
        columns=["day"],
        values=metric,
        aggfunc=aggfunc,
    )

    return table


def heatmap_chart(heatmap_table):
    fig = px.imshow(heatmap_table, x=heatmap_table.columns, y=heatmap_table.index)
    st.plotly_chart(fig)


def display_data_coverage_heatmap(df, metric="tempf", interval_minutes=60):
    """
    Show data density heatmap based on datapoint counts per interval.

    :param df: Filtered DataFrame with a datetime column "date".
    :param metric: A valid metric column (used for presence, not value).
    :param interval_minutes: Interval granularity in minutes.
    """
    interval_seconds = interval_minutes * 60
    table = better_heatmap_table(
        df, metric=metric, aggfunc="count", interval=interval_seconds
    )
    heatmap_chart(table)


def display_hourly_coverage_heatmap(df):
    """
    Render a grid-style heatmap showing the number of samples per hour per day.
    Missing days/hours are zero-filled to preserve visual continuity.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["date"].dt.hour
    df["day"] = df["date"].dt.date

    # Count samples per (day, hour)
    grouped = df.groupby(["day", "hour"]).size().reset_index(name="samples")

    # Create full (day, hour) index to fill gaps
    full_days = pd.date_range(df["day"].min(), df["day"].max(), freq="D").date
    full_hours = list(range(24))
    full_index = pd.MultiIndex.from_product(
        [full_days, full_hours], names=["day", "hour"]
    )

    full_df = (
        grouped.set_index(["day", "hour"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    pivot = full_df.pivot(index="day", columns="hour", values="samples")

    hover_text = [
        [
            f"Hour: {hour}<br>Date: {date}<br>Samples: {pivot.at[date, hour]}"
            for hour in pivot.columns
        ]
        for date in pivot.index
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(h) for h in pivot.columns],
            y=pivot.index.astype(str),
            text=hover_text,
            hoverinfo="text",
            colorscale="Blues",
            showscale=True,
            hoverongaps=False,
            zsmooth=False,  # No smoothing — preserves grid
        )
    )

    fig.update_traces(xgap=2, ygap=2)  # Visual cell spacing

    fig.update_layout(
        title="Hourly Coverage",
        xaxis=dict(
            title="Hour of Day",
            tickmode="linear",
            dtick=1,
            type="category",
            showgrid=True,
            gridcolor="lightgrey",
        ),
        yaxis=dict(
            title="Date",
            type="category",
            autorange="reversed",
            showgrid=True,
            gridcolor="lightgrey",
        ),
        margin=dict(l=60, r=20, t=40, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, width="stretch")




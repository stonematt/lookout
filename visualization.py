""" visualizations.py"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from colour import Color

gauge_defaults = {
    "temps": {
        "start_color": "#33CCFF",
        "end_color": "#FF3300",
        "steps": 100,
        "min_val": 0,
        "max_val": 120,
        "title": "Temperature (°F)",
    },
    "wind": {
        "start_color": "#CCEFFF",
        "end_color": "#6666FF",
        "steps": 5,
        "min_val": 0,
        "max_val": 100,
        "title": "Wind Speed (mph)",
    },
    "pressure": {
        "start_color": "#FFCCCC",
        "end_color": "#FF6666",
        "steps": 8,
        "min_val": 950,
        "max_val": 1050,
        "title": "Barometric Pressure (hPa)",
    },
    "rain": {
        "start_color": "#ADD8E6",  # Light blue, representing light rain
        "end_color": "#00008B",  # Dark blue, representing heavy rainfall
        "steps": 100,
        "min_val": 0,
        "max_val": 30,
        "title": "Rainfall (inches)",
    },
    "rain_rate": {
        "start_color": "#ADD8E6",  # Light blue, representing light rain
        "end_color": "#00008B",  # Dark blue, representing heavy rainfall
        "steps": 100,
        "min_val": 0,
        "max_val": 10,
        "title": "Rainfall (inches)",
    },
}


def generate_gradient_steps(start_color, end_color, steps):
    """
    Generate gradient steps for a Plotly gauge chart.

    :param start_color: Hex string for the start color (e.g., "#FF0000" for red).
    :param end_color: Hex string for the end color (e.g., "#00FF00" for green).
    :param steps: Number of gradient steps to generate.
    :return: A list of colors in hex format representing the gradient steps.
    """
    # Create Color objects for start and end
    start = Color(start_color)
    end = Color(end_color)

    # Generate colors in the gradient
    gradient_colors = list(start.range_to(end, steps))

    # Convert colors to hex format
    hex_colors = [color.get_hex() for color in gradient_colors]

    return hex_colors


def create_gauge_chart(
    value, metric_type, title="Gauge", gauge_defaults=gauge_defaults, chart_height=450
):
    """
    Create a Plotly gauge chart with gradient steps and a specified chart height.

    :param value: The value to display on the gauge.
    :param metric_type: The type of metric, which determines the gauge defaults.
    :param title: Title for the gauge chart.
    :param gauge_defaults: Default settings and preferences by metric type.
    :param chart_height: The height of the chart in pixels.
    :return: Plotly figure object for the gauge chart.
    """
    defaults = gauge_defaults[metric_type]
    gradient_colors = generate_gradient_steps(
        defaults["start_color"], defaults["end_color"], defaults["steps"]
    )
    step_values = np.linspace(
        defaults["min_val"], defaults["max_val"], len(gradient_colors) + 1
    )
    gauge_steps = [
        {"range": [step_values[i], step_values[i + 1]], "color": gradient_colors[i]}
        for i in range(len(gradient_colors))
    ]

    # Create the gauge figure
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title},
            gauge={
                "axis": {"range": [defaults["min_val"], defaults["max_val"]]},
                "bar": {"color": "black"},
                "steps": gauge_steps,
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": value,
                },
            },
        )
    )

    # Update the layout to set the chart height
    fig.update_layout(height=chart_height)

    return fig


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
    st.plotly_chart(fig, use_container_width=True)


def display_line_chart(df, metrics):
    """
    Display a line chart for selected metrics from the data.

    :param df: DataFrame containing the data.
    :param metrics: List of metric names to include in the chart.
    """
    fig = px.line(df, x="date", y=metrics, title="Historical Data")
    st.plotly_chart(fig, use_container_width=True)

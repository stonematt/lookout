"""
Main streamlit.io application
"""

import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dateutil import parser

import ambient_client

# my modules
# import storj_df_s3 as sj
import awn_controller as awn
import data_processing as lo_dp
import visualization as lo_viz
from log_util import app_logger

logger = app_logger(__name__)


st.set_page_config(
    page_title="Weather Station Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

AMBIENT_ENDPOINT = st.secrets["AMBIENT_ENDPOINT"]
AMBIENT_API_KEY = st.secrets["AMBIENT_API_KEY"]
AMBIENT_APPLICATION_KEY = st.secrets["AMBIENT_APPLICATION_KEY"]


# %%
# define variables
sec_in_hour = 3600 * 1000
bucket = "lookout"
auto_refresh_min = 6  # minutes to wait for auto update
auto_refresh_max = 3 * 24 * 60  # 3 days in minutes

keys = {
    "temp_keys": [
        "tempf",
    ],
    "atmos_keys": [],
    "wind_keys": [
        "winddir",
        "winddir_avg10m",
        "windspeedmph",
        "windspdmph_avg10m",
        "windgustmph",
        "maxdailygust",
    ],
    "rain_keys": [
        "hourlyrainin",
        "eventrainin",
        "dailyrainin",
        "weeklyrainin",
        "monthlyrainin",
        "yearlyrainin",
    ],
    "all_keys": [
        "tempf",
        "humidity",
        "winddir",
        "winddir_avg10m",
        "windspeedmph",
        "windspdmph_avg10m",
        "windgustmph",
        "maxdailygust",
        "hourlyrainin",
        "eventrainin",
        "dailyrainin",
        "weeklyrainin",
        "monthlyrainin",
        "yearlyrainin",
        "solarradiation",
        "uv",
        "temp1f",
        "humidity1",
        "feelsLike",
        "dewPoint",
        "feelsLike1",
        "dewPoint1",
        "feelsLikein",
        "dewPointin",
        "lastRain",
    ],
}


def update_session_data(device, hist_df=None, limit=250, pages=10):
    """
    Update session with latest historical data and reset session counter.

    :param device: Object representing the device.
    :param hist_df: DataFrame of current historical data, defaults to session state history.
    :param limit: int - Max records to fetch per call, default 250.
    :param pages: int - Number of pages to fetch, default 10.
    :return: None.
    """
    try:
        # Use provided or session state history
        current_df = (
            hist_df
            if hist_df is not None
            else st.session_state.get("history_df", pd.DataFrame())
        )

        # Fetch updated history
        updated_df = awn.get_history_since_last_archive(
            device, current_df, limit=limit, pages=pages
        )

        # Update session state
        st.session_state["history_df"] = updated_df
        st.session_state["history_max_dateutc"] = int(
            st.session_state["history_df"]["dateutc"].max().timestamp() * 1000
        )

        logger.info("Session data updated successfully.")
    except Exception as e:
        logger.error(f"Failed to update session data: {e}")
        st.error("An error occurred while updating session data. Please try again.")


def to_date(date_string: str):
    """
    Convert a date string to a datetime object.

    :param date_string: str - The date string to parse.
    :return: datetime - Parsed datetime object.
    :raises: Exception if date string parsing fails.
    """
    try:
        return parser.parse(date_string)
    except Exception as e:
        logger.error(f"Error parsing date string: {e}", exc_info=True)
        raise


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


def load_or_update_data(
    device, bucket, file_type, auto_update, short_minutes, long_minutes
):
    """
    Wrapper function to load or update historical data with status messages.

    :param device: Object representing the device.
    :param bucket: str - The S3 bucket name for archive storage.
    :param file_type: str - The file type for archive storage (e.g., 'parquet').
    :param auto_update: bool - Whether to perform updates automatically.
    :param short_minutes: int - Minimum age threshold for updates (in minutes).
    :param long_minutes: int - Maximum age threshold for updates (in minutes).
    :return: None - Updates Streamlit session state directly.
    """
    update_message = st.empty()

    # Initial load
    if "history_df" not in st.session_state:
        update_message.text("Getting archive data...")
        st.session_state["history_df"] = awn.load_archive_for_device(
            device, bucket, file_type
        )

        # Initialize session state variables
        st.session_state["history_max_dateutc"] = int(
            st.session_state["history_df"]["dateutc"].max().timestamp() * 1000
        )
        st.session_state["cloud_max_dateutc"] = int(
            st.session_state["history_df"]["dateutc"].max().timestamp() * 1000
        )
        st.session_state["session_counter"] = 0

        logger.info("Initial archive load completed.")
        update_message.empty()

    # Fetch interim data if conditions are met
    history_df = st.session_state["history_df"]
    history_max_dateutc = st.session_state["history_max_dateutc"]

    if should_update_history(
        device_last_dateutc=device["lastData"]["dateutc"],
        history_max_dateutc=history_max_dateutc,
        short_minutes=short_minutes,
        long_minutes=long_minutes,
        auto_update=auto_update,
    ):
        update_message.text("Updating historical data...")
        update_session_data(device, history_df)
        st.session_state["history_max_dateutc"] = int(
            st.session_state["history_df"]["dateutc"].max().timestamp() * 1000
        )

        logger.info("Historical data updated successfully.")
        update_message.empty()


def should_update_history(
    device_last_dateutc, history_max_dateutc, short_minutes, long_minutes, auto_update
):
    """
    Determines if the historical data should be updated based on age thresholds.

    :param device_last_dateutc: int - Last data timestamp from the device (in milliseconds).
    :param history_max_dateutc: int - Maximum data timestamp in the archive (in milliseconds).
    :param short_minutes: int - Minimum age (in minutes) required to trigger an update.
    :param long_minutes: int - Maximum age (in minutes) for which updates are valid.
    :param auto_update: bool - Whether auto-update is enabled.
    :return: bool - True if history should be updated, False otherwise.
    """
    if not auto_update:
        return False

    delta_ms = device_last_dateutc - history_max_dateutc
    short_ms = short_minutes * 60 * 1000  # Convert minutes to milliseconds
    long_ms = long_minutes * 60 * 1000  # Convert minutes to milliseconds

    return short_ms <= delta_ms < long_ms


def make_column_gauges(gauge_list, chart_height=300):
    """
    Take a list of metrics and produce a row of gauges, with min, median, and max values displayed below each gauge.

    :param gauge_list: list of dicts with metrics, titles to render as gauges, and their types.
    :param chart_height: height of the charts in the row
    """
    # Create columns for gauges
    cols = st.columns(len(gauge_list))

    for i, gauge in enumerate(gauge_list):
        metric = gauge["metric"]
        title = gauge["title"]
        metric_type = gauge["metric_type"]

        # Retrieve the last value for the metric
        value = last_data.get(metric, 0)

        # Calculate min, median, max for the current metric from history_df
        min_val = history_df[metric].min()
        median_val = history_df[metric].median()
        max_val = history_df[metric].max()

        # Create the gauge chart for the current metric
        gauge_fig = lo_viz.create_gauge_chart(
            value=value, metric_type=metric_type, title=title, chart_height=chart_height
        )

        # Plot the gauge in the respective column, fitting it to the column width
        with cols[i]:
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Use markdown to display min, median, and max values below the gauge with less vertical space
            stats_md = f"""<small>
            <b>Min:</b> {min_val:.2f} <br>
            <b>Median:</b> {median_val:.2f} <br>
            <b>Max:</b> {max_val:.2f}
            </small>"""
            st.markdown(stats_md, unsafe_allow_html=True)


# Setup and get data ########################

devices = ambient_client.get_devices()
device = False
device_last_dateutc = 0
last_data = {}

device_menu = "98:CD:AC:22:0D:E5"
if len(devices) == 1:
    device = devices[0]
    device_menu = device["macAddress"]
    device_name = device["info"]["name"]
    last_data = device["lastData"]
    st.header(f"Weather Station:  {device_name}")
    logger.debug(f"One device found:  {device['info']['name']}")

    # Compare device's last data UTC with the archive max dateutc
    device_last_dateutc = device["lastData"].get("dateutc")

# if we dont' get a device from ambient. blow up.
if not device:
    st.write("No connection to Ambient Network")
    exit()

file_type = "parquet"
device_mac = device_menu
hist_file = f"{device_mac}.{file_type}"
# lookout/98:CD:AC:22:0D:E5.json

# pause for ambient.
time.sleep(1)

# %%
auto_update = st.sidebar.checkbox("Auto-Update", value=True)


# Call the wrapper to load or update the weather station data
load_or_update_data(
    device=device,
    bucket=bucket,
    file_type=file_type,
    auto_update=auto_update,
    short_minutes=auto_refresh_min,
    long_minutes=auto_refresh_max,
)

# Access the updated history_df and max_dateutc
history_df = st.session_state["history_df"]
history_max_dateutc = st.session_state["history_max_dateutc"]

st.sidebar.write(f"History as of: {history_df.date.max()}")

history_age_h = lo_dp.get_human_readable_duration(
    device_last_dateutc, history_max_dateutc
)

# Display in the sidebar
st.sidebar.write(f"Archive is {history_age_h} old.")
# %%


# Gauge configurations
temp_gauges = [
    {"metric": "tempf", "title": "Temp Outside", "metric_type": "temps"},
    {"metric": "tempinf", "title": "Temp Bedroom", "metric_type": "temps"},
    {"metric": "temp1f", "title": "Temp Office", "metric_type": "temps"},
]

rain_guages = [
    {"metric": "hourlyrainin", "title": "Hourly Rain", "metric_type": "rain_rate"},
    {"metric": "eventrainin", "title": "Event Rain", "metric_type": "rain"},
    {"metric": "dailyrainin", "title": "Daily Rain", "metric_type": "rain"},
    {"metric": "weeklyrainin", "title": "Weekly Rain", "metric_type": "rain"},
    {"metric": "monthlyrainin", "title": "Monthly Rain", "metric_type": "rain"},
    {"metric": "yearlyrainin", "title": "Yearly Rain", "metric_type": "rain"},
]


# the "metric" that may be boxplotted
box_plot = [
    {"metric": "tempf", "title": "Temp Outside", "metric_type": "temps"},
    {"metric": "tempinf", "title": "Temp Bedroom", "metric_type": "temps"},
    {"metric": "temp1f", "title": "Temp Office", "metric_type": "temps"},
    {"metric": "solarradiation", "title": "Solar Radiation", "metric_type": "temps"},
]

# Present the dashboard ########################

row1 = st.columns(2)

with row1[0]:

    temp_bars = lo_dp.get_history_min_max(history_df, "date", "tempf", "temp")
    lo_viz.draw_horizontal_bars(temp_bars, label="Temperature (°F)")

with row1[1]:

    # Parameters for the polar chart
    # value_col = "windspeedmph"
    # direction_col = "winddir"
    # Define valid column pairs for the polar chart
    valid_pairs = [
        ("windspeedmph", "winddir", "Wind Speed"),
        ("windspdmph_avg10m", "winddir_avg10m", "10m Average Wind"),
        ("windgustmph", "winddir", "Wind Gust"),
        ("maxdailygust", "winddir_avg10m", "Max Daily Gust"),
    ]

    # Initialize session state for the selected pair
    if "selected_pair" not in st.session_state:
        st.session_state["selected_pair"] = valid_pairs[0]  # Default to the first pair

    # Unpack the selected pair from session state
    value_col, direction_col, wind_description = st.session_state["selected_pair"]

    # Use the wrapper function to prepare data
    grouped_data, value_labels, direction_labels = lo_dp.prepare_polar_chart_data(
        history_df,
        value_col,
        direction_col,
    )

    # Create and display the chart
    fig = lo_viz.create_windrose_chart(
        grouped_data, value_labels, color_palette="wind", title=value_col
    )

    st.plotly_chart(fig, use_container_width=True)

    # Dropdown for selecting the pair (below the chart)
    selected_pair = st.selectbox(
        "Wind Metric:",
        valid_pairs,
        format_func=lambda pair: pair[2],
        index=valid_pairs.index(
            st.session_state["selected_pair"]
        ),  # Set current session value
    )

    # Update session state if the user changes the dropdown value
    if selected_pair != st.session_state["selected_pair"]:
        st.session_state["selected_pair"] = selected_pair
        st.rerun()


# rain_bars = lo_dp.get_history_min_max(history_df, data_column= , )

# Display the header
st.subheader("Current")

if last_data:
    make_column_gauges(temp_gauges)
    # make_column_gauges(rain_guages)


st.subheader("Temps Plots")
# Let the user select multiple metrics for comparison
metric_titles = [metric["title"] for metric in box_plot]
selected_titles = st.multiselect(
    "Select metrics for the box plot:", metric_titles, default=metric_titles[0]
)

# Find the selected metrics based on the titles
selected_metrics = [metric for metric in box_plot if metric["title"] in selected_titles]

# User selects a box width
box_width_option = st.selectbox("Select box width:", ["hour", "day", "week", "month"])

if selected_metrics and "date" in history_df.columns:
    # Convert 'date' column to datetime if it's not already
    history_df["date"] = pd.to_datetime(history_df["date"])
    group_column = ""

    # Group by the selected box width option
    if box_width_option == "hour":
        history_df["hour"] = history_df["date"].dt.hour
        group_column = "hour"
    elif box_width_option == "day":
        history_df["day"] = history_df["date"].dt.dayofyear
        group_column = "day"
    elif box_width_option == "week":
        history_df["week"] = history_df["date"].dt.isocalendar().week
        group_column = "week"
    elif box_width_option == "month":
        history_df["month"] = history_df["date"].dt.month
        group_column = "month"

    # Create and render the box plot for each selected metric
    fig = go.Figure()
    for metric in selected_metrics:
        # Filter the DataFrame for the selected metric
        df_filtered = history_df[["date", group_column, metric["metric"]]].dropna()

        # Create a box plot for the current metric
        fig.add_trace(
            go.Box(
                x=df_filtered[group_column],
                y=df_filtered[metric["metric"]],
                name=metric["title"],
            )
        )

    # Update plot layout
    fig.update_layout(
        title=f"Comparison of Selected Metrics by {box_width_option.capitalize()}",
        xaxis_title=box_width_option.capitalize(),
        yaxis_title="Value",
    )

    st.plotly_chart(fig)

    st.write(device["lastData"])

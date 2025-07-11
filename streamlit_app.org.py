# %%
import time

import pandas as pd
import plotly.express as px
import streamlit as st
from ambient_api.ambientapi import AmbientAPI
from dateutil import parser

import lookout.api.awn_controller as awn

# my modules
import lookout.storage.storj as sj
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


st.set_page_config(
    page_title="Weather Station Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

AMBIENT_ENDPOINT = st.secrets["AMBIENT_ENDPOINT"]
AMBIENT_API_KEY = st.secrets["AMBIENT_API_KEY"]
AMBIENT_APPLICATION_KEY = st.secrets["AMBIENT_APPLICATION_KEY"]

api = AmbientAPI(
    # log_level="INFO",
    AMBIENT_ENDPOINT=AMBIENT_ENDPOINT,
    AMBIENT_API_KEY=AMBIENT_API_KEY,
    AMBIENT_APPLICATION_KEY=AMBIENT_APPLICATION_KEY,
)

# %%
# define variables
sec_in_hour = 3600 * 1000
bucket = "lookout"

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


def update_session_data(device, hist_df):
    # todo: convert this to just get interim data.
    """
    Update session with latest historical data and reset session counter.

    :param device: Object representing the device.
    :param hist_df: DataFrame of current historical data.
    :return: None.
    """

    st.session_state["history_df"] = awn.get_history_since_last_archive(device, hist_df)
    st.session_state["session_counter"] = 0


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


def initial_load_device_history(device, bucket, file_type, auto_update):
    """
    Load archive file from s3 resource and optionally update data to the current time.

    :param device: The device object to load history for.
    :param bucket: The name of the S3 bucket where the archive is stored.
    :param file_type: The file type of the archive.
    :param auto_update: Boolean flag to control whether to fetch interim data.
    """
    update_message = st.empty()
    update_message.text("Getting archive data")

    # Get archive
    st.session_state["history_df"] = awn.load_archive_for_device(
        device, bucket, file_type
    )
    df = st.session_state["history_df"]

    df_stats = st.empty()
    df_stats.text(f"Archive date range: {df.date.min()} to {df.date.max()}")

    # Fetch interim data if auto-update is enabled
    if auto_update:
        update_message.text("Getting data since last archive")
        st.session_state["history_df"] = awn.get_history_since_last_archive(
            device, st.session_state["history_df"], sleep=True
        )
        # Throttling due to Ambient API limitations
        time.sleep(1)
        st.session_state["session_counter"] = 0

    df_stats.empty()
    update_message.empty()


# %%
# Present the dashboard ########################
devices = api.get_devices()
device = False

device_menu = "98:CD:AC:22:0D:E5"
if len(devices) == 1:
    device = devices[0]
    device_menu = device.mac_address
    device_name = device.info["name"]
    st.header(f"Weather Station:  {device_name}")
    print(f"One device found:  {device.info['name']}")
# else:
#     device_menu = st.sidebar.selectbox(
#         "Select a device:", [device["macAddress"] for device in devices]
#     )

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
# Streamlit sidebar for auto-update toggle
auto_update = st.sidebar.checkbox("Auto-Update", value=False)

# start dashboard
if "history_df" not in st.session_state:
    initial_load_device_history(device, bucket, file_type, auto_update)

history_df = st.session_state["history_df"]

# Only fetch interim data if 'auto_update' is True
# might have to rethink the session counter logic for first run after auto_update
if auto_update and st.session_state["session_counter"] >= 1:
    st.session_state["history_df"] = awn.get_history_since_last_archive(
        device, history_df
    )
    history_df = st.session_state["history_df"]


save_df = st.sidebar.button("Save Archive")
if save_df:
    sj.save_df_to_s3(
        df=st.session_state["history_df"],
        bucket=bucket,
        key=hist_file,
        file_type=file_type,
    )
    save_df = False
# %%

st.subheader("Current")
guages = [
    {"tempf": "Temp Outside"},
    {"tempinf": "Temp Inside"},
    {"temp1f": "Temp Bedroom"},
]
# st.columns(len(guages))
st.write(device.last_data)
# heatmap_metric = st.selectbox("pick a metric", history_df.keys())
st.subheader("History")
fig = px.line(history_df, x="date", y=["tempinf", "tempf", "temp1f"], title="temp")
st.plotly_chart(fig)

# fig = px.line(
#     history_df,
#     x="date",
#     y=["eventrainin", "dailyrainin", "weeklyrainin", "monthlyrainin", "yearlyrainin"],
#     title="rain",
# )
# st.plotly_chart(fig)


# %%
def create_heatmap_date_hour_df(df, data_column):
    heatmap_df = df.loc[:, ["date", data_column]].copy()
    heatmap_df = heatmap_df[["date", data_column]]
    heatmap_df["datetime"] = pd.to_datetime(heatmap_df["date"])
    heatmap_df["date"] = heatmap_df["datetime"].dt.date
    heatmap_df["hour"] = heatmap_df["datetime"].dt.hour

    return heatmap_df


# heatmap_data_column = "eventrainin"
heatmap_data_column = st.selectbox("Heatmap data column", keys["all_keys"])
heatmap_df = create_heatmap_date_hour_df(history_df, heatmap_data_column)

# %%
grouped_df = heatmap_df.groupby(by=["date", "hour"]).max().reset_index()

# Pivot the dataframe
pivot_df = pd.pivot_table(
    heatmap_df,
    values=grouped_df.columns[2],
    index="date",
    columns="hour",
    aggfunc="max",
)


# %%
# Create a heat map using Ploty
fig = px.imshow(pivot_df)
fig.update_layout(
    title="Heat Map of Maximum Values by Day and Hour",
    xaxis_title="Hour",
    yaxis_title="Day",
)
st.subheader(f"Heat Map of Maximum Values by Day and Hour for {heatmap_data_column}")
st.write(fig)


st.write(history_df.describe())
st.write(history_df)
# %%
# pd.merge(h1_df,h2_df, on=h1_df.keys().to_list(), how='outer')

# %%

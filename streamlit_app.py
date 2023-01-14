# %%
# import libs
# from aioambient.api import API
from ambient_api.ambientapi import AmbientAPI
import time
from dateutil import parser
import pandas as pd
import plotly.express as px
import streamlit as st
import logging
import storj_df_s3 as sj
from datetime import datetime

# import pytz
from collections import defaultdict

# import numpy as np
# import plotly as pt
# import plotly.graph_objects as go
# from datetime import datetime
# from datetime import date, datetime,

AMBIENT_ENDPOINT = st.secrets["AMBIENT_ENDPOINT"]
AMBIENT_API_KEY = st.secrets["AMBIENT_API_KEY"]
AMBIENT_APPLICATION_KEY = st.secrets["AMBIENT_APPLICATION_KEY"]

# todo: delete aioambient:
# api = API(AMBIENT_APPLICATION_KEY, AMBIENT_API_KEY)
# api = AmbientAPI()
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

# load data


# engineer data

# build dashboard
def to_date(date_string: str):
    try:
        date = parser.parse(date_string)
        return date
    except Exception as e:
        print(f"Error parsing date string: {e}")
        raise e


# %%
def get_device_history_to_date(device, end_date=None, limit=288):
    history = []
    if end_date:
        # st.write(f"end date: {end_date} for history page")
        history = device.get_data(end_date=end_date, limit=limit)
    else:
        # st.write("no end date submitted")
        history = device.get_data(limit=limit)

    if history:
        return history
    else:
        st.write("someting broke")
        return history


# %%
# todo: chage to date range
def get_all_history_for_device(
    device, days_to_get=5, end_date=None, progress_message=st.empty()
):
    all_history = []
    last_min_date = ""
    progress_count = st.empty()
    progress_bar = st.progress(0)
    progress = 0

    min_date = datetime.fromtimestamp(end_date / 1000.0) if end_date else datetime.now()
    progress_message.text(f"Get data from Ambient to {min_date}")
    logging.warning(
        f"get_all_history_for_device days_to_get: {days_to_get}, end_date: {end_date}"
    )

    if end_date:
        last_history_retreived = get_device_history_to_date(device, end_date=end_date)
    else:
        last_history_retreived = get_device_history_to_date(device)
    all_history.extend(last_history_retreived)

    if last_history_retreived:
        last_min_date = last_history_retreived[-1]["dateutc"]

    while progress < days_to_get:
        progress += 1
        progress_count.text(f"Getting page {progress}")
        logging.warning(f"Getting page {progress}")
        progress_bar.progress(progress / days_to_get)
        time.sleep(1)
        next_history_page = get_device_history_to_date(device, end_date=last_min_date)
        next_history_min_date = next_history_page[-1]["dateutc"]
        logging.warning(
            f"next page len: {len(next_history_page)} for {next_history_min_date}"
        )

        # to do make sure we don't add the last page twice
        if next_history_min_date != last_min_date:
            last_min_date = next_history_min_date
            all_history = merge_distinct_items(all_history, next_history_page)
        else:
            # if the last two pages are the same, stop.
            break

    progress_count.empty()
    progress_bar.empty()
    progress_message = st.empty()
    # return all_history_df, all_history
    return all_history


def merge_distinct_items(old_list, new_data_list):
    """combine lists, return list of uniques

    Args:
        old_dict (list): dictionary to extend
        new_data_dict (list): dictionary of new data

    Returns:
        list: list of uniques in the tww dicts
    """
    logging.warning(f"Merging {len(old_list)} and {len(new_data_list)}")
    merged_list = {}

    if old_list and new_data_list:
        for data_point in new_data_list:
            if data_point not in old_list:
                old_list.append(data_point)
        merged_list = old_list

    if not old_list:
        merged_list = new_data_list

    return merged_list


# %%
st.cache(show_spinner=True)


def find_extreme_value_in_json(device_history, d_key):
    # Initialize value with the first dictionary's key value
    min_value = device_history[0][d_key]
    max_value = device_history[0][d_key]

    for data_point in device_history:
        if d_key in data_point and data_point[d_key] < min_value:
            min_value = data_point[d_key]
        if d_key in data_point and data_point[d_key] > max_value:
            max_value = data_point[d_key]

    # Return the values found
    return min_value, max_value


def heatmap(df, metric):
    df["hour"] = df["date"].dt.hour
    table = df.pivot_table(index="hour", columns="date", values=metric)
    fig = px.imshow(table, x=table.columns, y=table.index)
    st.plotly_chart(fig)


def make_history_df(history_json, tz):
    history_df = pd.json_normalize(history_json)
    history_df.sort_values(by="dateutc", inplace=True)

    # Convert 'date' column to local time
    history_df["date"] = pd.to_datetime(history_df["date"])
    history_df["date"] = history_df["date"].dt.tz_convert(tz)

    return history_df


# %%
def process_historical_data(hist_file, progress_message=st.empty()):
    """download historical data from storj,
    if successfull, return dict and min/max dates

    Args:
        hist_file (str): path to history file

    Returns:
        dict: device history from the backup
        datetime: device history minimun date
        datetime: device history maximum date
    """
    progress_message.text("Getting History from stroj.io")
    device_history = []

    device_history = sj.get_file_as_dict(hist_file)
    if device_history:
        device_history_min_date, device_history_max_date = find_extreme_value_in_json(
            device_history, "dateutc"
        )
        min_date = datetime.fromtimestamp(device_history_min_date / 1000.0)
        max_date = datetime.fromtimestamp(device_history_max_date / 1000.0)
        st.write(f"range in history form storj: min: {min_date}, max: {max_date}")
        logging.warning(
            f"range in history form storj: min: {min_date}, max: {max_date}"
        )
    else:
        logging.warning(f"no device history at {hist_file}")
        min_date = datetime.now() * 1000
        device_history_min_date = time.time() * 1000
        device_history_max_date = time.time() * 1000

    return device_history, device_history_min_date, device_history_max_date


# %%
def get_data(device, hist_file):
    tz = device.last_data.get("tz")
    progress_message = st.empty()

    # Retreive historical data from backup drive on Storj
    (
        device_history,
        device_history_min_date,
        device_history_max_date,
    ) = process_historical_data(hist_file, progress_message)

    # Get a few more days of historical data
    # todo: this could be optional...
    history_json = get_all_history_for_device(
        device,
        days_to_get=3,
        end_date=device_history_min_date,
        progress_message=progress_message,
    )

    # get data since the last time we ran the script.
    missing_hours = (time.time() * 1000 - device_history_max_date) / sec_in_hour
    progress_message.text(
        f"Missing {missing_hours} hours of recent history. Getting it now"
    )
    time.sleep(1)
    new_history = get_all_history_for_device(
        device, days_to_get=int(missing_hours / 24), progress_message=progress_message
    )

    # add new dataa to history
    progress_message.text("Combining datasets")
    history_json = merge_distinct_items(history_json, new_history)
    history_json = merge_distinct_items(device_history, history_json)

    progress_message.text("Making dataframe")
    history_df = make_history_df(history_json, tz)

    # todo: might be good to make this optional.
    progress_message.text("Saving history to storj.io")
    # sj.save_dict_to_fs(history_json, hist_file)

    progress_message.empty()

    return history_df, history_json


# %%
# Present the dashboard ########################

devices = api.get_devices()
device = False

device_menu = "98:CD:AC:22:0D:E5"
if len(devices) == 1:
    device = devices[0]
    device_menu = device.mac_address
    st.header(f"Weather Station:  {device.info['name']}")
    print(f"One device found:  {device.info['name']}")
# else:
#     device_menu = st.sidebar.selectbox(
#         "Select a device:", [device["macAddress"] for device in devices]
#     )

# if we dont' get a device from ambient. blow up.
if not device:
    st.write("No connection to Ambient Network")
    exit()


device_mac = device_menu
# device_mac = "98:CD:AC:22:0D:E5"
hist_file = bucket + "/" + device_mac + ".json"
# lookout/98:CD:AC:22:0D:E5.json

# pause for ambient.
time.sleep(1)

# %%
# start dashboard
# get data (save history_json and history_df for this session?)
if "history_json" not in st.session_state:
    st.write("no hist in session")

    st.session_state["history_df"], st.session_state["history_json"] = get_data(
        device, hist_file
    )
    st.session_state['session_counter'] = 0

history_json = st.session_state["history_json"]
history_df = st.session_state["history_df"]

st.session_state['session_counter'] += 1
if st.session_state['session_counter'] == 5:
    progress_message = st.empty()
    progress_message.text("Saving history to storj.io")
    sj.save_dict_to_fs(history_json, hist_file)

    st.session_state['session_counter'] = 0

    progress = st.empty()

# %%

# heatmap_metric = st.selectbox("pick a metric", history_df.keys())
# heatmap(history_df, heatmap_metric )

fig = px.line(history_df, x="date", y=["tempinf", "tempf", "temp1f"], title="temp")
st.plotly_chart(fig)

fig = px.line(
    history_df,
    x="date",
    y=["eventrainin", "dailyrainin", "weeklyrainin", "monthlyrainin", "yearlyrainin"],
    title="rain",
)
st.plotly_chart(fig)
# candlestick_tmp = history_df.groupby(history_df["date"].dt.date).agg(
#     {"temp": {"open": "first", "high": "max", "low": "min", "close": "last"}}
# )
# st.write(candlestick_tmp)

# %%


def create_heatmap_date_hour_df(df, data_column):
    heatmap_df = df[["date", data_column]]
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
# pivot_df = grouped_df.pivot(columns="hour", index="date", values=grouped_df.columns[2])
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


#  keys = [      "dateutc",      "tempinf",      "humidityin",      "baromrelin",      "baromabsin",
#      "tempf",      "humidity",      "winddir",      "winddir_avg10m",
#      "windspeedmph",      "windspdmph_avg10m",      "windgustmph",      "maxdailygust",
#      "hourlyrainin",      "eventrainin",      "dailyrainin",      "weeklyrainin",
#      "monthlyrainin",      "yearlyrainin",      "solarradiation",
#      "uv",      "temp1f",      "humidity1",      "batt1",      "batt_25",      "batt_co2",
#      "feelsLike",      "dewPoint",      "feelsLike1",      "dewPoint1",      "feelsLikein",      "dewPointin",
#      "lastRain",      "date",  ]
st.write(history_df.describe())
st.write(history_df)
# %%
# pd.merge(h1_df,h2_df, on=h1_df.keys().to_list(), how='outer')

# %%

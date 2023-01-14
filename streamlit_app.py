# %%
# import libs
# from aioambient.api import API
from ambient_api.ambientapi import AmbientAPI
import time
from dateutil import parser
import pandas as pd
import plotly.express as px
import streamlit as st
import storj_df_s3 as sj
import awn_controller as awn

# from datetime import datetime
# import logging

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


def heatmap(df, metric):
    df["hour"] = df["date"].dt.hour
    table = df.pivot_table(index="hour", columns="date", values=metric)
    fig = px.imshow(table, x=table.columns, y=table.index)
    st.plotly_chart(fig)


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


def update_session_data(device, hist_file):
    st.session_state["history_df"], st.session_state["history_json"] = awn.get_data(
        device, hist_file
    )
    st.session_state["session_counter"] = 0


# %%
# start dashboard
# get data (save history_json and history_df for this session?)
if "history_json" not in st.session_state:
    st.write("no hist in session")
    update_session_data(device, hist_file)


history_json = st.session_state["history_json"]
history_df = st.session_state["history_df"]

st.session_state["session_counter"] += 1
if st.session_state["session_counter"] >= 3:
    update_message = st.empty()
    update_message.text("Updating and refreshing...")
    progress_message = st.empty()
    update_session_data(device, hist_file)

    progress_message.text("Saving history to storj.io")
    sj.save_dict_to_fs(history_json, hist_file)

    st.session_state["session_counter"] = 0

    progress_message.empty()
    update_message.empty()


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

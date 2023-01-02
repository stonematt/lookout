# %%
# import libs
# from aioambient.api import API
from ambient_api.ambientapi import AmbientAPI
import time
from dateutil import parser
import pandas as pd
import plotly.express as px
import streamlit as st

# import logging
import storj_df_s3 as sj

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
bucket = "lookout"

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
# def get_all_history_for_device(device, end_date=pd.Timestamp.utcnow(), start_date=None):
# todo: chage to date range
def get_all_history_for_device(device, days_to_get=5):
    all_history = []
    last_min_date = ""
    progress_count = st.empty()
    progress_bar = st.progress(0)
    progress = 0

    last_history_retreived = get_device_history_to_date(device)
    all_history.extend(last_history_retreived)

    # last_min_date = find_earliest_date_in_history(last_history_retreived, "dateutc")
    # find date of last record
    last_min_date = last_history_retreived[-1]["dateutc"]

    while progress < days_to_get:
        progress += 1
        progress_count.text(f"Getting page {progress}")
        progress_bar.progress(progress / days_to_get)
        time.sleep(1)
        next_history_page = get_device_history_to_date(device, end_date=last_min_date)
        next_history_min_date = next_history_page[-1]["dateutc"]

        # to do make sure we don't add the last page twice
        if next_history_min_date != last_min_date:
            last_min_date = next_history_min_date
            for data_point in next_history_page:
                if data_point not in all_history:
                    all_history.append(data_point)

    all_history_df = pd.json_normalize(all_history)
    all_history_df.set_index("dateutc", inplace=True)

    progress_count.empty()
    progress_bar.empty()
    return all_history_df, all_history


# %%
st.cache(show_spinner=True)


def find_earliest_date_in_history(device_history, date_key):
    # Initialize smallest_value with the first dictionary's key value
    smallest_value = device_history[0][date_key]

    # scan history, if a date value is smaller, update smallest_value
    for data_point in device_history:
        if date_key in data_point and data_point[date_key] < smallest_value:
            smallest_value = data_point[date_key]

    # Return the smallest value found
    smallest_value = to_date(smallest_value)
    return smallest_value


# %%
devices = api.get_devices()

if len(devices) == 1:
    device = devices[0]
    device_menu = device.mac_address
    st.write(f"One device found:  {device.info['name']}")
    print(f"One device found:  {device.info['name']}")
# else:
#     device_menu = st.sidebar.selectbox(
#         "Select a device:", [device["macAddress"] for device in devices]
#     )
device_mac = device_menu
# device_mac = "98:CD:AC:22:0D:E5"
st.write(device_mac)
hist_file = bucket + "/" + device_mac + ".json"
# lookout/98:CD:AC:22:0D:E5.json

# pause for ambient.
time.sleep(1)

# %%
device_history = sj.get_file_as_dict(hist_file)
st.write(len(device_history))
st.write(device_history[1])

history_df, history_json = get_all_history_for_device(device, days_to_get=4)
history_df[:1]

# todo: add history_json to device_history
sj.save_dict_to_fs(history_json, hist_file)


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

# st.dataframe(history_df)

# keys = [ #     "dateutc", #     "tempinf", #     "humidityin", #     "baromrelin", #     "baromabsin",
#     "tempf", #     "humidity", #     "winddir", #     "winddir_avg10m", #     "windspeedmph", #     "windspdmph_avg10m", #     "windgustmph", #     "maxdailygust",
#     "hourlyrainin", #     "eventrainin", #     "dailyrainin", #     "weeklyrainin", #     "monthlyrainin", #     "yearlyrainin", #     "solarradiation",
#     "uv", #     "temp1f", #     "humidity1", #     "batt1", #     "batt_25", #     "batt_co2",
#     "feelsLike", #     "dewPoint", #     "feelsLike1", #     "dewPoint1", #     "feelsLikein", #     "dewPointin",
#     "lastRain", #     "date", # ]


# %%
# pd.merge(h1_df,h2_df, on=h1_df.keys().to_list(), how='outer')

# %%

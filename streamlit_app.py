# %%
# import libs
from aioambient.api import API
import asyncio
from dateutil import parser
import pandas as pd
import plotly.express as px
import streamlit as st
import logging
import storj_df_s3 as sj
# import numpy as np
# import plotly as pt
# import plotly.graph_objects as go
# from datetime import datetime
# from datetime import date, datetime,

AMBIENT_ENDPOINT = st.secrets["AMBIENT_ENDPOINT"]
AMBIENT_API_KEY = st.secrets["AMBIENT_API_KEY"]
AMBIENT_APPLICATION_KEY = st.secrets["AMBIENT_APPLICATION_KEY"]

api = API(AMBIENT_APPLICATION_KEY, AMBIENT_API_KEY)

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


# from aiocache import Cache
# from aiocache import cached

# @cached(ttl=None, cache=Cache.MEMORY)
async def get_devices():
    try:
        devices = await api.get_devices()
    except TimeoutError:
        logging.info("TimeoutError: get_devices")
    else:
        return devices


async def get_device_history_to_date(macAddress, end_date=None, limit=288):
    history = []
    if end_date:
        # st.write(f"end date: {end_date} for history page")
        try:
            history = await api.get_device_details(
                macAddress, end_date=end_date, limit=limit
            )
        except TimeoutError:
            logging.info("TimeoutError: get_device_history_to_date")
    else:
        # st.write("no end date submitted")
        try:
            history = await api.get_device_details(macAddress, limit=limit)
        except TimeoutError:
            logging.info("TimeoutError: get_device_history_to_date")

    return history if history else None


async def get_all_history_for_device(macAddress, days_to_get=15):
    all_history = []
    last_min_date = ""
    progress_count = st.empty()
    progress_bar = st.progress(0)
    progress = 0

    last_history_retreived = await get_device_history_to_date(macAddress)
    all_history.extend(last_history_retreived)

    last_min_date = await find_earliest_date_in_history(last_history_retreived, "date")
    # last_min_date = to_date(last_min_date)

    while progress < days_to_get:
        progress += 1
        progress_count.text(f"Getting page {progress}")
        progress_bar.progress(progress / days_to_get)
        try:
            next_history_page = await get_device_history_to_date(
                macAddress, end_date=last_min_date
            )
            next_history_min_date = await find_earliest_date_in_history(
                next_history_page, "date"
            )
        except TimeoutError:
            logging.info("TimeoutError: get_device_history_to_date")
        else:
            # to do make sure we don't add the last page twice
            if next_history_min_date != last_min_date:
                last_min_date = next_history_min_date
                all_history.extend(next_history_page)

    progress_count.empty()
    progress_bar.empty()
    return all_history


st.cache(show_spinner=True)


async def find_earliest_date_in_history(device_history, date_key):
    # Initialize smallest_value with the first dictionary's key value
    smallest_value = device_history[0][date_key]

    # scan history, if a date value is smaller, update smallest_value
    for data_point in device_history:
        if date_key in data_point and data_point[date_key] < smallest_value:
            smallest_value = data_point[date_key]

    # Return the smallest value found
    smallest_value = to_date(smallest_value)
    return smallest_value


async def main():
    devices = []
    # devices = await get_devices()
    devices = await api.get_devices()

    if len(devices) == 1:
        device = devices[0]
        device_menu = device["macAddress"]
        st.write(f"One device found:  {device['info']['name']}")
    else:
        device_menu = st.sidebar.selectbox(
            "Select a device:", [device["macAddress"] for device in devices]
        )
    device_mac = device_menu

    history = await get_all_history_for_device(device_mac, days_to_get=4)
    # st.write(history)
    # st.write(history[0].keys())
    history_df = pd.json_normalize(history)
    fig = px.line(history_df, x="date", y=["tempinf", "tempf", "temp1f"], title="temp")
    st.plotly_chart(fig)

    # candlestick_tmp = history_df.groupby(history_df["date"].dt.date).agg(
    #     {"temp": {"open": "first", "high": "max", "low": "min", "close": "last"}}
    # )
    # st.write(candlestick_tmp)

    # st.dataframe(history_df)

    # keys = [
    #     "dateutc",
    #     "tempinf",
    #     "humidityin",
    #     "baromrelin",
    #     "baromabsin",
    #     "tempf",
    #     "humidity",
    #     "winddir",
    #     "winddir_avg10m",
    #     "windspeedmph",
    #     "windspdmph_avg10m",
    #     "windgustmph",
    #     "maxdailygust",
    #     "hourlyrainin",
    #     "eventrainin",
    #     "dailyrainin",
    #     "weeklyrainin",
    #     "monthlyrainin",
    #     "yearlyrainin",
    #     "solarradiation",
    #     "uv",
    #     "temp1f",
    #     "humidity1",
    #     "batt1",
    #     "batt_25",
    #     "batt_co2",
    #     "feelsLike",
    #     "dewPoint",
    #     "feelsLike1",
    #     "dewPoint1",
    #     "feelsLikein",
    #     "dewPointin",
    #     "lastRain",
    #     "date",
    # ]


asyncio.run(main())

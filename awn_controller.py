# %%
from ambient_api.ambientapi import AmbientAPI
import time
import pandas as pd
import streamlit as st
import logging
import storj_df_s3 as sj
from datetime import datetime


# %%
AMBIENT_ENDPOINT = st.secrets["AMBIENT_ENDPOINT"]
AMBIENT_API_KEY = st.secrets["AMBIENT_API_KEY"]
AMBIENT_APPLICATION_KEY = st.secrets["AMBIENT_APPLICATION_KEY"]

api = AmbientAPI(
    # log_level="INFO",
    AMBIENT_ENDPOINT=AMBIENT_ENDPOINT,
    AMBIENT_API_KEY=AMBIENT_API_KEY,
    AMBIENT_APPLICATION_KEY=AMBIENT_APPLICATION_KEY,
)

sec_in_hour = 3600 * 1000


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
        logging.warning(
            f"range in history form storj: min: {min_date}, max: {max_date}"
        )
    else:
        logging.warning(f"no device history at {hist_file}")
        min_date = datetime.now().timestamp() * 1000
        device_history_min_date = time.time() * 1000
        device_history_max_date = time.time() * 1000

    return device_history, device_history_min_date, device_history_max_date


def make_history_df(history_json, tz):
    history_df = pd.json_normalize(history_json)
    history_df.sort_values(by="dateutc", inplace=True)

    # Convert 'date' column to local time
    history_df["date"] = pd.to_datetime(history_df["date"])
    history_df["date"] = history_df["date"].dt.tz_convert(tz)

    return history_df


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
    # history_json = get_all_history_for_device(
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

    progress_message.empty()

    return history_df, history_json


# %%
def main():
    return True


if __name__ == "__main__":
    main()
# %%

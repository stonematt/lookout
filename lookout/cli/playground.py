# %%
import logging
import time
from datetime import datetime

import pandas as pd
import socketio
import streamlit as st
from ambient_api.ambientapi import AmbientAPI

import lookout.api.awn_controller as awn
import lookout.storage.storj as sj

# %%
# open Ambient Weather Network
AMBIENT_ENDPOINT = st.secrets["AMBIENT_ENDPOINT"]
AMBIENT_API_KEY = st.secrets["AMBIENT_API_KEY"]
AMBIENT_APPLICATION_KEY = st.secrets["AMBIENT_APPLICATION_KEY"]

# api = AmbientAPI()
api = AmbientAPI(
    # log_level="WARNING",
    AMBIENT_ENDPOINT=AMBIENT_ENDPOINT,
    AMBIENT_API_KEY=AMBIENT_API_KEY,
    AMBIENT_APPLICATION_KEY=AMBIENT_APPLICATION_KEY,
)
# %%
devices = api.get_devices()

device = devices[0]

bucket = "lookout"
device_mac = device.mac_address
file_type = "parquet"
hist_file = f"{bucket}/{device.mac_address}.{file_type}"

# %%
time.sleep(1)  # pause for a second to avoid API limits

# data = device.get_data()
tz = device.last_data.get("tz")


# %% combine DFs
def combine_df(df1, df2):
    df1 = (
        pd.concat([df1, df2], ignore_index=True)
        .drop_duplicates()
        .sort_values(by="dateutc", ascending=False)
        .reset_index(drop=True)
    )

    return df1


# %%
def load_archive_for_device(device):
    key = f"{device.mac_address}.{file_type}"
    device_archive = sj.get_df_from_s3(bucket, key, file_type=file_type)
    return device_archive


# %%
def save_archive_for_device(df, device):
    key = f"{device.mac_address}.{file_type}"
    sj.save_df_to_s3(df, bucket, key, file_type=file_type)


# %%
# better check on these.
col_types = {
    "tempinf": "float32",
    "tempf": "float32",
    "temp1f": "float32",
    "humidityin": "float32",
    "humidity": "float32",
    "humidity1": "float32",
}


def set_df_data_types(df, types):
    df_copy = df.copy()
    for col in df.columns:
        if col in types:
            df_copy[col] = df_copy[col].astype(types[col])
    return df_copy


# %%
def heatmap(df, metric, aggfunc="max", interval="day"):
    """
    Create a pivot table of aggregate values for a given metric, with the row index
    as a time stamp for every `interval`-second interval and the column index as the unique dates in the "date" column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing a "date" column and a column with the desired `metric`.
    metric : str
        The name of the column in `df` containing the desired metric.
    aggfunc : str or function
        The aggregation function to use when computing the pivot table. Can be a string
        representing a built-in function (e.g., "mean", "sum", "count"), or a custom function.
    interval : int
        The number of seconds for each interval. For example, `interval=15` would create an
        interval of 15 seconds.

    Returns
    -------
    pandas.DataFrame
        A pivot table where the row index is a time stamp for every `interval`-second interval,
        and the column index is the unique dates in the "date" column. The values are the
        aggregate value of the `metric` column for each interval and each date.
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


# %%
def timestamp_generator(start, end, interval=12):
    # Parse start and end date strings into datetime objects
    start_date = datetime.strptime(start, "%m/%d/%Y")
    end_date = datetime.strptime(end, "%m/%d/%Y")

    # Convert start and end dates to Unix timestamps
    start_timestamp = int(time.mktime(start_date.timetuple()))
    end_timestamp = int(time.mktime(end_date.timetuple()))

    # Yield timestamps at 12 hour intervals
    while start_timestamp <= end_timestamp:
        yield start_timestamp
        start_timestamp += interval * 60 * 60


# %%
def merge_distinct(old_list, new_data_list):
    logging.warning(f"Merging {len(old_list)} and {len(new_data_list)}")
    for data_point in new_data_list:
        if data_point not in old_list:
            old_list.append(data_point)

    return old_list


# %%
def get_all_data(device):
    requestlog = {}

    dev_history = device.get_data()
    # last_data_point = dev_history[-1]["dateutc"]
    last_data_point = 1664585820000
    page_counter = 0
    requestlog[page_counter] = {
        "last_date": last_data_point,
        "page_records": len(dev_history),
        "total_records": len(dev_history),
    }
    run = True
    while run:
        page_counter += 1
        time.sleep(1.1)
        next_page = device.get_data(end_date=last_data_point)
        next_page_last_date = next_page[-1]["dateutc"]

        requestlog[page_counter] = {
            "last_date": last_data_point,
            "next_page_last_date": next_page_last_date,
            "page_records": len(next_page),
        }

        if next_page_last_date != last_data_point:
            last_data_point = next_page_last_date
            dev_history += next_page
            requestlog[page_counter]["total_records"] = len(dev_history)
            print(requestlog)
            print(f"counter: {page_counter}, len: {len(dev_history)}")
            print(f"lastutc: {last_data_point}, nextpageutc: {next_page_last_date}")
        else:
            run = False

        if page_counter > 5:
            run = False
    # todo: dedupe
    print("")
    logging.warning(requestlog)
    return dev_history


# %%
bucket = "lookout"
file_type = "parquet"
hist_file = f"{bucket}/{device.mac_address}.{file_type}"


# %%
def init(device):
    df = load_archive_for_device(device)

    return df


# %%


# %%
time.sleep(1)
# history_json = get_all_data(device)

# %%
# sj.save_dict_to_fs(history_json, hist_file)
# %%

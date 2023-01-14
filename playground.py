# %%
from ambient_api.ambientapi import AmbientAPI
import time
from datetime import datetime
import streamlit as st
import storj_df_s3 as sj
import logging

# %%
# open Ambient Weather Network
AMBIENT_ENDPOINT = st.secrets["AMBIENT_ENDPOINT"]
AMBIENT_API_KEY = st.secrets["AMBIENT_API_KEY"]
AMBIENT_APPLICATION_KEY = st.secrets["AMBIENT_APPLICATION_KEY"]

# api = AmbientAPI()
api = AmbientAPI(
    log_level="INFO",
    AMBIENT_ENDPOINT=AMBIENT_ENDPOINT,
    AMBIENT_API_KEY=AMBIENT_API_KEY,
    AMBIENT_APPLICATION_KEY=AMBIENT_APPLICATION_KEY,
)
# %%

devices = api.get_devices()

device = devices[0]

# %%
time.sleep(1)  # pause for a second to avoid API limits

data = device.get_data()
tz = device.last_data.get("tz")


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
hist_file = bucket + "/" + device.mac_address + "-2.json"


# %%
time.sleep(1)
history_json = get_all_data(device)

# %%
sj.save_dict_to_fs(history_json, hist_file)
# %%

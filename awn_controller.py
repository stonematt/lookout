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


def get_archive(archive_file):
    """get archive from local fs
    return df"""
    archive_df = pd.read_parquet(archive_file)

    return archive_df


# todo: keep this one
# %% Load archive from s3
def load_archive_for_device(device, bucket, file_type="parquet"):
    """return df of device archive"""
    key = f"{device.mac_address}.{file_type}"
    device_archive = sj.get_df_from_s3(bucket, key, file_type=file_type)
    return device_archive


def _df_column_to_datetime(df, column, tz):
    df[column] = pd.to_datetime(df[column]).dt.tz_convert(tz)


# todo: keep this one
# %%
def get_device_history_to_date(device, end_date=None, limit=288):
    """returns a tz converted df for a device a date"""
    df = pd.DataFrame()
    print("making df")
    try:
        if end_date:
            # st.write(f"end date: {end_date} for history page")
            df = pd.json_normalize(device.get_data(end_date=end_date, limit=limit))
        else:
            # st.write("no end date submitted")
            df = pd.json_normalize(device.get_data(limit=limit))
    except e:
        raise e
    else:
        df.sort_values(by="dateutc", inplace=True)

        # Convert 'date' column to local time
        dt_columns = ["date", "lastRain"]
        for c in dt_columns:
            _df_column_to_datetime(df, c, device.last_data.get("tz"))

        return df


# todo: delete this
# %%
# todo: keep this one
def get_interim_data_for_device(device, archive_df, limit=288, sleep=False):
    """Returns a dataframe with the full history of a device, combining archived data with interim data"""

    update_message = st.empty()
    update_message.text("Updating and refreshing...")
    progress_message = st.empty()
    # Find the most recent date in the archive
    max_archive_date = archive_df["dateutc"].max()
    page = 0
    #     # get data since the last time we ran the script.
    #     missing_hours = (time.time() * 1000 - history_max_date) / sec_in_hour
    #     progress_message.text(
    #         f"Missing {missing_hours} hours of recent history. Getting it now"
    #     )

    # Call get_device_history_to_date() until we have all data up to the archive's max date
    interim_df = pd.DataFrame()
    while True:
        progress_message.text(f"getting page {page}")
        #         progress += 1
        #         progress_count.text(f"Getting page {progress}")
        #         logging.warning(f"Getting page {progress}")
        #         progress_bar.progress(progress / days_to_get)
        # throttle subsequent api calls
        if sleep:
            print(f"sleep: {sleep}, page: {page}")
            time.sleep(1)
        # Get the next page of history
        end_date = (
            interim_df["dateutc"].min()
            if not interim_df.empty
            else int(datetime.now().timestamp() * 1000)
        )
        new_data = get_device_history_to_date(device, end_date=end_date, limit=limit)

        # Stop if we've reached the end of the data
        if new_data.empty or new_data["dateutc"].min() < max_archive_date:
            break

        # Append the new data to the interim dataframe
        interim_df = combine_df(interim_df, new_data)
        page += 1

    # Combine the archive and interim dataframes
    full_history_df = combine_df(archive_df, interim_df)

    update_message.empty()
    progress_message.empty()

    return full_history_df


# todo: keep this one
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
def main():
    api = AmbientAPI(
        log_level="INFO",
        AMBIENT_ENDPOINT=AMBIENT_ENDPOINT,
        AMBIENT_API_KEY=AMBIENT_API_KEY,
        AMBIENT_APPLICATION_KEY=AMBIENT_APPLICATION_KEY,
    )
    # %%

    devices = api.get_devices()

    device = devices[0]

    df = load_archive_for_device(device, "lookout", "parquet")
    print(df.date.min())
    print(df.date.max())

    df = get_interim_data_for_device(device, df)
    print(df.date.min())
    print(df.date.max())

    return True


if __name__ == "__main__":
    main()
# %%

# %%
import pandas as pd
import streamlit as st
import s3fs
import json
import logging

logging.basicConfig(level=logging.INFO)


ACCESS_KEY_ID = st.secrets["lookout_storage_options"]["ACCESS_KEY_ID"]
SECRET_ACCESS_KEY = st.secrets["lookout_storage_options"]["SECRET_ACCESS_KEY"]
ENDPOINT_URL = st.secrets["lookout_storage_options"]["ENDPOINT_URL"]


storage_options = {
    "key": ACCESS_KEY_ID,
    "secret": SECRET_ACCESS_KEY,
    "client_kwargs": {"endpoint_url": ENDPOINT_URL},
}

fs = s3fs.S3FileSystem(
    client_kwargs={
        "endpoint_url": ENDPOINT_URL,
        "aws_access_key_id": ACCESS_KEY_ID,
        "aws_secret_access_key": SECRET_ACCESS_KEY,
    }
)

f = "s3://lookout/98:CD:AC:22:0D:E5.json"


def get_file_as_dict(file_path, fs):
    try:
        with fs.open(file_path, "rb") as f:
            file_contents = f.read()
        return json.loads(file_contents)
    except FileNotFoundError:
        return {}


# %%
def get_dataframe_from_s3_json(bucket, key):
    """
    Returns a pandas DataFrame from a JSON file stored in an S3 bucket.

    Parameters:
        bucket (str): The name of the S3 bucket.
        key (str): The key of the JSON file in the S3 bucket.

    Returns:
        pandas.DataFrame: The DataFrame created from the JSON file.
    """
    f = bucket + "/" + key

    try:
        logging.info(f"Getting {f}")
        # df = pd.read_json(f"s3://{bucket}/{key}", storage_options=storage_options)
        df = pd.read_json(f"s3://{f}", storage_options=storage_options)
    except FileNotFoundError:
        logging.info(f"File not found: {f}")
    else:
        return df


# %%
def save_df_to_s3_json(df, bucket, key):
    """Save a datafram to an storj.io bucket with 3fs

    Args:
        df (dataframe): pandas data frame
        bucket (str): bucket name
        key (str): path and filename to save
    """
    f = bucket + "/" + key

    logging.info(f"Saving {f}")
    df.to_json(f"s3://{f}", storage_options=storage_options)
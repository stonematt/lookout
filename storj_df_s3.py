import pandas as pd
import streamlit as st
import s3fs
import json
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)

# Retrieving secrets for S3 access
ACCESS_KEY_ID = st.secrets["lookout_storage_options"]["ACCESS_KEY_ID"]
SECRET_ACCESS_KEY = st.secrets["lookout_storage_options"]["SECRET_ACCESS_KEY"]
ENDPOINT_URL = st.secrets["lookout_storage_options"]["ENDPOINT_URL"]

# Storage options for pandas direct S3 access
storage_options = {
    "key": ACCESS_KEY_ID,
    "secret": SECRET_ACCESS_KEY,
    "client_kwargs": {"endpoint_url": ENDPOINT_URL},
}

# Initializing S3 file system for s3fs
fs = s3fs.S3FileSystem(
    client_kwargs={
        "endpoint_url": ENDPOINT_URL,
        "aws_access_key_id": ACCESS_KEY_ID,
        "aws_secret_access_key": SECRET_ACCESS_KEY,
    }
)


def read_json_from_path(file_path: str, local: bool = False) -> dict:
    """
    Reads a JSON file from a given path and returns its content as a dictionary.

    :param file_path: File path to read from.
    :param local: Flag to indicate if the file is local or in S3.
    :return: Dictionary containing the JSON file content.
    """
    if local:
        read_function = open
    else:
        read_function = fs.open
        file_path = "s3://" + file_path

    try:
        with read_function(file_path, "rb") as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return {}


def save_json_to_path(dict_data: dict, file_path: str, local: bool = False) -> bool:
    """
    Saves a dictionary as a JSON file to a specified path.

    :param dict_data: Dictionary to save.
    :param file_path: File path to save the dictionary to.
    :param local: Flag to indicate if the file should be saved locally or in S3.
    :return: True if the save operation was successful, False otherwise.
    """
    if local:
        write_function = open
    else:
        write_function = fs.open
        file_path = "s3://" + file_path

    try:
        with write_function(file_path, "w") as file:
            json.dump(dict_data, file)
            return True
    except PermissionError:
        logging.error(f"Permission denied: Unable to save dictionary to {file_path}")
        return False


def get_df_from_s3(bucket: str, key: str, file_type: str = "json") -> pd.DataFrame:
    """
    Retrieves a pandas DataFrame from a JSON or Parquet file stored in an S3 bucket.

    :param bucket: The name of the S3 bucket.
    :param key: The key of the file in the S3 bucket.
    :param file_type: The type of the file ('json' or 'parquet').
    :return: DataFrame obtained from the file.
    """
    file_path = f"s3://{bucket}/{key}"

    try:
        logging.info(f"Getting {file_path}")
        if file_type == "parquet":
            return pd.read_parquet(file_path, storage_options=storage_options)
        else:
            return pd.read_json(file_path, storage_options=storage_options)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return pd.DataFrame()


def save_df_to_s3(
    df: pd.DataFrame, bucket: str, key: str, file_type: str = "parquet"
) -> None:
    """
    Saves a pandas DataFrame to an S3 bucket in JSON or Parquet format.

    :param df: DataFrame to save.
    :param bucket: Bucket name to save the DataFrame to.
    :param key: Path and filename for the saved DataFrame.
    :param file_type: The type of file to save ('json' or 'parquet').
    """
    file_path = f"s3://{bucket}/{key}"

    logging.info(f"Saving {file_path}")
    if file_type == "parquet":
        df.to_parquet(file_path, storage_options=storage_options)
    else:
        df.to_json(file_path, storage_options=storage_options)


def list_bucket_items(bucket_name: str) -> list:
    """
    List all items in a specified S3 bucket.

    :param bucket_name: The name of the bucket to list items from.
    :return: A list of file paths in the specified bucket.
    """
    try:
        # Constructing the bucket path
        bucket_path = f"{bucket_name}/"

        # Listing items in the bucket
        items = fs.ls(bucket_path)
        logging.info(f"Items in bucket '{bucket_name}': {items}")

        return items
    except Exception as e:
        logging.error(f"Error listing items in bucket '{bucket_name}': {e}")
        return []


# Example usage
def main() -> None:
    """
    Main function to execute module logic.
    """
    # Implement module logic here
    bucket_items = list_bucket_items("lookout")
    print(bucket_items)

    pass


if __name__ == "__main__":
    main()

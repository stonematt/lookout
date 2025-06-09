"""
storj_df_s3.py: This module provides functionalities to interact with S3 storage,
specifically for handling dataframes (pandas) and JSON objects. It includes utilities
for reading and writing data to/from S3 buckets using s3fs and pandas. It also features
logger setup for monitoring file operations and a backup mechanism for S3 data. The
module integrates with Streamlit for secret management, enhancing the security aspect
of S3 access. Designed for applications where S3 storage interaction is crucial,
this module streamlines tasks like data backup, retrieval, and storage
of structured data.

Functions:
- Read and write JSON files to/from S3.
- Convert pandas DataFrames to/from JSON or Parquet format for S3 storage.
- List items in an S3 bucket.
- Backup data from S3 to a designated backup directory.
"""

import datetime
import json
import os

import pandas as pd
import s3fs
import streamlit as st

from log_util import app_logger

# For general logging with console output
logger = app_logger(__name__)

# For logging with both console and file output
backup_logger = app_logger("backup_logger", log_file="backup.log")


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
        logger.error(f"File not found: {file_path}")
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
        logger.error(f"Permission denied: Unable to save dictionary to {file_path}")
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
        logger.info(f"Getting {file_path}")
        if file_type == "parquet":
            return pd.read_parquet(file_path, storage_options=storage_options)
        else:
            return pd.read_json(file_path, storage_options=storage_options)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
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

    logger.info(f"Saving {file_path}")
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
        logger.info(f"Items in bucket '{bucket_name}': {items}")

        return items
    except Exception as e:
        logger.error(f"Error listing items in bucket '{bucket_name}': {e}")
        return []


def backup_data(
    bucket: str,
    prefix: str,
    force_backup: bool = False,
    dry_run: bool = False,
    backup_root: str = ".backups",
):
    """
    Backs up all files with a given prefix in an S3 bucket to a specified backup folder,
    if the backup folder does not exist or if force_backup is True. If dry_run is True,
    performs all steps except the actual file copy.

    :param bucket: Name of the S3 bucket.
    :param prefix: Prefix of the files to back up.
    :param force_backup: Boolean flag to force backup even if the folder exists.
    :param dry_run: Boolean flag for a dry run without actual file copying.
    :param backup_root: Root directory for backups, default to '.backups'.
    """
    today = datetime.datetime.now().strftime("%Y%m%d")
    # backup_folder = f"{backup_root}/{today}_{prefix}/"
    backup_folder = f"{backup_root}/{prefix}/{today}/"
    backup_logger.info(f"Starting Backup of {backup_folder}")

    # Check if backup folder already exists
    backup_folder_exists = fs.exists(f"{bucket}/{backup_folder}")
    if backup_folder_exists and not force_backup:
        backup_logger.info(
            f"Backup folder {backup_folder} already exists. Backup skipped."
        )
        return

    # List all files with the device ID using a glob pattern
    glob_pattern = f"{bucket}/{prefix}*"
    files_to_backup = fs.glob(glob_pattern)

    # Create the backup folder and copy files (or simulate if dry run)
    for file_path in files_to_backup:
        file_name = os.path.basename(file_path)
        backup_path = f"{bucket}/{backup_folder}{file_name}"
        # Get file size and modified date for logger
        file_info = fs.info(file_path)
        file_size = file_info.get("Size") or file_info.get("size", "Unknown size")

        modified_date = file_info.get("LastModified", "Unknown date")
        backup_logger.debug(file_info)

        if not dry_run:
            try:
                fs.copy(file_path, backup_path)

                # Log file backup details on successful copy
                backup_logger.info(
                    f"SUCCESS: {file_name} ({file_size} bytes, "
                    f"Last Modified: {modified_date})"
                )
            except Exception as e:
                backup_logger.error(f"Failed to back up {file_name}: {e}")
                continue
        else:
            # Log dry run details
            backup_logger.info(
                f"DRY RUN: {file_name} ({file_size} bytes, "
                f"Last Modified: {modified_date})"
            )

    if dry_run:
        backup_logger.info(f"Dry run completed for folder {backup_folder}")
    else:
        backup_logger.info(f"Backup completed to folder {backup_folder}")


# Example usage
def main() -> None:
    """
    Main function to execute various tests..
    """
    # Implement module logic here
    # Bucket name
    bucket_name = "lookout"
    # Device ID (used as the prefix)
    device_id = "98:CD:AC:22:0D:E5"

    bucket_items = list_bucket_items(bucket_name)
    # print(bucket_items)

    # Perform a dry run of the backup
    backup_data(bucket=bucket_name, prefix=device_id)

    print(bucket_items)
    pass


if __name__ == "__main__":
    main()

"""
storj.py: This module provides functionalities to interact with S3 storage,
specifically for handling pandas DataFrames and JSON objects. It includes utilities
for reading and writing data to/from S3 buckets using boto3. It also features
logger setup for monitoring file operations and a backup mechanism for S3 data.
The module integrates with Streamlit for secret management, enhancing the security
aspect of S3 access. Designed for applications where S3 storage interaction is crucial,
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
from io import BytesIO

import boto3
import pandas as pd
import streamlit as st

from lookout.utils.log_util import app_logger

# For general logging with console output
logger = app_logger(__name__)

# For logging with both console and file output
backup_logger = app_logger("backup_logger", log_file="backup.log")


# Retrieving secrets for S3 access
ACCESS_KEY_ID = st.secrets["lookout_storage_options"]["ACCESS_KEY_ID"]
SECRET_ACCESS_KEY = st.secrets["lookout_storage_options"]["SECRET_ACCESS_KEY"]
ENDPOINT_URL = st.secrets["lookout_storage_options"]["ENDPOINT_URL"]


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY,
        region_name="us-east-1",
    )


def read_json_from_path(file_path: str, local: bool = False) -> dict:
    """
    Reads a JSON file from a given path and returns its content as a dictionary.

    :param file_path: Full S3 key or local path.
    :param local: Flag to indicate if the file is local or from S3.
    :return: Dictionary containing the JSON file content.
    """
    if local:
        try:
            with open(file_path, "rb") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Local file not found: {file_path}")
            return {}
    else:
        try:
            # Extract bucket and key
            if file_path.startswith("s3://"):
                _, bucket, *key_parts = file_path.split("/")
                key = "/".join(key_parts)
            else:
                logger.error(f"Invalid S3 path: {file_path}")
                return {}

            s3 = get_s3_client()

            obj = s3.get_object(Bucket=bucket, Key=key)
            return json.load(obj["Body"])
        except Exception as e:
            logger.error(f"Failed to read JSON from {file_path}: {e}")
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
        try:
            with open(file_path, "w") as f:
                json.dump(dict_data, f)
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON locally: {e}")
            return False
    else:
        try:
            # Extract bucket and key
            if file_path.startswith("s3://"):
                _, bucket, *key_parts = file_path.split("/")
                key = "/".join(key_parts)
            else:
                logger.error(f"Invalid S3 path: {file_path}")
                return False

            s3 = get_s3_client()
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(dict_data),
                ContentType="application/json",
            )
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save JSON to {file_path}: {e}")
            return False


def get_df_from_s3(bucket: str, key: str, file_type: str = "json") -> pd.DataFrame:
    """
    Retrieves a pandas DataFrame from a JSON or Parquet file stored in an S3 bucket.

    :param bucket: The name of the S3 bucket.
    :param key: The key of the file in the S3 bucket.
    :param file_type: The type of the file ('json' or 'parquet').
    :return: DataFrame obtained from the file.
    """
    try:
        logger.info(f"Getting s3://{bucket}/{key}")
        s3 = get_s3_client()
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()

        if file_type == "parquet":
            return pd.read_parquet(BytesIO(body), engine="pyarrow")
        elif file_type == "json":
            return pd.read_json(BytesIO(body), orient="records", lines=True)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        logger.error(f"❌ Failed to load DataFrame from s3://{bucket}/{key}: {e}")
        return pd.DataFrame()


def save_df_to_s3(
    df: pd.DataFrame, bucket: str, key: str, file_type: str = "parquet"
) -> None:
    """
    Saves a pandas DataFrame to an S3-compatible bucket using boto3 and BytesIO,
    ensuring compatibility with Storj.io and AWS S3 behavior.

    :param df: The DataFrame to save.
    :param bucket: The name of the S3 bucket.
    :param key: The key (path/filename) to write the file under.
    :param file_type: The file type to save ('parquet' or 'json').
    :return: None
    """
    file_path = f"s3://{bucket}/{key}"
    logger.info(f"Saving {file_path}")

    try:
        buffer = BytesIO()

        if file_type == "parquet":
            df.to_parquet(buffer, engine="pyarrow", compression="snappy")
            content_type = "application/octet-stream"
        elif file_type == "json":
            df.to_json(buffer, orient="records", lines=True)
            content_type = "application/json"
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        buffer.seek(0)

        s3 = get_s3_client()

        s3.upload_fileobj(
            Fileobj=buffer,
            Bucket=bucket,
            Key=key,
            ExtraArgs={"ContentType": content_type},
        )

    except Exception as e:
        logger.error(f"❌ Failed to save {key} to bucket {bucket}: {e}")


def list_bucket_items(bucket_name: str, depth: int = 2) -> list:
    """
    List items in a specified S3 bucket up to a specified depth.

    :param bucket_name: The name of the bucket to list items from.
    :param depth: Maximum depth of path segments to include (default: 2).
    :return: A list of dictionaries with keys 'path' and 'size'.
    """
    s3 = get_s3_client()
    try:
        paginator = s3.get_paginator("list_objects_v2")
        result = []

        for page in paginator.paginate(Bucket=bucket_name):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.count("/") <= depth:
                    result.append({"path": f"{bucket_name}/{key}", "size": obj["Size"]})

        if result:
            formatted = "\n".join(
                f"    {entry['path']:<60} {entry['size']:>10,} bytes"
                for entry in result
            )
            logger.info(f"Bucket contents:\n{formatted}")

        return result

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
    Backs up all files with a given prefix in an S3 bucket to a specified backup folder
    using boto3. If dry_run is True, simulates the backup without copying files.

    :param bucket: Name of the S3 bucket.
    :param prefix: Prefix of the files to back up.
    :param force_backup: Boolean flag to force backup even if the folder exists.
    :param dry_run: Boolean flag for a dry run without actual file copying.
    :param backup_root: Root directory for backups, default to '.backups'.
    """
    today = datetime.datetime.now().strftime("%Y%m%d")
    backup_folder = f"{backup_root}/{prefix}/{today}/"
    backup_logger.info(f"Starting Backup of {backup_folder}")

    s3 = get_s3_client()

    # Check if backup folder exists
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=backup_folder, MaxKeys=1)
        if response.get("Contents") and not force_backup:
            backup_logger.info(
                f"Backup folder {backup_folder} already exists. Backup skipped."
            )
            return
    except Exception as e:
        backup_logger.error(f"Failed to check backup folder: {e}")
        return

    # List objects to back up
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                file_name = os.path.basename(key)
                backup_key = f"{backup_folder}{file_name}"
                file_size = obj["Size"]
                last_modified = obj["LastModified"]

                if dry_run:
                    backup_logger.info(
                        f"DRY RUN: {file_name} ({file_size} bytes, Last Modified: {last_modified})"
                    )
                    continue

                try:
                    s3.copy_object(
                        Bucket=bucket,
                        CopySource={"Bucket": bucket, "Key": key},
                        Key=backup_key,
                    )
                    backup_logger.info(
                        f"SUCCESS: {file_name} ({file_size} bytes, Last Modified: {last_modified})"
                    )
                except Exception as e:
                    backup_logger.error(f"Failed to back up {file_name}: {e}")

        if dry_run:
            backup_logger.info(f"Dry run completed for folder {backup_folder}")
        else:
            backup_logger.info(f"Backup completed to folder {backup_folder}")

    except Exception as e:
        backup_logger.error(f"Error listing or copying files for backup: {e}")


# Example usage
def main() -> None:
    """
    Main function to execute various tests.
    """
    bucket_name = "lookout"
    device_id = "98:CD:AC:22:0D:E5"

    # List bucket contents
    bucket_items = list_bucket_items(bucket_name, depth=3)
    print(f"Items in bucket: {len(bucket_items)}")

    # Perform dry run backup for the device
    backup_data(bucket=bucket_name, prefix=device_id)

    # Test saving a dummy DataFrame to S3
    test_df = pd.DataFrame(
        {
            "sensor": ["temp", "humid"],
            "value": [72.5, 38.2],
            "timestamp": [pd.Timestamp.now(), pd.Timestamp.now()],
        }
    )
    test_key = f"{device_id}/test_upload.parquet"
    save_df_to_s3(test_df, bucket_name, test_key)

    # Clean up test file
    try:
        s3 = get_s3_client()
        s3.delete_object(Bucket=bucket_name, Key=test_key)
        logger.info(f"Deleted test upload: s3://{bucket_name}/{test_key}")
    except Exception as e:
        logger.warning(f"Failed to delete test upload: {e}")


if __name__ == "__main__":
    main()

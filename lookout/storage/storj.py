"""
storj.py: This module provides functionalities to interact with S3 storage,
specifically for handling pandas DataFrames and JSON objects. It includes utilities
for reading and writing data to/from S3 buckets using boto3. It also features
logger setup for monitoring file operations and a backup mechanism for S3 data.
Storj credentials come from `lookout.secrets_loader.get_secret`, read each time a
client is built, never at import. Designed for applications where S3 storage
interaction is crucial, this module streamlines tasks like data backup, retrieval,
and storage of structured data.

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
from typing import Optional

import boto3
import pandas as pd

from lookout.config import HISTORY_FILE_TYPE
from lookout.secrets_loader import get_secret
from lookout.utils.log_util import app_logger

# For general logging with console output
logger = app_logger(__name__)

# For logging with both console and file output
backup_logger = app_logger("backup_logger", log_file="backup.log")


def get_s3_client():
    """
    Build an S3 client for the Storj gateway, reading credentials at call time.

    :return: A boto3 S3 client.
    :raises MissingSecretError: If a Storj secret is not configured.
    """
    return boto3.client(
        "s3",
        endpoint_url=get_secret("STORJ_ENDPOINT_URL"),
        aws_access_key_id=get_secret("STORJ_ACCESS_KEY_ID"),
        aws_secret_access_key=get_secret("STORJ_SECRET_ACCESS_KEY"),
        region_name="us-east-1",
    )


class StorageReadError(RuntimeError):
    """A stored object could not be read, or read back as nothing usable."""


class StorageWriteError(RuntimeError):
    """A save to storage failed; the stored object is as it was."""


class ArchiveShrinkError(StorageWriteError):
    """A save would leave the archive with fewer rows than were loaded."""


class BackupError(RuntimeError):
    """A backup copy failed; whatever it guarded must not be saved."""


def _is_missing(error: Exception) -> bool:
    """
    Whether an S3 error says the object is not there (as opposed to failing).

    :param error: The exception a client call raised.
    :return: True for a `NoSuchKey` / 404 response.
    """
    response = getattr(error, "response", None) or {}
    code = str(response.get("Error", {}).get("Code", ""))
    return code in ("NoSuchKey", "404", "NotFound")


def read_df_from_s3(bucket: str, key: str, file_type: str = "json") -> pd.DataFrame:
    """
    Read a DataFrame from a JSON-lines or Parquet object, failing loudly.

    For writers that must not mistake a failed read for an empty one
    (catchup, #160); `get_df_from_s3` is the forgiving wrapper readers use.

    :param bucket: The name of the S3 bucket.
    :param key: The key of the file in the S3 bucket.
    :param file_type: The type of the file ('json' or 'parquet').
    :return: The DataFrame the object holds (possibly empty).
    :raises StorageReadError: If the object is missing, cannot be fetched, or
        cannot be parsed.
    """
    logger.info(f"Getting s3://{bucket}/{key}")
    if file_type not in ("parquet", "json"):
        raise ValueError(f"Unsupported file type: {file_type}")
    try:
        body = get_s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        if file_type == "parquet":
            return pd.read_parquet(BytesIO(body), engine="pyarrow")
        return pd.read_json(BytesIO(body), orient="records", lines=True)
    except Exception as e:
        state = "missing" if _is_missing(e) else "unreadable"
        raise StorageReadError(f"s3://{bucket}/{key} is {state}: {e}") from e


def get_df_from_s3(bucket: str, key: str, file_type: str = "json") -> pd.DataFrame:
    """
    Retrieves a pandas DataFrame from a JSON or Parquet file stored in an S3 bucket.

    Forgiving: a missing or unreadable object reads as an empty frame (logged
    at ERROR). Writers use `read_df_from_s3` or `load_archive` instead.

    :param bucket: The name of the S3 bucket.
    :param key: The key of the file in the S3 bucket.
    :param file_type: The type of the file ('json' or 'parquet').
    :return: DataFrame obtained from the file; empty on any failure.
    """
    try:
        return read_df_from_s3(bucket, key, file_type)
    except Exception as e:
        logger.error(f"Failed to load DataFrame from s3://{bucket}/{key}: {e}")
        return pd.DataFrame()


def load_archive(bucket: str, mac: str) -> pd.DataFrame:
    """
    Read a station's archive for a run that will save over it (#160).

    :param bucket: The bucket.
    :param mac: The station's MAC address.
    :return: The archive as stored, with at least one row.
    :raises StorageReadError: If `<mac>.<HISTORY_FILE_TYPE>` is missing,
        unreadable or holds no rows: a run that saved after such a read
        would overwrite the archive with nothing.
    """
    key = f"{mac}.{HISTORY_FILE_TYPE}"
    archive = read_df_from_s3(bucket, key, file_type=HISTORY_FILE_TYPE)
    if archive.empty:
        raise StorageReadError(f"s3://{bucket}/{key} holds no rows")
    return archive


def read_json_from_s3(bucket: str, key: str) -> Optional[dict]:
    """
    Read a JSON object, telling a missing object apart from a failed read.

    :param bucket: The name of the S3 bucket.
    :param key: The key of the JSON file in the S3 bucket.
    :return: The parsed JSON; None when no object is stored at `key`.
    :raises StorageReadError: If the object exists but cannot be fetched or
        parsed.
    """
    logger.info(f"Getting JSON from s3://{bucket}/{key}")
    try:
        obj = get_s3_client().get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception as e:
        if _is_missing(e):
            return None
        raise StorageReadError(f"s3://{bucket}/{key} is unreadable: {e}") from e


def get_json_from_s3(bucket: str, key: str) -> dict:
    """
    Retrieves a JSON dict from an S3-compatible bucket using boto3.

    :param bucket: The name of the S3 bucket.
    :param key: The key of the JSON file in the S3 bucket.
    :return: Dictionary containing the JSON content, empty dict if failed.
    """
    try:
        return read_json_from_s3(bucket, key) or {}
    except Exception as e:
        logger.error(f"Failed to get JSON from s3://{bucket}/{key}: {e}")
        return {}


def save_json_to_s3(dict_data: dict, bucket: str, key: str) -> bool:
    """
    Saves a JSON dict to an S3-compatible bucket using boto3.

    :param dict_data: Dictionary to save as JSON.
    :param bucket: The name of the S3 bucket.
    :param key: The key (path/filename) to write the JSON file under.
    :return: True if successful, False if failed.
    """
    file_path = f"s3://{bucket}/{key}"
    logger.info(f"Saving JSON to {file_path}")

    try:
        s3 = get_s3_client()
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(dict_data),
            ContentType="application/json",
        )
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        return False


def object_exists(bucket: str, key: str) -> bool:
    """
    Check whether an object exists in an S3-compatible bucket.

    :param bucket: The name of the S3 bucket.
    :param key: The object's key.
    :return: True if `head_object` finds it, False on any error.
    """
    try:
        s3 = get_s3_client()
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def save_df_to_s3(
    df: pd.DataFrame, bucket: str, key: str, file_type: str = "parquet"
) -> bool:
    """
    Saves a pandas DataFrame to an S3-compatible bucket using boto3 and BytesIO,
    ensuring compatibility with Storj.io and AWS S3 behavior.

    :param df: The DataFrame to save.
    :param bucket: The name of the S3 bucket.
    :param key: The key (path/filename) to write the file under.
    :param file_type: The file type to save ('parquet' or 'json').
    :return: True when saved; False when the save failed (logged at ERROR).
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
        return True

    except Exception as e:
        logger.error(f"Failed to save {key} to bucket {bucket}: {e}")
        return False


def save_archive(
    df: pd.DataFrame, bucket: str, mac: str, loaded_rows: Optional[int] = None
) -> None:
    """
    Save a station's archive to `<mac>.<HISTORY_FILE_TYPE>`, newest first.

    The one save step every archive writer goes through (#133): the stored
    archive has a single defined order, newest `dateutc` first, whatever
    order the caller built it in. The sort is stable and the index is reset;
    the rows and columns are saved as handed.

    :param df: The archive, any row order.
    :param bucket: The bucket.
    :param mac: The station's MAC address.
    :param loaded_rows: The rows the caller loaded the archive with, for a
        save that may only add rows (catchup); None skips the check.
    :raises ArchiveShrinkError: If `df` has fewer than `loaded_rows` rows;
        nothing is written (#160).
    :raises StorageWriteError: If the save fails.
    """
    key = f"{mac}.{HISTORY_FILE_TYPE}"
    if loaded_rows is not None and len(df) < loaded_rows:
        raise ArchiveShrinkError(
            f"Refusing to save s3://{bucket}/{key}: {len(df)} rows, "
            f"fewer than the {loaded_rows} loaded"
        )
    if "dateutc" in df.columns:
        df = df.sort_values("dateutc", ascending=False, kind="stable").reset_index(
            drop=True
        )
    if not save_df_to_s3(df, bucket, key):
        raise StorageWriteError(f"Failed to save s3://{bucket}/{key}")


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
    folder: Optional[str] = None,
    exact_key: bool = False,
):
    """
    Backs up all files with a given prefix in an S3 bucket to a specified backup folder
    using boto3. If dry_run is True, simulates the backup without copying files.

    :param bucket: Name of the S3 bucket.
    :param prefix: Prefix of the files to back up.
    :param force_backup: Boolean flag to force backup even if the folder exists.
    :param dry_run: Boolean flag for a dry run without actual file copying.
    :param backup_root: Root directory for backups, default to '.backups'.
    :param folder: Name of the folder under `backup_root`; the prefix when None.
    :param exact_key: Back up only the object whose key is exactly `prefix`.
    :raises BackupError: If the backup folder cannot be checked, the objects
        cannot be listed, or any copy fails (#160); a caller that saves
        after the backup must not save then.
    """
    today = datetime.datetime.now().strftime("%Y%m%d")
    backup_folder = f"{backup_root}/{folder or prefix}/{today}/"
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
        raise BackupError(f"Failed to check backup folder {backup_folder}: {e}") from e

    # List objects to back up
    failed = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if exact_key and key != prefix:
                    continue
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
                    failed.append(file_name)

    except Exception as e:
        backup_logger.error(f"Error listing or copying files for backup: {e}")
        raise BackupError(f"Failed to list {prefix} for backup: {e}") from e

    if failed:
        raise BackupError(f"Failed to back up {', '.join(failed)} to {backup_folder}")
    if dry_run:
        backup_logger.info(f"Dry run completed for folder {backup_folder}")
    else:
        backup_logger.info(f"Backup completed to folder {backup_folder}")


def backup_archive(
    bucket: str, mac: str, force_backup: bool = False, dry_run: bool = False
) -> None:
    """
    Back up the station's archive, and only the archive, to its dated folder.

    The copy lands at `.backups/<mac>/<YYYYMMDD>/<mac>.<HISTORY_FILE_TYPE>`,
    once a day unless forced. Catalogs share the bare-MAC prefix but are not
    copied here. Each catalog is backed up on its own route: those saved through
    `backup_and_save_catalog` (the generic store, and the energy catalog's
    updaters), the Event Catalog by `RainEventCatalog.backup_catalog`, the
    energy catalog's `EnergyCatalog._create_backup`, and the day periods
    catalog by `backup_day_periods_catalog`. The last three write timestamped
    copies under `backups/`.

    :param bucket: Name of the S3 bucket.
    :param mac: The station's MAC address.
    :param force_backup: Back up even if today's folder already exists.
    :param dry_run: Log what would be copied without copying it.
    :raises BackupError: If the backup fails.
    """
    backup_data(
        bucket=bucket,
        prefix=f"{mac}.{HISTORY_FILE_TYPE}",
        force_backup=force_backup,
        dry_run=dry_run,
        folder=mac,
        exact_key=True,
    )


def backup_and_save_catalog(
    df: pd.DataFrame,
    catalog_type: str,
    bucket: Optional[str] = None,
    mac_address: Optional[str] = None,
) -> None:
    """
    Back up the stored catalog, then save the new version over it.

    :param df: The catalog to save.
    :param catalog_type: The catalog's name, e.g. "energy".
    :param bucket: The bucket; `DEFAULT_BUCKET` when None.
    :param mac_address: The station; the pinned station
        (`lookout.config.get_station_mac`) when None.
    :raises RuntimeError: If the backup fails; nothing is saved then.
    :raises StorageWriteError: If the save fails.
    """
    from lookout.config import DEFAULT_BUCKET, DEFAULT_FILE_TYPE, get_station_mac

    if bucket is None:
        bucket = DEFAULT_BUCKET
    if mac_address is None:
        mac_address = get_station_mac()

    # Create prefix and path for catalog files
    catalog_prefix = f"{mac_address}.{catalog_type}_catalog"
    catalog_path = f"{mac_address}.{catalog_type}_catalog.{DEFAULT_FILE_TYPE}"

    # Use existing backup_data for daily deduplication
    try:
        backup_data(bucket=bucket, prefix=catalog_prefix)
    except Exception as e:
        logger.error(f"Catalog backup failed for {catalog_prefix}: {e}")
        raise RuntimeError("Aborting catalog save: backup failed.")

    # Save new catalog
    if not save_df_to_s3(df, bucket, catalog_path, DEFAULT_FILE_TYPE):
        raise StorageWriteError(f"Failed to save s3://{bucket}/{catalog_path}")


def backup_and_save_history(
    df: pd.DataFrame, device: dict, bucket: str = "lookout"
) -> None:
    """
    Back up the old archive and save the updated DataFrame to S3.

    :param df: The archive to save.
    :param device: The station's `/devices` entry.
    :param bucket: The bucket.
    :raises RuntimeError: If the backup fails; nothing is saved then.
    :raises StorageWriteError: If the save fails.
    """
    mac = device["macAddress"]

    try:
        backup_archive(bucket=bucket, mac=mac)
    except Exception as e:
        logger.error(f"Backup failed for {mac}: {e}")
        raise RuntimeError("Aborting save: backup failed.")

    save_archive(df, bucket, mac)


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
    backup_archive(bucket=bucket_name, mac=device_id)

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

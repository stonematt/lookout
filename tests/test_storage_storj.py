"""
Characterizes `lookout.storage.storj` against a moto-faked Storj gateway:
the parquet round trip, the archive's key path, and the backup layout.

These pin today's behaviour, quirks included; a test that reads oddly says so.
"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import lookout.api.awn_controller as awn
from lookout.storage import storj
from tests.fixtures.archive import FAKE_MAC, scenario_archive
from tests.fixtures.real import load_real_slice
from tests.fixtures.storage import BUCKET, bucket_keys

ARCHIVE_KEY = f"{FAKE_MAC}.parquet"


@pytest.mark.parametrize(
    "build", [load_real_slice, scenario_archive], ids=["real", "synthetic"]
)
def test_parquet_round_trip_preserves_rows_and_schema(fake_s3, build):
    df = build()

    storj.save_df_to_s3(df, BUCKET, ARCHIVE_KEY)
    back = storj.get_df_from_s3(BUCKET, ARCHIVE_KEY, file_type="parquet")

    assert_frame_equal(back, df)
    assert list(back.dtypes.astype(str)) == list(df.dtypes.astype(str))


def test_parquet_is_written_as_octet_stream(fake_s3):
    storj.save_df_to_s3(scenario_archive(), BUCKET, ARCHIVE_KEY)
    head = fake_s3.head_object(Bucket=BUCKET, Key=ARCHIVE_KEY)
    assert head["ContentType"] == "application/octet-stream"


def test_json_lines_round_trip_keeps_rows(fake_s3):
    df = pd.DataFrame(
        {"dateutc": [1767600000000, 1767600300000], "tempf": [50.1, 50.2]}
    )
    storj.save_df_to_s3(df, BUCKET, "frame.json", file_type="json")
    back = storj.get_df_from_s3(BUCKET, "frame.json", file_type="json")
    assert_frame_equal(back, df)


def test_missing_key_reads_as_an_empty_frame(fake_s3):
    assert storj.get_df_from_s3(BUCKET, "absent.parquet", "parquet").empty


def test_json_dict_save_get_and_exists(fake_s3):
    assert storj.json_exists_in_s3(BUCKET, "doc.json") is False
    assert storj.get_json_from_s3(BUCKET, "doc.json") == {}

    assert storj.save_json_to_s3({"a": [1, 2]}, BUCKET, "doc.json") is True

    assert storj.json_exists_in_s3(BUCKET, "doc.json") is True
    assert storj.get_json_from_s3(BUCKET, "doc.json") == {"a": [1, 2]}


@pytest.mark.xfail(
    strict=True,
    reason="s3:// path parsing yields '' as the bucket (read/save_json_to_path)",
)
def test_json_by_s3_path_round_trips(fake_s3):
    assert storj.save_json_to_path({"a": 1}, f"s3://{BUCKET}/doc.json") is True
    assert storj.read_json_from_path(f"s3://{BUCKET}/doc.json") == {"a": 1}


def test_archive_is_read_from_mac_dot_parquet(fake_s3, monkeypatch):
    monkeypatch.setattr(awn, "is_dev_mode", lambda: False)
    df = scenario_archive()
    storj.save_df_to_s3(df, BUCKET, ARCHIVE_KEY)

    back = awn.load_archive_for_device({"macAddress": FAKE_MAC}, bucket=BUCKET)

    assert_frame_equal(back, df)


def test_save_history_backs_up_then_writes_mac_dot_parquet(fake_s3, frozen_today):
    old = scenario_archive().iloc[100:]
    new = scenario_archive()
    storj.save_df_to_s3(old, BUCKET, ARCHIVE_KEY)

    storj.backup_and_save_history(new, {"macAddress": FAKE_MAC}, bucket=BUCKET)

    backup_key = f".backups/{FAKE_MAC}/{frozen_today}/{ARCHIVE_KEY}"
    assert bucket_keys(fake_s3) == sorted([ARCHIVE_KEY, backup_key])
    assert_frame_equal(storj.get_df_from_s3(BUCKET, ARCHIVE_KEY, "parquet"), new)
    assert_frame_equal(storj.get_df_from_s3(BUCKET, backup_key, "parquet"), old)


def test_archive_backup_sweeps_in_everything_sharing_the_mac_prefix(
    fake_s3, frozen_today
):
    # Quirk: the archive's prefix is the bare MAC, so catalogs come along.
    storj.save_df_to_s3(scenario_archive(), BUCKET, ARCHIVE_KEY)
    fake_s3.put_object(
        Bucket=BUCKET, Key=f"{FAKE_MAC}.event_catalog.parquet", Body=b"x"
    )

    storj.backup_data(bucket=BUCKET, prefix=FAKE_MAC)

    folder = f".backups/{FAKE_MAC}/{frozen_today}/"
    assert [k for k in bucket_keys(fake_s3) if k.startswith(folder)] == [
        f"{folder}{FAKE_MAC}.event_catalog.parquet",
        f"{folder}{ARCHIVE_KEY}",
    ]


def test_backup_runs_once_a_day_unless_forced(fake_s3, frozen_today):
    folder_key = f".backups/{FAKE_MAC}/{frozen_today}/{ARCHIVE_KEY}"
    fake_s3.put_object(Bucket=BUCKET, Key=ARCHIVE_KEY, Body=b"first")
    storj.backup_data(bucket=BUCKET, prefix=FAKE_MAC)

    fake_s3.put_object(Bucket=BUCKET, Key=ARCHIVE_KEY, Body=b"second")
    storj.backup_data(bucket=BUCKET, prefix=FAKE_MAC)
    body = fake_s3.get_object(Bucket=BUCKET, Key=folder_key)["Body"].read()
    assert body == b"first"

    storj.backup_data(bucket=BUCKET, prefix=FAKE_MAC, force_backup=True)
    body = fake_s3.get_object(Bucket=BUCKET, Key=folder_key)["Body"].read()
    assert body == b"second"


def test_dry_run_backup_copies_nothing(fake_s3, frozen_today):
    fake_s3.put_object(Bucket=BUCKET, Key=ARCHIVE_KEY, Body=b"x")
    storj.backup_data(bucket=BUCKET, prefix=FAKE_MAC, dry_run=True)
    assert bucket_keys(fake_s3) == [ARCHIVE_KEY]


def test_list_bucket_items_respects_depth(fake_s3):
    fake_s3.put_object(Bucket=BUCKET, Key=ARCHIVE_KEY, Body=b"abc")
    fake_s3.put_object(Bucket=BUCKET, Key="a/b/c/deep.parquet", Body=b"x")

    items = storj.list_bucket_items(BUCKET, depth=2)

    assert items == [{"path": f"{BUCKET}/{ARCHIVE_KEY}", "size": 3}]

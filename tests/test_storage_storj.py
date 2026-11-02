"""
Characterizes `lookout.storage.storj` against a moto-faked Storj gateway:
the parquet round trip, the archive's key path, and the backup layout.

These pin today's behaviour, quirks included; a test that reads oddly says so.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import lookout
import lookout.api.awn_controller as awn
from lookout.storage import storj
from tests.fixtures.archive import FAKE_MAC, scenario_archive
from tests.fixtures.real import load_real_slice
from tests.fixtures.storage import BUCKET, bucket_keys
from tests.fixtures.storage import flaky_storj

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


def test_json_dict_save_and_read(fake_s3):
    assert storj.read_json_from_s3(BUCKET, "doc.json") is None

    assert storj.save_json_to_s3({"a": [1, 2]}, BUCKET, "doc.json") is True

    assert storj.read_json_from_s3(BUCKET, "doc.json") == {"a": [1, 2]}


def test_storage_has_no_separate_existence_check():
    # Callers read once and act on the read outcome (#185, ADR-0004).
    assert not hasattr(storj, "object_exists")


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

    storj.backup_and_save_history(
        new, {"macAddress": FAKE_MAC}, bucket=BUCKET, stored=old
    )

    backup_key = f".backups/{FAKE_MAC}/{frozen_today}/{ARCHIVE_KEY}"
    assert bucket_keys(fake_s3) == sorted([ARCHIVE_KEY, backup_key])
    assert_frame_equal(storj.get_df_from_s3(BUCKET, ARCHIVE_KEY, "parquet"), new)
    assert_frame_equal(storj.get_df_from_s3(BUCKET, backup_key, "parquet"), old)


def test_save_history_stores_an_oldest_first_frame_newest_first(fake_s3, frozen_today):
    # #133: the Diagnostics CLI `fill-gap`, the Streamlit Diagnostics save and
    # the rain repair all save through here, in whatever order they built.
    new = scenario_archive()
    oldest_first = new.iloc[::-1].reset_index(drop=True)

    storj.backup_and_save_history(
        oldest_first, {"macAddress": FAKE_MAC}, bucket=BUCKET, stored=new
    )

    stored = storj.get_df_from_s3(BUCKET, ARCHIVE_KEY, "parquet")
    assert stored["dateutc"].is_monotonic_decreasing
    assert_frame_equal(stored, new)


def test_save_archive_writes_mac_dot_parquet_newest_first(fake_s3):
    new = scenario_archive()
    shuffled = new.sample(frac=1, random_state=7)

    storj.save_archive(shuffled, BUCKET, FAKE_MAC, stored=new)

    assert bucket_keys(fake_s3) == [ARCHIVE_KEY]
    assert_frame_equal(storj.get_df_from_s3(BUCKET, ARCHIVE_KEY, "parquet"), new)


def test_archive_backup_copies_the_archive_and_not_the_catalogs(fake_s3, frozen_today):
    # Decided in #114: an archive backup is the archive only; catalogs are
    # backed up by their own catalog-specific path.
    storj.save_df_to_s3(scenario_archive(), BUCKET, ARCHIVE_KEY)
    fake_s3.put_object(
        Bucket=BUCKET, Key=f"{FAKE_MAC}.event_catalog.parquet", Body=b"x"
    )

    storj.backup_archive(bucket=BUCKET, mac=FAKE_MAC)

    folder = f".backups/{FAKE_MAC}/{frozen_today}/"
    assert [k for k in bucket_keys(fake_s3) if k.startswith(folder)] == [
        f"{folder}{ARCHIVE_KEY}"
    ]


def test_archive_backup_copies_the_archive_key_exactly(fake_s3, frozen_today):
    fake_s3.put_object(Bucket=BUCKET, Key=ARCHIVE_KEY, Body=b"x")
    fake_s3.put_object(Bucket=BUCKET, Key=f"{ARCHIVE_KEY}.bak", Body=b"y")

    storj.backup_archive(bucket=BUCKET, mac=FAKE_MAC)

    folder = f".backups/{FAKE_MAC}/{frozen_today}/"
    assert [k for k in bucket_keys(fake_s3) if k.startswith(folder)] == [
        f"{folder}{ARCHIVE_KEY}"
    ]


def test_save_history_backup_leaves_the_catalogs_out(fake_s3, frozen_today):
    storj.save_df_to_s3(scenario_archive(), BUCKET, ARCHIVE_KEY)
    catalog_key = f"{FAKE_MAC}.energy_catalog.parquet"
    fake_s3.put_object(Bucket=BUCKET, Key=catalog_key, Body=b"x")

    storj.backup_and_save_history(
        scenario_archive(),
        {"macAddress": FAKE_MAC},
        bucket=BUCKET,
        stored=scenario_archive(),
    )

    backup_key = f".backups/{FAKE_MAC}/{frozen_today}/{ARCHIVE_KEY}"
    assert bucket_keys(fake_s3) == sorted([ARCHIVE_KEY, backup_key, catalog_key])


def test_dry_run_archive_backup_copies_nothing(fake_s3, frozen_today):
    fake_s3.put_object(Bucket=BUCKET, Key=ARCHIVE_KEY, Body=b"x")
    storj.backup_archive(bucket=BUCKET, mac=FAKE_MAC, dry_run=True)
    assert bucket_keys(fake_s3) == [ARCHIVE_KEY]


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


# ---------- telling a missing object from a failed read (#160) ----------


def test_read_json_is_none_for_a_missing_object(fake_s3):
    assert storj.read_json_from_s3(BUCKET, "absent.json") is None


def test_read_json_raises_on_an_unparseable_object(fake_s3):
    fake_s3.put_object(Bucket=BUCKET, Key="bad.json", Body=b"{not json")

    with pytest.raises(storj.StorageReadError):
        storj.read_json_from_s3(BUCKET, "bad.json")


# The reader contract (#172, ADR-0004): found returns the value, missing
# returns None, failed raises `StorageReadError`.


@pytest.mark.parametrize("file_type", ["parquet", "json"])
def test_read_df_returns_the_frame_when_found(fake_s3, file_type):
    df = pd.DataFrame({"dateutc": [1767600000000], "tempf": [50.1]})
    storj.save_df_to_s3(df, BUCKET, "frame", file_type=file_type)

    assert_frame_equal(storj.read_df_from_s3(BUCKET, "frame", file_type), df)


@pytest.mark.parametrize("file_type", ["parquet", "json"])
def test_read_df_returns_none_when_missing(fake_s3, file_type):
    assert storj.read_df_from_s3(BUCKET, "absent", file_type) is None


def test_read_df_raises_when_the_read_fails(fake_s3, monkeypatch):
    storj.save_df_to_s3(scenario_archive(), BUCKET, ARCHIVE_KEY)
    flaky_storj(monkeypatch, fake_s3, read=ARCHIVE_KEY)

    with pytest.raises(storj.StorageReadError, match="read failed"):
        storj.read_df_from_s3(BUCKET, ARCHIVE_KEY, "parquet")
    assert storj.get_df_from_s3(BUCKET, ARCHIVE_KEY, "parquet").empty


def test_read_df_raises_when_the_object_is_corrupt(fake_s3):
    fake_s3.put_object(Bucket=BUCKET, Key="bad.parquet", Body=b"not parquet")

    with pytest.raises(storj.StorageReadError, match="unreadable"):
        storj.read_df_from_s3(BUCKET, "bad.parquet", "parquet")


def test_read_json_raises_when_the_read_fails(fake_s3, monkeypatch):
    storj.save_json_to_s3({"a": 1}, BUCKET, "doc.json")
    flaky_storj(monkeypatch, fake_s3, read="doc.json")

    with pytest.raises(storj.StorageReadError, match="read failed"):
        storj.read_json_from_s3(BUCKET, "doc.json")


def test_backup_object_returns_the_copy_key_when_found(fake_s3):
    fake_s3.put_object(Bucket=BUCKET, Key="src", Body=b"as stored")

    assert storj.backup_object(BUCKET, "src", "dst") == "dst"
    copied = fake_s3.get_object(Bucket=BUCKET, Key="dst")["Body"].read()
    assert copied == b"as stored"


def test_backup_object_returns_none_when_the_source_is_missing(fake_s3):
    assert storj.backup_object(BUCKET, "absent", "dst") is None
    assert bucket_keys(fake_s3) == []


def test_backup_object_raises_backup_error_when_the_copy_fails(fake_s3, monkeypatch):
    fake_s3.put_object(Bucket=BUCKET, Key="src", Body=b"x")
    flaky_storj(monkeypatch, fake_s3, copy=True)

    with pytest.raises(storj.BackupError):
        storj.backup_object(BUCKET, "src", "dst")
    assert bucket_keys(fake_s3) == ["src"]


def test_load_archive_raises_for_a_missing_archive(fake_s3):
    with pytest.raises(storj.StorageReadError, match="missing"):
        storj.load_archive(BUCKET, FAKE_MAC)


def test_load_archive_counts_rows_after_dropping_duplicates(fake_s3):
    # One duplicate row must not make every later guarded save refuse.
    archive = scenario_archive()
    storj.save_df_to_s3(pd.concat([archive, archive.head(1)]), BUCKET, ARCHIVE_KEY)

    assert len(storj.load_archive(BUCKET, FAKE_MAC)) == len(archive)


def test_load_archive_raises_for_an_archive_with_no_valid_rows(fake_s3):
    storj.save_df_to_s3(pd.DataFrame({"dateutc": [0, -5]}), BUCKET, ARCHIVE_KEY)

    with pytest.raises(storj.StorageReadError, match="no rows"):
        storj.load_archive(BUCKET, FAKE_MAC)


def test_save_df_reports_whether_it_saved(fake_s3):
    assert storj.save_df_to_s3(scenario_archive(), BUCKET, ARCHIVE_KEY) is True
    assert storj.save_df_to_s3(scenario_archive(), "no-such-bucket", "x") is False


# ---------- every archive save is guarded against the stored archive (#190) ---


def stored_archive(fake_s3) -> pd.DataFrame:
    """Seed the bucket with the SCENARIO week's later half; return it as read."""
    storj.save_df_to_s3(scenario_archive().iloc[:-100], BUCKET, ARCHIVE_KEY)
    return storj.load_archive(BUCKET, FAKE_MAC)


def stored_now() -> pd.DataFrame:
    """The archive as it sits in the bucket now."""
    return storj.get_df_from_s3(BUCKET, ARCHIVE_KEY, "parquet")


def test_an_archive_save_must_hand_over_the_stored_archive(fake_s3):
    with pytest.raises(TypeError):
        storj.save_archive(scenario_archive(), BUCKET, FAKE_MAC)
    with pytest.raises(TypeError):
        storj.save_archive(scenario_archive(), BUCKET, FAKE_MAC, loaded_rows=0)
    assert bucket_keys(fake_s3) == []


def test_save_history_refuses_a_save_that_drops_a_stored_reading(fake_s3, frozen_today):
    stored = stored_archive(fake_s3)
    before = stored_now()

    with pytest.raises(storj.ArchiveLossError, match="drop 1 stored reading"):
        storj.backup_and_save_history(
            stored.iloc[1:],
            {"macAddress": FAKE_MAC},
            bucket=BUCKET,
            stored=stored,
        )

    assert_frame_equal(stored_now(), before)


def test_a_save_that_swaps_a_stored_reading_is_refused_at_the_same_length(fake_s3):
    stored = stored_archive(fake_s3)
    before = stored_now()
    later = scenario_archive().iloc[[-100]]
    swapped = pd.concat([stored.iloc[1:], later], ignore_index=True)
    assert len(swapped) == len(stored)

    with pytest.raises(storj.ArchiveLossError, match="drop 1 stored reading"):
        storj.save_archive(swapped, BUCKET, FAKE_MAC, stored=stored)

    assert_frame_equal(stored_now(), before)


def test_a_save_that_blanks_a_stored_value_is_refused_though_rows_grow(fake_s3):
    # The April gap fill: rows added, and a stored value blanked in place.
    stored = stored_archive(fake_s3)
    before = stored_now()
    grown = scenario_archive()
    row = grown.index[grown["dateutc"] == stored["dateutc"].iloc[0]][0]
    grown.loc[row, "tempf"] = float("nan")
    assert len(grown) > len(stored)

    with pytest.raises(storj.ArchiveLossError, match="blank 1 stored value"):
        storj.save_archive(grown, BUCKET, FAKE_MAC, stored=stored)

    assert_frame_equal(stored_now(), before)


def test_a_save_without_a_stored_column_blanks_its_values_and_is_refused(fake_s3):
    stored = stored_archive(fake_s3)
    before = stored_now()

    with pytest.raises(storj.ArchiveLossError, match="tempf"):
        storj.save_archive(
            stored.drop(columns=["tempf"]), BUCKET, FAKE_MAC, stored=stored
        )

    assert_frame_equal(stored_now(), before)


def test_a_save_that_only_updates_and_adds_values_is_saved(fake_s3):
    stored = stored_archive(fake_s3)
    new = scenario_archive()
    new["tempf"] = new["tempf"] + 0.5
    new["pm25"] = 7.0
    stored_nan = stored.assign(humidity=float("nan"))
    # A value missing on both sides is not a blank.
    new.loc[new["dateutc"].isin(stored_nan["dateutc"]), "humidity"] = float("nan")

    storj.save_archive(new, BUCKET, FAKE_MAC, stored=stored_nan)

    assert_frame_equal(stored_now(), new)


def test_the_guard_counts_the_stored_archive_after_normalize(fake_s3):
    # A stored duplicate is not a reading the save could drop.
    stored = scenario_archive()
    with_duplicate = pd.concat([stored, stored.iloc[[0]]], ignore_index=True)

    storj.save_archive(stored, BUCKET, FAKE_MAC, stored=with_duplicate)

    assert_frame_equal(stored_now(), stored)


def test_the_repair_allowance_lets_a_dropping_and_blanking_save_through(
    fake_s3, frozen_today
):
    stored = stored_archive(fake_s3)
    repaired = stored.iloc[1:].copy()
    repaired["tempf"] = float("nan")

    storj.backup_and_save_history(
        repaired,
        {"macAddress": FAKE_MAC},
        bucket=BUCKET,
        stored=stored,
        repair=True,
    )

    newest_first = repaired.sort_values("dateutc", ascending=False)
    assert_frame_equal(stored_now(), newest_first.reset_index(drop=True))


def _calls_passing_repair() -> set:
    """
    (file, enclosing function) of every archive save that passes `repair`,
    outside the storage module, whose save path only forwards it.
    """
    root = Path(lookout.__file__).parent
    found = set()
    for path in sorted(root.rglob("*.py")):
        if path == Path(storj.__file__):
            continue
        tree = ast.parse(path.read_text())
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(func):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "attr", getattr(node.func, "id", None))
                if name not in ("save_archive", "backup_and_save_history"):
                    continue
                if any(k.arg == "repair" for k in node.keywords):
                    found.add((path.relative_to(root.parent).as_posix(), func.name))
    return found


def test_only_the_rain_repair_and_repair_archive_pass_the_repair_allowance():
    assert _calls_passing_repair() == {
        ("lookout/cli/catchup.py", "main"),
        ("lookout/cli/repair_rain.py", "main"),
    }


def test_a_failed_history_backup_raises_backup_error_and_saves_nothing(
    fake_s3, monkeypatch, frozen_today
):
    stored = scenario_archive().iloc[100:]
    storj.save_df_to_s3(stored, BUCKET, ARCHIVE_KEY)
    flaky_storj(monkeypatch, fake_s3, copy=True)

    with pytest.raises(storj.BackupError, match="Failed to back up"):
        storj.backup_and_save_history(
            scenario_archive(),
            {"macAddress": FAKE_MAC},
            bucket=BUCKET,
            stored=stored,
        )

    assert bucket_keys(fake_s3) == [ARCHIVE_KEY]
    assert_frame_equal(storj.get_df_from_s3(BUCKET, ARCHIVE_KEY, "parquet"), stored)


def test_a_failed_catalog_backup_raises_backup_error_and_saves_nothing(
    fake_s3, monkeypatch, frozen_today
):
    key = f"{FAKE_MAC}.energy_catalog.parquet"
    fake_s3.put_object(Bucket=BUCKET, Key=key, Body=b"old")
    flaky_storj(monkeypatch, fake_s3, copy=True)

    with pytest.raises(storj.BackupError, match="Failed to back up"):
        storj.backup_and_save_catalog(
            pd.DataFrame({"a": [1]}), "energy", bucket=BUCKET, mac_address=FAKE_MAC
        )

    assert bucket_keys(fake_s3) == [key]
    assert fake_s3.get_object(Bucket=BUCKET, Key=key)["Body"].read() == b"old"

"""
make_real_fixtures.py: Regenerate the frozen real slice and Ambient responses.

Reads a local archive parquet already on disk (never Storj, never Ambient),
cuts a fixed window, scrubs station-identifying values, and writes:

- `real_slice.parquet`: archive rows in [SLICE_START, SLICE_END), every column
  and dtype as on disk, newest row first.
- `ambient/history.json`: rows in [SLICE_END, AMBIENT_END) reshaped as the
  Ambient REST `/devices/{mac}` body, newest first, so live data runs ahead of
  the archive slice.
- `ambient/devices.json`: the `/devices` body, with the newest history record
  as `lastData`.

Scrubbing: `passkey` and the passkey inside `loc` become FAKE_PASSKEY; the MAC
becomes FAKE_MAC; station name and coordinates are invented.

Usage (from the repo root):
    python -m tests.fixtures.make_real_fixtures ../lookout/data/<MAC>.parquet
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from tests.fixtures.archive import FAKE_MAC, STATION_TZ

HERE = Path(__file__).parent
SLICE_START = "2025-11-17T08:00:00Z"  # 2025-11-17 00:00 local
SLICE_END = "2025-11-19T05:00:00Z"
AMBIENT_END = "2025-11-19T08:00:00Z"
FAKE_PASSKEY = "0" * 32

# Columns the archive carries that Ambient's history endpoint does not send.
ARCHIVE_ONLY = ("passkey", "loc", "time", "value_bin", "direction_bin", "hour")


def _window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Rows with start <= dateutc < end, newest first."""
    lo = pd.Timestamp(start).value // 1_000_000
    hi = pd.Timestamp(end).value // 1_000_000
    out = df[(df["dateutc"] >= lo) & (df["dateutc"] < hi)]
    return out.sort_values("dateutc", ascending=False).reset_index(drop=True)


def scrub_archive(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the station passkey wherever it appears."""
    df = df.copy()
    if "passkey" in df.columns:
        df["passkey"] = df["passkey"].where(df["passkey"].isna(), FAKE_PASSKEY)
    if "loc" in df.columns:
        df["loc"] = df["loc"].str.replace(r"^[0-9a-f]{32}", FAKE_PASSKEY, regex=True)
    return df


def _iso_z(ts: pd.Timestamp) -> str:
    """Ambient's timestamp format: UTC with milliseconds and a Z."""
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z")


def to_ambient_records(df: pd.DataFrame) -> list:
    """Archive rows as Ambient history records: no NaN fields, ISO dates."""
    records = []
    for row in df.drop(columns=[c for c in ARCHIVE_ONLY if c in df.columns]).to_dict(
        "records"
    ):
        record = {}
        for key, value in row.items():
            if value is None or (not isinstance(value, str) and pd.isna(value)):
                continue
            if isinstance(value, pd.Timestamp):
                value = _iso_z(value)
            elif hasattr(value, "item"):
                value = value.item()
            record[key] = value
        records.append(record)
    return records


def devices_body(last_data: dict) -> list:
    """The `/devices` body for one invented station."""
    return [
        {
            "macAddress": FAKE_MAC,
            "lastData": {**last_data, "tz": STATION_TZ},
            "info": {
                "name": "Fixture Station",
                "coords": {
                    "coords": {"lat": 45.0, "lon": -123.0},
                    "address": "Fixture Address",
                    "location": "Fixture",
                    "elevation": 50.0,
                    "geo": {"type": "Point", "coordinates": [-123.0, 45.0]},
                },
            },
        }
    ]


def main() -> None:
    """Write the fixtures from the archive parquet named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("source", type=Path, help="Local archive parquet to read")
    args = parser.parse_args()

    df = pd.read_parquet(args.source)
    archive = scrub_archive(_window(df, SLICE_START, SLICE_END))
    archive.to_parquet(HERE / "real_slice.parquet", engine="pyarrow", index=False)

    history = to_ambient_records(_window(df, SLICE_END, AMBIENT_END))
    (HERE / "ambient").mkdir(exist_ok=True)
    with open(HERE / "ambient" / "history.json", "w") as f:
        json.dump(history, f, indent=1)
        f.write("\n")
    with open(HERE / "ambient" / "devices.json", "w") as f:
        json.dump(devices_body(history[0]), f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()

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

from tests.fixtures.ambient import devices_body, to_ambient_records
from tests.fixtures.real import AMBIENT_END, FAKE_PASSKEY, SLICE_END, SLICE_START

HERE = Path(__file__).parent


def _window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Cut a time window from an archive.

    :param df: Archive rows.
    :param start: ISO-8601 UTC start, inclusive.
    :param end: ISO-8601 UTC end, exclusive.
    :return: The rows in the window, newest first.
    """
    lo = pd.Timestamp(start).value // 1_000_000
    hi = pd.Timestamp(end).value // 1_000_000
    out = df[(df["dateutc"] >= lo) & (df["dateutc"] < hi)]
    return out.sort_values("dateutc", ascending=False).reset_index(drop=True)


def scrub_archive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the station passkey wherever it appears.

    :param df: Archive rows.
    :return: A scrubbed copy.
    """
    df = df.copy()
    if "passkey" in df.columns:
        df["passkey"] = df["passkey"].where(df["passkey"].isna(), FAKE_PASSKEY)
    if "loc" in df.columns:
        df["loc"] = df["loc"].str.replace(r"^[0-9a-f]{32}", FAKE_PASSKEY, regex=True)
    return df


def main() -> None:
    """Write the fixtures from the archive parquet named on the command line."""
    parser = argparse.ArgumentParser(description="Regenerate the real fixtures")
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

#!/usr/bin/env python3
"""
rain_corrections_cli.py: Rain carryover detection and catalog management.

Detects and corrects rain accumulation errors caused by equipment outages
spanning midnight. When the station is offline at midnight, dailyrainin
fails to reset, carrying forward the previous day's total.

Usage:
    python -m lookout.cli.rain_corrections_cli [--bucket lookout] detect
    python -m lookout.cli.rain_corrections_cli [--bucket lookout] generate [--dry-run]
    python -m lookout.cli.rain_corrections_cli [--bucket lookout] show

Exits 1 when the pinned station (STATION_MAC) is not on the Ambient account.
"""

import argparse
import sys
from typing import Tuple

import pandas as pd

import lookout.api.ambient_client as ambient_client
import lookout.core.rain_corrections as lo_rc
import lookout.storage.storj as lo_storj
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def get_station() -> Tuple[dict, str, str]:
    """
    Get the pinned station (STATION_MAC) from the Ambient API, or exit 1.

    :return: (device, mac, name).
    :raises SystemExit: With code 1, after printing why the station was not
        found (`StationNotFoundError`'s message).
    """
    try:
        device = ambient_client.get_station_device()
    except ambient_client.StationNotFoundError as e:
        logger.error(str(e))
        print(f"❌ {e}")
        sys.exit(1)

    mac = device.get("macAddress")
    name = device.get("info", {}).get("name", "Unnamed Device")

    logger.info(f"Selected device: {name} ({mac})")
    return device, mac, name


def load_archive_s3(mac_address: str, bucket: str = "lookout") -> pd.DataFrame:
    """Load archive from S3 production storage."""
    logger.info(f"Loading archive for {mac_address} from S3")
    try:
        return lo_storj.get_df_from_s3(bucket, f"{mac_address}.parquet", "parquet")
    except Exception as e:
        logger.error(f"Failed to load archive from S3: {e}")
        return pd.DataFrame()


def handle_detect(args):
    """Handle detect command for the pinned station."""
    device, mac, name = get_station()

    print(f"🔍 Scanning archive for: {name} ({mac})")

    df = load_archive_s3(mac, args.bucket)
    if df.empty:
        print("❌ Archive is empty or not found.")
        return

    print(f"📊 Scanning {len(df)} records for carryovers...")
    carryovers = lo_rc.detect_carryovers(df)

    if not carryovers:
        print("✅ No carryovers detected.")
        return

    print(f"\n🚨 Found {len(carryovers)} carryover(s):\n")
    for c in carryovers:
        print(f"  📅 {c['affected_date']}: {c['carryover_amount']:.3f}\"")
        print(f"     Gap: {c['gap_start']} → {c['gap_end']}")


def handle_generate(args):
    """Handle generate command for the pinned station."""
    device, mac, name = get_station()

    print(f"🔍 Scanning archive for: {name} ({mac})")

    df = load_archive_s3(mac, args.bucket)
    if df.empty:
        print("❌ Archive is empty or not found.")
        return

    print(f"📊 Scanning {len(df)} records for carryovers...")
    carryovers = lo_rc.detect_carryovers(df)

    if not carryovers:
        print("✅ No carryovers detected. No catalog created.")
        return

    print(f"\n🚨 Found {len(carryovers)} carryover(s):")
    for c in carryovers:
        print(f"  📅 {c['affected_date']}: {c['carryover_amount']:.3f}\"")

    if args.dry_run:
        print("\n🔒 --dry-run: Catalog not saved.")
        return

    catalog_data = {
        "version": "1.0",
        "mac_address": mac,
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "corrections": carryovers,
    }

    if lo_rc.save_catalog(catalog_data, mac, args.bucket):
        print(f"\n💾 Saved catalog with {len(carryovers)} corrections to S3.")
    else:
        print("\n❌ Failed to save catalog.")


def handle_show(args):
    """Handle show command for the pinned station."""
    device, mac, name = get_station()

    print(f"📋 Showing catalog for: {name} ({mac})")

    if not lo_rc.catalog_exists(mac, args.bucket):
        print("📭 No corrections catalog found.")
        return

    data = lo_rc.load_catalog(mac, args.bucket)
    print(f"Version: {data.get('version')}")
    print(f"Updated: {data.get('updated_at')}")
    print(f"Corrections: {len(data.get('corrections', []))}\n")

    for c in data.get("corrections", []):
        print(f"  📅 {c['affected_date']}: {c['carryover_amount']:.3f}\"")
        print(f"     Gap: {c['gap_start']} → {c['gap_end']}")


def main():
    parser = argparse.ArgumentParser(
        description="Rain carryover corrections management"
    )
    parser.add_argument("--bucket", default="lookout", help="S3 bucket name")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # detect subcommand
    detect_parser = subparsers.add_parser(
        "detect", help="Scan archive for carryover gaps"
    )

    # generate subcommand
    gen_parser = subparsers.add_parser(
        "generate", help="Detect carryovers and save catalog to S3"
    )
    gen_parser.add_argument(
        "--dry-run", action="store_true", help="Detect but do not save"
    )

    # show subcommand
    show_parser = subparsers.add_parser(
        "show", help="Display current corrections catalog from S3"
    )

    args = parser.parse_args()

    if args.command == "detect":
        handle_detect(args)
    elif args.command == "generate":
        handle_generate(args)
    elif args.command == "show":
        handle_show(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

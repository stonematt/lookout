#!/usr/bin/env python3
"""
Manage Day Periods Catalog CLI

Manages pre-computed sunrise/sunset catalogs for solar analysis.
Requires astral library for generation.

Usage:
    python lookout/cli/manage_day_periods.py                    # List catalogs (default)
    python lookout/cli/manage_day_periods.py --generate
    python lookout/cli/manage_day_periods.py --delete
    python lookout/cli/manage_day_periods.py --dry-run

Examples:
  List existing catalogs (default):
    %(prog)s

  Generate new catalog (requires astral):
    %(prog)s --generate

  Delete existing catalog:
    %(prog)s --delete

  Dry run (show what would be generated):
    %(prog)s --dry-run
"""

import argparse
import sys
from datetime import datetime
import pandas as pd

# Lookout imports
from lookout.api.ambient_client import StationNotFoundError, get_station_device
from lookout.core.day_periods import (
    generate_day_periods_catalog,
    save_day_periods_catalog,
    backup_day_periods_catalog,
    delete_day_periods_catalog,
    get_catalog_path,
    load_day_periods_catalog,
)
from lookout.storage.storj import BackupError
from lookout.utils.log_util import app_logger

# Location configuration (Salem, OR)
LOCATION = {"latitude": 44.9429, "longitude": -123.0351}

logger = app_logger(__name__)


def get_station():
    """Get the pinned station (STATION_MAC) from the Ambient API, or exit 1."""
    try:
        device = get_station_device()
    except StationNotFoundError as e:
        logger.error(f"❌ {e}")
        sys.exit(1)

    mac = device.get("macAddress")
    name = device.get("info", {}).get("name", "Unnamed Device")

    logger.info(f"📡 Selected device: {name} ({mac})")
    return mac, device


def show_catalog_stats(catalog_df, mac_address, device_name):
    """Display catalog statistics."""
    if catalog_df.empty:
        logger.info("📋 No catalog data available")
        return

    logger.info("📊 Catalog statistics:")
    logger.info(f"   - Days in catalog: {len(catalog_df)}")
    logger.info(
        f"   - Date range: {catalog_df['date'].min().date()} to {catalog_df['date'].max().date()}"
    )
    logger.info(
        f"   - Average daylight: {catalog_df['daylight_minutes'].mean():.1f} minutes"
    )

    # Check for generation metadata if available
    if "generated_at" in catalog_df.columns:
        generated_time = catalog_df["generated_at"].iloc[0]
        if pd.notna(generated_time):
            logger.info(f"   - Generated: {generated_time}")

    logger.info(f"   - Device: {device_name} ({mac_address})")


def main():
    parser = argparse.ArgumentParser(
        description="Manage day periods catalog for solar analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  List existing catalogs (default):
    %(prog)s
  
  Generate new catalog (requires astral):
    %(prog)s --generate
  
  Delete existing catalog:
    %(prog)s --delete
  
  Dry run (show what would be generated):
    %(prog)s --dry-run
        """,
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate new day periods catalog",
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete existing catalog (after creating backup)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without actually doing it",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing day periods catalogs (default action)",
    )

    parser.add_argument(
        "--start-date",
        default="2023-01-01",
        help="Start date for catalog (default: 2023-01-01)",
    )

    parser.add_argument(
        "--end-date", help="End date for catalog (default: current year + 5 years)"
    )

    args = parser.parse_args()

    # Default action is list if no other action specified
    if not any([args.generate, args.delete, args.dry_run]):
        args.list = True

    # Handle list operation (default)
    if args.list:
        logger.info("📋 Listing existing day periods catalogs...")

        # Get the pinned station to check for catalogs
        try:
            mac_address, device = get_station()
            device_name = device.get("info", {}).get("name", "Unknown")

            # One read: found, missing, or failed (raises, reported below)
            catalog_df = load_day_periods_catalog(mac_address)
            if catalog_df is not None:
                logger.info("✅ Found 1 catalog(s):")
                logger.info(f"   - {get_catalog_path(mac_address)}")
                if not catalog_df.empty:
                    show_catalog_stats(catalog_df, mac_address, device_name)
            else:
                logger.info("📋 No day periods catalogs found")

            return 0
        except Exception as e:
            logger.error(f"❌ Error listing catalogs: {e}")
            return 1

    # Handle delete operation
    if args.delete:
        logger.info("🗑️  Delete operation requested")

        # Get the pinned station (STATION_MAC)
        mac_address, device = get_station()

        # Delete catalog (not implemented: it reports that nothing was deleted)
        if delete_day_periods_catalog(mac_address):
            logger.info("✅ Day periods catalog deleted successfully")
            return 0
        logger.error("❌ Day periods catalog not deleted")
        return 1

    # Handle dry run operation
    if args.dry_run:
        logger.info("🔍 Dry run mode - showing what would be generated...")

        # Get the pinned station (STATION_MAC)
        mac_address, device = get_station()

        # Use LOCATION coordinates (Salem, OR)
        lat = LOCATION["latitude"]
        lon = LOCATION["longitude"]

        logger.info(f"📍 Using location: lat={lat}, lon={lon}")
        logger.info(
            f"📡 Device: {device.get('info', {}).get('name', 'Unknown')} ({mac_address})"
        )
        logger.info(
            f"📅 Date range: {args.start_date} to {args.end_date or 'current year + 5 years'}"
        )

        # Check if astral is available
        try:
            import astral

            logger.info("✅ astral library available for precise calculations")

            # Show sample of what would be generated
            from datetime import timedelta

            sample_start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
            sample_end = sample_start + timedelta(days=7)  # Show 7 days sample

            sample_catalog = generate_day_periods_catalog(
                lat=lat,
                lon=lon,
                start_date=args.start_date,
                end_date=sample_end.strftime("%Y-%m-%d"),
                use_astral=True,
            )

            if not sample_catalog.empty:
                logger.info("📊 Sample catalog preview:")
                logger.info(f"   - Sample days: {len(sample_catalog)}")
                logger.info(
                    f"   - Date range: {sample_catalog['date'].min().date()} to {sample_catalog['date'].max().date()}"
                )
                logger.info(
                    f"   - Sample daylight range: {sample_catalog['daylight_minutes'].min():.0f} - {sample_catalog['daylight_minutes'].max():.0f} minutes"
                )
                logger.info(
                    "💡 This would generate a full catalog with all days in the specified range"
                )
            else:
                logger.error("❌ Would generate empty catalog")

        except ImportError:
            logger.warning("⚠️ astral library not available")
            logger.warning("Install with: pip install astral")
            logger.warning("This is required for day periods catalog generation")

        return 0

    # Handle generate operation
    if args.generate:
        logger.info("🌅 Generating day periods catalog...")

        # Get the pinned station (STATION_MAC)
        mac_address, device = get_station()

        # Use LOCATION coordinates (Salem, OR)
        lat = LOCATION["latitude"]
        lon = LOCATION["longitude"]

        logger.info(f"📍 Using location: lat={lat}, lon={lon}")

        # Require astral
        try:
            import astral

            logger.info("✅ Using astral library for precise calculations")
        except ImportError:
            logger.error("❌ astral library not found")
            logger.error("Install with: pip install astral")
            logger.error("This is required for day periods catalog generation")
            return 1

        # Generate catalog
        try:
            start_time = datetime.now()

            catalog_df = generate_day_periods_catalog(
                lat=lat,
                lon=lon,
                start_date=args.start_date,
                end_date=args.end_date,
                use_astral=True,  # Always use astral
            )

            if catalog_df.empty:
                logger.error("❌ Failed to generate catalog - empty result")
                return 1

            # Back up the stored catalog; a failed backup stops the save
            try:
                backup_day_periods_catalog(mac_address)
            except BackupError as e:
                logger.error(f"❌ Backup failed; catalog not saved: {e}")
                return 1

            # Save new catalog
            if save_day_periods_catalog(catalog_df, mac_address):
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                logger.info("✅ Day periods catalog generated and saved successfully")
                show_catalog_stats(
                    catalog_df,
                    mac_address,
                    device.get("info", {}).get("name", "Unknown"),
                )
                logger.info(f"   - Processing time: {duration:.1f} seconds")

                return 0
            else:
                logger.error("❌ Failed to save day periods catalog")
                return 1

        except KeyboardInterrupt:
            logger.info("⚠️  Generation interrupted by user")
            return 1
        except Exception as e:
            logger.error(f"❌ Error generating day periods catalog: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

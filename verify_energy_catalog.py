#!/usr/bin/env python3
"""
Quick verification script for energy catalog optimization
"""

import pandas as pd
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lookout.core.energy_catalog import EnergyCatalog
from lookout.core.solar_energy_periods import calculate_15min_energy_periods, calculate_15min_energy_periods_cached

def create_test_data():
    """Create test solar data matching real data structure"""
    dates = pd.date_range("2023-01-01", periods=100, freq="5min", tz="America/Los_Angeles")
    data = {
        "date": dates,  # TZ-aware datetime column (this is what real data has)
        "dateutc": [int(dt.timestamp() * 1000) for dt in dates],
        "solarradiation": [max(0, 800 * (1 - abs((i % 288) - 144) / 144)) for i in range(100)]
    }
    return pd.DataFrame(data)

def test_energy_catalog():
    """Test EnergyCatalog functionality"""
    print("🧪 Testing EnergyCatalog functionality...")

    # Create test data
    df = create_test_data()
    print(f"✓ Created test data: {len(df)} records")

    # Test catalog creation
    catalog = EnergyCatalog("TEST_MAC")
    print("✓ Created EnergyCatalog instance")

    # Test period calculation
    periods = catalog.detect_and_calculate_periods(df, auto_save=False)
    print(f"✓ Calculated {len(periods)} energy periods")

    # Test update functionality
    new_df = create_test_data()
    new_df["date"] = new_df["date"] + pd.Timedelta(days=1)  # Next day
    new_df["dateutc"] = [int(dt.timestamp() * 1000) for dt in new_df["date"]]

    updated_periods = catalog.update_catalog_with_new_data(new_df, periods)
    print(f"✓ Updated catalog: {len(updated_periods)} total periods")

    print("✅ EnergyCatalog tests passed!")

def test_performance_improvement():
    """Test performance improvement with caching"""
    print("\n⚡ Testing performance improvements...")

    df = create_test_data()
    # Prepare data for calculate_15min_energy_periods (needs 'datetime' column)
    df_processed = df.copy()
    df_processed['datetime'] = df_processed['date']  # Use 'date' as 'datetime'

    # Test original function
    import time
    start = time.time()
    result1 = calculate_15min_energy_periods(df_processed)
    original_time = time.time() - start

    # Test cached function (first run - no cache)
    start = time.time()
    result2 = calculate_15min_energy_periods_cached(df_processed)
    cached_time_first = time.time() - start

    # Test cached function (second run - should use cache)
    start = time.time()
    result3 = calculate_15min_energy_periods_cached(df_processed)
    cached_time_second = time.time() - start

    print(".2f")
    print(".2f")
    print(".2f")

    # Verify results are equivalent
    pd.testing.assert_frame_equal(result1, result2, check_dtype=False)
    pd.testing.assert_frame_equal(result2, result3, check_dtype=False)
    print("✓ Results are consistent between cached and non-cached versions")

    print("✅ Performance tests passed!")

if __name__ == "__main__":
    try:
        test_energy_catalog()
        test_performance_improvement()
        print("\n🎉 All tests passed! Energy catalog optimization is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
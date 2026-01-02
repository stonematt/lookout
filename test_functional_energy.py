#!/usr/bin/env python3
"""
Test functional energy catalog approach with mocking.
"""

import pandas as pd
import sys
from unittest.mock import Mock, patch
from datetime import datetime, timezone

# Mock streamlit session state before importing functional modules
mock_st = Mock()
mock_st.secrets = {
    "lookout_storage_options": {
        "ACCESS_KEY_ID": "mock_key",
        "SECRET_ACCESS_KEY": "mock_secret", 
        "ENDPOINT_URL": "mock_endpoint"
    }
}
mock_st.session_state = {
    "device": {"macAddress": "TEST_MAC"}
}

# Mock streamlit at module level
sys.modules['streamlit'] = mock_st

def test_functional_energy_catalog():
    """Test functional energy catalog with mocked session state."""
    
    # Import after mocking
    from lookout.core.energy_catalog_func import (
        detect_and_calculate_periods,
        load_energy_catalog,
        save_energy_catalog,
        energy_catalog_exists
    )
    
    print("✅ Functional energy catalog imported successfully")
    
    # Test MAC address extraction indirectly via function calls
    exists = energy_catalog_exists()
    print("✅ MAC address extraction works via catalog check")
    
    # Test with sample data
    sample_data = pd.DataFrame({
        "datetime": pd.date_range("2023-01-01", periods=96, freq="15min", tz="America/Los_Angeles"),
        "solarradiation": [100.0 if 6 <= i // 4 <= 18 else 0.0 for i in range(96)]
    })
    
    # Test detect_and_calculate_periods (without auto_save)
    periods = detect_and_calculate_periods(sample_data, auto_save=False)
    
    assert isinstance(periods, pd.DataFrame)
    assert not periods.empty
    assert "period_start" in periods.columns
    assert "period_end" in periods.columns
    assert "energy_kwh" in periods.columns
    print("✅ detect_and_calculate_periods works")
    
    print("✅ All functional energy catalog tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_functional_energy_catalog()
        print("\n🎉 Functional energy catalog is working correctly!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
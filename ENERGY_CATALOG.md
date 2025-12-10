# Solar Energy Catalog System

## Overview

The Solar Energy Catalog system provides high-performance caching for 15-minute solar energy period calculations, eliminating the 30-second bottleneck that previously occurred on every page load.

### Key Benefits
- **30x faster loads**: <1s cached vs 30s uncached
- **Incremental updates**: Only processes new data since last period
- **Persistent storage**: Automatic saving to Storj with backup strategy
- **Session-first approach**: Instant loads after initial calculation

## Architecture

### Core Components

#### EnergyCatalog Class (`lookout/core/energy_catalog.py`)
- **Purpose**: Manages storage, retrieval, and updates of energy catalogs
- **Storage**: `{mac_address}.energy_catalog.parquet` in Storj bucket
- **Backup**: `backups/{mac_address}.energy_catalog_{timestamp}.parquet`
- **Methods**:
  - `catalog_exists()`: Check if catalog exists in storage
  - `load_catalog()`: Load existing catalog from storage
  - `detect_and_calculate_periods()`: Generate new catalog from data
  - `update_catalog_with_new_data()`: Incremental updates
  - `save_catalog()`: Persist to storage with backup

#### Optimized Algorithm (`lookout/core/solar_energy_periods.py`)
- **Vectorized operations**: Uses `pandas.resample()` instead of O(n²) iteration
- **Streamlit caching**: `@st.cache_data()` with 20-entry, 2-hour TTL
- **Timezone handling**: Proper UTC→Pacific conversion

#### Data Pipeline Integration (`lookout/core/data_processing.py`)
- **Auto-loading**: Loads catalog during app initialization
- **Incremental updates**: Updates with new archive data
- **Auto-save**: Saves when catalog >2 days old
- **User feedback**: Progress messages during operations

#### UI Management (`lookout/ui/diagnostics.py`)
- **Status display**: Shows catalog metrics and date range
- **Management controls**: Regenerate and save buttons
- **Details view**: Expandable catalog details

## Usage Patterns

### Automatic Operation
```python
# During app startup - automatic loading
if "energy_catalog" not in st.session_state:
    catalog = EnergyCatalog(device_mac)
    if catalog.catalog_exists():
        periods_df = catalog.load_catalog()
        # Update with any new data
        updated_df = catalog.update_catalog_with_new_data(history_df, periods_df)
        st.session_state["energy_catalog"] = updated_df
    else:
        # Generate new catalog
        periods_df = catalog.detect_and_calculate_periods(history_df)
        st.session_state["energy_catalog"] = periods_df
```

### Manual Management
```python
# From diagnostics UI
catalog = EnergyCatalog(device_mac)
periods_df = catalog.detect_and_calculate_periods(history_df, auto_save=False)
catalog.save_catalog(periods_df)
```

## Performance Characteristics

| Scenario | Time | Improvement |
|----------|------|-------------|
| First load (no cache) | ~30s | Same |
| Cached loads | <1s | **30x faster** |
| Incremental updates | <5s | **6x faster** |
| Algorithm only | ~6s | **5x faster** |

## Storage Strategy

### Primary Storage
- **Location**: `lookout` bucket in Storj
- **Format**: Parquet with snappy compression
- **Naming**: `{mac_address}.energy_catalog.parquet`

### Backup Strategy
- **Location**: `backups/` folder in same bucket
- **Naming**: `{mac_address}.energy_catalog_{timestamp}.parquet`
- **Trigger**: Automatic before overwriting existing catalog

### Auto-Save Policy
- **Threshold**: >2 days since last save
- **Trigger**: During data pipeline loading
- **Purpose**: Ensure data durability without excessive I/O

## Data Structure

### Catalog DataFrame
```python
period_start: datetime64[ns, America/Los_Angeles]  # Start of 15min period
period_end: datetime64[ns, America/Los_Angeles]    # End of 15min period
energy_kwh: float                                  # Energy in kWh
updated_at: datetime                               # Last update timestamp
catalog_version: str                               # Version identifier
```

### Source Data Requirements
```python
date: datetime64[ns, America/Los_Angeles]          # TZ-aware datetime
dateutc: int64                                     # Milliseconds timestamp
solarradiation: float                              # W/m²
```

## Troubleshooting

### Common Issues

#### "Missing required columns: ['datetime']"
**Cause**: Data doesn't have expected 'datetime' column
**Solution**: Ensure data has 'date' column (TZ-aware datetime)

#### "Energy catalog not loaded"
**Cause**: Catalog loading failed during startup
**Solution**: Check diagnostics tab for error details, try manual regeneration

#### Slow initial loads
**Cause**: First-time catalog generation
**Solution**: Expected behavior, subsequent loads will be cached

### Manual Recovery

#### Force catalog regeneration:
1. Go to Diagnostics tab
2. Click "🔄 Regenerate Energy Catalog"
3. Wait for completion message

#### Check catalog status:
1. Go to Diagnostics tab
2. View "Solar Energy Catalog Management" section
3. Check metrics and date range

#### Clear session cache:
1. Refresh browser page
2. Catalog will reload from storage

## Development Notes

### Testing
- **Unit tests**: `tests/test_energy_catalog.py` (13 tests)
- **Integration tests**: Solar UI and data pipeline tests
- **Performance tests**: `verify_energy_catalog.py` script

### Code Locations
- **Core logic**: `lookout/core/energy_catalog.py`
- **Algorithm**: `lookout/core/solar_energy_periods.py`
- **Pipeline**: `lookout/core/data_processing.py`
- **UI**: `lookout/ui/diagnostics.py`

### Future Enhancements
- **Compression optimization**: Further reduce storage size
- **Memory optimization**: Reduce memory footprint for large catalogs
- **Parallel processing**: Multi-threaded calculation for very large datasets
- **Advanced caching**: LRU eviction for memory-constrained environments
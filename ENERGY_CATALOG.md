# Solar Energy Catalog

The energy catalog stores 15-minute solar energy periods so the refresh run does not recompute them from the whole archive. The refresh run brings it up to the archive on every run and saves it when the stored copy is stale.

## Components

- **`EnergyCatalog`** (`lookout/core/energy_catalog.py`) manages storage, retrieval and updates:
  - `load_catalog()` reads the stored catalog once: the catalog when found, `None` when none is stored. A failed read raises `StorageReadError` (ADR-0004).
  - `detect_and_calculate_periods()` builds a catalog from archive data.
  - `update_catalog_with_new_data()` adds only the periods after the catalog's last one.
  - `save_catalog()` backs up the stored catalog, then replaces it.
- **Algorithm** (`lookout/core/solar_energy_periods.py`): vectorized with `pandas.resample()`, in station-local time.
- **Refresh run** (`lookout/cli/refresh.py`, with `lookout/core/energy_catalog_func.py`): loads the stored catalog (`load_catalogs`), updates it in memory every run, and saves only when the stored one is older than `ENERGY_SAVE_AGE` (2 days).
- **Manual** (`lookout/cli/diagnostics.py`): `diagnostics energy-catalog` reports the catalog's metrics and date range; `--save --dry-run` says what a save would do; `--save` backs up and replaces the stored catalog. Both read and write live Storj; see `docs/agents/operations.md` first.

## Storage

- **Catalog**: `<MAC>.energy_catalog.parquet` in the `lookout` bucket, parquet with snappy compression.
- **Backups**: `backups/<MAC>.energy_catalog_<timestamp>.parquet`, written before every overwrite.

## Data

Catalog columns:

```python
period_start: datetime64[ns, <station zone>]  # start of the 15-minute period
period_end: datetime64[ns, <station zone>]    # end of the 15-minute period
energy_kwh: float                             # energy in kWh
updated_at: datetime                          # last update
catalog_version: str                          # version identifier
```

Source columns read from the archive: `dateutc` (epoch ms) and `solarradiation` (W/m²).

## Tests

- `tests/test_energy_catalog.py`: the catalog class and algorithm
- `tests/test_refresh.py`: the load, update and save inside the refresh run

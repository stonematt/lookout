# AI Coding Guidelines for Lookout Weather Dashboard

## Archive Data Handling (AI Focus)

### Critical Column Name Rules

**NEVER assume these columns exist:**
- `datetime` - **DOES NOT EXIST** in archive data
- `timestamp` - Use `dateutc` for timestamps
- `date_only` - Use `date` (already datetime)

**ALWAYS use these columns:**
- `date` - Primary TZ-aware datetime column
- `dateutc` - UTC milliseconds timestamp
- `solarradiation` - Solar radiation data

### Timezone Handling Patterns

```python
# ✅ CORRECT: Use existing TZ-aware column
def process_archive_data(df):
    if 'date' in df.columns:
        # Already TZ-aware, no conversion needed
        time_col = df['date']
        return time_col

# ❌ WRONG: Creating non-existent datetime column
def process_archive_data_wrong(df):
    if 'dateutc' in df.columns:
        # DON'T DO THIS - 'date' column already exists as datetime
        df['datetime'] = pd.to_datetime(df['dateutc'], unit='ms', utc=True)
        return df['datetime']
```

### Data Validation Checklist

**Before accessing any column, AI agents MUST:**
```python
# ✅ REQUIRED: Check column existence
if 'date' in df.columns:
    time_data = df['date']
else:
    raise ValueError("Required 'date' column missing")

# ✅ REQUIRED: Check DataFrame state
if df.empty:
    return pd.DataFrame()  # Or appropriate empty response

# ✅ REQUIRED: Validate data types
if not pd.api.types.is_datetime64_any_dtype(df['date']):
    raise ValueError("'date' column must be datetime type")
```

### Common AI Pitfalls to Avoid

#### 1. Datetime Column Creation
```python
# ❌ DON'T create 'datetime' column
df['datetime'] = pd.to_datetime(df['dateutc'], unit='ms', utc=True)

# ✅ USE existing 'date' column
time_col = df['date']  # Already datetime and TZ-aware
```

#### 2. Timezone Conversion Errors
```python
# ❌ DON'T convert already-correct timezone
df['date'] = df['date'].dt.tz_convert('America/Los_Angeles')

# ✅ TRUST the existing timezone
# df['date'] is already America/Los_Angeles TZ-aware
```

#### 3. Column Name Confusion
```python
# ❌ DON'T use non-existent columns
start_time = df['datetime'].min()

# ✅ USE correct column names
start_time = df['date'].min()
```

### Test Data Creation Templates

#### Standard Archive Mock
```python
def create_test_archive_data():
    """Create test data matching actual archive structure"""
    dates = pd.date_range("2023-01-01", periods=100, freq="5min", tz="America/Los_Angeles")
    return pd.DataFrame({
        "date": dates,  # TZ-aware datetime (PRIMARY)
        "dateutc": [int(dt.timestamp() * 1000) for dt in dates],  # Milliseconds
        "solarradiation": [max(0, 800 * (1 - abs((i % 288) - 144) / 144)) for i in range(100)],
        "tempf": [70 + 10 * (i % 24) / 24 for i in range(100)],
        "dailyrainin": [0.0] * 100,
        "baromrelin": [30.0 + 0.1 * (i % 10) for i in range(100)],
    })
```

#### Minimal Solar Test Data
```python
def create_minimal_solar_test():
    """Minimal test data for solar functionality"""
    dates = pd.date_range("2023-01-01 06:00", periods=50, freq="15min", tz="America/Los_Angeles")
    return pd.DataFrame({
        "date": dates,
        "dateutc": [int(dt.timestamp() * 1000) for dt in dates],
        "solarradiation": [i * 50 for i in range(50)],  # Increasing solar radiation
    })
```

### Error Prevention Checklist

**AI agents MUST verify these before committing code:**

- [ ] Used `df['date']` instead of `df['datetime']`?
- [ ] Checked `if 'column' in df.columns` before access?
- [ ] Avoided redundant timezone conversions?
- [ ] Used actual archive column names from schema?
- [ ] Tested with real archive data structure?
- [ ] Handled empty DataFrames gracefully?

### Debugging Patterns for AI

#### When Time Operations Fail
```python
# Check column existence and types
print("Available columns:", df.columns.tolist())
print("Date column type:", df['date'].dtype if 'date' in df.columns else "MISSING")
print("DateUTC column type:", df['dateutc'].dtype if 'dateutc' in df.columns else "MISSING")

# Verify timezone
if 'date' in df.columns:
    print("Date timezone:", df['date'].dt.tz)
    print("Sample dates:", df['date'].head(3).tolist())
```

#### When Solar Data is Missing
```python
# Check solar data availability
if 'solarradiation' in df.columns:
    solar_count = df['solarradiation'].notna().sum()
    print(f"Solar readings: {solar_count}/{len(df)}")
    if solar_count == 0:
        print("WARNING: No solar data available")
else:
    print("ERROR: 'solarradiation' column missing")
```

### AI Code Generation Rules

**When writing code for Lookout, AI agents MUST:**

1. **Always check column existence** before accessing any column
2. **Use 'date' column** for all datetime operations
3. **Never create 'datetime' column** from 'dateutc'
4. **Trust existing timezones** - don't convert 'date' column
5. **Include error handling** for missing columns
6. **Test with actual data structure** before finalizing

### Performance Considerations

```python
# ✅ EFFICIENT: Use vectorized operations
df_filtered = df[df['date'] >= start_date]

# ❌ INEFFICIENT: Row-by-row processing
for idx, row in df.iterrows():
    if row['date'] >= start_date:
        # Process row
```

### Import Requirements

```python
# ✅ REQUIRED imports for archive handling
import pandas as pd
from typing import Optional

# ✅ OPTIONAL but recommended
import numpy as np
from datetime import datetime, timezone
```

This document is specifically optimized for AI coding agents to prevent the recurring `date`/`datetime`/`dateutc` confusion that has caused multiple implementation issues.
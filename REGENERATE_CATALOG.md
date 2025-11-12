# Regenerate Event Catalog

After switching from indices to timestamps (commit b3ec536), the event catalog needs to be regenerated to:
1. Remove stale `start_idx` and `end_idx` fields
2. Recalculate quality metrics using correct timestamp-based data extraction

## Command

```bash
cd /Users/mstone/src/github.com/stonematt/lookout

PYTHONPATH=. python -c "
from lookout.core.rain_events import RainEventCatalog
from lookout.storage.storj import get_s3_client
import pandas as pd
import io

print('Loading archive from Storj...')
client = get_s3_client()
response = client.get_object(Bucket='lookout', Key='98:CD:AC:22:0D:E5.parquet')
archive_df = pd.read_parquet(io.BytesIO(response['Body'].read()))
print(f'Loaded {len(archive_df)} records')

print('\\nBacking up existing catalog...')
catalog = RainEventCatalog('98:CD:AC:22:0D:E5', 'parquet')
backup_path = catalog.backup_catalog()
print(f'Backup saved: {backup_path}')

print('\\nRegenerating catalog with timestamp-based extraction...')
events_df = catalog.detect_and_catalog_events(archive_df, auto_save=True, backup_existing=False)

print(f'\\n✓ Regenerated catalog: {len(events_df)} events')
print(f'✓ Catalog saved to Storj')
print(f'\\nNew catalog schema (should NOT have start_idx/end_idx):')
print(events_df.columns.tolist())
"
```

## Expected Results

**Before regeneration:**
- Nov 10-17, 2024 event: Excellent quality, 100% complete, 0 min gap
- Catalog has `start_idx` and `end_idx` columns
- 4 events with mid-event resets show incorrect quality metrics

**After regeneration:**
- Nov 10-17, 2024 event: Poor quality, 79.5% complete, 1205 min gap (correct!)
- Catalog has NO `start_idx` or `end_idx` columns
- All events show accurate quality metrics based on actual data

## Verification

Check a few events to verify quality metrics:

```bash
PYTHONPATH=. python -c "
from lookout.storage.storj import get_s3_client
import pandas as pd
import io

client = get_s3_client()
response = client.get_object(Bucket='lookout', Key='98:CD:AC:22:0D:E5.event_catalog.parquet')
catalog = pd.read_parquet(io.BytesIO(response['Body'].read()))

print('Catalog schema:')
print(catalog.columns.tolist())
print()

# Check Nov 10-17 event (should be around index 125)
nov_events = catalog[catalog['start_time'].str.contains('2024-11-1')]
if len(nov_events) > 0:
    event = nov_events.iloc[0]
    print(f'Nov 10-17 event:')
    print(f'  Quality: {event[\"quality_rating\"]}')
    print(f'  Completeness: {event[\"data_completeness\"]*100:.1f}%')
    print(f'  Max gap: {event[\"max_gap_minutes\"]:.1f} min')
"
```

Expected output:
```
Quality: poor
Completeness: 79.5%
Max gap: 1205.0 min
```

## Auto-save Behavior

After regeneration, the catalog will auto-save to Storj when ongoing events complete.
This keeps historical catalog current while minimizing Storj writes during active rain.

**Triggers auto-save:**
- Last event in catalog has `ongoing=True` (rain event in progress)
- New archive data shows `eventrainin=0` (rain stopped)
- UI automatically saves completed event to Storj

**Does NOT auto-save:**
- Event still ongoing (`eventrainin > 0`) - only updates session state
- No new events detected - catalog unchanged
- New event starts - new event marked `ongoing=True`, not saved until complete

**Manual save:**
- Users can manually save anytime via "Save to Storj" button in Rain Events tab
- Useful for saving newly detected events or manual catalog regeneration

**Rationale:**
- Raw archive data is critical and persisted immediately by catchup CLI
- Event catalog is derived data that can be regenerated from archive
- Auto-saving completed events keeps historical catalog current
- Avoiding saves during active rain minimizes Storj I/O costs
- Typical behavior: 1-2 auto-saves per week during rainy season

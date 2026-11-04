# Wind visualization prototype

A throwaway prototype of six options for wind on the dashboard, drawn from the real archive. It lives on the `prototype/wind-viz` branch, which is never merged. It is here to inform triage of the wind issue.

- Published page: https://claude.ai/artifact/MUdeqXK8b4aZvYTrLpQSjK (private to the maintainer)
- Open locally: `open docs/prototypes/wind/wind-studies.html`

## Options

- **A. Compass**: direction now, the last hour's directions on the rim, the usual winter direction as a band. For the quick-fact tile.
- **B. Wind rose**: the last 7 days stacked by speed, against a season's outline, with calm in the centre. Switches between speed and gusts.
- **C. Speed against gusts, 24 hours**: the 15-minute average speed, with a tick up to each 15-minute peak gust, over the usual winter gust band, with downwind arrows by hour.
- **D. Today's top gust against winter days**: the range-bar grammar.
- **E. When it blows, by hour**: the last 7 days, summer and winter.
- **F. Quick-fact copy**: four sentences, no chart.

## What the data says

- Calm is normal. The median speed is 1.6 mph, and 21–33% of readings are 0 mph. A rose leaves calm out of its petals.
- Direction is steady. Every season leans E–SE, and summer adds a smaller WNW lobe.
- Gusts run about 1.9 times the average speed, so speed with gusts says more than speed alone.
- The by-hour pattern is summer-only: still nights and an afternoon breeze. Winter is flat.
- The instant `windspeedmph` is often 0 while `windspdmph_avg10m` is not. The snapshot carries only the instant fields.

## Rebuild

`wind_agg.py` reads a local archive parquet and prints the aggregates as JSON. The committed data came from `data/98-CD-AC-22-0D-E5.parquet` (Feb 2023 to Jan 5, 2026), so "now" is the last archived reading, not live.

```bash
.venv/bin/python docs/prototypes/wind/wind_agg.py data/98-CD-AC-22-0D-E5.parquet > docs/prototypes/wind/wind.json
python3 docs/prototypes/wind/build.py
```

The slate-teal wind colours are a proposal. They are not yet in the DESIGN.md contrast list, and the palest rose fill likely fails 3:1 on the tile.

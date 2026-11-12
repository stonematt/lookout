# 🌤️ Lookout

**A long-memory dashboard for your Ambient Weather station.** Lookout keeps
your station's complete record and shows what is happening outside now
compared with recent weeks and with this time in earlier years.

Live at [lookout.stonecover.com](https://lookout.stonecover.com). It runs from
February 2023 and has more than 370,000 readings at 5-minute resolution.

<!-- TODO: screenshot of the Overview, phone and desktop -->

---

## Why use it

The Ambient Weather dashboard shows you today and recent history. Lookout keeps
everything your station has ever recorded and does something with it.

- **Your full record, kept for you.** Lookout pages back through Ambient's
  history to build your station's archive. After that it catches up every
  5 minutes, including anything it missed while the cron host was down.
- **Almanac comparisons.** This month against the same month in earlier years,
  and today against this date in past years. Also rolling windows of 7, 30, 90
  and 365 days.
- **Rain as events, not just totals.** Each storm is a rain event in an Event
  Catalog, with its own rain curve and total. Each one is compared
  with your station's usual event. When it rains, the current event leads the
  page.
- **Solar energy over time.** Solar energy in kWh/m², tracked in an energy
  catalog across the whole record.
- **Honest about gaps.** Every station has missing periods. Lookout marks a
  comparison over a gap as not valid and never fills it in quietly.
- **More than one station.** Add every station on your Ambient account. A
  station selector in the header chooses which one the page shows.
- **Measured data only.** There is no forecast. Every number is something your
  station recorded.
- **Built for a phone.** It loads fast and reads in a glance, with dark and
  light themes. On a desktop, the same findings show as a grid of tiles.

The site has four tabs: **Overview**, **Rain**, **Rain Events** and **Solar**.

## How it works

```
Ambient Weather API ──► refresh run (cron, every 5 min, on any small Linux host)
                           │  catch up the archive, update the catalogs,
                           │  build the dashboard snapshot
                           ▼
                        S3-compatible storage (Storj)
                           │  archive, catalogs, snapshot JSON
                           ▼
                        static site + read functions (Vercel)
```

All the analysis runs in Python on the cron host. The site only reads the
snapshot and draws it ([ADR-0003](docs/adr/0003-vps-cron-precomputes-vercel-only-reads.md)).
There is no database and no server-side rendering. The running cost is a small
VPS, a few GB of object storage, and Vercel's free tier.

Your data is safe from the code that writes it:

- Only one archive writer runs at a time. A second writer stops with an error
  that names the lock file.
- A save that would drop or blank a stored reading is refused
  (`ArchiveLossError`).
- The archive is backed up daily. The catalogs are backed up before each write.
- A failed storage read stops the run. It is never treated as "no data"
  ([ADR-0004](docs/adr/0004-failed-storage-read-aborts.md)).
- Analysis runs in your station's local time. The time zone comes from
  Ambient, so you never set it
  ([ADR-0002](docs/adr/0002-station-local-time-as-analytical-frame.md),
  [ADR-0005](docs/adr/0005-station-time-zone-from-ambient-per-build.md)).

---

## Run your own

### What you need

- An Ambient Weather station and an
  [Ambient API key and application key](https://ambientweather.docs.apiary.io/).
- An S3-compatible bucket. Lookout is built and run on
  [Storj](https://www.storj.io/). Name the bucket `lookout`, because that is the
  default every command uses.
- A Linux host for the cron job, with Python 3.12 and
  [uv](https://docs.astral.sh/uv/).
- A [Vercel](https://vercel.com/) account for the site.
- Optional: a [Healthchecks.io](https://healthchecks.io/) project to alert you
  when the cron stops. Without a key, the pings are skipped.

### 1. Install

```bash
git clone https://github.com/stonematt/lookout.git
cd lookout
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Add your secrets

```bash
mkdir -p ~/.config/lookout
cp secrets.example.toml ~/.config/lookout/secrets.toml
chmod 600 ~/.config/lookout/secrets.toml
```

Fill in the Ambient keys and the storage keys. Every name can also be set as an
env var of the same name (`AMBIENT_API_KEY`, `STORJ_ENDPOINT_URL`, ...), and an
env var wins over the file.

### 3. Add your stations

List the stations on your Ambient account:

```bash
python -m lookout.cli.stations scan
```

Then add each one you want, by its MAC:

```bash
python -m lookout.cli.stations add <MAC>
```

`add` builds the station's first archive from Ambient's history, paging back
until Ambient has no older readings. This takes a while for a station with
years of data. The first station you add is the default station, the one the
site shows first. The stations are kept in a station inventory on your bucket
([ADR-0006](docs/adr/0006-station-inventory-on-storj-written-by-hand.md)).

### 4. Run the refresh once, then schedule it

```bash
python -m lookout.cli.refresh
```

This catches up every station, builds the catalogs and publishes the snapshot.
When it ends with one "published" line per station, add it to your crontab:

```cron
*/5 * * * * cd /path/to/lookout && lookout/bin/cron_run.sh lookout/cli/refresh.py >> ~/refresh.log 2>&1
```

The run takes its own lock, so the crontab needs no `flock`.

### 5. Deploy the site

Import the repo into Vercel with **`web`** as the root directory. Set these
environment variables:

| Variable                                         | Value                                     |
| ------------------------------------------------ | ----------------------------------------- |
| `STORJ_ACCESS_KEY_ID`, `STORJ_SECRET_ACCESS_KEY` | Storage keys (read access is enough)      |
| `STORJ_ENDPOINT_URL`                             | Your S3 gateway URL                       |
| `STORJ_BUCKET`                                   | `lookout`                                 |
| `AMBIENT_API_KEY`, `AMBIENT_APPLICATION_KEY`     | Ambient keys, used for the live reading   |

The site reads its station list from the snapshot, so you never configure
stations on Vercel. `/status` returns the last refresh run's outcome as JSON.

---

## Develop

```bash
uv pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -q          # no secrets needed
cd web && npm ci && npm run dev               # the site on test fixtures, no secrets
cd web && npm run format:check && npm run lint && npm run build && npm test
```

The dev server serves fixed test snapshots. Add `?fixture=raining` to see a
rain event, or `?theme=dark` to see the dark theme. [`web/README.md`](web/README.md)
lists the other switches.

Where to read more:

- [`CONTRIBUTING.md`](CONTRIBUTING.md): the workflow and how tests are written.
- [`CLAUDE.md`](CLAUDE.md): the code rules and the checks CI runs. It is written
  for coding agents, but people can use it too.
- [`GLOSSARY.md`](GLOSSARY.md): the domain terms, such as rain event, Event
  Catalog and dashboard snapshot.
- [`docs/adr/`](docs/adr/): the architecture decisions.
- [`DESIGN.md`](DESIGN.md) and [`design-system/`](design-system/): the visual
  design.

Built with Python (pandas, PyArrow, boto3) and a Vite + TypeScript site with
Plotly.js.

---

## Author and license

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Z8Z41G13PX)

Maintained by [@stonematt](https://github.com/stonematt).
Licensed under the [MIT License](LICENSE).

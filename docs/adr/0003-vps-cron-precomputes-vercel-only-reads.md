# A VPS cron precomputes the dashboard; Vercel only reads

## Status

accepted (2026-10-30). Amended the same day: the snapshot is a JSON file
on Storj, not a Neon table. Decided in "How do secrets and state work
without Streamlit?" (#56).

## Context

Lookout is moving off Streamlit. Today every page open reads the live
`lastData` from Ambient and, if the history is more than 6 minutes old,
fetches the gap and reruns the pandas pipeline in the Streamlit process.
The new front end is a static site, so that work needs a new home.

Measured on a laptop (2026-10-29), one full run of the derived pipeline
takes about 15 s and peaks near 1.2 GB RSS. About 6-7 s of that is
downloading the 12.8 MB archive and the 2.9 MB energy catalog from Storj.

## Decision

1. A cron on the existing VPS runs every 5 minutes. Each run does catchup,
   then the full derived pipeline (Event Catalog update and its lazy
   auto-save, rain and solar aggregations), then publishes a dashboard
   snapshot and a small context block as JSON files on Storj, under a
   `snapshot/` prefix next to the archive.
2. The front end is Vite + TypeScript with no UI framework, charts in
   Plotly.js, built to static files.
3. Vercel (Hobby) serves the static build and thin read functions on one
   domain. No pandas runs on Vercel.
   - `/dashboard` returns the snapshot, with
     `Cache-Control: s-maxage=300, stale-while-revalidate=300` so the
     CDN serves it and Storj is read about once per 5 minutes per region.
   - `/current` returns the live `lastData` plus the context block. The
     page header and any widget read it.
4. The Storj archive and its parquet layout are unchanged. The cron is
   the only writer to Storj. Vercel holds a separate Storj credential
   that is read-only and limited to the `snapshot/` prefix.
5. An off-VPS monitor must alert when the snapshot goes stale or the cron
   stops reporting, because the VPS is now the only source of freshness.
   It is two Healthchecks.io heartbeats with email alerts:
   - A cron heartbeat, pinged by `cron_run.sh` before the run and with
     its exit status after the snapshot is published. Period 5 min,
     grace 10 min.
   - A station heartbeat, pinged by the run only while the station's
     live reading is under 15 min old. Period 5 min, grace 30 min. Its
     slug is the station's human-readable name, not its MAC address.
   The page header shows the age of the newest observation and turns
   stale past 15 min. `/current` carries the observation time and the
   snapshot build time.
6. Storj data, the snapshot and `/current` are keyed by station MAC
   address, so a later multi-station effort extends them rather than
   breaking them. Multi-station support itself is out of scope here.

## Consequences

- The UI no longer needs a "Save to Storj" button: Event Catalog
  persistence happens in the cron.
- A dead VPS means a dashboard that stops updating while still serving.
  The monitor and a visible data age are what catch it.
- A healthy cron with an offline station also gives an old dashboard.
  The separate station heartbeat tells that case apart from a dead VPS.
- Renaming a station in Ambient creates a new station heartbeat, and the
  old one alerts as down. That is accepted.
- A 5-minute cron re-downloads the archive on each run, about 140 GB a
  month of Storj egress, or about $1. That is accepted. A local archive
  copy on the VPS, checked by ETag, can remove it later.
- Storj GET latency is not on the user's path. Measured from a laptop
  (2026-10-30): median 343 ms for 50 KB and about 600 ms for 0.5-2 MB.
  The CDN cache hides it, and stale-while-revalidate covers cold regions.
- Sorting, filtering and reshaping the snapshot happen in the browser.
  The pandas pipeline that builds it stays in Python.

## Considered alternatives

- **Per-request pandas on Vercel behind stale-while-revalidate.**
  Workable: Hobby allows 300 s and 2 GB. Rejected because it carries the
  pandas cold start, the memory headroom, and the 500 MB bundle limit on
  every refresh, and a widget's short fetch window can't wait on it.
- **Cron precompute at a long interval.** Rejected: it can't match
  today's 6-minute freshness.
- **GitHub Actions as the scheduler.** Rejected: scheduled workflows are
  disabled after 60 days without repo activity, which this repo has hit
  five times, and GitHub's terms limit Actions to developing and testing
  the project. A well-known keepalive tool's repo was disabled by GitHub.
- **Move the archive to Postgres.** Rejected: Storj costs about $0.38 a
  month, and a 5-minute cron would keep a Neon compute awake past the
  free plan's compute hours. Out of scope for the Streamlit migration.
- **Publish the snapshot to a Neon table.** The first version of this
  ADR. Rejected: it added a database, two Postgres roles and a
  connection string per host, only to hold one small document. A cached
  JSON file on the existing store does the same job.
- **Public Storj linkshare fetched by the browser.** Rejected: no CDN
  cache in front of Storj, and CORS must be configured on the bucket.
- **Cloudflare Python Workers.** Rejected in the hosting research: 128 MB
  memory and no boto3.

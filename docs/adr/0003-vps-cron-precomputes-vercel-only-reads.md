# A VPS cron precomputes the dashboard; Vercel only reads

## Status

accepted (2026-10-30)

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
   snapshot and a small context block to a Neon table.
2. The front end is Vite + TypeScript with no UI framework, charts in
   Plotly.js, built to static files.
3. Vercel (Hobby) serves the static build and thin read functions on one
   domain. No pandas runs on Vercel.
   - `/dashboard` returns the snapshot.
   - `/current` returns the live `lastData` plus the context block. The
     page header and any widget read it.
4. The Storj archive and its parquet layout are unchanged. The cron is
   the only writer to Storj, so Vercel needs read access to Neon only.
5. An off-VPS monitor must alert when the snapshot goes stale or the cron
   stops reporting, because the VPS is now the only source of freshness.

## Consequences

- The UI no longer needs a "Save to Storj" button: Event Catalog
  persistence happens in the cron.
- A dead VPS means a dashboard that stops updating while still serving.
  The monitor and a visible data age are what catch it.
- A 5-minute cron re-downloads the archive on each run, about 140 GB a
  month of Storj egress, or about $1. That is accepted. A local archive
  copy on the VPS, checked by ETag, can remove it later.

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
- **Cloudflare Python Workers.** Rejected in the hosting research: 128 MB
  memory and no boto3.

# Lookout product brief

The product lead reads this on every run. Keep it short: point at the docs that already say a thing rather than restating it.

## Goals

- A glance on a phone gives current conditions in the context of recent history.
- Deeper views for rain, Rain Events and solar energy, each stating its finding.
- The dashboard runs off Streamlit: a VPS cron precomputes it, Vercel serves it (ADR-0003).

Product context: `PRODUCT.md` (users, principles, constraints), `DESIGN.md` (The Annotated Almanac), `GLOSSARY.md`, `docs/adr/`.

## Initiatives

One line each. The product lead infers the rung; write one here only when Matt overrides it.

- **Move Lookout off Streamlit** — replace the Streamlit app with a cron-built snapshot on Storj and a Vite + TypeScript + Plotly.js site on Vercel · containers: map #51, spec #66 and its sub-issues #67–#86, bug #63 (sub-issue of #66)

## Where the tickets live

Tracker: GitHub Issues on `stonematt/lookout`. Labels: one category (`bug`, `enhancement`) and one state (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`); see `docs/agents/`. When a fix leaves bad data in Storj, the cleanup is its own `ready-for-human` ticket, blocked by the fix; the product lead queues it as a separate session once the blockers merge (example: #143 after #132). Specs: issues titled `Spec: …`, tickets as native sub-issues. Maps: issues labelled `wayfinder:map`. No milestones.

Integration branch: `dev` · release branch: `main` · deploy branch: `live` (Streamlit Cloud serves it until #86 retires Streamlit). Research notes: `docs/research/`, wayfinder research on unmerged `research/<name>` branches.

## Drift

| Last architecture review | Date | Threshold |
|---|---|---|
| `007d960` (baseline at setup; no review yet) | 2026-10-30 | 30 files or 1,500 lines |

Threshold history: none yet.

## Vault page

Vault: `tyee` · path: `product-leads/lookout.md` (director convention, 2026-10-30) · area: none

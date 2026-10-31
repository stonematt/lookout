# Lookout web

The static site and its two read functions, deployed on Vercel. The VPS
cron builds the snapshot in Python and publishes it on Storj; this side only
reads it (ADR-0003, `docs/adr/0003-vps-cron-precomputes-vercel-only-reads.md`).
Vite and TypeScript with no UI framework. Plotly.js is installed for the tab
charts. The design system is `DESIGN.md` at the repo root.

## Layout

`web/` is the Vercel project root.

| Path | What it is |
| --- | --- |
| `index.html`, `src/main.ts` | The page: header plus tab shell, fed by `/dashboard` and then `/current` |
| `src/snapshot.ts` | The snapshot contract types and `parseDashboard` / `parseContext` / `parseCurrent`, shared by the browser and the functions |
| `src/load.ts` | `fetchDashboard` / `fetchCurrent` from the browser |
| `src/header.ts` | `mountHeader`: station name, reading time, data age (stale past 15 min) |
| `src/tabs.ts` | `mountTabs`: the four sticky tabs, roving tabindex, URL hash |
| `src/format.ts` | The One Format Rule: `formatStamp`, `formatDuration`, `dataAge` |
| `src/theme.ts` | The `?theme=light\|dark` override (the default follows the OS) |
| `src/styles/tokens.css` | DESIGN.md tokens, light plus dark |
| `src/styles/base.css` | Base type, page column, header, tabs |
| `api/dashboard.ts`, `api/current.ts` | The Vercel functions; `vercel.json` rewrites `/dashboard` and `/current` to them |
| `server/handlers.ts` | `handleDashboard` / `handleCurrent`, with no Vercel or env in them |
| `server/storj.ts`, `server/ambient.ts` | The Storj (SigV4 via `aws4fetch`) and Ambient readers |
| `server/env.ts` | Builds the readers from `process.env` and answers 500 naming unset variables |
| `server/dev-fixtures.ts` | The lane server's `/dashboard` and `/current`, served from the goldens |
| `tests/` | Vitest. `tests/golden.ts` loads `tests/golden/snapshots/` from the repo root, in place |
| `scripts/shots.mjs` | Headless screenshots at 390px and 1280px, light and dark |

## Commands

```bash
cd web
npm ci                 # install from the lockfile
npm run dev -- --port 5177   # the lane server (below)
npm run typecheck      # tsc --noEmit
npm run build          # typecheck, then vite build into dist/
npm test               # vitest run
npm run shots -- http://localhost:5177 <out-dir> "" "fixture=raining&age=3"
```

The web gate is `npm run build && npm test`. `build` runs the typecheck.
`npm run shots` needs the browser once: `npx playwright install chromium`.

## The lane server

```bash
cd web && npm run dev -- --port <port>
```

This serves this worktree's `web/` on that port. `/dashboard` and `/current`
come from the golden snapshot documents in `tests/golden/snapshots/`, run
through the real handlers. It needs no secrets and makes no network calls.
Stop it with Ctrl-C, or `kill` its PID.

- `?fixture=dry|raining` on the page picks the golden. The default is `dry`,
  or whatever `LOOKOUT_FIXTURE` names.
- `?age=<minutes>` shifts every `_ms` time so that the newest observation is
  that many minutes old (or set `LOOKOUT_AGE_MIN`). Without it the goldens are
  served as recorded, which means long stale.
- `?theme=light|dark` forces a theme.
- `/current` has `last_data: null` here, because Ambient is never called in
  dev.

Example: `http://localhost:5177/?fixture=raining&age=3#rain`.

## Read functions

Both take an optional `?station=<MAC>`. Without it they use `STATION_MAC`.

- `GET /dashboard`: the station's `snapshot/<MAC>.dashboard.json`, as Storj
  holds it, with `Cache-Control: s-maxage=300, stale-while-revalidate=300`.
  It answers 404 when the station has no snapshot, 502 when Storj fails or the
  document breaks the contract, 400 for a malformed station and 500 when an
  env var is unset.
- `GET /current`: `{schema_version, kind: "current", station, observed_at_ms,
  built_at_ms, last_data, context}`. `last_data` is Ambient's live `lastData`,
  and `context` is `snapshot/<MAC>.context.json`. Either can be null when its
  source fails; it answers 502 only when both fail. `observed_at_ms` is the
  newer of the live `dateutc` and the context's observation. When both halves are
  read, it is cached with `s-maxage=60, stale-while-revalidate=60` so that
  Ambient is called at most about once a minute per region. A partial body
  is `no-store`, so the CDN does not keep serving a blip.

## Environment variables

These are set on Vercel for Production and Preview. `.env.example` lists them
with no values.

| Name | Used by | Value |
| --- | --- | --- |
| `STORJ_ACCESS_KEY_ID` | both | read-only S3 credential, limited to `sj://lookout/snapshot/` |
| `STORJ_SECRET_ACCESS_KEY` | both | same credential |
| `STORJ_ENDPOINT_URL` | both | that credential's S3 gateway endpoint |
| `STORJ_BUCKET` | both | `lookout` |
| `STATION_MAC` | both | the station's MAC address, the same value as on the VPS |
| `AMBIENT_API_KEY` | `/current` | Vercel's own Ambient key, or the VPS key |
| `AMBIENT_APPLICATION_KEY` | `/current` | the existing application key |
| `AMBIENT_ENDPOINT` | `/current` | optional, default `https://rt.ambientweather.net/v1` |

`HC_PING_KEY` is for the VPS only and is not set on Vercel.

## Deploy checklist (owner)

1. In Storj, create an S3 access grant that is **read-only** and limited to
   `sj://lookout/snapshot/`. Keep its access key, secret and gateway endpoint.
2. In Vercel (account `mstone-3228`, team "Matt Stone's projects"), import
   `stonematt/lookout` as a new project. Set **Root Directory** to `web`.
   The framework preset is Vite, which `vercel.json` also sets. Set the
   production branch.
3. Add the environment variables above for Production and Preview.
4. Deploy. Check `https://<domain>/dashboard` (JSON with the cache header),
   `https://<domain>/current` (JSON with `last_data` not null) and the page
   (station name, reading time, a fresh data age).
5. Make sure the production URL is public: Deployment Protection must not
   cover Production. Preview deployments stay behind Vercel login, which is
   the Hobby default.

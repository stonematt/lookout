# Lookout web

The static site and its read functions, deployed on Vercel. The VPS
cron builds the snapshot in Python and publishes it on Storj; this side only
reads it (ADR-0003, `docs/adr/0003-vps-cron-precomputes-vercel-only-reads.md`).
Vite and TypeScript with no UI framework. Plotly.js is installed for the tab
charts. The design system is `DESIGN.md` at the repo root.

## Layout

`web/` is the Vercel project root.

| Path                                   | What it is                                                                                                                    |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `index.html`, `src/main.ts`            | The page: header plus tab shell, fed by `/dashboard`, then `/current`, and `/stations`; choosing a station writes `?station=` |
| `debug.html`, `src/debug.ts`           | The debug page: the raw `/current` answer for `?station=`, as tiles plus the body as received, with the station selector      |
| `src/snapshot.ts`                      | The snapshot contract types and `parseDashboard` / `parseContext` / `parseCurrent`, shared by the browser and the functions   |
| `src/stations.ts`                      | The station index contract: `parseStationIndex`, `findStation`, `shownStation` (the default station when none is named)       |
| `src/load.ts`                          | `fetchDashboard` / `fetchCurrent` / `fetchStations` from the browser                                                          |
| `src/header.ts`                        | `mountHeader`: station name (the station selector with two or more stations), reading time, data age, archive start           |
| `src/tabs.ts`                          | `mountTabs`: the four sticky tabs, roving tabindex, URL hash                                                                  |
| `src/format.ts`                        | The One Format Rule: `formatStamp`, `formatDuration`, `dataAge`                                                               |
| `src/theme.ts`                         | The `?theme=light\|dark` override (the default follows the OS)                                                                |
| `src/styles/tokens.css`                | DESIGN.md tokens, light plus dark                                                                                             |
| `src/styles/base.css`                  | Base type, page column, header, tabs                                                                                          |
| `api/*.ts`                             | The Vercel functions; `vercel.json` rewrites `/dashboard`, `/current`, `/event`, `/status` and `/stations` to them            |
| `server/handlers.ts`                   | `handleDashboard` / `handleCurrent` / `handleStatus`, with no Vercel or env in them                                           |
| `server/storj.ts`, `server/ambient.ts` | The Storj (SigV4 via `aws4fetch`) and Ambient readers                                                                         |
| `server/env.ts`                        | Builds the readers from `process.env` and answers 500 naming unset variables                                                  |
| `server/dev-fixtures.ts`               | The lane server's read functions, served from the goldens                                                                     |
| `tests/`                               | Vitest. `tests/golden.ts` loads `tests/golden/snapshots/` from the repo root, in place                                        |
| `scripts/layout-check.mjs`             | Layout facts on every tab: sideways scroll, chart width, touching stat values (`npm run layout-check`)                        |
| `scripts/shots.mjs`                    | Headless screenshots at 390px and 1280px, light and dark                                                                      |

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

`npm run layout-check` builds with `VITE_LANE_SERVER=1`, serves the build itself
and visits every tab at 320, 390, 760 and 1440px, light and dark, on the `dry`
and `raining` goldens and a `live` case (the raining golden with every Rain and
Solar window valid and holding wide, 5-digit values, as live data does). It
exits 1, naming the view, on page sideways scroll, a chart narrower than the
box that holds it, or stat values whose text boxes touch. It takes about a
minute and is a CI step. `-- --prove` adds the pre-fix Solar window CSS (#395)
to the `live` case and passes only if the check then fails there (a failure in
any other case still fails the run). Given a base URL, `node scripts/layout-check.mjs
http://localhost:5177` checks that running server (a `VITE_LANE_SERVER=1`
build under `vite preview`, or the lane server) and skips the build and preview.

## The lane server

```bash
cd web && npm run dev -- --port <port>
```

This serves this worktree's `web/` on that port. `/dashboard`, `/current` and
`/event` come from the golden snapshot documents in `tests/golden/snapshots/`,
and `/status` from `tests/golden/status/`, run through the real handlers. It needs no secrets and makes no network calls.
Stop it with Ctrl-C, or `kill` its PID.

- `?fixture=dry|raining|seasoned` on the page picks the golden (`seasoned` has
  two earlier years around its date). The default is `dry`,
  or whatever `LOOKOUT_FIXTURE` names.
- `?age=<minutes>` shifts every `_ms` time so that the newest observation is
  that many minutes old (or set `LOOKOUT_AGE_MIN`). Without it the goldens are
  served as recorded, which means long stale. Only the `_ms` times move.
  Decided values such as month labels and window texts (the Rain by month
  grid, the "so far" sentences) keep the date the golden was built, so a
  shifted date can sit beside another month's label. A real snapshot boxes
  the reading's own month.
- `?theme=light|dark` forces a theme.
- `/current` has `last_data: null` here, because Ambient is never called in
  dev.

Example: `http://localhost:5177/?fixture=raining&age=3#rain`.

Vercel deploys only `main`, `goal/**` and `preview/**` branches. `git.deploymentEnabled` in `vercel.json` stops the other named prefixes (`dev`, `feat/**`, `fix/**`, `docs/**`, `chore/**`, `lane/**`, `shots/**`, `prototype/**`) from creating a deployment, because a skipped build still counts toward the Hobby deployment budget. `ignoreCommand` then runs `scripts/vercel-ignore.sh`: `main` and `preview/**` always build, a `goal/**` push builds only when it changed `web/` since the last deployment, and any other branch skips. Push to `preview/<slug>` to get a preview of other work.

## Security headers

`vercel.json` sets `Content-Security-Policy`, `X-Content-Type-Options`,
`Referrer-Policy` and `X-Frame-Options` on every path. Scripts, fetches,
fonts and images come from this origin only; styles also allow
`'unsafe-inline'`, because Plotly injects a `<style>` element and sets style
attributes, as does the temperature chart's gradient. Neither `vite` nor
`vite preview` applies `vercel.json`, so check the page against the policy
with the built site:

```
VITE_LANE_SERVER=1 npm run build && npx vite preview --port 5177 --strictPort
node scripts/csp-check.mjs http://localhost:5177 [screenshot-dir]
```

It opens every tab at 390px and 1280px in both themes over both goldens,
draws a live and a completed Rain Event's charts, and fails on any CSP
violation. `LOOKOUT_NO_HEADERS=1` runs it without the headers, for baseline
screenshots. `vite preview` serves the goldens the same way the dev server
does. The build needs `VITE_LANE_SERVER=1`: without it the page, as
deployed, does not send `fixture` or `age`, which the deployed functions reject.

## Read functions

The CDN keys its cache on the whole URL, so each read function reads Storj or
Ambient only at one URL per request: the bare path, plus its own parameters
in one spelling (`/event`'s `?start=<ms>`, then `station`). Before any read,
an unknown or repeated query parameter answers 400 (uncached), and another
spelling of the same request answers 308 to the one URL. `fixture` and `age`
are the lane server's only: the deployed functions refuse them.

`/dashboard`, `/current`, `/event` and `/status` serve one station (#272): the
one `?station=<MAC>` names, if the station index lists it, or the default
station (the index's first) without it. The one spelling of `station` is the
MAC in upper case, URL-encoded (`?station=98%3ACD%3AAC%3A22%3A0D%3AE5`); any
other spelling is a 308 to it. A malformed MAC is a 400 and a MAC the index
does not list a 404, both uncached; a station in the index with no snapshot
yet is a 404 cached `s-maxage=15`, so the next 5-minute refresh run fills it.
The default station named outright is served at its own URL, not redirected
to the bare path. Each function instance reads the station index at most
once a minute and keeps the last good copy through a failed read; before any
good read, a station read answers 502 (`s-maxage=15`), and before the
refresh run has published an index, 404 (`s-maxage=15`).

- `GET /dashboard`: the station's `snapshot/<MAC>.dashboard.json`, as Storj
  holds it, with `Cache-Control: s-maxage=300, stale-while-revalidate=300,
stale-if-error=300`. It answers 404 when the station has no snapshot, 502
  when Storj fails or the document breaks the contract (both `s-maxage=15`),
  and 500 when an env var is unset.
- `GET /current`: `{schema_version, kind: "current", station, observed_at_ms,
built_at_ms, last_data, context}`. `last_data` is Ambient's live `lastData`,
  and `context` is `snapshot/<MAC>.context.json`. Either can be null when its
  source fails; it answers 502 only when both fail. `observed_at_ms` is the
  newer of the live `dateutc` and the context's observation. When both halves are
  read, it is cached with `s-maxage=60, stale-while-revalidate=60,
stale-if-error=300` so that Ambient is called at most about once a minute
  per region. A partial body (an Ambient 429 included) and the 502 are cached
  with `s-maxage=15`, so that a rate-limit episode does not feed itself with
  a call per visitor, and a blip is not served on for long.

- `GET /stations`: the station index, `snapshot/stations.json`, as the
  refresh run publishes it last on every run (#272): `{schema_version: 1,
kind: "station_index", built_at_ms, stations: [{mac, name, tz,
newest_reading_ms, archive_start_ms}]}`, in station inventory order, the
  first being the default station. It has its own `schema_version`, outside
  the dashboard contract. It takes no parameters. Cached like `/dashboard`;
  404 before the refresh run has published one, 502 when Storj fails or the
  index breaks its contract (both `s-maxage=15`).
- `GET /status`: the refresh run's status document,
  `snapshot/<MAC>.status.json`, as Storj holds it (#268): `{kind: "status",
station_mac, version, sha, started_at_ms, ended_at_ms, outcome, error,
built_at_ms}`. The refresh run writes it at the end of every run, success or
  failure: `outcome` is `ok` or `failed`, `error` the failure's exception
  class name (never its message), `sha` the checkout's `git rev-parse HEAD`
  (null when git cannot say) and `built_at_ms` the published snapshot's (null
  when the run published none). It is outside the snapshot contract and the
  page never reads it. Cached `s-maxage=60, stale-if-error=300`; 404 before
  any run has written one, 502 when Storj fails or holds something other than
  a status document (both `s-maxage=15`).

Vercel's CDN does not cache a 5xx response, so a 502's `s-maxage=15` holds
only for other caches; there, `stale-if-error` lets the CDN keep serving the
last good copy instead. Each Ambient and Storj fetch times out after 5
seconds, so a hung upstream fails the read rather than stalling it.

## Environment variables

These are set on Vercel for Production and Preview. `.env.example` lists them
with no values.

| Name                      | Used by    | Value                                                        |
| ------------------------- | ---------- | ------------------------------------------------------------ |
| `STORJ_ACCESS_KEY_ID`     | both       | read-only S3 credential, limited to `sj://lookout/snapshot/` |
| `STORJ_SECRET_ACCESS_KEY` | both       | same credential                                              |
| `STORJ_ENDPOINT_URL`      | both       | that credential's S3 gateway endpoint                        |
| `STORJ_BUCKET`            | both       | `lookout`                                                    |
| `AMBIENT_API_KEY`         | `/current` | Vercel's own Ambient key, or the VPS key                     |
| `AMBIENT_APPLICATION_KEY` | `/current` | the existing application key                                 |
| `AMBIENT_ENDPOINT`        | `/current` | optional, default `https://rt.ambientweather.net/v1`         |

`HC_PING_KEY` is for the VPS only and is not set on Vercel. There is no
`STATION_MAC` (#272): the station index names the stations and the default
one, so a station added to the station inventory is served after its first
refresh run, with no Vercel change.

## Deploy checklist (owner)

1. In Storj, create an S3 access grant that is **read-only** and limited to
   `sj://lookout/snapshot/`. Keep its access key, secret and gateway endpoint.
2. In Vercel (account `mstone-3228`, team "Matt Stone's projects"), import
   `stonematt/lookout` as a new project. Set **Root Directory** to `web`.
   The framework preset is Vite, which `vercel.json` also sets. Set the
   production branch.
3. Add the environment variables above for Production and Preview.
4. Deploy. Check `https://<domain>/stations` (the station index),
   `https://<domain>/dashboard` (JSON with the cache header),
   `https://<domain>/current` (JSON with `last_data` not null),
   `https://<domain>/status` (the last refresh run's outcome) and the page
   (station name, reading time, a fresh data age).
5. Make sure the production URL is public: Deployment Protection must not
   cover Production. Preview deployments stay behind Vercel login, which is
   the Hobby default.

# Research: where can Lookout's Python compute and front end be hosted?

Ticket: #52 (map #51). Input to the stack and hosting decision in #55; not the decision itself.
Researched 2026-10-26 from official docs. Items marked (inferred) are my reasoning, not a documented claim.

## Constraints recap

pandas + pyarrow; read/write parquet and JSON on Storj (S3 API); server-side Storj write; public no-auth URL; public JSON current-conditions feed; mostly phone use; near-zero cost; existing Vercel, Neon, Cloudflare accounts, no Python host. Current `requirements.txt` has pandas<3, numpy, plotly, boto3==1.34.99, streamlit.

## Host facts

### Vercel Python functions
- Runs ASGI/WSGI apps (FastAPI, Flask, Django) as Vercel Functions; Python 3.12 (default), 3.13, 3.14. [Python runtime](https://vercel.com/docs/functions/runtimes/python)
- Bundle limit is 500 MB uncompressed for Python (250 MB for other runtimes); up to 5 GB with Large Functions (beta). pandas + numpy + pyarrow + boto3 should fit comfortably (inferred; typical combined install is well under 500 MB). [Python runtime](https://vercel.com/docs/functions/runtimes/python), [limits](https://vercel.com/docs/functions/limitations)
- Memory: Hobby 2 GB / 1 vCPU. Max duration Hobby 300 s. Request/response body max 4.5 MB. [limits](https://vercel.com/docs/functions/limitations)
- Hobby included usage (guideline): 1,000,000 invocations, 4 h Active CPU, 360 GB-hrs provisioned memory. [fair use](https://vercel.com/docs/limits/fair-use-guidelines)
- Hobby is restricted to non-commercial personal use. A personal weather dashboard fits. [fair use](https://vercel.com/docs/limits/fair-use-guidelines)
- Serverless, so no persistent process: Storj reads happen per cold instance; in-memory caches survive only while an instance is warm (inferred). This matters for the lazy Event Catalog.
- "Services" lets a Python backend and a frontend share one project and domain. [Python runtime](https://vercel.com/docs/functions/runtimes/python)
- Secrets: project environment variables.
- Cold start: no published number. Vercel precompiles `.pyc` into the bundle to cut init time. Importing pandas + pyarrow typically costs about a second or more (inferred; measure before deciding).
- Fit: same account and dashboard as the existing TS sites. Best fit of the options.

### Cloudflare Python Workers
- Python Workers install pure-Python and Pyodide packages; FastAPI is supported. [packages](https://developers.cloudflare.com/workers/languages/python/packages/)
- Pyodide ships numpy 2.4.6, pandas 3.0.2, pyarrow 22.0.0; boto3 and plotly are not in its list. [Pyodide packages](https://pyodide.org/en/stable/usage/packages-in-pyodide.html) The Cloudflare docs fetched do not explicitly confirm pandas or pyarrow on Workers, so treat that as unverified.
- Limits: 64 MiB uncompressed Worker size, 128 MB memory per isolate (both plans). CPU per request 10 ms on Free; Paid default 30 s, up to 5 min. [limits](https://developers.cloudflare.com/workers/platform/limits/)
- Verdict (inferred): pandas + pyarrow wheels plus parquet data in 128 MB is very tight, and boto3 would have to be replaced (e.g. signed `fetch` calls). The 10 ms Free CPU cap rules out pandas work on Free. Not recommended for pandas.
- Cloudflare Containers is a different path: real containers (lite 256 MiB up to standard-4 12 GiB), scale to zero, billed through the $5/month Workers Paid plan. [Containers pricing](https://developers.cloudflare.com/containers/pricing/)
- A static front end on Cloudflare remains a good fit for shape (b) (inferred from the existing account).

### Fly.io
- Docker VMs with autostop. shared-cpu-1x: 256 MB $2.19/mo, 512 MB $3.69/mo, 1 GB $6.70/mo if running all month. New orgs have no free allowance, only a 2 hour / 7 day trial. Autostop bills running time plus $0.15/GB-month rootfs while stopped. [pricing](https://docs.fly.io/about/pricing)
- With autostop and light use, likely about $0.20 to $1/month (inferred). Cold start is a machine start (seconds, unmeasured).

### Render
- Free web service: 750 instance hours/month, spins down after 15 minutes idle, about one minute to spin back up. RAM not stated on the page. [Render free](https://render.com/docs/free) A one-minute wake on a phone is poor UX.

### Google Cloud Run
- Request-based free tier: 180,000 vCPU-seconds, 360,000 GiB-seconds, 2M requests per month; beyond that $0.000024/vCPU-s, $0.0000025/GiB-s, $0.40/M requests. [pricing](https://cloud.google.com/run/pricing), [free program](https://docs.cloud.google.com/free/docs/free-cloud-features) (the direct page fetch was truncated; figures confirmed from search results citing these pages)
- Scales to zero; full Docker, so any framework including websocket apps. Needs a new GCP account with billing. Free usage is ample for this load (inferred).

### Railway
- Free plan: $1 credit/month, 0.5 GB RAM, 1 vCPU. Hobby: $5/month with $5 included usage, up to 48 GB. [plans](https://docs.railway.com/reference/pricing/plans) Free credit is probably too small for an always-on service; Hobby is a fixed $5/month (inferred).

## Framework facts (shape c)

- NiceGUI: documented deployment assumes a long-running process (Docker/uvicorn, systemd); serverless not addressed. [NiceGUI deployment](https://nicegui.io/documentation/section_configuration_deployment)
- Reflex: backend uses WebSockets; the frontend can be exported static, but the backend needs a long-running server; serverless not addressed. [Reflex self-hosting](https://reflex.dev/docs/hosting/self-hosting/)
- Dash and Shiny for Python (not fetched): server-process frameworks (inferred), same host class. So shape (c) points to Fly, Cloud Run, Render or Railway, not Vercel.

## Storj access facts

- Storj's S3-compatibility page lists the bucket CORS APIs (GetBucketCors, PutBucketCors, DeleteBucketCors) as not supported. [Storj S3 compatibility](https://storj.dev/dcs/api/s3/s3-compatibility) Implication (inferred): a browser cannot read parquet or JSON straight from the S3 gateway unless Storj's separate public-link hosting handles CORS; verify before relying on direct browser reads. This weakens a pure static front end reading Storj; a server hop or a mirror to Cloudflare-served storage is needed.
- Vercel's 4.5 MB body cap applies to function responses, so large parquet must not be returned through a function; return aggregated JSON. [limits](https://vercel.com/docs/functions/limitations)

## Shape assessment

- (a) TS front end on Vercel + small Python API: Python API as Vercel functions on the same project meets all constraints. Server-side Storj write works with env-var secrets. Fits existing accounts. Risk: cold-start import time, and no persistent in-memory Event Catalog (each cold instance reloads from Storj).
- (b) Front end reading precomputed artifacts: cheapest and fastest on a phone (CDN-cached JSON) and gives the widget feed naturally (a `current.json` written by cron). Still needs a tiny write path for "Save to Storj" and catalog auto-save, so it collapses to (a) for that endpoint. No documented Storj CORS, so artifacts need proxying or mirroring.
- (c) All-Python framework: one language, but needs a persistent websocket host and has the same phone cold-start issue as Streamlit; the widget feed is a side route. Least aligned with existing accounts and adds a new host.

## Comparison

| Option | pandas+pyarrow | Cold start | Cost (one user) | Storj write | Existing accounts |
|---|---|---|---|---|---|
| Vercel Python (Hobby) | Yes, 500 MB bundle, 2 GB RAM | Serverless; import time, measure | $0 within Hobby usage | Yes, env vars | Yes |
| Cloudflare Python Workers | In Pyodide, unverified on Workers; 128 MB, 64 MiB | Unmeasured | $0 or $5 | boto3 not available | Yes, poor fit |
| Cloudflare Containers | Yes (Docker) | Container wake | $5/mo Workers Paid | Yes | Partly |
| Fly.io autostop | Yes | Machine start, seconds | about $0.2 to $1/mo, no free tier | Yes | New |
| Render free | Yes | About 1 min | $0 | Yes | New |
| Cloud Run | Yes | Container start | $0 in free tier | Yes | New (GCP) |
| Railway Hobby | Yes | Always-on | $5/mo | Yes | New |

## Recommended shortlist (input to #55)

1. Vercel: TS/static front end plus Python FastAPI functions, with the cron job precomputing the feed and aggregates to Storj (blend of a and b). No new accounts, fits limits, secrets in env vars. Measure cold start with pandas + pyarrow first.
2. Cloud Run (container): safe fallback if Vercel cold starts or the 4.5 MB cap hurt, or if shape (c) wins. Scales to zero, free tier covers the load, no framework restrictions; cost is a new GCP account.
3. Fly.io with autostop: cheapest long-running-style Docker option at about a dollar a month, preferred if a persistent process is wanted without GCP.

Not recommended: Cloudflare Python Workers (memory and size limits against pandas/pyarrow, no boto3), Render free (one-minute wake), Railway (fixed $5 for little gain).

Open items for #55: measure Vercel cold start; confirm Storj browser/CORS behavior via link sharing; decide where the widget feed lives (a static `current.json` is the cheapest).

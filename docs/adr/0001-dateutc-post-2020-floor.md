# Keep the post-2020 floor on `dateutc` even after pinning pandas < 3

## Status

accepted (2026-05-04)

## Context

In April 2026, Streamlit Cloud's resolver picked up `pandas==3.0.2`, which
silently changed parquet timestamp resolution. Real `dateutc` values like
`1,776,725,051,936` ms got read back as `1,776,665` ms. After tz conversion
those tiny positive integers landed on **1969-12-31 PT**, collapsing the
rain-slider date range and raising `StreamlitAPIException`.

The first hotfix filtered `dateutc > 0`. That was insufficient: the corrupt
values were small *positive* integers, so they passed the `> 0` guard and
still poisoned the slider. The follow-up hotfix
([`4ad61a7`](../../../commit/4ad61a7)) raised the floor to **2020-01-01 UTC
in epoch ms** (`1577836800000`). Pandas was later pinned `< 3`
([`6ab1d3d`](../../../commit/6ab1d3d)), closing the specific corruption
path.

## Decision

The post-2020 floor stays. The dateutc normalizer
(`lookout/utils/dateutc.py`) drops any row with `dateutc <
POST_2020_MS = 1577836800000`. The floor is exposed as a default override
parameter (`min_ms=`) for tests, but every production caller takes the
default.

## Why we kept it despite the pandas pin

The pandas pin closes one source of small-int corruption. The floor defends
against the **whole class** of bugs that produce small positive ints in
`dateutc` — partial parquet writes, Ambient API hiccups returning 0 or
near-zero, future dependency drift, CSV import edges. We don't have
confidence we've enumerated every source. The cost of the floor is one
comparison per row at load time; the cost of removing it is rediscovering
the April 2026 outage shape next time.

## Considered alternatives

- **Drop the floor, rely on pandas pin alone.** Rejected: re-introduces a
  whole class of bugs the moment a different small-int path appears.
- **Per-device install-date floor.** Rejected: that's catalog-layer
  metadata, not normalizer concern. Single global floor matches the failure
  mode (corruption, not legitimate-but-old data).
- **Keep `> 0` only, log warnings on suspicious values.** Rejected:
  warnings don't prevent the slider from collapsing; we need rows
  *removed*, not flagged.

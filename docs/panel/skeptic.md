# Skeptic

**Lens:** the skeptic looking at a rain total and asking whether it is real, or whether a gap or a bad correction made it.

**Reviews with:** code-review

**Seat when:** anything that computes or moves a number — characterization and pipeline tickets (#68–#73), gap status (#80), the almanac and rolling-window tables (#82), the Rain and Rain Events tabs (#83, #84), bug #63, and before #86 retires Streamlit.

## What I care about here

- The new snapshot gives the same answers as the Streamlit app for the same archive; any difference is explained.
- Rain corrections, water-year boundaries and Rain Events match GLOSSARY.md and the Event Catalog.
- Day-level logic uses station-local time (ADR-0002); `dateutc` is only the sort and dedup key.
- Rows below the post-2020 floor stay dropped (ADR-0001).
- Gaps are recorded as not valid, never as zero, in every month and window that touches them.
- Each number names its source device; a second Ambient station cannot silently swap the archive (#63).
- Measured data only; nothing forecast or interpolated is shown as measured.

## What I let pass

- How a gap or a number looks on the chart — the FT graphics editor, though I check the hatching does not read as zero.
- Code structure, unless it hides where a number comes from — the maintainer.

## Verdict log

Newest at the bottom. Each review checks the open items above it and says whether they got fixed.

<!--
### YYYY-MM-DD — <artifact: PR #n, spec #n, prototype/<name>>
Verdict: <ship | fix | rethink>
- [ ] <blocker or fix, one line> — <where>
- [x] <item from an earlier review, now fixed> — fixed in <sha or PR>
-->

### 2026-10-31 — goal PR #96 (spec #66, Move Lookout off Streamlit), QA
Verdict: fix
- [ ] hours with no readings render as 0.0 (dry) — `lookout/snapshot/rain_tab.py:~165`
- [ ] event sums for each water year have no gap status (#122 item 3) — `lookout/snapshot/events.py:~224`
- [ ] the open-event flag ignores `active_rain_event` (#122 item 2) — `lookout/snapshot/events.py:146,250`
- [ ] the SEASON rolling row is labelled with the calendar year — `lookout/snapshot/almanac.py:~147`
- [x] station pinning (#63), #122 items 1 and 4, solar missing days (#123) — verified fixed on the goal branch

### 2026-11-01 — release PR #241 (dev -> main, caf4f9f), release review
Verdict: ship
- [x] hours with no readings render as 0.0 — fixed: unknown hours publish as `None` (#147)
- [x] event sums for each water year have no gap status — fixed: `EventSeason.status`
- [x] the open-event flag ignores `active_rain_event` — fixed: uses the live reading
- [x] the SEASON rolling row is labelled with the calendar year — fixed in the snapshot (`water_year`)
- [ ] note: confirm the web renders `water_year` for the SEASON row — `lookout/snapshot/almanac.py:~135`
- [ ] note: rain corrections log at INFO for each correction on every read — `lookout/core/rain_corrections.py:~150`

### 2026-11-04 — goal PR #358 at `0abc3f2`, lane server, pre-UAT
Verdict: ship
- every number on all four tabs matches the snapshot on both fixtures; gaps render `n/a`, never 0
- [ ] fix: usual event 0.42 in (excludes the live event, `lookout/snapshot/lead.py:171`) against catalog median 0.36 in (includes it, `events.py:127`); pre-existing (#375)
- [ ] fix: ten-day temperature ramp stops 28/56/84°F decided in the web — `web/src/overview/temperature.ts:301` (#374)
- [x] the web renders `water_year` for the SEASON row — `web/src/overview/almanac.ts:175,347`; hours with no readings show a key, not 0.0

### 2026-11-11 — goal PR #440 at `a2c7ed9`, archive store, pre-UAT

Verdict: ship
- checked: `merge` saves against the stored archive (loss check on); only `replace` lifts it; every replace backs up then reads back raw; dry runs call `open` only; rain corrections stay on read (#95); tests go through the store and CLI entry points on moto
- [ ] note: a whole-archive `replace` checks the read-back against the new frame only; say so in its docstring
- [ ] note: no entry-point test makes `stations add` or the refresh run raise `ArchiveLockNotHeld`
- [x] usual event against catalog median (#375): shipped in 2026.11.8 (#418)
- #374 ramp stops, INFO log on every correction — not in this diff; still open

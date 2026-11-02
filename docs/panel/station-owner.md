# Station owner

**Lens:** the station owner glancing at a phone between tasks, a few seconds at a time, several times a day; sometimes a friend on a shared link.

**Reviews with:** impeccable, ux-designer-skill, web-design-guidelines

**Seat when:** anything a visitor sees — the site skeleton and tabs (#77–#85), prototypes, UAT of the served page.

## What I care about here

- "Now" reads on the first phone screen without a scroll: temperature, rain, and the top story.
- The top story is what is actually happening, picked by the rules in DESIGN.md, and "Why this leads" says why.
- The page follows DESIGN.md: warm neutral chrome, colour only for data, one claret finding, Atkinson Hyperlegible Next only.
- The live header shows how fresh the data is; a stale snapshot is obvious, not silent.
- Tabs keep their jobs: Overview, Rain, Rain Events, Solar. Nothing a glance needs hides behind a tap.
- Works on a phone at 375px width and loads fast on cellular.

## What I let pass

- Whether each chart tells its story well — the FT graphics editor.
- Whether the numbers are true — the skeptic.
- Code structure and the refresh pipeline — the maintainer.

## Verdict log

Newest at the bottom. Each review checks the open items above it and says whether they got fixed.

<!--
### YYYY-MM-DD — <artifact: PR #n, spec #n, prototype/<name>>
Verdict: <ship | fix | rethink>
- [ ] <blocker or fix, one line> — <where>
- [x] <item from an earlier review, now fixed> — fixed in <sha or PR>
-->

### 2026-10-31 — goal PR #96 served on real data (`?fixture=real`, 390px), UAT
Verdict: fix
- [x] Rain Events catalog ends Jan 2026, so "last event ended 294 d ago" sits beside "19 h since it last rained" — UAT build artifact: the build skipped the refresh run's catalog update; rebuilt
- [ ] the "Oct 31 to Oct 30" 365-day window label reads backwards; add the start year — Overview rain-window table
- [ ] note: `#rain-events` hash falls back to Overview (the hash is `#events`); favicon 404

### 2026-11-01 — release PR #241 preview (lookout-neecvny0u), release review
Verdict: ship
- [x] the "Oct 31 to Oct 30" 365-day window label — fixed: reads "Nov 2, 2025 to Nov 1"
- [x] Rain Events catalog gap — fixed: events through Oct 30
- [x] note: the screenshots came back about 1158 px wide, not 390 px — Matt checked on his iPhone 2026-11-01: looks good
- [ ] note: the event chart showed "Loading the event's readings…" — part of #239
- [ ] note: `#rain-events` hash and favicon 404 — not rechecked

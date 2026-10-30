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

---
version: 1
slug: "prototype-dashboard-index-html"
primary_target: "prototype/dashboard/index.html"
related_targets: ["prototype/dashboard/variant-a.js","prototype/dashboard/variant-b.js","prototype/dashboard/variant-c.js"]
---

# Lookout dashboard (prototype)

Scope: the whole dashboard — Overview front page plus Rain, Rain Events, Solar. Mode: Operate. Throwaway prototype answering "What should the new dashboards look like?" (wayfinder ticket #53). Three structural variants share one world.

Audience/job: the station owner on a phone, glancing several times a day: what is it doing now, and how normal is that. Constraints: measured data only, no forecast; gaps make comparisons invalid and must show as such; imperial units; station-local time.

## Direction contract

THESIS: A personal weather record that reads like Dark Sky and argues like the FT. Refuses the category default of a gauge wall and tile grid of raw numbers: every view leads with a plain-language finding, then the evidence.

OWN-WORLD: The Paper world: a warm off-white ground by day and a warm near-black in dark mode, following the system. Ink is warm near-black, never grey-on-grey. Data colour does the work: a desaturated precipitation blue ramp (light drizzle to downpour), a temperature ramp from cold blue to warm terracotta, a sun amber for solar, and one claret annotation colour for the finding on a chart. Headlines state findings in a short, quiet sans (the serif is dropped for Variant D); UI, narrative and numerals share one dyslexia-friendly family, Atkinson Hyperlegible Next, whose `tnum` gives tabular figures (checked: all digits equal width); tabular digits are used only in tables, stats and axes, and headlines and body text use normal digits. Variants A-C still use Source Serif 4 and Schibsted Grotesk. Thin hairline axes, direct labels, no legends, no chart boxes.

STORY: The visitor sees the summary sentence and current temperature, understands today against recent days and against the same date in earlier years, sees rain promoted when an event is live, and drills into Rain, Events or Solar for the story behind each.

FIRST VIEWPORT: Phone width. The top third holds the station name and timestamp, a one-sentence condition summary in the serif, and a large current temperature with feels-like, wind and rain today set beside it. Below that sits the signature condition ribbon: a 48-hour horizontal band coloured by rain intensity and daylight, with temperature traced over it and "now" marked at the right edge. When an event is live, the rain event block takes the lead position above the ribbon.

FORM: World pinned by the user's brief (Dark Sky presentation, FT storytelling); the variants vary organisation only. the concept roll was skipped because the owner pinned the references (Dark Sky presentation, FT storytelling); the build was led by code with no comps.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Style refinement (Variant D, 2026-10-28)

Owner request: dyslexia-friendly text, muted options, "the data is the point; headlines and narratives are context (FT style)". Typeface change: Source Serif 4 and Schibsted Grotesk are replaced by Atkinson Hyperlegible Next for all text in D (Google Fonts; OpenDyslexic rejected, weak evidence). Hierarchy: short finding headline, one-line standfirst, chart, source line; headline size and weight reduced so the chart is read first. Readability: left-aligned, no italics, no long all-caps, body line-height 1.55, narrative max 60-62ch, small text floor 12px, slightly open letter and word spacing, off-white and off-black grounds.

The style is being chosen among three token-override options, switched with `?variant=D&style=1|2|3` (styles-d.css, html[data-style]): 1 Paper (warm off-white, near-greyscale chrome, desaturated blue and terracotta, claret annotation), 2 Slate (cool grey, ink-blue chrome, restrained cool ramp, plum annotation), 3 Mist (lowest-contrast chrome, no tinted panels, data colour is the only colour, neutral annotation). Annotation is no longer red in any option. Decision made: the owner chose style 1, Paper; the other two are to be deleted.

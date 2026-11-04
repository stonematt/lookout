# Prototype: Tiles That Talk (Nov 2026)

The design rounds behind the Lookout redesign spec. `DESIGN.md` and `PRODUCT.md` on `dev` hold the decisions; this folder holds the evidence: every canvas board, rendered images, the first design-system attempt, and the contrast checker.

Nothing here is built into the site. Every number on every board is a made-up sample shaped like the snapshot, and the samples do not fully agree with each other across boards.

## Live artifacts (private until shared)

- **Lookout Prototypes canvas:** https://claude.ai/artifact/3L81ndVLwwTKZP1jE4Ecmp. Its files are copied in `canvas/` as published on Nov 4, 2026.
- **Lookout design system (first attempt):** https://claude.ai/artifact/VcvtpuqYRHutK3r6HWzQvi. It describes the earlier Paper world plus the first "phone edition" ideas, not the chosen design. Its files are in `design-system/`.

## How the design moved

1. **Before.** The site as it ran after the Streamlit port, on the golden fixtures at phone width: `images/before/`. The Overview ran about five phone screens, and a third of the words were method.
2. **Four phone directions** (canvas row 1): `images/directions/`.
   - A, Morning Paper: a light editorial page, headline and paragraph first. Closest to the FT.
   - B, Sky Field: dark, a big temperature, one number and one sentence per section. Its palette was kept.
   - C, The Log: the night as a timeline. Dropped.
   - D, Tiles That Talk: Apple-style tiles whose headline is a sentence. Its layout was kept.
3. **Row E: D's tiles, B's palette, production's charts** (desktop, 1440 wide): `images/desktop/*-1440.png`.
   - Overview: the now tile with the 48-hour ribbon, the lead (rain event against the usual storm), the six-window temperature ranges, now against this date, ten days, the rain almanac, rolling windows, the solar trace, and four quick-fact tiles.
   - Rain: windows and the 90-day daily bars, water-year lines, dry spells, rain by the hour.
   - Rain events: catalog finding, year and sort filters, the selected event's two charts, the events table with a Peak column.
   - Solar: the 24-hour trace, daily energy for 90 days, one day by the quarter hour, the solar almanac.
   - macOS widgets (small, medium, large): `images/widgets/`.
   - Every source paragraph folds into a one-line "Source" disclosure under its tile.
4. **Contrast pass.** All text was already 4.5:1; many graphics were not 3:1 (bands, hatching, borders, the palest ramp steps). They were fixed on every board, and the ribbon's rain moved from an overprint to its own track. `contrast-check.py` holds the dark and light tokens and the pairs checked.
5. **The desktop boards at phone width** showed shrunken chart text (4 to 6px): `images/desktop/*-at-390-before-phone-pass.png`. That led to row F.
6. **Row F: the phone pattern**: `images/phone/`. A short fixed top, then an exclusive accordion of findings, and a bottom tab bar. Charts are drawn at phone width with 12px text. Rain and Solar default to 30 days with a 30/90 switch.
7. **Rain events list.** The list is how you move between events: each event is an accordion row whose curve opens inside it, newest open by default, with a pinned year select and a Newest/Wettest/Longest sort. `images/phone/Events-accordion-list.png`. The filter stays at year and sort.

## Files

- `canvas/`: `canvas.json` plus one `.dc.html` per board. They are Design-canvas components; opened as plain HTML in a browser they render close to the canvas (the `support.js` runtime is not included, so interactive controls do nothing).
- `images/`: as above. Phone images are 2× renders at 390 px; desktop at 1440 px.
- `design-system/`: tokens, README, components and previews of the first design-system artifact.
- `contrast-check.py`: `python3 contrast-check.py` prints every failing pair; it should print `failures 0`.

## Known gaps in the prototypes

- On the phone Overview, "Against this date" and "The last ten days" open to a one-line note instead of their charts, and the Solar row links to the Solar tab.
- Only the newest event row has a real curve.
- The macOS widget board uses 10 to 11 px labels, under the 12 px floor in DESIGN.md.
- No light-theme boards were drawn; the light tokens exist only in DESIGN.md and `contrast-check.py`.
- The station selector (GLOSSARY.md) is not drawn on any board.

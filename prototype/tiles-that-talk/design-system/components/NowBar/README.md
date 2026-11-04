# NowBar
Proposed. A 44px line above the sticky tabs that keeps the live reading in view after NowHero scrolls off.

- Content: live dot, temperature (Stat), the condition in three words, age at the right. Nothing else.
- Appears only once the hero leaves the viewport (IntersectionObserver), so the first screen is unchanged.
- Stale (past 15 min): hollow dot and the hatched age chip, the same stale grammar as the masthead.
- The condition words come from the snapshot (a short form beside the full sentence).

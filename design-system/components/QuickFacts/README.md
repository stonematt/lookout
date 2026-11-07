The quick-fact tiles show wind, pressure, air (humidity and dew point) and sun and moon, each as a kicker, one large value in `fact` type and one sentence in `ink-2`. On desktop they are a row of four one-column tiles; on a phone, a 2x2.

**Consumer provides.** The kicker, the value (and an optional word unit shown in `stat`), and the sentence. A station with no coordinates has no sun and moon times: its tile keeps the kicker, shows no value, and says the station reports no location.

**Use.** One value per tile, in `ink`, weight 300 and tabular digits. Under 360px wide, the sun and moon times drop to `stat` so "2:50 pm to 12:13 am" does not scroll the page sideways. Keep sentences short; they are the short forms Python writes.

Static rendition of `web/src/overview/quick-facts.ts` and `quick-facts.css`.

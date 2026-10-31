/**
 * The condition sentence beside the current temperature: what it feels like
 * outside, then what the rain is doing, in one plain sentence.
 *
 * Every word comes from the snapshot's `current`, the live-stitched reading,
 * so it agrees with the temperature and the Right-now strip beside it. Rules,
 * in order:
 *
 * 1. Temperature word: under 32° freezing, 45° cold, 55° cool, 65° mild,
 *    80° warm, else hot. Humidity 90% and up adds "damp", 35% and under
 *    "dry". Wind under 3 mph is "still", under 10 "lightly breezy", else
 *    "windy". A rising or falling day-on-day trend adds "warmer/cooler than
 *    this time yesterday".
 * 2. Rain: raining now (a rate above 0) gives the rate and today's total;
 *    rain earlier today gives the total and when the last fell; a dry day
 *    names when it last rained.
 */
import { formatStamp } from "../format.js";
import type { SnapshotConditions } from "../snapshot.js";

function temperatureWord(f: number): string {
  if (f < 32) return "Freezing";
  if (f < 45) return "Cold";
  if (f < 55) return "Cool";
  if (f < 65) return "Mild";
  if (f < 80) return "Warm";
  return "Hot";
}

function humidityWord(pct: number | null): string | null {
  if (pct === null) return null;
  if (pct >= 90) return "damp";
  if (pct <= 35) return "dry";
  return null;
}

function windWord(mph: number | null): string | null {
  if (mph === null) return null;
  if (mph < 3) return "still";
  if (mph < 10) return "lightly breezy";
  return "windy";
}

const TREND_WORDS = {
  rising: "warmer than this time yesterday",
  falling: "cooler than this time yesterday",
} as const;

/** "a", "a and b", "a, b and c". */
function listed(words: string[]): string {
  if (words.length < 2) return words.join("");
  return `${words.slice(0, -1).join(", ")} and ${words.at(-1)}`;
}

function capitalised(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function feel(c: SnapshotConditions): string | null {
  const words = [
    c.temperature_f === null ? null : temperatureWord(c.temperature_f),
    humidityWord(c.humidity_pct),
    windWord(c.wind_speed_mph),
  ].filter((w): w is string => w !== null);
  if (!words.length) return null;
  const trend =
    c.temperature_f !== null && (c.temp_trend === "rising" || c.temp_trend === "falling")
      ? `, ${TREND_WORDS[c.temp_trend]}`
      : "";
  return `${listed(words).toLowerCase()}${trend}`;
}

function rain(c: SnapshotConditions): string | null {
  const today = c.rain_today_in;
  if (c.rain_rate_in_hr !== null && c.rain_rate_in_hr > 0) {
    const sofar = today === null ? "" : `, ${today.toFixed(2)} in so far today`;
    return `raining at ${c.rain_rate_in_hr.toFixed(2)} in/hr${sofar}`;
  }
  if (today !== null && today > 0) {
    const last = c.last_rain_ms === null ? "" : `, the last at ${formatStamp(c.last_rain_ms)}`;
    return `${today.toFixed(2)} in of rain today${last}`;
  }
  if (today === null) return null;
  const last = c.last_rain_ms === null ? "" : `, the last on ${formatStamp(c.last_rain_ms)}`;
  return `no rain today${last}`;
}

/**
 * The condition sentence for the current reading.
 *
 * @param current - The snapshot's current conditions.
 * @returns One sentence; "No conditions reported." with nothing known.
 */
export function conditionSentence(current: SnapshotConditions): string {
  const parts = [feel(current), rain(current)].filter((p): p is string => p !== null);
  return parts.length ? `${capitalised(parts.join("; "))}.` : "No conditions reported.";
}

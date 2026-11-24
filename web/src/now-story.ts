/**
 * The now story in words (ADR-0011, #694): last rain, or while Python tells
 * the story (`rain_now.rate_in_hr` set, last rain within 3 hours) raining
 * with its rate or rain stopped, then the event's rain since it started.
 * The Overview's rain tile and its chart, and the Rain tab head (and so the
 * rain zoom) all word `rain_now` through this one module, so one snapshot
 * never reads two ways.
 *
 * The format is the C boards': how long ago in minutes under an hour, whole
 * hours under two days, whole days after ("Rain stopped 40 min ago", "Last
 * rain 9 h ago"); the rate in "in/hr" through `inchesPerHour` (#647 call
 * 1); the event's start as a time today, else with its day. Every value is
 * the snapshot's; the site only words it.
 *
 * Offers `agoWords`, `sinceWords`, `toldStory`, `ToldStory`, `nowStory`,
 * `NowStory` and `storyLine`.
 */
import { formatStamp, formatTime, inches, inchesPerHour, localDay } from "./format.js";
import { rainNow } from "./snapshot.js";
import type { DashboardSnapshot, RainNow } from "./snapshot.js";

/**
 * How long ago: minutes under an hour, whole hours under two days, whole
 * days after.
 *
 * @param ms - The time since, in milliseconds.
 * @returns E.g. "40 min", "9 h", "3 d".
 */
export function agoWords(ms: number): string {
  const minutes = Math.floor(Math.max(0, ms) / 60_000);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  return hours < 48 ? `${hours} h` : `${Math.floor(hours / 24)} d`;
}

/**
 * When an event started: a time on the build's day, else with its day.
 *
 * @param startMs - The event's start.
 * @param builtMs - The build's time.
 * @returns E.g. "2:40 pm", "Nov 22, 9:05 pm".
 */
export function sinceWords(startMs: number, builtMs: number): string {
  return localDay(startMs) === localDay(builtMs) ? formatTime(startMs) : formatStamp(startMs);
}

/** `rain_now` while Python tells the now story: it carries a rate. */
export type ToldStory = RainNow & { rate_in_hr: number };

/**
 * The snapshot's `rain_now` while it tells the now story, or null: before
 * #659 there is none, and outside the story the rate is null.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The story.
 */
export function toldStory(snapshot: DashboardSnapshot): ToldStory | null {
  const rain = rainNow(snapshot);
  if (rain === null || rain.rate_in_hr === null) return null;
  return { ...rain, rate_in_hr: rain.rate_in_hr };
}

/** The now story's words, in the parts a view sets apart. */
export interface NowStory {
  /** "Raining · 0.08 in/hr", "Rain stopped 40 min ago", "Last rain 9 h ago". */
  lead: string;
  /** The event's rain and its start: ["0.31 in", "since 2:40 pm"]; null without one. */
  event: [string, string] | null;
}

/**
 * The now story in words, or null for a snapshot without `rain_now`.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns The lead and the event's part; "No rain in the record yet" with
 *   no last rain.
 */
export function nowStory(snapshot: DashboardSnapshot): NowStory | null {
  const rain = rainNow(snapshot);
  if (!rain) return null;
  if (rain.last_rain_ms === null) return { lead: "No rain in the record yet", event: null };
  const built = snapshot.built_at_ms;
  const ago = agoWords(built - rain.last_rain_ms);
  const story = toldStory(snapshot);
  if (story === null) return { lead: `Last rain ${ago} ago`, event: null };
  const lead = story.raining
    ? `Raining · ${inchesPerHour(story.rate_in_hr)}`
    : `Rain stopped ${ago} ago`;
  const event: [string, string] | null =
    story.event_start_ms === null || story.event_total_in === null
      ? null
      : [
          inches(story.event_total_in, { digits: 2 }),
          `since ${sinceWords(story.event_start_ms, built)}`,
        ];
  return { lead, event };
}

/**
 * The now story as one line.
 *
 * @param story - The story's words.
 * @returns E.g. "Raining · 0.12 in/hr · 0.30 in since 2:40 pm".
 */
export function storyLine(story: NowStory): string {
  return story.event ? `${story.lead} · ${story.event.join(" ")}` : story.lead;
}

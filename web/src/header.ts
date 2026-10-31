/**
 * The page header: station name, the newest reading's time and its data age,
 * and, when they have fallen behind it, the age of the figures on the page.
 *
 * Two clocks run here, both from the browser's clock on every `tick()`:
 *
 * - **The reading**: the newest observation either read function has
 *   brought, the snapshot's or `/current`'s. It keeps counting when the
 *   snapshot itself has stopped updating (a dead VPS cannot hide behind a
 *   cached page).
 * - **The figures**: when the snapshot behind every value on the page was
 *   observed. The page's figures come only from `/dashboard`, so a fresh
 *   `/current` must not lend them its freshness. Their line appears only
 *   when they are stale and the reading line would otherwise suggest they
 *   are fresher than they are: the reading is fresh, or more than 15
 *   minutes newer. When the two agree there is one line, as before.
 *
 * Each line turns to the stale style past 15 minutes on its own clock.
 */
import { STALE_AFTER_MS, dataAge, formatStamp } from "./format.js";
import type { CurrentReading, DashboardSnapshot, Station } from "./snapshot.js";
import { newestObservation } from "./snapshot.js";

/** A mounted header. */
export interface Header {
  /** Show the snapshot from `/dashboard`: its station, reading and figures. */
  showSnapshot(snapshot: DashboardSnapshot): void;
  /** Show `/current`'s reading; an older one than the reading shown is ignored. */
  showCurrent(current: CurrentReading): void;
  /** Say the snapshot could not be loaded, with a button that calls `retry`. */
  fail(retry: () => void): void;
  /** Re-compute both ages against the clock. */
  tick(): void;
}

const FALLBACK_NAME = "Lookout";

/**
 * When the page's figures were observed: the snapshot's newest observation,
 * or, with none, when it was built.
 *
 * @param snapshot - The dashboard snapshot.
 * @returns Epoch ms UTC.
 */
export function figuresTime(snapshot: DashboardSnapshot): number {
  return snapshot.observed_at_ms ?? snapshot.built_at_ms;
}

/** One line of the header: a dot, a labelled time and an age. */
interface AgeLine {
  root: HTMLElement;
  when: HTMLElement;
  age: HTMLElement;
}

function ageLine(root: HTMLElement): AgeLine {
  return {
    root,
    when: root.querySelector<HTMLElement>(".masthead-when")!,
    age: root.querySelector<HTMLElement>(".masthead-age")!,
  };
}

/** Write "<label> <time>" into a line's `when`. */
function stamp(line: AgeLine, label: string, ms: number): void {
  const time = document.createElement("time");
  time.dateTime = new Date(ms).toISOString();
  time.textContent = formatStamp(ms);
  line.when.replaceChildren(`${label} `, time);
}

/**
 * Render the header into `host`.
 *
 * @param host - The element to render into; its content is replaced.
 * @param now - The clock, epoch ms; injected so tests can run it.
 * @returns The header's controls.
 */
export function mountHeader(host: HTMLElement, now: () => number = Date.now): Header {
  host.innerHTML = `
    <header class="masthead">
      <h1 class="masthead-station">${FALLBACK_NAME}</h1>
      <div class="masthead-ages">
        <p class="masthead-line masthead-reading" data-stale="false">
          <span class="masthead-dot" aria-hidden="true"></span>
          <span class="masthead-when">Loading the newest reading</span>
          <span class="masthead-age"></span>
        </p>
        <p class="masthead-line masthead-figures" data-stale="true" hidden>
          <span class="masthead-dot" aria-hidden="true"></span>
          <span class="masthead-when"></span>
          <span class="masthead-age"></span>
        </p>
      </div>
    </header>`;
  const name = host.querySelector<HTMLElement>(".masthead-station")!;
  const reading = ageLine(host.querySelector<HTMLElement>(".masthead-reading")!);
  const figures = ageLine(host.querySelector<HTMLElement>(".masthead-figures")!);

  let station: Station | null = null;
  let observedAtMs: number | null = null;
  let figuresAtMs: number | null = null;
  let shown = false;

  function tick(): void {
    if (!shown) return;
    const clock = now();
    const age = dataAge(observedAtMs, clock);
    reading.age.textContent = age.text;
    reading.root.dataset.stale = String(age.stale);

    const behind =
      observedAtMs !== null && figuresAtMs !== null ? observedAtMs - figuresAtMs : 0;
    const own = dataAge(figuresAtMs, clock);
    figures.root.hidden = !(
      observedAtMs !== null &&
      own.stale &&
      (!age.stale || behind > STALE_AFTER_MS)
    );
    figures.age.textContent = own.text;
    figures.root.dataset.stale = String(own.stale);
    // With two lines, "Reading" alone could be taken for the figures' time.
    if (observedAtMs === null) reading.when.replaceChildren();
    else {
      const label = figures.root.hidden ? "Reading" : "Station last reported";
      stamp(reading, label, observedAtMs);
    }
  }

  function render(): void {
    name.textContent = station?.name || FALLBACK_NAME;
    if (figuresAtMs !== null) stamp(figures, "Numbers below from", figuresAtMs);
    tick();
  }

  function take(next: Station, observed: number | null): void {
    // A reading without a name (Ambient unnamed) never erases a known one.
    if (next.name || !station?.name) station = next;
    observedAtMs = newestObservation(observedAtMs, observed);
    shown = true;
    render();
  }

  return {
    showSnapshot(snapshot) {
      figuresAtMs = figuresTime(snapshot);
      take(snapshot.station, snapshot.observed_at_ms);
    },
    showCurrent(current) {
      take(current.station, current.observed_at_ms);
    },
    fail(retry) {
      shown = false;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "masthead-retry";
      button.textContent = "Try again";
      button.addEventListener("click", retry);
      reading.when.replaceChildren("The dashboard could not be loaded.", " ", button);
      reading.age.textContent = "";
      reading.root.dataset.stale = "true";
      figures.root.hidden = true;
    },
    tick,
  };
}

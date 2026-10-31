/**
 * The page header: station name, the newest reading's time, and its data age.
 *
 * The age is computed here, in the browser, from the newest observation
 * time, and re-computed on every `tick()`, so it keeps counting when the
 * snapshot itself has stopped updating (a dead VPS cannot hide behind a
 * cached page). Past 15 minutes it switches to the stale style.
 */
import { dataAge, formatStamp } from "./format.js";
import type { Station } from "./snapshot.js";
import { newestObservation } from "./snapshot.js";

/** What the header shows: from `/dashboard`, then `/current`. */
export interface HeaderReading {
  /** The station, or null to keep the one already shown. */
  station: Station | null;
  /** The newest observation time, or null when unknown. */
  observedAtMs: number | null;
}

/** A mounted header. */
export interface Header {
  /** Show a reading; an older observation than the one shown is ignored. */
  show(reading: HeaderReading): void;
  /** Say the snapshot could not be loaded, with a button that calls `retry`. */
  fail(retry: () => void): void;
  /** Re-compute the data age against the clock. */
  tick(): void;
}

const FALLBACK_NAME = "Lookout";

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
      <p class="masthead-reading" data-stale="false">
        <span class="masthead-dot" aria-hidden="true"></span>
        <span class="masthead-when">Loading the newest reading</span>
        <span class="masthead-age"></span>
      </p>
    </header>`;
  const name = host.querySelector<HTMLElement>(".masthead-station")!;
  const reading = host.querySelector<HTMLElement>(".masthead-reading")!;
  const when = host.querySelector<HTMLElement>(".masthead-when")!;
  const age = host.querySelector<HTMLElement>(".masthead-age")!;

  let station: Station | null = null;
  let observedAtMs: number | null = null;
  let shown = false;

  function tick(): void {
    if (!shown) return;
    const current = dataAge(observedAtMs, now());
    age.textContent = current.text;
    reading.dataset.stale = String(current.stale);
  }

  function render(): void {
    name.textContent = station?.name || FALLBACK_NAME;
    when.replaceChildren();
    if (observedAtMs !== null) {
      const time = document.createElement("time");
      time.dateTime = new Date(observedAtMs).toISOString();
      time.textContent = formatStamp(observedAtMs);
      when.append("Reading ", time);
    }
    tick();
  }

  return {
    show(next) {
      station = next.station ?? station;
      observedAtMs = newestObservation(observedAtMs, next.observedAtMs);
      shown = true;
      render();
    },
    fail(retry) {
      shown = false;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "masthead-retry";
      button.textContent = "Try again";
      button.addEventListener("click", retry);
      when.replaceChildren("The dashboard could not be loaded.", " ", button);
      age.textContent = "";
      reading.dataset.stale = "true";
    },
    tick,
  };
}

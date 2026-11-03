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
 * - **The figures** (the "numbers below" in the copy): when the snapshot
 *   behind every value on the page was
 *   observed. The page's figures come only from `/dashboard`, so a fresh
 *   `/current` must not lend them its freshness. Their line appears only
 *   when they are stale and the reading line would otherwise suggest they
 *   are fresher than they are: the reading is fresh, or more than 15
 *   minutes newer. When the two agree there is one line, as before.
 *
 * Each line turns to the stale style past 15 minutes on its own clock.
 *
 * With the station index (#273) the header is also the station's details
 * strip: its name, the reading line above (the newest reading time) and its
 * archive start. With two or more stations the name is the station
 * selector, a dropdown of the index in order; with one there is nothing to
 * choose and the name stays a heading.
 */
import { STALE_AFTER_MS, dataAge, dayText, formatStamp, localDay } from "./format.js";
import type { CurrentReading, DashboardSnapshot, StationId } from "./snapshot.js";
import { newestObservation } from "./snapshot.js";
import type { StationIndex } from "./stations.js";
import { findStation } from "./stations.js";

/** A mounted header. */
export interface Header {
  /** Show the snapshot from `/dashboard`: its station, reading and figures. */
  showSnapshot(snapshot: DashboardSnapshot): void;
  /**
   * Show `/current`'s reading; an older one than the reading shown is
   * ignored, and after `fail` it waits for the snapshot.
   */
  showCurrent(current: CurrentReading): void;
  /**
   * Say the snapshot could not be loaded, or `message` when given, with a
   * button that calls `retry`.
   */
  fail(retry: () => void, message?: string): void;
  /** Re-compute both ages against the clock. */
  tick(): void;
  /**
   * Show the station index: the selector (two or more stations), with
   * `chosen` selected (the default station when null), and the chosen
   * station's archive start. `onChoose` is called with the MAC a visitor
   * picks.
   */
  showStations(index: StationIndex, chosen: string | null, onChoose: (mac: string) => void): void;
  /** Forget the shown station's reading and figures: another was chosen. */
  reset(): void;
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

/**
 * Whether the figures need their own line: they are stale, and the reading
 * line would otherwise suggest they are fresher than they are (the reading
 * is fresh, or more than 15 minutes newer).
 *
 * @param readingMs - The newest observation from either source, or null.
 * @param figuresMs - When the page's figures were observed, or null.
 * @param nowMs - The browser's clock.
 * @returns True to show the figures' line.
 */
export function figuresLag(
  readingMs: number | null,
  figuresMs: number | null,
  nowMs: number,
): boolean {
  if (readingMs === null || figuresMs === null) return false;
  if (!dataAge(figuresMs, nowMs).stale) return false;
  return !dataAge(readingMs, nowMs).stale || readingMs - figuresMs > STALE_AFTER_MS;
}

/** One line of the header: a dot, a labelled time and an age. */
interface AgeLine {
  root: HTMLElement;
  when: HTMLElement;
  age: HTMLElement;
}

/** The parts of one rendered header line. */
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
      <h1 class="masthead-station"><span class="masthead-name"></span></h1>
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
      <p class="masthead-archive" hidden></p>
    </header>`;
  const heading = host.querySelector<HTMLElement>(".masthead-station")!;
  const name = host.querySelector<HTMLElement>(".masthead-name")!;
  name.textContent = FALLBACK_NAME;
  const archive = host.querySelector<HTMLElement>(".masthead-archive")!;
  let picker: HTMLSelectElement | null = null;
  let listed = "";
  let choose: (mac: string) => void = () => undefined;
  const reading = ageLine(host.querySelector<HTMLElement>(".masthead-reading")!);
  const figures = ageLine(host.querySelector<HTMLElement>(".masthead-figures")!);

  let station: StationId | null = null;
  let observedAtMs: number | null = null;
  let figuresAtMs: number | null = null;
  let shown = false;
  let failed = false;

  function tick(): void {
    if (!shown) return;
    delete reading.root.dataset.failed;
    const clock = now();
    const age = dataAge(observedAtMs, clock);
    reading.age.textContent = age.text;
    reading.root.dataset.stale = String(age.stale);

    const own = dataAge(figuresAtMs, clock);
    figures.root.hidden = !figuresLag(observedAtMs, figuresAtMs, clock);
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

  function take(next: StationId, observed: number | null): void {
    // A reading without a name (Ambient unnamed) never erases a known one.
    if (next.name || !station?.name) station = next;
    observedAtMs = newestObservation(observedAtMs, observed);
    shown = true;
    render();
  }

  /** The selector for an index of two or more stations, drawn once per list. */
  function showPicker(index: StationIndex): HTMLSelectElement | null {
    const list = JSON.stringify(index.stations.map((s) => [s.mac, s.name]));
    if (list === listed) return picker;
    listed = list;
    if (index.stations.length < 2) {
      picker = null;
      heading.classList.remove("masthead-picks");
      heading.replaceChildren(name);
      return null;
    }
    picker = document.createElement("select");
    picker.className = "station-select";
    picker.setAttribute("aria-label", "Station");
    for (const entry of index.stations) {
      const option = document.createElement("option");
      option.value = entry.mac;
      // An unnamed station (Ambient allows it) is told apart by its MAC's end.
      option.textContent = entry.name ?? `Unnamed station …${entry.mac.slice(-5)}`;
      picker.append(option);
    }
    picker.addEventListener("change", () => choose(picker!.value));
    heading.classList.add("masthead-picks");
    heading.replaceChildren(picker);
    return picker;
  }

  /** The chosen station's archive start, as a station-local day; none when unknown. */
  function showArchive(startMs: number | null): void {
    archive.hidden = startMs === null;
    if (startMs === null) return;
    const time = document.createElement("time");
    const day = localDay(startMs);
    time.dateTime = day;
    time.textContent = dayText(day, true);
    archive.replaceChildren("Archive from ", time);
  }

  return {
    showStations(index, chosen, onChoose) {
      choose = onChoose;
      const entry = chosen === null ? index.stations[0] : findStation(index, chosen);
      const shownPicker = showPicker(index);
      if (shownPicker) {
        if (entry) shownPicker.value = entry.mac;
        else shownPicker.selectedIndex = -1;
      }
      showArchive(entry?.archive_start_ms ?? null);
    },
    reset() {
      station = null;
      observedAtMs = null;
      figuresAtMs = null;
      shown = false;
      failed = false;
      name.textContent = FALLBACK_NAME;
      reading.when.replaceChildren("Loading the newest reading");
      reading.age.textContent = "";
      reading.root.dataset.stale = "false";
      delete reading.root.dataset.failed;
      figures.root.hidden = true;
    },
    showSnapshot(snapshot) {
      failed = false;
      figuresAtMs = figuresTime(snapshot);
      take(snapshot.station, snapshot.observed_at_ms);
    },
    showCurrent(current) {
      // With no snapshot on the page, a live reading must not paper over the
      // failure: keep it for when the snapshot arrives.
      if (failed) observedAtMs = newestObservation(observedAtMs, current.observed_at_ms);
      else take(current.station, current.observed_at_ms);
    },
    fail(retry, message = "The dashboard could not be loaded.") {
      shown = false;
      failed = true;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "masthead-retry";
      button.textContent = "Try again";
      button.addEventListener("click", retry);
      reading.when.replaceChildren(message, " ", button);
      reading.age.textContent = "";
      reading.root.dataset.stale = "true";
      reading.root.dataset.failed = "true";
      figures.root.hidden = true;
    },
    tick,
  };
}

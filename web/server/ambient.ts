/**
 * Read the live reading from the Ambient Weather Network.
 *
 * One call to `/devices` returns every station on the account with its
 * newest `lastData`; the station is picked by MAC address, never by position.
 */
import type { LastData } from "../src/snapshot.js";

/** Ambient's endpoint when `AMBIENT_ENDPOINT` is not set. */
export const DEFAULT_AMBIENT_ENDPOINT = "https://rt.ambientweather.net/v1";

/** What Ambient needs to answer. */
export interface AmbientConfig {
  apiKey: string;
  applicationKey: string;
  endpoint: string;
}

/** One station's live reading as Ambient reports it. */
export interface LiveReading {
  lastData: LastData;
  /** The station's name in Ambient, or null. */
  name: string | null;
}

/**
 * Read one station's live reading; null when Ambient does not list it.
 * Rejects with an `AmbientReadError` when Ambient fails.
 */
export type LiveReader = (mac: string) => Promise<LiveReading | null>;

/** Ambient did not answer with a device list. */
export class AmbientReadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AmbientReadError";
  }
}

/**
 * A rejected fetch's name and message with both keys, plain or URL-encoded,
 * replaced by "[redacted]".
 *
 * @param error - What the fetch rejected with.
 * @param config - Holds the keys to take out.
 * @returns "<name>: <message>" without the keys.
 */
function redact(error: unknown, config: AmbientConfig): string {
  let text = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  for (const key of [config.apiKey, config.applicationKey]) {
    if (!key) continue;
    for (const form of new Set([key, encodeURIComponent(key)])) {
      text = text.split(form).join("[redacted]");
    }
  }
  return text;
}

/** The parts of an Ambient `/devices` entry this module reads. */
interface AmbientDevice {
  macAddress?: unknown;
  info?: { name?: unknown } | null;
  lastData?: unknown;
}

/**
 * A reader for live readings from Ambient.
 *
 * @param config - The API and application keys and the endpoint.
 * @param fetchImpl - The fetch to call Ambient with.
 * @returns The reader.
 */
export function ambientReader(
  config: AmbientConfig,
  fetchImpl: typeof fetch = fetch,
): LiveReader {
  return async (mac) => {
    const url = new URL(`${config.endpoint.replace(/\/+$/, "")}/devices`);
    url.searchParams.set("apiKey", config.apiKey);
    url.searchParams.set("applicationKey", config.applicationKey);
    let response: Response;
    try {
      response = await fetchImpl(url);
    } catch (error) {
      // A network error can carry the URL, and the URL carries the keys.
      throw new AmbientReadError(
        `Ambient /devices fetch failed: ${redact(error, config)}`,
      );
    }
    if (!response.ok) {
      throw new AmbientReadError(`Ambient ${response.status} on /devices`);
    }
    const devices = (await response.json()) as unknown;
    if (!Array.isArray(devices)) {
      throw new AmbientReadError("Ambient /devices did not return a list");
    }
    const device = (devices as AmbientDevice[]).find(
      (d) => typeof d.macAddress === "string" && d.macAddress.toUpperCase() === mac,
    );
    const lastData = device?.lastData as LastData | undefined;
    if (!device || !lastData || typeof lastData.dateutc !== "number") return null;
    const name = device.info?.name;
    return { lastData, name: typeof name === "string" ? name : null };
  };
}

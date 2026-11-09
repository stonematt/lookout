/**
 * Read the live reading from the Ambient Weather Network.
 *
 * One call to `/devices` returns every station on the account with its
 * newest `lastData`; the station is picked by MAC address, never by position.
 */
import type { LastData } from "../src/snapshot.js";
import { describeError } from "./errors.js";

/** Ambient's endpoint when `AMBIENT_ENDPOINT` is not set. */
export const DEFAULT_AMBIENT_ENDPOINT = "https://rt.ambientweather.net/v1";

/** How long one Ambient call may take, body included, before it fails. */
export const AMBIENT_TIMEOUT_MS = 5_000;

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

/** The parts of an Ambient `/devices` entry this module reads. */
interface AmbientDevice {
  macAddress?: unknown;
  info?: { name?: unknown } | null;
  lastData?: unknown;
}

/** One device on the Ambient account, with its newest reading. */
export interface LiveDevice {
  /** Its MAC address, upper case. */
  mac: string;
  /** Its name in Ambient, or null. */
  name: string | null;
  /** Its newest reading; null when Ambient has none with a `dateutc`. */
  lastData: LastData | null;
}

/**
 * Read every device on the account in one `/devices` call.
 * Rejects with an `AmbientReadError` when Ambient fails.
 */
export type DevicesReader = () => Promise<LiveDevice[]>;

/**
 * A reader for every device on the Ambient account: one `/devices` call.
 *
 * @param config - The API and application keys and the endpoint.
 * @param fetchImpl - The fetch to call Ambient with.
 * @param timeoutMs - How long one call may take before it fails.
 * @returns The reader.
 */
export function ambientDevicesReader(
  config: AmbientConfig,
  fetchImpl: typeof fetch = fetch,
  timeoutMs: number = AMBIENT_TIMEOUT_MS,
): DevicesReader {
  return async () => {
    const url = new URL(`${config.endpoint.replace(/\/+$/, "")}/devices`);
    url.searchParams.set("apiKey", config.apiKey);
    url.searchParams.set("applicationKey", config.applicationKey);
    let response: Response;
    try {
      response = await fetchImpl(url, { signal: AbortSignal.timeout(timeoutMs) });
    } catch (error) {
      // A network error can carry the URL, and the URL carries the keys.
      const keys = [config.apiKey, config.applicationKey];
      throw new AmbientReadError(`Ambient /devices fetch failed: ${describeError(error, keys)}`);
    }
    if (!response.ok) {
      throw new AmbientReadError(`Ambient ${response.status} on /devices`);
    }
    const devices = (await response.json()) as unknown;
    if (!Array.isArray(devices)) {
      throw new AmbientReadError("Ambient /devices did not return a list");
    }
    const live: LiveDevice[] = [];
    for (const device of devices as AmbientDevice[]) {
      if (typeof device.macAddress !== "string") continue;
      const lastData = device.lastData as LastData | undefined;
      const name = device.info?.name;
      live.push({
        mac: device.macAddress.toUpperCase(),
        name: typeof name === "string" ? name : null,
        lastData: lastData && typeof lastData.dateutc === "number" ? lastData : null,
      });
    }
    return live;
  };
}

/**
 * A reader for one station's live reading, picked by MAC from one
 * `/devices` call.
 *
 * @param config - The API and application keys and the endpoint.
 * @param fetchImpl - The fetch to call Ambient with.
 * @param timeoutMs - How long one call may take before it fails.
 * @returns The reader.
 */
export function ambientReader(
  config: AmbientConfig,
  fetchImpl: typeof fetch = fetch,
  timeoutMs: number = AMBIENT_TIMEOUT_MS,
): LiveReader {
  const readDevices = ambientDevicesReader(config, fetchImpl, timeoutMs);
  return async (mac) => {
    const device = (await readDevices()).find((d) => d.mac === mac);
    if (!device || !device.lastData) return null;
    return { lastData: device.lastData, name: device.name };
  };
}

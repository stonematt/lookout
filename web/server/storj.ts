/**
 * Read snapshot documents from Storj's S3 gateway.
 *
 * Vercel holds a read-only credential limited to `snapshot/`, so this module
 * only ever GETs keys under that prefix. Requests are signed with SigV4 by
 * `aws4fetch` and sent with the `fetch` it is given, which tests fake.
 */
import { AwsClient } from "aws4fetch";

import type { RainEventRow } from "../src/snapshot.js";
import { parseDashboard } from "../src/snapshot.js";
import { describeError } from "./errors.js";

/** The prefix every snapshot document lives under. */
export const SNAPSHOT_PREFIX = "snapshot/";

/** How long one Storj read may take, body included, before it fails. */
export const STORJ_TIMEOUT_MS = 5_000;

/** The two documents a refresh run publishes per station. */
export type SnapshotKind = "dashboard" | "context";

/** What Storj needs to sign a read. */
export interface StorjConfig {
  accessKeyId: string;
  secretAccessKey: string;
  endpointUrl: string;
  bucket: string;
}

/**
 * Read one snapshot document's text; null when Storj has no such key.
 * Rejects with a `StorjReadError` on any other failure.
 */
export type SnapshotReader = (mac: string, kind: SnapshotKind) => Promise<string | null>;

/** A start the Event Catalog does not list: its curve file was not read. */
export const UNLISTED: unique symbol = Symbol("unlisted");

/**
 * Read one completed event's curve file; null when it is not written yet,
 * `UNLISTED` when the Event Catalog does not list the start.
 * Rejects with a `StorjReadError` on any other failure.
 */
export type CurveReader = (
  mac: string,
  startMs: number,
) => Promise<string | null | typeof UNLISTED>;

/** Storj answered, but not with the document. */
export class StorjReadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "StorjReadError";
  }
}

/**
 * The Storj key of one station's document, as `lookout/snapshot/publish.py`
 * writes it.
 *
 * @param mac - The station's MAC address.
 * @param kind - "dashboard" or "context".
 * @returns e.g. `snapshot/00:11:22:33:44:55.dashboard.json`.
 */
export function snapshotKey(mac: string, kind: SnapshotKind): string {
  return `${SNAPSHOT_PREFIX}${mac}.${kind}.json`;
}

/**
 * The Storj key of one event's curve file, as `lookout/snapshot/curves.py`
 * writes it.
 *
 * @param mac - The station's MAC address.
 * @param startMs - The event's start, its key.
 * @returns e.g. `snapshot/00:11:22:33:44:55.events/1767758400000.json`.
 */
export function eventCurveKey(mac: string, startMs: number): string {
  return `${SNAPSHOT_PREFIX}${mac}.events/${startMs}.json`;
}

/**
 * Read any key under `snapshot/` from Storj.
 *
 * @param config - The read-only credential, endpoint and bucket.
 * @param fetchImpl - The fetch to send signed requests with.
 * @param timeoutMs - How long one read may take before it fails.
 * @returns A reader: the key's text, or null when Storj has no such key.
 */
function storjKeyReader(
  config: StorjConfig,
  fetchImpl: typeof fetch,
  timeoutMs: number,
): (key: string) => Promise<string | null> {
  const client = new AwsClient({
    accessKeyId: config.accessKeyId,
    secretAccessKey: config.secretAccessKey,
    service: "s3",
    // Storj ignores the region; the Python client signs with this one too.
    region: "us-east-1",
  });
  const base = config.endpointUrl.replace(/\/+$/, "");
  return async (key) => {
    const url = `${base}/${config.bucket}/${encodeURIComponent(key).replace(/%2F/g, "/")}`;
    const request = await client.sign(url, { method: "GET" });
    const response = await fetchImpl(request, { signal: AbortSignal.timeout(timeoutMs) });
    if (response.status === 404) return null;
    if (!response.ok) {
      throw new StorjReadError(`Storj ${response.status} reading ${key}`);
    }
    return response.text();
  };
}

/**
 * How often, at most, one function instance reads a station's dashboard to
 * learn the Event Catalog's starts: the bound on what `/event` misses cost.
 */
export const EVENT_STARTS_REFRESH_MS = 60_000;

/**
 * What one function instance knows of a station's curve files: the starts
 * the Event Catalog lists, learned from the dashboard document, which it
 * reads at most once per `EVENT_STARTS_REFRESH_MS`.
 */
class KnownStarts {
  /** The starts with a curve file; null before a good dashboard read. */
  private starts: Set<number> | null = null;
  /** When the last dashboard read began, whatever its outcome. */
  private askedAtMs = -Infinity;
  /** The last dashboard read failed: its starts may be out of date. */
  private failed = false;
  /** The dashboard read under way, shared by requests that arrive together. */
  private pending: Promise<void> | null = null;

  /**
   * @param readDashboard - Reads the station's dashboard document's text.
   */
  constructor(private readonly readDashboard: () => Promise<string | null>) {}

  /**
   * Whether the Event Catalog lists a start, reading the dashboard only for
   * a start not already known, and only when the interval allows.
   *
   * @param startMs - The start asked for.
   * @returns True when it has a curve file to read.
   * @throws StorjReadError when the dashboard read fails, or failed last
   *   time, and the start is not already known.
   */
  async lists(startMs: number): Promise<boolean> {
    if (this.starts?.has(startMs)) return true;
    if (!this.pending && Date.now() - this.askedAtMs >= EVENT_STARTS_REFRESH_MS) {
      this.askedAtMs = Date.now();
      this.pending = this.refresh();
    }
    if (this.pending) await this.pending;
    else if (this.failed) {
      throw new StorjReadError("the Event Catalog's starts could not be read");
    }
    return this.starts?.has(startMs) ?? false;
  }

  private async refresh(): Promise<void> {
    try {
      this.starts = curveStarts(await this.readDashboard());
      this.failed = false;
    } catch (error) {
      this.failed = true;
      throw error;
    } finally {
      this.pending = null;
    }
  }
}

/**
 * By endpoint, bucket and station (`checkQuery` admits only the configured
 * one). Module state, so it lasts as long as a warm function instance
 * does, though the reader is built per request.
 */
const knownStarts = new Map<string, KnownStarts>();

/**
 * The starts a dashboard document says have a curve file: the Event
 * Catalog's completed events with readings. The live event's file is
 * written only once the catalog closes it.
 *
 * @param text - The dashboard document's text, or null when there is none.
 * @returns The starts.
 * @throws StorjReadError when the document breaks the contract.
 */
function curveStarts(text: string | null): Set<number> {
  if (text === null) return new Set();
  let events: RainEventRow[];
  try {
    events = parseDashboard(JSON.parse(text)).rain_events.events;
  } catch (error) {
    throw new StorjReadError(`dashboard breaks the contract: ${describeError(error)}`);
  }
  return new Set(events.filter((e) => e.has_curve && !e.live).map((e) => e.start_ms));
}

/**
 * A reader for event curve files on Storj.
 *
 * It reads a curve file only for a start the Event Catalog lists, as the
 * station's dashboard document has it, so a caller varying `start` cannot
 * drive Storj reads (#231). An instance reads the dashboard when asked for
 * a start it does not know, at most once per `EVENT_STARTS_REFRESH_MS`;
 * in between, such a start is `UNLISTED` without any read.
 *
 * @param config - The read-only credential, endpoint and bucket.
 * @param fetchImpl - The fetch to send signed requests with.
 * @param timeoutMs - How long one read may take before it fails.
 * @returns The reader. For a start it does not know, it rejects with a
 *   `StorjReadError` while the last dashboard read failed.
 */
export function storjCurveReader(
  config: StorjConfig,
  fetchImpl: typeof fetch = fetch,
  timeoutMs: number = STORJ_TIMEOUT_MS,
): CurveReader {
  const read = storjKeyReader(config, fetchImpl, timeoutMs);
  return async (mac, startMs) => {
    const id = `${config.endpointUrl} ${config.bucket} ${mac}`;
    let known = knownStarts.get(id);
    if (!known) {
      known = new KnownStarts(() => read(snapshotKey(mac, "dashboard")));
      knownStarts.set(id, known);
    }
    return (await known.lists(startMs)) ? read(eventCurveKey(mac, startMs)) : UNLISTED;
  };
}

/**
 * A reader for snapshot documents on Storj.
 *
 * @param config - The read-only credential, endpoint and bucket.
 * @param fetchImpl - The fetch to send signed requests with.
 * @param timeoutMs - How long one read may take before it fails.
 * @returns The reader.
 */
export function storjReader(
  config: StorjConfig,
  fetchImpl: typeof fetch = fetch,
  timeoutMs: number = STORJ_TIMEOUT_MS,
): SnapshotReader {
  const read = storjKeyReader(config, fetchImpl, timeoutMs);
  return (mac, kind) => read(snapshotKey(mac, kind));
}

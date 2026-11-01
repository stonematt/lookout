/**
 * Read snapshot documents from Storj's S3 gateway.
 *
 * Vercel holds a read-only credential limited to `snapshot/`, so this module
 * only ever GETs keys under that prefix. Requests are signed with SigV4 by
 * `aws4fetch` and sent with the `fetch` it is given, which tests fake.
 */
import { AwsClient } from "aws4fetch";

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

/**
 * Read one completed event's curve file; null when it is not written yet.
 * Rejects with a `StorjReadError` on any other failure.
 */
export type CurveReader = (mac: string, startMs: number) => Promise<string | null>;

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
 * A reader for event curve files on Storj.
 *
 * @param config - The read-only credential, endpoint and bucket.
 * @param fetchImpl - The fetch to send signed requests with.
 * @param timeoutMs - How long one read may take before it fails.
 * @returns The reader.
 */
export function storjCurveReader(
  config: StorjConfig,
  fetchImpl: typeof fetch = fetch,
  timeoutMs: number = STORJ_TIMEOUT_MS,
): CurveReader {
  const read = storjKeyReader(config, fetchImpl, timeoutMs);
  return (mac, startMs) => read(eventCurveKey(mac, startMs));
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

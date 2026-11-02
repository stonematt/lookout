/**
 * Errors as log lines the read functions can safely write: name and message
 * only, never the stack, cause or an attached request, with secrets removed.
 */

/** Query parameters whose values are secrets, as Ambient's URL carries them. */
const SECRET_PARAM = /\b(apiKey|applicationKey)=[^&\s"']*/gi;

/**
 * An error as one log line, with key-shaped query parameters redacted and
 * any given secret removed, plain or URL-encoded.
 *
 * @param error - Anything a promise rejected with.
 * @param secrets - Values to take out wherever they appear.
 * @returns "<name>: <message>", safe to log.
 */
export function describeError(error: unknown, secrets: readonly string[] = []): string {
  let text =
    error instanceof Error ? `${error.name}: ${error.message}` : `non-Error: ${String(error)}`;
  for (const secret of secrets) {
    if (!secret) continue;
    for (const form of new Set([secret, encodeURIComponent(secret)])) {
      text = text.split(form).join("[redacted]");
    }
  }
  return text.replace(SECRET_PARAM, "$1=[redacted]");
}

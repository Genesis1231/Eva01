import { LiveFeed } from "./LiveFeed";

/**
 * Build the feed for the current mode. Every feed exposes the same interface:
 *   start(), stop(), onEvent(handler) -> unsubscribe
 *
 * Both modes are live SSE — they differ only in WHERE the Room listens:
 *   local  → "/api/stream" on this same dev server (Eva running on this machine)
 *   remote → config.feed.remoteUrl (a deployed/remote Eva's relay)
 * Same events.js shapes either way, so the Room needs no changes.
 */
export function makeFeed(config) {
  const { mode = "local", remoteUrl } = config?.feed || {};
  if (mode === "remote" && remoteUrl) return new LiveFeed({ url: remoteUrl });
  return new LiveFeed(); // local: same-origin /api/stream
}

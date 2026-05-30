/**
 * LiveFeed — consumes Eva's live feed over SSE (GET /api/stream by default, or a
 * remote relay URL). Implements the Feed interface — start() / stop() / onEvent()
 * — so the Room is transport-blind: local vs remote changes only the URL.
 *
 * Read-only: the browser only receives. EventSource reconnects on its own if the
 * dev server restarts, so there's no reconnect logic to maintain here.
 */
export class LiveFeed {
  constructor({ url = "/api/stream" } = {}) {
    this._url = url;
    this._handlers = new Set();
    this._source = null;
  }

  onEvent(handler) {
    this._handlers.add(handler);
    return () => this._handlers.delete(handler);
  }

  start() {
    if (this._source) return;
    this._source = new EventSource(this._url);
    this._source.onmessage = (e) => {
      let event;
      try {
        event = JSON.parse(e.data);
      } catch {
        return; // ignore malformed frames (SSE comments never reach here)
      }
      for (const h of this._handlers) h(event); // relay already stamped ts
    };
  }

  stop() {
    if (this._source) {
      this._source.close();
      this._source = null;
    }
  }
}

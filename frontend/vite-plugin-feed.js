/**
 * Eva's Room feed — a dev-only relay living in the Vite dev server.
 *
 * One pipe, two ends:
 *   POST /api/post    ← Eva (the Python backend) posts every event here
 *                       (mood, sense/speech lines, canvas surfaces) as one stream.
 *   GET  /api/stream  → the Room subscribes here over SSE and receives that
 *                       same mixed stream; it routes each event to a section
 *                       by `kind` (Room.jsx). The server is a dumb relay.
 *
 * The last mood (and artifact) is retained and replayed to a freshly-opened
 * Room so the inner-state bar isn't blank on load.
 *
 * Dev-only: a built/deployed frontend has no Node process — a headless Room
 * would need a standalone relay instead. Fine while the Room is a localhost tool.
 */

import { CANVAS_KINDS } from "./src/feed/events.js"; // single source of truth

const clients = new Set(); // open SSE responses
const retained = new Map(); // slot -> last event, for snapshot-on-connect

const MAX_BODY = 1 << 20; // 1 MB — Eva's events are tiny; this only stops runaways

// Write one SSE frame. Returns false if the socket is dead, so the caller can
// drop that client instead of letting one stale connection break the fan-out.
function sse(res, event) {
  try {
    res.write(`data: ${JSON.stringify(event)}\n\n`);
    return true;
  } catch {
    return false;
  }
}

// Read, parse, and validate one posted event, then hand it to `done`. Guards the
// ingest boundary: oversized body → 413, malformed JSON or non-string kind/text
// → 400. Accumulation stops at MAX_BODY so a runaway producer can't grow memory.
function readEvent(req, res, done) {
  let body = "";
  let over = false;
  req.on("data", (chunk) => {
    if (over) return;
    body += chunk;
    if (body.length > MAX_BODY) {
      over = true;
      res.statusCode = 413;
      res.end();
    }
  });
  req.on("end", () => {
    if (over) return;
    let event;
    try {
      event = JSON.parse(body);
    } catch {
      res.statusCode = 400;
      return res.end();
    }
    if (typeof event.kind !== "string" || typeof event.text !== "string") {
      res.statusCode = 400;
      return res.end();
    }
    done(event);
  });
}

export default function evaFeed() {
  return {
    name: "eva-feed",
    configureServer(server) {
      // ── out: the Room listens here (SSE) ──────────────────────────────
      server.middlewares.use("/api/stream", (req, res) => {
        res.writeHead(200, {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        });
        res.flushHeaders?.();
        res.write(":ok\n\n"); // open the stream
        clients.add(res);
        // replay retained state so a new Room paints immediately
        for (const event of retained.values()) sse(res, event);
        req.on("close", () => clients.delete(res));
      });

      // ── in: Eva posts here (POST) ─────────────────────────────────────
      server.middlewares.use("/api/post", (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          return res.end();
        }
        readEvent(req, res, (event) => {
          event.ts = Date.now(); // authoritative server time
          if (event.kind === "mood") {
            retained.set("mood", event);
          } else if (CANVAS_KINDS.has(event.kind)) {
            retained.set("artifact", event); // latest desk surface
          }
          // sense / speech are transient log lines — not retained
          for (const client of clients) {
            if (!sse(client, event)) clients.delete(client);
          }
          res.statusCode = 204;
          res.end();
        });
      });
    },
  };
}

/**
 * The feed contract. Eva's backend emits these and LiveFeed receives them raw
 * over SSE — nothing on the frontend constructs them, so this file is mostly
 * documentation (plus the CANVAS_KINDS set below, which Room + CanvasPage AND the
 * dev-server relay all import — keep this file dependency-free so the relay can).
 *
 * Every event is the same flat shape:
 *
 *     { kind, text, ts }
 *
 * Everything Eva emits is text. `kind` is the type; `text` is the content (for a
 * canvas surface, its source path or url); `ts` (epoch ms) is stamped by the
 * relay on ingest. One method posts them all: `feeder.post(kind, text)`.
 *
 *   kind                       text                        where it goes
 *   ────────────────────────   ─────────────────────────   ────────────────────────
 *   "mood"                     "curious, tender"           → InnerState bar
 *   "sense"                    "Adam said: morning, eva"   → StreamFeed (roman)
 *   "speech"                   "Morning, Adam."            → StreamFeed (italic + quotes)
 *   "image"|"doc"|"video"|     a file path or a url        → Canvas (kind = the surface,
 *     "browser"|"art"                                        no guessing)
 *
 * Routing rule: "mood" → bar, a CANVAS_KINDS member → the desk, anything else
 * (sense / speech / a future "thought") → the stream. To add a surface, add its
 * kind here and a case in Canvas's Surface — the wire shape never changes.
 */

export const CANVAS_KINDS = new Set(["image", "doc", "video", "browser", "art"]);

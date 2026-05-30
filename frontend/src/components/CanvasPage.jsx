import { useEffect, useState } from "react";
import { useFeed } from "../feed/useFeed";
import { CANVAS_KINDS } from "../feed/events";
import Canvas from "./Canvas";

/**
 * Standalone /canvas route — the deterministic, artifact-only capture surface
 * that feeds the image-novelty channel. `capture-mode` (on <body>) freezes
 * animation/transitions and strips grain, lamp-glow, and cursor; `variant=full`
 * is borderless/shadowless. The same artifact captures to identical pixels →
 * zero novelty. metadata (title/by) stays on the event, never baked into pixels.
 *
 * `/canvas?artifact=<text>` pins one artifact: the backend screenshotter passes
 * the exact artifact string it's embedding, so it captures THAT one, not whatever
 * happens to be current (closes the per-mount race). No param ⇒ track the latest.
 * The screenshotter should wait for [data-ready="true"] before capturing.
 */
export default function CanvasPage() {
  const wanted = new URLSearchParams(window.location.search).get("artifact");
  const [art, setArt] = useState(null);

  useFeed((e) => {
    if (!CANVAS_KINDS.has(e.kind)) return;
    if (!wanted || e.text === wanted) setArt(e);
  });

  useEffect(() => {
    document.body.classList.add("capture-mode");
    return () => document.body.classList.remove("capture-mode");
  }, []);

  return (
    <div
      data-ready={!!art}
      data-artifact={art?.text || ""}
      className="flex h-screen w-screen items-center justify-center bg-ink"
    >
      {art && <Canvas kind={art.kind} text={art.text} variant="full" />}
    </div>
  );
}

import { useRef, useState } from "react";
import { useFeed } from "../feed/useFeed";
import { CANVAS_KINDS } from "../feed/events";
import InnerState from "./InnerState";
import StreamFeed from "./StreamFeed";
import Canvas from "./Canvas";

const STREAM_CAP = 16; // keep the feed to what fits the column; oldest dissolve out

/**
 * Eva's Room — her self-perception surface. Subscribes to the feed and routes
 * each event: stream → the log, mood → the inner-state bar, artifact → the desk.
 * Two columns: a 2/5 stream and a 3/5 canvas (the canvas is the focal piece).
 */
export default function Room() {
  const [items, setItems] = useState([]);
  const [mood, setMood] = useState("");
  const [art, setArt] = useState(null);
  const idRef = useRef(0);

  // Every event is { kind, text, ts }. mood → the bar, a canvas surface → the
  // desk, everything else (sense / speech) → the stream log. memo(Canvas) keys on
  // kind+text, so re-emitting the same surface (relay replay) is a no-op.
  useFeed((e) => {
    if (e.kind === "mood") {
      setMood(e.text);
    } else if (CANVAS_KINDS.has(e.kind)) {
      setArt(e);
    } else {
      setItems((s) => [...s, { ...e, id: ++idRef.current }].slice(-STREAM_CAP));
    }
  });

  return (
    <div className="relative z-10 mx-auto flex h-screen max-w-7xl flex-col px-10 pt-10 pb-9">
      <InnerState mood={mood} />
      <div className="grid min-h-0 flex-1 grid-cols-5 gap-10 pt-8">
        <StreamFeed items={items} />
        <section className="col-span-3 min-h-0 border-l border-rule pl-10">
          <Canvas kind={art?.kind} text={art?.text} />
        </section>
      </div>
    </div>
  );
}

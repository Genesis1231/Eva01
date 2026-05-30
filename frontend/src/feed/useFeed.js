import { useEffect, useRef } from "react";
import config from "../config";
import { makeFeed } from "./Feed";

/**
 * Subscribe to Eva's feed for the lifetime of a component. Creates its own feed
 * instance (its own SSE subscription), forwards every event to `handler`, and
 * tears down on unmount. Used by both Room and the standalone CanvasPage.
 */
export function useFeed(handler) {
  const ref = useRef(handler);
  useEffect(() => {
    ref.current = handler;
  });

  useEffect(() => {
    const feed = makeFeed(config);
    const off = feed.onEvent((event) => ref.current(event));
    feed.start();
    return () => {
      off();
      feed.stop();
    };
  }, []);
}

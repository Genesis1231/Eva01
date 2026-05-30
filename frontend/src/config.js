/**
 * EVA App Configuration
 * Central configuration file for the EVA React application
 */

const config = {
  websocket: {
    baseUrl: import.meta.env.VITE_EVA_WS_URL || "ws://localhost:8080",
    reconnectInterval: 3000,
    reconnectAttempts: 5,
  },
  api: {
    baseUrl: import.meta.env.VITE_EVA_API_URL || "http://localhost:8080",
    downloadPath: "/download",
  },
  feed: {
    // "local" → same-origin /api/stream (Eva on this machine);
    // "remote" → VITE_EVA_FEED_URL (a deployed Eva's relay).
    mode: import.meta.env.VITE_FEED_MODE || "local",
    remoteUrl: import.meta.env.VITE_EVA_FEED_URL || "",
  },
};

export default config;

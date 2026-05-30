import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import evaFeed from "./vite-plugin-feed.js";

export default defineConfig({
  plugins: [react(), evaFeed()],
  server: {
    port: 3000,
  },
});
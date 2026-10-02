// The review UI is served by the PassageWatch API in production (same origin).
// During development, `npm run dev` proxies API calls to a running service.
import { defineConfig } from "vitest/config";

const api = process.env.PASSAGEWATCH_API ?? "http://127.0.0.1:8010";

export default defineConfig({
  server: {
    proxy: { "/v1": api, "/health": api },
  },
  build: { outDir: "dist", sourcemap: true },
  test: { environment: "node" },
});

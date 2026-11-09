import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

import { goldenReadFunctions } from "./server/dev-fixtures.js";

export default defineConfig({
  plugins: [goldenReadFunctions()],
  server: {
    // The lane server: `npm run dev -- --port <port>`.
    strictPort: true,
  },
  build: {
    outDir: "dist",
    target: "es2022",
    rollupOptions: {
      // The site, and the debug page that dumps each station's lastData.
      input: {
        main: fileURLToPath(new URL("./index.html", import.meta.url)),
        debug: fileURLToPath(new URL("./debug.html", import.meta.url)),
      },
    },
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    setupFiles: ["tests/setup.ts"],
  },
});

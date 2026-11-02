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
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    setupFiles: ["tests/setup.ts"],
  },
});

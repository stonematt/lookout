// ESLint for the site (#260). Types are tsc's job (`npm run typecheck`), so
// this runs typescript-eslint's untyped `recommended` set. Formatting is
// Prettier's (`npm run format:check`).
//
// no-unsanitized guards the rule from #148: HTML built from snapshot values
// goes through `escapeHtml` (`src/dom.ts`, imported as `esc`). Any other
// value assigned to innerHTML, outerHTML or insertAdjacentHTML fails lint.
import js from "@eslint/js";
import nounsanitized from "eslint-plugin-no-unsanitized";
import globals from "globals";
import tseslint from "typescript-eslint";

const escaping = { escape: { methods: ["esc", "escapeHtml"] } };

export default tseslint.config(
  { ignores: ["dist/", "node_modules/"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    plugins: { "no-unsanitized": nounsanitized },
    languageOptions: { globals: { ...globals.browser, ...globals.node } },
    rules: {
      "no-unsanitized/method": ["error", escaping],
      "no-unsanitized/property": ["error", escaping],
      // `const { gone: _gone, ...rest } = x` drops a key on purpose.
      "@typescript-eslint/no-unused-vars": ["error", { ignoreRestSiblings: true }],
    },
  },
);

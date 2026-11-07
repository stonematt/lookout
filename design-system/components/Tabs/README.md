The section tabs are Overview, Rain, Rain Events and Solar. On desktop they are pills on the right of the header: 40px tall, `ink-2` text, and the current tab an `ink` pill with `ground` text and weight 600. On a phone they are a sticky bottom bar of four equal 48px targets.

**Consumer provides.** The four labels and which is open; the open tab is tracked in the URL hash.

**Use.** Build a WAI-ARIA tablist with a roving tabindex and arrows to move. The phone bar is padded below by `env(safe-area-inset-bottom)` (the page sets `viewport-fit=cover`) and its labels use `annotation` size so each fits one line at 390px. The bar names the tab, so the header always names the station.

Static rendition of `web/src/tabs.ts` and `web/src/styles/base.css`.

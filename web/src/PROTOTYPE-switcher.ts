/**
 * PROTOTYPE (issue #619, round 5): a static note, dev builds only, shown when
 * `?rainwindows=` feeds fake rain totals. Nothing is left to switch. Throwaway.
 */
if (!import.meta.env.PROD) {
  const over = new URLSearchParams(location.search).get("rainwindows");
  if (over) {
    const bar = document.createElement("div");
    bar.className = "pg-switcher";
    bar.dataset.prototype = "rain-gauge";
    bar.textContent = `PROTOTYPE · FAKE rain totals (?rainwindows=${over})`;
    document.body.append(bar);
  }
}

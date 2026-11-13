/**
 * PROTOTYPE (issue #619, round 3): a small note, dev builds only, that says
 * the rain totals are fake when `?rainwindows=` is set. No variant switching
 * any more. Throwaway.
 */
if (!import.meta.env.PROD) {
  const q = new URLSearchParams(location.search);
  const over = q.get("rainwindows");
  if (over) {
    const bar = document.createElement("div");
    bar.className = "pg-switcher";
    bar.dataset.prototype = "rain-gauge";
    bar.textContent = `PROTOTYPE: FAKE rain totals (?rainwindows=${over})`;
    document.body.append(bar);
  }
}

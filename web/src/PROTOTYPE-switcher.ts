/**
 * PROTOTYPE (issue #619, round 6): the floating switcher. Arrows and the
 * left/right keys cycle `?years=` between bars and dots (reloads the page).
 * Small, bottom-left, dev builds only. Throwaway.
 */
const ORDER = ["bars", "dots"];

function go(): void {
  const q = new URLSearchParams(location.search);
  const cur = q.get("years") === "dots" ? "dots" : "bars";
  q.set("years", cur === "bars" ? "dots" : "bars");
  location.replace(`${location.pathname}?${q.toString()}${location.hash}`);
}

if (!import.meta.env.PROD) {
  const q = new URLSearchParams(location.search);
  const cur = q.get("years") === "dots" ? "dots" : "bars";
  const bar = document.createElement("div");
  bar.className = "pg-switcher";
  bar.dataset.prototype = "rain-gauge";
  const label = document.createElement("span");
  const [a, b] = ORDER.map((m) => (m === cur ? `[${m}]` : m));
  label.textContent = `PROTOTYPE years: ${a}|${b}${q.get("rainwindows") ? " · FAKE totals" : ""}`;
  const mk = (txt: string, aria: string) => {
    const btn = document.createElement("button");
    btn.textContent = txt;
    btn.setAttribute("aria-label", aria);
    btn.addEventListener("click", go);
    return btn;
  };
  bar.append(mk("←", "Previous years style"), label, mk("→", "Next years style"));
  document.addEventListener("keydown", (e) => {
    const t = e.target as HTMLElement | null;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") go();
  });
  document.body.append(bar);
}

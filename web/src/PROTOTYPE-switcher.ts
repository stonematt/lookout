/**
 * PROTOTYPE (issue #619, round 4): the floating switcher. Arrows and the
 * left/right keys cycle `?hero=` through off, a, b, c (reloads the page); the
 * gauge stays B. Small, bottom-left, dev builds only. Throwaway.
 */
const ORDER = ["", "a", "b", "c"];

function go(delta: number): void {
  const q = new URLSearchParams(location.search);
  const cur = q.get("hero") === "1" ? "c" : (q.get("hero") ?? "");
  const i = Math.max(ORDER.indexOf(cur), 0);
  const next = ORDER[(i + delta + ORDER.length) % ORDER.length] as string;
  if (next) q.set("hero", next);
  else q.delete("hero");
  location.replace(`${location.pathname}?${q.toString()}${location.hash}`);
}

if (!import.meta.env.PROD) {
  const q = new URLSearchParams(location.search);
  const bar = document.createElement("div");
  bar.className = "pg-switcher";
  bar.dataset.prototype = "rain-gauge";
  const label = document.createElement("span");
  const fake = q.get("rainwindows") ? " | FAKE rain totals" : "";
  label.textContent = `PROTOTYPE hero ${q.get("hero") ?? "off"} | gauge B${fake}`;
  const mk = (txt: string, aria: string, d: number) => {
    const b = document.createElement("button");
    b.textContent = txt;
    b.setAttribute("aria-label", aria);
    b.addEventListener("click", () => go(d));
    return b;
  };
  bar.append(mk("←", "Previous hero option", -1), label, mk("→", "Next hero option", 1));
  document.addEventListener("keydown", (e) => {
    const t = e.target as HTMLElement | null;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
    if (e.key === "ArrowLeft") go(-1);
    else if (e.key === "ArrowRight") go(1);
  });
  document.body.append(bar);
}

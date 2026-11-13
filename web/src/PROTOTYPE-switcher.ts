/**
 * PROTOTYPE (issue #619): the floating variant switcher. Arrows and keys
 * cycle `?variant=` through off, A to D (reloads the page); hidden in
 * production builds. Throwaway.
 */
const ORDER = ["", "A", "B", "C", "D"];
const LABELS: Record<string, string> = {
  "": "off (window strip)",
  A: "A shared linear axis",
  B: "B tubes, own scale",
  C: "C shared log axis",
  D: "D one tall gauge",
};

function go(delta: number): void {
  const q = new URLSearchParams(location.search);
  const i = ORDER.indexOf(q.get("variant") ?? "");
  const next = ORDER[(Math.max(i, 0) + delta + ORDER.length) % ORDER.length] as string;
  if (next) q.set("variant", next);
  else q.delete("variant");
  location.replace(`${location.pathname}?${q.toString()}${location.hash}`);
}

if (!import.meta.env.PROD) {
  const q = new URLSearchParams(location.search);
  const cur = q.get("variant") ?? "";
  const bar = document.createElement("div");
  bar.className = "pg-switcher";
  bar.dataset.prototype = "rain-gauge";
  const fake = q.get("rainwindows") ? " | FAKE totals" : "";
  const label = document.createElement("span");
  label.textContent = `PROTOTYPE ${LABELS[cur] ?? cur}${fake}`;
  const prev = document.createElement("button");
  prev.textContent = "←";
  prev.setAttribute("aria-label", "Previous variant");
  prev.addEventListener("click", () => go(-1));
  const next = document.createElement("button");
  next.textContent = "→";
  next.setAttribute("aria-label", "Next variant");
  next.addEventListener("click", () => go(1));
  bar.append(prev, label, next);
  document.addEventListener("keydown", (e) => {
    const t = e.target as HTMLElement | null;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
    if (e.key === "ArrowLeft") go(-1);
    else if (e.key === "ArrowRight") go(1);
  });
  document.body.append(bar);
}

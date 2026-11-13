/**
 * PROTOTYPE (issue #619): the floating switcher. Arrows and keys cycle
 * `?variant=` through off, B, C; the button "hero" cycles `?hero=` through
 * off, 1, 2 (reloads the page). Hidden in production builds. Throwaway.
 */
const ORDER = ["", "B", "C"];
const HERO = ["", "1", "2"];
const LABELS: Record<string, string> = {
  "": "off",
  B: "B tubes",
  C: "C log axis",
};

function reload(q: URLSearchParams): void {
  location.replace(`${location.pathname}?${q.toString()}${location.hash}`);
}

function go(delta: number): void {
  const q = new URLSearchParams(location.search);
  const i = ORDER.indexOf(q.get("variant") ?? "");
  const next = ORDER[(Math.max(i, 0) + delta + ORDER.length) % ORDER.length] as string;
  if (next) q.set("variant", next);
  else q.delete("variant");
  reload(q);
}

function goHero(): void {
  const q = new URLSearchParams(location.search);
  const i = HERO.indexOf(q.get("hero") ?? "");
  const next = HERO[(Math.max(i, 0) + 1) % HERO.length] as string;
  if (next) q.set("hero", next);
  else q.delete("hero");
  reload(q);
}

if (!import.meta.env.PROD) {
  const q = new URLSearchParams(location.search);
  const cur = q.get("variant") ?? "";
  const bar = document.createElement("div");
  bar.className = "pg-switcher";
  bar.dataset.prototype = "rain-gauge";
  const fake = q.get("rainwindows") ? " | FAKE totals" : "";
  const label = document.createElement("span");
  label.textContent = `PROTOTYPE gauge ${LABELS[cur] ?? cur} | hero ${q.get("hero") ?? "off"}${fake}`;
  const mk = (txt: string, aria: string, fn: () => void) => {
    const b = document.createElement("button");
    b.textContent = txt;
    b.setAttribute("aria-label", aria);
    b.addEventListener("click", fn);
    return b;
  };
  bar.append(
    mk("←", "Previous variant", () => go(-1)),
    label,
    mk("→", "Next variant", () => go(1)),
    mk("hero", "Cycle hero placement", goHero),
  );
  document.addEventListener("keydown", (e) => {
    const t = e.target as HTMLElement | null;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
    if (e.key === "ArrowLeft") go(-1);
    else if (e.key === "ArrowRight") go(1);
  });
  document.body.append(bar);
}

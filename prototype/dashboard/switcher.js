// PROTOTYPE switcher. Loads data.json and exactly one variant script, chosen by ?variant=.
// Variant contract: variant-<key>.js sets
//   window.LOOKOUT_VARIANT = { name: "Short name", render(root, data) { ... } }
// and scopes all of its CSS under its own root class. Switching reloads the page,
// so variants never share styles or state.
(function () {
  const VARIANTS = ["A", "B", "C", "D"];
  const params = new URLSearchParams(location.search);
  let key = (params.get("variant") || "A").toUpperCase();
  if (!VARIANTS.includes(key)) key = "A";

  // Variant D only: ?style=1|2|3 picks a muted style (1 Paper, 2 Slate, 3 Mist) via html[data-style].
  const STYLES = { 1: "Paper", 2: "Slate", 3: "Mist" };
  let style = key === "D" ? String(params.get("style") || "1") : "";
  if (key === "D" && !STYLES[style]) style = "1";
  if (style) document.documentElement.setAttribute("data-style", style);

  // ?theme=light|dark overrides the OS setting via html[data-theme]; absent, the OS decides.
  const theme = params.get("theme");
  if (theme === "light" || theme === "dark") document.documentElement.setAttribute("data-theme", theme);
  const dark = theme ? theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;

  function go(delta) {
    const i = (VARIANTS.indexOf(key) + delta + VARIANTS.length) % VARIANTS.length;
    params.set("variant", VARIANTS[i]);
    location.search = params.toString();
  }

  function bar(name) {
    const el = document.createElement("div");
    el.setAttribute("style", [
      "position:fixed", "left:50%", "bottom:max(12px, env(safe-area-inset-bottom))",
      "transform:translateX(-50%)", "z-index:99999", "display:flex", "gap:6px",
      "align-items:center", "padding:6px 8px", "border-radius:999px",
      "background:#ff2d95", "color:#fff", "font:600 13px/1 system-ui, sans-serif",
      "box-shadow:0 4px 18px rgba(0,0,0,.35)",
    ].join(";"));
    const btn = (label, d) => {
      const b = document.createElement("button");
      b.textContent = label;
      b.setAttribute("style", "all:unset;cursor:pointer;padding:6px 10px;border-radius:999px;background:rgba(0,0,0,.2)");
      b.onclick = () => go(d);
      return b;
    };
    const label = document.createElement("span");
    label.textContent = `${key}${name ? ` (${name})` : ""}`;
    label.style.padding = "0 6px"; label.style.whiteSpace = "nowrap";
    el.append(btn("←", -1), label, btn("→", 1));
    if (style) {
      const sep = document.createElement("span");
      sep.setAttribute("style", "width:1px;height:18px;background:rgba(255,255,255,.5);margin:0 2px");
      el.append(sep);
      Object.keys(STYLES).forEach((n) => {
        const b = document.createElement("button");
        b.textContent = n;
        b.title = STYLES[n];
        b.setAttribute("aria-label", `Style ${n}, ${STYLES[n]}`);
        b.setAttribute("aria-pressed", String(n === style));
        b.setAttribute("style", `all:unset;cursor:pointer;padding:6px 10px;border-radius:999px;background:${n === style ? "#fff" : "rgba(0,0,0,.2)"};color:${n === style ? "#c4156b" : "#fff"}`);
        b.onclick = () => { params.set("style", n); location.search = params.toString(); };
        el.append(b);
      });
    }
    const sep2 = document.createElement("span");
    sep2.setAttribute("style", "width:1px;height:18px;background:rgba(255,255,255,.5);margin:0 2px");
    const tb = document.createElement("button");
    tb.textContent = dark ? "☀ Light" : "☾ Dark";
    tb.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
    tb.setAttribute("style", "all:unset;cursor:pointer;padding:6px 10px;border-radius:999px;background:rgba(0,0,0,.2);white-space:nowrap");
    tb.onclick = () => { params.set("theme", dark ? "light" : "dark"); location.search = params.toString(); };
    el.append(sep2, tb);
    document.body.appendChild(el);
  }

  document.addEventListener("keydown", (e) => {
    const t = e.target;
    if (t && (t.matches("input, textarea, select") || t.isContentEditable)) return;
    if (e.key === "ArrowLeft") go(-1);
    if (e.key === "ArrowRight") go(1);
  });

  const script = document.createElement("script");
  script.src = `variant-${key.toLowerCase()}.js`;
  script.onload = async () => {
    const v = window.LOOKOUT_VARIANT || {};
    bar(v.name);
    const data = await (await fetch("data.json")).json();
    v.render(document.getElementById("app"), data);
  };
  script.onerror = () => bar("missing");
  document.head.appendChild(script);
})();

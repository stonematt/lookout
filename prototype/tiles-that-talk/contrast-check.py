"""Check the DESIGN.md colour tokens against Lookout's contrast rules.

Text pairs need 4.5:1 and graphics that carry meaning need 3:1 (WCAG 1.4.11),
in both themes. A usual band passes by its fill (dark) or by its edge around
a pale fill (light). Text in a ramp cell passes if white or near-black does.

Run: python3 contrast-check.py   (prints each failure, then the count)
"""


def luminance(hex_colour: str) -> float:
    """Relative luminance of a #rrggbb colour.

    :param hex_colour: the colour, as "#rrggbb".
    :return: its WCAG relative luminance.
    """
    h = hex_colour.lstrip("#")
    channels = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def ratio(a: str, b: str) -> float:
    """WCAG contrast ratio between two colours.

    :param a: one colour.
    :param b: the other.
    :return: the ratio, 1 to 21.
    """
    hi, lo = sorted([luminance(a), luminance(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


THEMES = {
    "dark": dict(
        ground="#11161c", tile="#1a2129", now_surface="#3b4a5a", ink="#eef2f5", ink_2="#c3ccd5",
        ink_3="#aab5c0", ink_on_now="#dfe6ec", rule_strong="#6b7886", hatch="#64717e", now="#f2a65e",
        median="#11161c", rain_line="#d6e9fb", rain_bar="#7fb0e0", rain_band="#55779c",
        rain_band_edge="#55779c", temp_band="#606d7a", temp_band_edge="#606d7a", temp_cool="#b4cde6",
        temp_mild="#d3dfd8", temp_warm="#f5bd8a", temp_iqr="#9fb9d3", sun_line="#f6cf74",
        sun_bar="#c99d45", sun_band="#7d6b40", sun_band_edge="#7d6b40", ribbon_night="#1c242d",
        ribbon_day="#8a9aab", ribbon_rain="#a9d0f5",
        rain=["#4f7299", "#5f86b0", "#7aa3cc", "#9fc2e4", "#cfe3f6"],
        sun=["#7d6b40", "#a88b45", "#c99d45", "#e3b75a", "#f7d98f"],
    ),
    "light": dict(
        ground="#f3f5f7", tile="#ffffff", now_surface="#e2e9f0", ink="#11161c", ink_2="#353e48",
        ink_3="#4d5763", ink_on_now="#2c353f", rule_strong="#7d8995", hatch="#7d8995", now="#b04f12",
        median="#11161c", rain_line="#1d4a7a", rain_bar="#3b6fa6", rain_band="#dce7f3",
        rain_band_edge="#6f8fb3", temp_band="#e8ecf0", temp_band_edge="#7d8995", temp_cool="#3f6890",
        temp_mild="#5e6e66", temp_warm="#b5602a", temp_iqr="#5a7da1", sun_line="#8a6413",
        sun_bar="#a87a24", sun_band="#f5e9c8", sun_band_edge="#a1802f", ribbon_night="#2c3640",
        ribbon_day="#8797a8", ribbon_rain="#2f5f94", ribbon_edge="#5d6976",
        rain=["#6a8fb7", "#52789f", "#3d6289", "#2a4b6f", "#1a3352"],
        sun=["#a8873d", "#8e6f2c", "#75591f", "#5c4515", "#42310d"],
    ),
}

TEXT = 4.5
GRAPHIC = 3.0

PAIRS = [
    ("ink on ground", "ink", "ground", TEXT),
    ("ink on tile", "ink", "tile", TEXT),
    ("ink-2 on tile", "ink_2", "tile", TEXT),
    ("ink-3 on tile", "ink_3", "tile", TEXT),
    ("ink-3 on ground", "ink_3", "ground", TEXT),
    ("ink on now surface", "ink", "now_surface", TEXT),
    ("ink-on-now on now surface", "ink_on_now", "now_surface", TEXT),
    ("now on tile", "now", "tile", TEXT),
    ("rule-strong on tile", "rule_strong", "tile", GRAPHIC),
    ("hatch on tile", "hatch", "tile", GRAPHIC),
    ("rain band edge on tile", "rain_band_edge", "tile", GRAPHIC),
    ("rain line on tile", "rain_line", "tile", GRAPHIC),
    ("rain line on rain band", "rain_line", "rain_band", GRAPHIC),
    ("median on rain band", "median", "rain_band", GRAPHIC),
    ("rain bars on tile", "rain_bar", "tile", GRAPHIC),
    ("temp band edge on tile", "temp_band_edge", "tile", GRAPHIC),
    ("cool bar on temp band", "temp_cool", "temp_band", GRAPHIC),
    ("mild bar on temp band", "temp_mild", "temp_band", GRAPHIC),
    ("warm bar on temp band", "temp_warm", "temp_band", GRAPHIC),
    ("middle-half bar on tile", "temp_iqr", "tile", GRAPHIC),
    ("sun band edge on tile", "sun_band_edge", "tile", GRAPHIC),
    ("sun line on tile", "sun_line", "tile", GRAPHIC),
    ("sun line on sun band", "sun_line", "sun_band", GRAPHIC),
    ("sun bars on tile", "sun_bar", "tile", GRAPHIC),
    ("ribbon night against day", "ribbon_night", "ribbon_day", GRAPHIC),
    ("ribbon rain on now surface", "ribbon_rain", "now_surface", GRAPHIC),
]


def check() -> int:
    """Print every failing pair and return how many failed.

    :return: the number of failures across both themes.
    """
    failures = 0
    for theme, t in THEMES.items():
        for name, a, b, need in PAIRS:
            r = ratio(t[a], t[b])
            if r < need:
                failures += 1
                print(f"{theme:5s} {name:28s} {r:5.2f} needs {need}")
        # The ribbon's day band must stand off the now surface, or be outlined.
        edge = t.get("ribbon_edge")
        if ratio(t["ribbon_day"], t["now_surface"]) < GRAPHIC and (
            edge is None or ratio(edge, t["now_surface"]) < GRAPHIC
        ):
            failures += 1
            print(f"{theme:5s} ribbon day on now surface, no 3:1 outline")
        for ramp in ("rain", "sun"):
            steps = t[ramp]
            if ratio(steps[0], t["tile"]) < GRAPHIC:
                failures += 1
                print(f"{theme:5s} {ramp} ramp step 0 on tile {ratio(steps[0], t['tile']):.2f}")
            for i, cell in enumerate(steps):
                best = max(ratio("#ffffff", cell), ratio("#11161c", cell))
                if best < TEXT:
                    failures += 1
                    print(f"{theme:5s} text in {ramp}-{i} cell {best:.2f}")
    print("failures", failures)
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if check() else 0)

/**
 * Boot the page (`page.ts`) with the browser's read functions: the page's
 * own `station`, and on the lane server `fixture` and `age`, passed on, read
 * from the URL at each call so a chosen station is followed.
 */
import "@fontsource-variable/lexend/wght.css";
import "./styles/tokens.css";
import "./styles/base.css";

import { fetchCurrent, fetchDashboard, fetchStations } from "./load.js";
import { startPage } from "./page.js";
import { applyThemeOverride } from "./theme.js";

applyThemeOverride(location.search);

startPage(document.getElementById("masthead")!, document.getElementById("sections")!, {
  dashboard: () => fetchDashboard(location.search),
  current: () => fetchCurrent(location.search),
  stations: () => fetchStations(location.search),
});

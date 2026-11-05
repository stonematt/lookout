// @vitest-environment happy-dom
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TABS, mountTabs } from "../src/tabs.js";

let host: HTMLElement;

beforeEach(() => {
  history.replaceState(null, "", "/");
  document.body.innerHTML = "<div id=host></div>";
  host = document.getElementById("host")!;
});

const tabs = () => [...host.querySelectorAll<HTMLButtonElement>("[role=tab]")];
const selected = () => tabs().find((t) => t.getAttribute("aria-selected") === "true");
const visiblePanels = () =>
  [...host.querySelectorAll<HTMLElement>("[role=tabpanel]")].filter((p) => !p.hidden);
const key = (target: Element, name: string) =>
  target.dispatchEvent(new KeyboardEvent("keydown", { key: name, bubbles: true }));

describe("the tabs", () => {
  it("shows the four tabs with Overview open by default", () => {
    mountTabs(host);
    expect(tabs().map((t) => t.textContent)).toEqual(["Overview", "Rain", "Rain Events", "Solar"]);
    expect(host.querySelector("[role=tablist]")).not.toBeNull();
    expect(selected()?.textContent).toBe("Overview");
    expect(visiblePanels().map((p) => p.id)).toEqual(["panel-overview"]);
    expect(location.hash).toBe("#overview");
  });

  it("gives only the open tab a tab stop", () => {
    mountTabs(host);
    expect(tabs().map((t) => t.tabIndex)).toEqual([0, -1, -1, -1]);
  });

  it("opens the tab the URL hash names", () => {
    history.replaceState(null, "", "/#events");
    mountTabs(host);
    expect(selected()?.textContent).toBe("Rain Events");
    expect(visiblePanels().map((p) => p.id)).toEqual(["panel-events"]);
  });

  it("falls back to Overview for an unknown hash", () => {
    history.replaceState(null, "", "/#nope");
    mountTabs(host);
    expect(selected()?.textContent).toBe("Overview");
    expect(location.hash).toBe("#overview");
  });

  it("tracks a clicked tab in the hash", () => {
    mountTabs(host);
    tabs()[3]!.click();
    expect(selected()?.textContent).toBe("Solar");
    expect(location.hash).toBe("#solar");
    expect(tabs().map((t) => t.tabIndex)).toEqual([-1, -1, -1, 0]);
  });

  it("moves with the arrow keys, wrapping, and with Home and End", () => {
    mountTabs(host);
    const [overview] = tabs();
    overview!.focus();

    key(overview!, "ArrowRight");
    expect(selected()?.textContent).toBe("Rain");
    expect(document.activeElement).toBe(selected());

    key(document.activeElement!, "End");
    expect(selected()?.textContent).toBe("Solar");

    key(document.activeElement!, "ArrowRight");
    expect(selected()?.textContent).toBe("Overview");

    key(document.activeElement!, "ArrowLeft");
    expect(selected()?.textContent).toBe("Solar");

    key(document.activeElement!, "Home");
    expect(selected()?.textContent).toBe("Overview");
    expect(location.hash).toBe("#overview");
  });

  it("follows a hash change, such as the back button", () => {
    mountTabs(host);
    history.replaceState(null, "", "/#rain");
    window.dispatchEvent(new HashChangeEvent("hashchange"));
    expect(selected()?.textContent).toBe("Rain");
  });

  it("tells its caller which panel opened", () => {
    const onSelect = vi.fn();
    const control = mountTabs(host, onSelect);
    expect(onSelect).toHaveBeenLastCalledWith("overview", control.panel("overview"));
    tabs()[1]!.click();
    expect(onSelect).toHaveBeenLastCalledWith("rain", control.panel("rain"));
    expect(control.current()).toBe("rain");
  });

  it("labels each panel by its tab", () => {
    mountTabs(host);
    for (const { id } of TABS) {
      const panel = host.querySelector(`#panel-${id}`)!;
      expect(panel.getAttribute("aria-labelledby")).toBe(`tab-${id}`);
      expect(host.querySelector(`#tab-${id}`)!.getAttribute("aria-controls")).toBe(`panel-${id}`);
    }
  });
});

describe("the tab bar beside its host (barBeforeHost)", () => {
  it("puts the bar just before the host and keeps the panels inside it", () => {
    mountTabs(host, undefined, { barBeforeHost: true });

    const bar = document.querySelector("nav.tabs")!;
    expect(bar.nextElementSibling).toBe(host);
    expect(host.querySelector("nav.tabs")).toBeNull();
    expect(host.querySelectorAll("[role=tabpanel]")).toHaveLength(4);
  });

  it("replaces its own last bar when mounted again on the same host", () => {
    mountTabs(host, undefined, { barBeforeHost: true });
    mountTabs(host, undefined, { barBeforeHost: true });

    expect(document.querySelectorAll("nav.tabs")).toHaveLength(1);
    expect(host.querySelectorAll("[role=tabpanel]")).toHaveLength(4);
  });

  it("keeps the bar inside a host with no parent to sit beside", () => {
    const loose = document.createElement("div");
    mountTabs(loose, undefined, { barBeforeHost: true });

    expect(loose.firstElementChild?.matches("nav.tabs")).toBe(true);
  });
});

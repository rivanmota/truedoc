import assert from "node:assert/strict";
import { describe, it } from "node:test";
import vm from "node:vm";
import { DEFAULT_THEME, otherTheme, resolveTheme, THEME_BOOTSTRAP_SCRIPT, THEME_STORAGE_KEY } from "./theme.ts";

describe("theme resolution", () => {
  it("honours an explicit stored choice over the OS preference", () => {
    assert.equal(resolveTheme("light", false), "light");
    assert.equal(resolveTheme("dark", true), "dark");
  });
  it("falls back to the OS preference, then to the default", () => {
    assert.equal(resolveTheme(null, true), "light");
    assert.equal(resolveTheme(null, false), DEFAULT_THEME);
    assert.equal(resolveTheme("garbage", true), "light");
  });
  it("toggles between exactly two themes", () => {
    assert.equal(otherTheme("dark"), "light");
    assert.equal(otherTheme("light"), "dark");
  });
});

describe("pre-hydration bootstrap script", () => {
  function run(stored: string | null, prefersLight: boolean, storageThrows = false) {
    const attrs: Record<string, string> = {};
    const sandbox = {
      localStorage: {
        getItem: (k: string) => {
          if (storageThrows) throw new Error("blocked");
          return k === THEME_STORAGE_KEY ? stored : null;
        },
      },
      window: { matchMedia: () => ({ matches: prefersLight }) },
      document: { documentElement: { setAttribute: (k: string, v: string) => (attrs[k] = v) } },
    };
    vm.runInNewContext(THEME_BOOTSTRAP_SCRIPT, sandbox);
    return attrs["data-theme"];
  }
  it("matches resolveTheme() for every input combination", () => {
    for (const stored of ["dark", "light", null, "nope"]) {
      for (const prefersLight of [true, false]) {
        assert.equal(run(stored, prefersLight), resolveTheme(stored, prefersLight), `stored=${stored} prefersLight=${prefersLight}`);
      }
    }
  });
  it("still sets a theme when storage access throws", () => {
    assert.equal(run(null, true, true), "dark");
  });
});

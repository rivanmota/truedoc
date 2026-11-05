"use client";

import { useCallback, useSyncExternalStore } from "react";
import { DEFAULT_THEME, isTheme, otherTheme, THEME_STORAGE_KEY, type Theme } from "@/lib/theme";

/** Reads the live theme from <html data-theme>, which the bootstrap script sets pre-hydration. */
function readTheme(): Theme {
  const t = document.documentElement.getAttribute("data-theme");
  return isTheme(t) ? t : DEFAULT_THEME;
}

function subscribe(onChange: () => void): () => void {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  // Another tab changed the stored preference.
  const onStorage = (e: StorageEvent) => {
    if (e.key === THEME_STORAGE_KEY && isTheme(e.newValue)) {
      document.documentElement.setAttribute("data-theme", e.newValue);
    }
  };
  window.addEventListener("storage", onStorage);
  return () => {
    observer.disconnect();
    window.removeEventListener("storage", onStorage);
  };
}

function applyTheme(theme: Theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    /* private mode / storage disabled — the attribute alone still works for this session */
  }
}

export function ThemeToggle() {
  // useSyncExternalStore avoids a hydration mismatch: the server snapshot
  // is the default, the client snapshot is whatever the bootstrap set.
  const theme = useSyncExternalStore(subscribe, readTheme, () => DEFAULT_THEME);
  const next = otherTheme(theme);
  const toggle = useCallback(() => applyTheme(otherTheme(readTheme())), []);

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
      data-testid="theme-toggle"
      data-theme-current={theme}
      className="inline-flex h-8 items-center gap-2 rounded-lg border border-line bg-panel px-2.5 font-mono text-[10px] uppercase tracking-[0.12em] text-dim transition-colors hover:border-dim hover:text-paper"
    >
      {theme === "dark" ? (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5Z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
        </svg>
      ) : (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="4" stroke="currentColor" strokeWidth="1.6" />
          <path d="M12 2.5v2.5M12 19v2.5M2.5 12H5m14 0h2.5M5.3 5.3l1.8 1.8m9.8 9.8 1.8 1.8M5.3 18.7l1.8-1.8m9.8-9.8 1.8-1.8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
      )}
      <span>{theme}</span>
    </button>
  );
}

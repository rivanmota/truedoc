/**
 * Theme model: "dark" | "light", stored under THEME_STORAGE_KEY when the
 * user picks explicitly; otherwise follows the OS preference. The active
 * theme is exposed as `<html data-theme="…">`, which globals.css keys its
 * palette variables on.
 *
 * Pure helpers here; the React side is components/ThemeToggle.tsx and the
 * pre-hydration bootstrap is app/layout.tsx (THEME_BOOTSTRAP_SCRIPT).
 */
export type Theme = "dark" | "light";

export const THEME_STORAGE_KEY = "td-theme";
export const DEFAULT_THEME: Theme = "dark";

export function isTheme(value: unknown): value is Theme {
  return value === "dark" || value === "light";
}

export function otherTheme(theme: Theme): Theme {
  return theme === "dark" ? "light" : "dark";
}

/** Resolves the theme to apply from a stored value and the OS preference. */
export function resolveTheme(stored: unknown, prefersLight: boolean): Theme {
  if (isTheme(stored)) return stored;
  return prefersLight ? "light" : DEFAULT_THEME;
}

/**
 * Inline script run before paint so the first frame already has the right
 * palette (no dark→light flash). Must stay dependency-free and tiny. Keep
 * in sync with resolveTheme().
 */
export const THEME_BOOTSTRAP_SCRIPT = `(function(){try{var s=localStorage.getItem(${JSON.stringify(
  THEME_STORAGE_KEY,
)});var t=(s==="dark"||s==="light")?s:(window.matchMedia("(prefers-color-scheme: light)").matches?"light":"dark");document.documentElement.setAttribute("data-theme",t);}catch(e){document.documentElement.setAttribute("data-theme","dark");}})();`;

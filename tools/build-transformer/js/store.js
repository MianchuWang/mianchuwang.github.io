// Saved code and solved marks live in localStorage, one key per component.

const CODE_PREFIX = "bt:code:";
const SOLVED_PREFIX = "bt:solved:";

function safe(fn, fallback = null) {
  try {
    return fn();
  } catch {
    return fallback; // private mode / storage disabled — degrade quietly
  }
}

export function loadCode(id) {
  return safe(() => localStorage.getItem(CODE_PREFIX + id));
}

export function saveCode(id, code) {
  safe(() => localStorage.setItem(CODE_PREFIX + id, code));
}

export function clearCode(id) {
  safe(() => localStorage.removeItem(CODE_PREFIX + id));
}

export function isSolved(id) {
  return safe(() => !!localStorage.getItem(SOLVED_PREFIX + id), false);
}

export function markSolved(id) {
  safe(() => localStorage.setItem(SOLVED_PREFIX + id, new Date().toISOString()));
}

/** ISO date of the last solve, or null (also for pre-date "1" marks). */
export function solvedAt(id) {
  return safe(() => {
    const value = localStorage.getItem(SOLVED_PREFIX + id);
    return value && value !== "1" ? value : null;
  });
}

export function clearAll() {
  safe(() => {
    const keys = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && (key.startsWith(CODE_PREFIX) || key.startsWith(SOLVED_PREFIX))) keys.push(key);
    }
    keys.forEach((key) => localStorage.removeItem(key));
  });
}

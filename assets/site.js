/* Site-wide utilities with no dependencies: date / tag / id formatting,
   the theme switch, and the scroll spy shared by the homepage Index and the
   article Contents. Markdown rendering lives in md.js, which pulls its
   libraries from a CDN — keep this file import-free so the homepage never
   pays for them. */

/* --- formatting ----------------------------------------------------------
   scripts/prerender.py mirrors formatDate, tagClass and articleId when it
   bakes the homepage — change one, change the other. */

export function formatDate(iso) {
  if (!iso) return "";
  const d = new Date(iso + "T00:00:00");
  if (isNaN(d)) return iso;
  return d.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/* Tags with a dedicated chip style (matched case-insensitively). */
const SPECIAL_TAGS = { "wandb": "tag-wandb", "agent runbook": "tag-agent", "in progress": "tag-progress" };

export function tagClass(tag) {
  const extra = SPECIAL_TAGS[tag.toLowerCase()];
  return extra ? `tag ${extra}` : "tag";
}

/* Article id from the creation date: 2026-09-18 -> W260918. */
export function articleId(date) {
  return `W${date.replaceAll("-", "").slice(2)}`;
}

/* --- theme ---------------------------------------------------------------- */

/* localStorage can throw (blocked site data, some private modes). */
function storage(op, ...args) {
  try { return localStorage[op](...args); } catch { return null; }
}

const currentTheme = () =>
  document.documentElement.dataset.theme ||
  (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

/* The switch's knob position and sun/moon icon are pure CSS (base.css);
   this only flips data-theme, persists it, and keeps ARIA in sync. */
export function initTheme() {
  const saved = storage("getItem", "theme");
  if (saved) document.documentElement.dataset.theme = saved;
  const btn = document.querySelector(".theme-toggle");
  if (!btn) return;
  btn.setAttribute("role", "switch");
  const sync = () => btn.setAttribute("aria-checked", String(currentTheme() === "dark"));
  sync();
  btn.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    storage("setItem", "theme", next);
    sync();
  });
}

/* --- scroll spy ------------------------------------------------------------
   Highlights the side-index entry for the section in view, and jumps to a
   section on request.

     links    Map of key -> <a>, in document order. Key "" is the index
              title: it stands for "above the first section" and jumps to
              the top of the page.
     resolve  key -> the section's element (or null). Called on every use,
              so it survives the homepage re-rendering its sections.

   Two workarounds live here:
   - Jumps use an instant scrollIntoView, not fragment navigation: Chromium
     can silently drop smooth fragment scrolls, and a same-hash re-click
     never scrolls.
   - The spy is a scroll scanline, not an IntersectionObserver: an instant
     jump teleports headings past an IO band without firing it, and at the
     page bottom the last heading may never enter the band at all. */
export function createScrollSpy(links, resolve) {
  const keys = [...links.keys()];
  let lockUntil = 0; // a jump pins the highlight briefly; geometry may disagree
  let deferred = 0;

  const setActive = (key) => {
    links.forEach((a) => a.classList.remove("active"));
    links.get(key)?.classList.add("active");
  };

  const onScroll = () => {
    const now = Date.now();
    if (now < lockUntil) {
      // a scroll landed inside the jump-lock: re-check once it expires,
      // or the highlight would stay pinned wrong with no further events
      clearTimeout(deferred);
      deferred = setTimeout(onScroll, lockUntil - now + 20);
      return;
    }
    const atBottom = scrollY + innerHeight >= document.documentElement.scrollHeight - 2;
    if (atBottom) return setActive(keys[keys.length - 1]);
    const line = innerHeight * 0.3;
    let current = keys[0];
    for (const key of keys) {
      const el = key && resolve(key);
      if (el && el.getBoundingClientRect().top <= line) current = key;
    }
    setActive(current);
  };

  /* Jump to a key's section ("" = top of the page) and put the key in the
     URL. Returns false, doing nothing, when the key has no section. */
  const jumpTo = (key) => {
    const el = key ? resolve(key) : null;
    if (key && !el) return false;
    if (el) el.scrollIntoView({ behavior: "instant", block: "start" });
    else window.scrollTo({ top: 0, behavior: "instant" });
    history.replaceState(null, "", key ? `#${key}` : location.pathname + location.search);
    setActive(key);
    lockUntil = Date.now() + 500;
    return true;
  };

  addEventListener("scroll", onScroll, { passive: true });
  onScroll();
  return { jumpTo };
}

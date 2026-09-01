/* Article page controller (post.html?p=<slug>).

   Lifecycle: fetch writings/<slug>.md → parseFrontmatter → fill header
   (title, W-id badge, date, tags) → renderMarkdown into #article-body →
   enhancement passes, in order:
     upgradeCallouts   [!info]-style blockquotes → callout boxes   (md.js)
     initCharts        <div class="chart" data-src/data-metric>    (chart.js)
     addHeadingAnchors hover # links, unique ids
     upgradeExperimentBlocks  "E1 —" h3 sections → cards + chips
     addCopyButtons    on <pre>
     buildToc          floating contents (h2-h4) with scroll spy
   Styles live in assets/article.css. */

import {
  parseFrontmatter,
  renderMarkdown,
  upgradeCallouts,
  formatDate,
  tagClass,
  initTheme,
} from "./md.js";
import { initCharts } from "./chart.js";

initTheme();

// "Back" as real history navigation when we came from this site, so the
// homepage restores its scroll position; plain link otherwise.
document.querySelector(".back-link")?.addEventListener("click", (e) => {
  if (history.length > 1 && document.referrer.startsWith(location.origin)) {
    e.preventDefault();
    history.back();
  }
});

const params = new URLSearchParams(location.search);
const slug = params.get("p");

const titleEl = document.getElementById("article-title");
const metaEl = document.getElementById("article-meta");
const bodyEl = document.getElementById("article-body");
const tocEl = document.getElementById("toc");

async function main() {
  if (!slug || !/^[\w-]+$/.test(slug)) {
    bodyEl.innerHTML = `<div class="empty-note">Post not found.</div>`;
    return;
  }

  let raw;
  try {
    const res = await fetch(`writings/${slug}.md`, { cache: "no-cache" });
    if (!res.ok) throw new Error(res.status);
    raw = await res.text();
  } catch {
    bodyEl.innerHTML = `<div class="empty-note">Post not found.</div>`;
    return;
  }

  const { meta, body } = parseFrontmatter(raw);
  const title = meta.title || slug;
  document.title = `${title} · Mianchu Wang`;
  if (meta.lang) document.documentElement.lang = meta.lang;
  titleEl.textContent = title;

  if (meta.date) {
    const idEl = document.getElementById("article-id");
    idEl.textContent = `W${meta.date.replaceAll("-", "").slice(2)}`;
    idEl.hidden = false;
  }

  const tags = (meta.tags || [])
    .map?.((t) => `<span class="${tagClass(t)}">${t}</span>`)
    .join("") || "";
  metaEl.innerHTML = `<span>${formatDate(meta.date)}</span>${tags}`;

  bodyEl.innerHTML = renderMarkdown(body);
  upgradeCallouts(bodyEl);
  initCharts(bodyEl);
  addHeadingAnchors();
  upgradeExperimentBlocks();
  addCopyButtons();
  buildToc();
}

function slugify(text) {
  return (
    text
      .trim()
      .toLowerCase()
      .replace(/[^\p{L}\p{N}]+/gu, "-")
      .replace(/^-+|-+$/g, "") || "section"
  );
}

function addHeadingAnchors() {
  const seen = new Set();
  for (const h of bodyEl.querySelectorAll("h1, h2, h3, h4")) {
    let id = slugify(h.textContent);
    while (seen.has(id)) id += "-";
    seen.add(id);
    h.id = id;
    const a = document.createElement("a");
    a.className = "anchor";
    a.href = `#${id}`;
    a.textContent = "#";
    a.setAttribute("aria-hidden", "true");
    h.prepend(a);
  }
}

/* Wrap each "E1 — ..." style h3 section into a bordered card and turn the
   experiment number into an accent chip, so experiment blocks read as units. */
function upgradeExperimentBlocks() {
  for (const h of bodyEl.querySelectorAll("h3")) {
    const textNode = [...h.childNodes].find((n) => n.nodeType === Node.TEXT_NODE);
    const m = textNode?.textContent.match(/^(E\d+)\s*—\s*/);
    if (!m) continue;

    const chip = document.createElement("span");
    chip.className = "exp-chip";
    chip.textContent = m[1];
    textNode.textContent = " " + textNode.textContent.slice(m[0].length);
    h.insertBefore(chip, textNode);

    const section = document.createElement("section");
    section.className = "exp-block";
    h.before(section);
    let node = h;
    while (node && !(node !== h && /^H[123]$/.test(node.tagName || ""))) {
      const next = node.nextElementSibling;
      section.appendChild(node);
      node = next;
    }

    upgradeSubExperiments(section, m[1]);
  }
}

/* Within one experiment card: number the evidence list as E1.1/E1.2/…,
   and chip those sub-ids where results reference them (chart titles,
   bold "E1.2"-style markers). */
function upgradeSubExperiments(section, expId) {
  const subChip = (id) => {
    const c = document.createElement("span");
    c.className = "exp-subchip";
    c.textContent = id;
    return c;
  };

  const evidence = section.querySelector("ol");
  if (evidence) {
    evidence.classList.add("exp-evidence");
    [...evidence.children].forEach((li, k) => li.prepend(subChip(`${expId}.${k + 1}`)));
  }

  // chart titles get their chips in chart.js (they render async, after this pass)

  for (const b of section.querySelectorAll("strong")) {
    const ids = b.textContent.trim().match(/^E\d+\.\d+(\s*,\s*E\d+\.\d+)*$/);
    if (!ids) continue;
    const frag = document.createDocumentFragment();
    for (const id of b.textContent.split(",")) frag.appendChild(subChip(id.trim()));
    const after = b.nextSibling;
    if (after?.nodeType === Node.TEXT_NODE) {
      after.textContent = after.textContent.replace(/^\s*—\s*/, "");
    }
    b.replaceWith(frag);
  }
}

function addCopyButtons() {
  for (const pre of bodyEl.querySelectorAll("pre")) {
    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.textContent = "Copy";
    btn.addEventListener("click", async () => {
      const code = pre.querySelector("code");
      await navigator.clipboard.writeText(code ? code.innerText : pre.innerText);
      btn.textContent = "Copied";
      setTimeout(() => (btn.textContent = "Copy"), 1200);
    });
    pre.appendChild(btn);
  }
}

function buildToc() {
  const headings = [...bodyEl.querySelectorAll("h2, h3, h4")];
  if (headings.length < 2) return;

  tocEl.classList.add("has-items");
  tocEl.replaceChildren();
  const links = new Map();
  let lockUntil = 0;

  const setActive = (id) => {
    links.forEach((a) => a.classList.remove("active"));
    links.get(id)?.classList.add("active");
  };

  // "Contents" mirrors the homepage Index: click -> top + clean URL, and
  // it owns the bar while the page sits above the first heading (key "").
  const title = document.createElement("a");
  title.className = "toc-title";
  title.href = location.pathname + location.search;
  title.textContent = "Contents";
  title.addEventListener("click", (e) => {
    e.preventDefault();
    window.scrollTo({ top: 0, behavior: "instant" });
    history.replaceState(null, "", location.pathname + location.search);
    setActive("");
    lockUntil = Date.now() + 500;
  });
  tocEl.appendChild(title);
  links.set("", title);

  for (const h of headings) {
    const a = document.createElement("a");
    a.href = `#${h.id}`;
    // clone the rendered heading (KaTeX, chips) instead of textContent —
    // KaTeX's textContent duplicates every formula (MathML + HTML copies)
    for (const node of h.childNodes) {
      if (node.nodeType === 1 && node.classList.contains("anchor")) continue;
      if (node.nodeType === 1 && node.classList.contains("exp-chip")) {
        a.appendChild(document.createTextNode(node.textContent)); // plain "E1", no badge (following text node carries the space)
        continue;
      }
      a.appendChild(node.cloneNode(true));
    }
    a.className = `depth-${h.tagName[1]}`;
    a.addEventListener("click", (e) => {
      // same workaround as the homepage index: Chromium can drop smooth
      // fragment scrolls, and a same-hash re-click never scrolls
      e.preventDefault();
      h.scrollIntoView({ behavior: "instant", block: "start" });
      history.replaceState(null, "", `#${h.id}`);
      setActive(h.id);
      lockUntil = Date.now() + 500;
    });
    tocEl.appendChild(a);
    links.set(h.id, a);
  }

  // Scanline spy (same as the homepage index): an instant jump teleports
  // headings past an IntersectionObserver band without firing it.
  let deferred = 0;
  const onScroll = () => {
    const now = Date.now();
    if (now < lockUntil) {
      clearTimeout(deferred);
      deferred = setTimeout(onScroll, lockUntil - now + 20);
      return;
    }
    const atBottom = scrollY + innerHeight >= document.documentElement.scrollHeight - 2;
    if (atBottom) return setActive(headings[headings.length - 1].id);
    const line = innerHeight * 0.3;
    let current = ""; // above the first heading: the bar rests on Contents
    for (const h of headings) {
      if (h.getBoundingClientRect().top <= line) current = h.id;
    }
    setActive(current);
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();
}

main();

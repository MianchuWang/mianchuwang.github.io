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

  const tags = (meta.tags || [])
    .map?.((t) => `<span class="${tagClass(t)}">${t}</span>`)
    .join("") || "";
  metaEl.innerHTML = `<span>${formatDate(meta.date)}</span>${tags}`;

  bodyEl.innerHTML = renderMarkdown(body);
  upgradeCallouts(bodyEl);
  initCharts(bodyEl);
  addHeadingAnchors();
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
  tocEl.innerHTML = `<div class="toc-title">Contents</div>`;
  const links = new Map();
  for (const h of headings) {
    const a = document.createElement("a");
    a.href = `#${h.id}`;
    a.textContent = h.textContent.replace(/^#\s*/, "");
    a.className = `depth-${h.tagName[1]}`;
    tocEl.appendChild(a);
    links.set(h.id, a);
  }

  const observer = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          links.forEach((a) => a.classList.remove("active"));
          links.get(e.target.id)?.classList.add("active");
        }
      }
    },
    { rootMargin: "-10% 0px -80% 0px" }
  );
  headings.forEach((h) => observer.observe(h));
}

main();

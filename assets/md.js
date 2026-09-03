/* Shared utilities: markdown rendering (marked + KaTeX + highlight.js),
   frontmatter parsing, and theme toggling. */

import { marked } from "https://cdn.jsdelivr.net/npm/marked@14.1.4/lib/marked.esm.js";
import markedKatex from "https://cdn.jsdelivr.net/npm/marked-katex-extension@5.1.4/+esm";
import hljs from "https://cdn.jsdelivr.net/npm/highlight.js@11.10.0/+esm";

marked.use(
  markedKatex({
    throwOnError: false,
    nonStandard: true, // allow $x$ without surrounding spaces (natural in Chinese text)
  })
);

marked.use({
  gfm: true,
  renderer: {
    code({ text, lang }) {
      let html;
      if (lang && hljs.getLanguage(lang)) {
        html = hljs.highlight(text, { language: lang }).value;
      } else {
        html = escapeHtml(text);
      }
      return `<pre><code class="hljs">${html}</code></pre>`;
    },
  },
});

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/* --- frontmatter ---------------------------------------------------------
   Parses a leading YAML-lite block:
   ---
   title: ...
   date: 2026-07-22
   tags: [a, b]
   summary: ...
   ---
*/
export function parseFrontmatter(raw) {
  const meta = {};
  let body = raw;
  const m = raw.match(/^---\s*\n([\s\S]*?)\n---\s*\n?/);
  if (m) {
    body = raw.slice(m[0].length);
    for (const line of m[1].split("\n")) {
      const kv = line.match(/^(\w+)\s*:\s*(.*)$/);
      if (!kv) continue;
      const key = kv[1];
      let value = kv[2].trim();
      if (value.startsWith("[") && value.endsWith("]")) {
        value = value
          .slice(1, -1)
          .split(",")
          .map((s) => s.trim().replace(/^["']|["']$/g, ""))
          .filter(Boolean);
      } else {
        value = value.replace(/^["']|["']$/g, "");
      }
      meta[key] = value;
    }
  }
  return { meta, body };
}

/* --- callouts ------------------------------------------------------------
   Converts blockquotes beginning with [!note] / [!tip] / [!warn] etc.
   into Notion-style callout boxes. Runs on the rendered DOM.
*/
const CALLOUT_ICONS = {
  note: "💡",
  tip: "💡",
  info: "ℹ️",
  warn: "⚠️",
  warning: "⚠️",
  important: "📌",
  quote: "💬",
};

export function upgradeCallouts(rootEl) {
  for (const bq of rootEl.querySelectorAll("blockquote")) {
    const first = bq.querySelector("p");
    if (!first) continue;
    const m = first.innerHTML.match(/^\[!(\w+)\]\s*/);
    if (!m) continue;
    const kind = m[1].toLowerCase();
    const icon = CALLOUT_ICONS[kind] || "💡";
    first.innerHTML = first.innerHTML.slice(m[0].length);
    const callout = document.createElement("div");
    callout.className = `callout callout-${kind}`;
    callout.innerHTML = `<div class="callout-icon">${icon}</div>`;
    const content = document.createElement("div");
    content.className = "callout-content";
    while (bq.firstChild) content.appendChild(bq.firstChild);
    if (!first.textContent.trim()) first.remove();
    callout.appendChild(content);
    bq.replaceWith(callout);
  }
}

export function renderMarkdown(body) {
  return marked.parse(body);
}

/* Tags with a dedicated chip style (matched case-insensitively). */
const SPECIAL_TAGS = { "wandb": "tag-wandb", "agent runbook": "tag-agent", "in progress": "tag-progress" };

export function tagClass(tag) {
  const extra = SPECIAL_TAGS[tag.toLowerCase()];
  return extra ? `tag ${extra}` : "tag";
}

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

/* --- theme ---------------------------------------------------------------- */

/* localStorage can throw (blocked site data, some private modes). */
function storage(op, ...args) {
  try { return localStorage[op](...args); } catch { return null; }
}

export function initTheme() {
  const saved = storage("getItem", "theme");
  if (saved) document.documentElement.dataset.theme = saved;
  const btn = document.querySelector(".theme-toggle");
  if (!btn) return;
  const isSwitch = !!btn.querySelector(".theme-knob"); // knob styling is pure CSS
  const icon = () => {
    const t =
      document.documentElement.dataset.theme ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    if (isSwitch) {
      btn.setAttribute("role", "switch");
      btn.setAttribute("aria-checked", String(t === "dark"));
      return;
    }
    btn.textContent = t === "dark" ? "☀" : "☾";
  };
  icon();
  btn.addEventListener("click", () => {
    const current =
      document.documentElement.dataset.theme ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    storage("setItem", "theme", next);
    icon();
  });
}

/* Markdown pipeline for article pages: rendering (marked + KaTeX +
   highlight.js), frontmatter parsing, and callouts. The libraries come from
   a CDN, so only pages that render Markdown import this file; dependency-free
   helpers (theme, dates, tags, scroll spy) live in site.js. */

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
    // Fence info string: a language, then options. `lines=1467,1470-1471`
    // numbers the lines from source: numbers are handed out in order, and a
    // line that is only `...` (an elision) takes none and shows none. The
    // gutter is a separate element, so Copy still gets the bare code.
    code({ text, lang }) {
      const [language, ...opts] = (lang || "").split(/\s+/);
      const html = language && hljs.getLanguage(language)
        ? hljs.highlight(text, { language }).value
        : escapeHtml(text);
      const lines = opts.find((o) => o.startsWith("lines="));
      if (!lines) return `<pre><code class="hljs">${html}</code></pre>`;
      const gutter = lineNumbers(text, lines.slice("lines=".length)).join("\n");
      return `<pre class="has-ln"><span class="ln" aria-hidden="true">${gutter}</span><code class="hljs">${html}</code></pre>`;
    },
  },
});

function lineNumbers(text, spec) {
  const pool = [];
  for (const part of spec.split(",")) {
    const [a, b = a] = part.split("-").map(Number);
    for (let n = a; n <= b; n++) pool.push(n);
  }
  const out = text.replace(/\n$/, "").split("\n").map((line) =>
    line.trim() === "..." ? "" : (pool.shift() ?? "")
  );
  if (pool.length) console.warn(`lines=${spec}: ${pool.length} number(s) unused`);
  return out;
}

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

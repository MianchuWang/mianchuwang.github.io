/* Shared KaTeX rendering for dynamically injected HTML.

   Posts already render math inside the markdown pipeline (md.js); this module
   is for everything else on the site — tools and other pages that inject raw
   HTML and want $...$ / $$...$$ rendered in place, exactly like markdown.

   Usage (from any ES module):

     import { renderMathIn } from "../../../assets/math.js";  // path to site root
     el.innerHTML = html;
     renderMathIn(el);   // async; safe to fire-and-forget

   KaTeX is loaded lazily from the CDN on the first call, so pages without
   math pay nothing. `<pre>` and `<code>` contents are never touched, so a
   literal $ in code examples is safe. Errors render as source text rather
   than throwing. */

const KATEX_VERSION = "0.16.11"; // keep in sync with post.html
const BASE = `https://cdn.jsdelivr.net/npm/katex@${KATEX_VERSION}/dist/`;

const DELIMITERS = [
  { left: "$$", right: "$$", display: true },
  { left: "\\[", right: "\\]", display: true },
  { left: "$", right: "$", display: false },
  { left: "\\(", right: "\\)", display: false },
];

let katexReady = null;

function loadCss() {
  if (document.querySelector(`link[href^="${BASE}katex.min.css"]`)) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = `${BASE}katex.min.css`;
  document.head.appendChild(link);
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error(`failed to load ${src}`));
    document.head.appendChild(s);
  });
}

/** Load KaTeX + auto-render once; resolves to the renderMathInElement function. */
export function ensureKatex() {
  if (!katexReady) {
    katexReady = (async () => {
      loadCss();
      await loadScript(`${BASE}katex.min.js`);
      await loadScript(`${BASE}contrib/auto-render.min.js`);
      return window.renderMathInElement;
    })();
  }
  return katexReady;
}

/** Render $...$ and $$...$$ inside `el`. Async; resolves when done.
    Fails soft: with no network the source text simply stays visible. */
export async function renderMathIn(el) {
  try {
    const render = await ensureKatex();
    render(el, { delimiters: DELIMITERS, throwOnError: false });
  } catch {
    /* offline or CDN blocked — leave the raw text */
  }
}

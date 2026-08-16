# mianchuwang.github.io

Personal site. Content and UI are fully separated: posts are plain Markdown
files, and a small static frontend renders them in the browser (marked +
KaTeX + highlight.js — no build step, no framework).

## Writing a post

1. Add a Markdown file to `writings/`, e.g. `writings/2026-07-my-post.md`:

   ```markdown
   ---
   title: My Post
   date: 2026-07-22
   lang: en              # "zh" for posts written in Chinese
   tags: [machine-learning]
   summary: One-line summary describing the post.
   ---

   Inline math $E = mc^2$ and display math:

   $$
   \int_{-\infty}^{\infty} e^{-x^2}\, dx = \sqrt{\pi}
   $$
   ```

2. Commit and push. The GitHub Actions workflow rebuilds the post index
   (`writings/manifest.json`) and deploys automatically.

Add `draft: true` to the frontmatter to keep a post off the site.
See `writings/technical-writing-template.md` for everything the renderer
supports (callouts, tables, code highlighting, TOC, …).

## Local preview

```bash
python3 scripts/build_manifest.py   # refresh the post index
python3 -m http.server 8000         # then open http://localhost:8000
```

## One-time setup

In the repo settings on GitHub: **Settings → Pages → Source → GitHub Actions**.

## Layout

```
profile/           ← homepage content (JSON, no rebuild needed)
  profile.json         name, bio, links, education & experience (home hero)
  publications.json    publication list shown on the home page
  tools.json           "Tool Set" section: interactive tools hosted on the site
writings/          ← posts (Markdown + frontmatter); manifest.json is generated
tools/             ← self-contained web tools, one folder each
  cs336-quiz/          CS336 Learning Tools (quiz app, see its README)
  build-transformer/   ML coding practice: implement transformer components in the
                       browser with numpy + einops (see its README)
index.html         ← home page (post list)
post.html          ← article page (Markdown renderer, math, TOC)
assets/            ← styles and JS
  math.js              shared KaTeX loader: renderMathIn(el) renders $...$ and
                       $$...$$ in dynamically injected HTML (used by the tools;
                       posts render math in the markdown pipeline instead)
scripts/           ← manifest generator
.github/workflows/ ← auto-deploy on push
```

## Adding a tool

Drop a self-contained static app into `tools/<name>/` and add an entry to
`profile/tools.json` (`id`, `title`, `description`, `url`). It appears in the
home page's **Tool Set** section. The `id` is `T<yymmdd>` from the tool's
creation date (its first commit) — e.g. `T260816` — shown on the home page and
in the tool's own header, so tools can be referred to by number.

For LaTeX math in a tool, import the shared renderer and call it on any
element after injecting content:

```js
import { renderMathIn } from "../../../assets/math.js"; // from tools/<name>/js/
el.innerHTML = html;   // may contain $x^2$ and $$\int f$$
renderMathIn(el);      // async, fire-and-forget; <pre>/<code> are left alone
```

# mianchuwang.github.io

Personal site. Content and UI are separated: posts are plain Markdown files and
homepage data lives in JSON; a small static frontend renders everything in the
browser (marked + KaTeX + highlight.js — no framework). A prerender step bakes
the homepage content into static HTML so crawlers and LLM agents (which don't
run JS) see the real content, and generates `llms.txt` — an agent-readable
index that points at the raw Markdown of every post.

GitHub Pages serves `main` directly: **push = deploy**, and there is no CI —
run the prerender step locally before pushing.

## Writing a post

1. Add a Markdown file to `writings/`, e.g. `writings/2026-08-my-post.md`:

   ```markdown
   ---
   title: My Post
   date: 2026-08-30        # creation date — never bumped on edits
   lang: en                # "zh" for posts written in Chinese
   tags: [GPU, verl]       # auto-sorted alphabetically at build time
   summary: One line; shown in llms.txt (the homepage shows tags instead).
   ---
   ```

   The homepage shows a `W<yymmdd>` id derived from `date` (like tool ids).
   Add `draft: true` to keep a post off the site. See
   `writings/technical-writing-template.md` for everything the renderer
   supports (math, callouts, tables, code highlighting, TOC, …).

2. **Run `python3 scripts/prerender.py`** — it regenerates
   `writings/manifest.json` (never edit by hand), re-bakes the homepage's
   static HTML between the `<!--bake:*-->` markers in `index.html`, and
   rewrites `llms.txt`.

3. Commit and push.

To list an externally-hosted post (kept for future use), add an entry with a
`url` instead of a `slug` to `writings/external.json` and rerun the prerender.

## Local preview

```bash
python3 scripts/dev_server.py 8644 .    # then open http://localhost:8644
```

## Layout

```
index.html         ← home page; content baked between <!--bake:*--> markers,
                     re-rendered live by assets/home.js from the JSON below
post.html          ← article page (Markdown renderer, math, TOC)
llms.txt           ← generated agent-readable site index
profile/           ← homepage content (JSON)
  profile.json         name, bio, links, education & experience (home hero)
  publications.json    publication list
  tools.json           "Tool Set" section
writings/          ← posts (Markdown + frontmatter)
  manifest.json        generated post index — never edit by hand
  external.json        externally-hosted posts (entries carry a url)
  figures/             static figures referenced by posts
tools/             ← self-contained web tools, one folder each
  cs336-quiz/          CS336 Learning Tools (see its README)
  build-transformer/   implement transformer components in the browser (see its README)
assets/            ← styles and JS (home.js, post.js, md.js, math.js)
scripts/
  prerender.py         the one pre-push command (runs build_manifest itself)
  build_manifest.py    frontmatter → writings/manifest.json
  dev_server.py        local preview server
```

## Adding a tool

Drop a self-contained static app into `tools/<name>/` and add an entry to
`profile/tools.json` (`id`, `title`, `description`, `url`). It appears in the
home page's **Tool Set** section. The `id` is `T<yymmdd>` from the tool's
creation date — e.g. `T260816` — shown on the home page and in the tool's own
header, so tools can be referred to by number.

For LaTeX math in a tool, import the shared renderer and call it on any
element after injecting content:

```js
import { renderMathIn } from "../../../assets/math.js"; // from tools/<name>/js/
el.innerHTML = html;   // may contain $x^2$ and $$\int f$$
renderMathIn(el);      // async, fire-and-forget; <pre>/<code> are left alone
```

# mianchuwang.github.io

Personal site of Mianchu Wang, served by GitHub Pages straight from `main` —
no build pipeline, plain HTML/CSS/JS, no frameworks. Keep it that way.

## Layout

- `index.html` + `assets/home.js` — homepage; sections (Publications, Tool Set,
  Writing) render from JSON: `profile/*.json` and `writings/manifest.json`.
- `assets/` CSS is layered: `base.css` (theme tokens, nav, footer, badges —
  every page) + one page layer (`home.css` / `article.css`, charts add
  `chart.css`). JS: `md.js` shared markdown+theme utils, `home.js` and
  `post.js` page controllers (each has a header map), `chart.js` interactive
  figures, `math.js` KaTeX for tools. New styles go in the layer that owns
  the page; new tokens go in all three theme blocks of `base.css`.
- `writings/*.md` — articles, rendered client-side by `post.html?p=<slug>`
  (`assets/md.js`). Slug = filename without `.md`.
- `tools/` — self-contained single-page HTML tools, listed in `profile/tools.json`.
- `scripts/dev_server.py` — local preview (`.claude/launch.json`, name `site`,
  port 8644). `scripts/build_manifest.py` — see below.

## Conventions

- Article frontmatter: `title`, `date`, `tags`, `summary`.
  - `date` is the **creation date — never bump it on edits**.
  - `summary` is one short sentence; it feeds `llms.txt` (the homepage list
    shows tags instead).
- **After changing frontmatter or any `profile/*.json`, run
  `python3 scripts/prerender.py`.** It regenerates `writings/manifest.json`
  (never edit that by hand) and re-bakes the homepage's static HTML between
  the `<!--bake:*-->` markers in `index.html` — the baked copy is what
  crawlers and non-JS agents see, so a stale bake silently shows them old
  content. Never edit inside the bake markers by hand; `assets/home.js`
  re-renders the same markup at runtime, and the two must stay structurally
  identical (change one → change the other).
- Markdown supports Notion-style callouts: blockquote starting with `[!info]`,
  `[!warning]`, etc.
- Tool ids follow `T<yymmdd>`; article ids `W<yymmdd>` are derived from the
  `date` automatically (nothing to set). Tags are auto-sorted alphabetically
  by `build_manifest.py`.

## Publishing

Push to `main` = deploy. Verify in the local preview first (port 8644);
check both light and dark themes for style changes (`.theme-toggle`).

**Never push without an explicit push instruction from the user in the current
request** — a past "push it" does not carry over to later changes. **Never
publish a new article the user has not read** — push authorization covers
reviewed work, not sight-unseen publications. Commit in milestone-sized
batches (a finished feature, a completed analysis section), not per-edit —
dozens of micro-commits are noise. Pushing is the user's call, every time. Don't track or nag about unpushed commits;
only when a substantial piece of work lands — the kind the user would plausibly
want live — ask once whether to push.

# mianchuwang.github.io

Personal site of Mianchu Wang, served by GitHub Pages straight from `main` —
no build pipeline, plain HTML/CSS/JS, no frameworks. Keep it that way.

## Layout

- `index.html` + `assets/home.js` — homepage; sections (Publications, Tool Set,
  Writing) render from JSON: `profile/*.json` and `writings/manifest.json`.
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
- Tool ids follow `T<yymmdd>`.

## Publishing

Push to `main` = deploy. Verify in the local preview first (port 8644);
check both light and dark themes for style changes (`.theme-toggle`).

**Never push without an explicit push instruction from the user in the current
request** — a past "push it" does not carry over to later changes. Committing
small completed increments locally is fine at Claude's own discretion; pushing
is the user's call, every time. Don't track or nag about unpushed commits;
only when a substantial piece of work lands — the kind the user would plausibly
want live — ask once whether to push.

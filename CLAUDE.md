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
  - `summary` is one short sentence; it is displayed under the title on the
    homepage list, so keep it tight.
- **After changing any frontmatter, run `python3 scripts/build_manifest.py`.**
  The homepage reads `writings/manifest.json`; it is generated — never edit it
  by hand, and don't forget to regenerate it (a stale manifest silently shows
  old titles).
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

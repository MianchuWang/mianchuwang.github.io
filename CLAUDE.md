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

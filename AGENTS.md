# AGENTS.md

Instructions for any AI model working in this repository. This file describes
how the site is designed and built. It is independent of article content: do
not put notes about a specific article, or a session log, here.

## The site

Personal site of Mianchu Wang. GitHub Pages serves `main` directly:
**push = deploy**. There is no build pipeline and no framework, only plain
HTML, CSS and ES modules. Keep it that way.

Design principles:

- **Content and UI are separate.** Articles are Markdown files, homepage data
  is JSON, and a small static frontend renders both in the browser.
- **Readable without JavaScript.** The homepage content is also baked into
  `index.html` as static HTML, and `llms.txt` points agents at the raw
  Markdown of every article.
- **The homepage stays light.** It loads `home.js` and `site.js` only. It
  must never import `md.js`, which pulls marked, KaTeX and highlight.js from
  a CDN.
- **Light and dark are equal.** Every color comes from a theme token. Check
  both themes after any style change.
- **No visual change by accident.** A refactor must leave the rendered pages
  identical. Compare before and after.

## Commands

```bash
python3 scripts/dev_server.py 8644 .   # local preview, no-cache + live reload
python3 scripts/prerender.py           # rebuild manifest, homepage bake, llms.txt
```

Reuse a preview server that is already running on port 8644.

## Layout

```
index.html               homepage; static copy between <!--bake:*--> markers
post.html                article page: post.html?p=<slug>
llms.txt                 generated index for agents (never edit by hand)
profile/*.json           homepage data: profile, publications, tools
writings/*.md            articles; slug = filename without .md
writings/manifest.json   generated article index (never edit by hand)
writings/external.json   externally hosted posts (entries carry a url)
writings/figures/        chart data (JSON) and static figures
tools/<name>/            self-contained tools with their own CSS and JS
assets/                  shared CSS and JS (see below)
scripts/                 prerender.py, build_manifest.py, dev_server.py
```

`README.md` has the human workflow for adding an article or a tool.

## How pages render

**Homepage.** `assets/home.js` fetches `profile/*.json` and
`writings/manifest.json` and renders the hero, then the sections
Publications, Writing and Toolbox into `#sections`. `scripts/prerender.py`
writes the same markup between the bake markers.

- The two must stay structurally identical: change one, change the other.
  `postHtml`, `toolHtml` and `publicationHtml` in `home.js`, and
  `formatDate`, `tagClass` and `articleId` in `site.js`, each have a mirror
  in `prerender.py`.
- Never edit inside the bake markers by hand. Run `prerender.py`.
- If a fetch fails, the baked content stays on the page.

**Article page.** `assets/post.js` fetches `writings/<slug>.md`, parses the
frontmatter, renders the Markdown with `md.js`, and then runs these passes in
order: callouts, charts, heading anchors, experiment cards (`E1 —` headings),
copy buttons, and the Contents.

- The Contents lists `h2` and `h3` only. `h4` headings keep their anchors.
- Callouts are blockquotes that start with `[!info]`, `[!note]`, `[!warning]`
  and so on.
- Math is KaTeX through `marked-katex-extension`; `$x$` works without spaces.
- A code fence may carry `lines=1467,1470-1471` after the language to show
  source line numbers in a gutter. Numbers go to the lines in order; a line
  that is only `...` marks an elision and gets none.
- Charts are `<div class="chart" data-src="…json" data-metric="…">`, rendered
  by `assets/chart.js`. Its header comment holds a compatibility contract:
  add features as new options whose default keeps old charts unchanged.

**Tools.** Each tool under `tools/` is self-contained and does not load the
site's CSS. The only shared file is `assets/math.js` (KaTeX for injected
HTML). Its KaTeX version must match the one in `post.html`.

## CSS and JS structure

- CSS is layered: `base.css` (theme tokens, reset, nav, theme switch, tags,
  id badges, footer) is loaded on every page, plus one page layer:
  `home.css` or `article.css`. Article pages add `chart.css`. A new style
  goes in the layer that owns the page.
- Theme tokens live in `base.css` in three blocks: light (`:root`), system
  dark (`@media (prefers-color-scheme: dark)` on
  `:root:not([data-theme="light"])`), and manual dark
  (`:root[data-theme="dark"]`). A new themed token goes in all three. A token
  that does not depend on the theme, such as `--font-mono`, goes in `:root`
  only.
- The theme switch is pure CSS. `initTheme()` only flips `data-theme`, saves
  it, and keeps ARIA in sync. An inline script in each HTML head applies the
  saved theme before first paint.
- JS modules: `site.js` has no imports (theme, dates, tags, ids, and the
  scroll spy shared by the homepage Index and the article Contents). `md.js`
  is the Markdown pipeline. `home.js` and `post.js` are the page controllers;
  each starts with a header comment that maps the file.
- Wide screens (1280px and up) show a fixed left index on both page types.
  Narrow screens show the homepage sections in the top nav and the article
  Contents inline under the title.

## Content conventions

- Frontmatter: `title`, `date`, `tags`, `summary`, optional `lang` and
  `draft`.
  - `date` is the creation date. **Never change it on an ordinary edit.**
  - `summary` is one short sentence. It feeds `llms.txt`; the homepage shows
    tags instead.
  - `draft: true` keeps an article off the homepage. An article that is
    listed while being written carries the `In Progress` tag instead.
- Ids: articles get `W<yymmdd>` from `date` automatically. Tools set
  `T<yymmdd>` in `profile/tools.json`.
- Tags are sorted alphabetically at build time; status tags such as
  `In Progress` go last. `wandb`, `Agent Runbook` and `In Progress` have
  their own chip style (`SPECIAL_TAGS`).
- **After changing frontmatter or any `profile/*.json`, run
  `python3 scripts/prerender.py`.** A stale bake silently shows old content
  to crawlers and agents.

## Checking a change

- Look at the page in the local preview, in both themes, at a wide and a
  narrow width. Check the browser console for errors.
- Two draft articles are regression fixtures: `?p=chart-gallery` (every chart
  case) and `?p=technical-writing-template` (every Markdown feature). Open
  them after touching `chart.js`, `md.js`, `post.js` or `article.css`.
- After touching `home.js` or `prerender.py`, run `prerender.py` and confirm
  that `index.html` and `llms.txt` change only where intended.

## Git and publishing

- **Ask before every commit.** Prefer a few commits over many; do not split
  work into a neat series.
- **Never push without an explicit push instruction in the current request.**
  An earlier "push it" does not carry over.
- **Never publish a new article the owner has not read.**
- Do not track or nag about unpushed commits.

## Working with the owner

- Discuss in Chinese. Articles are written in English.
- Write simple, direct, concise English: short sentences, common words, terms
  defined before use.
- Work one step at a time. Do not fill in sections ahead of the discussion.

## Maintaining this file

`AGENTS.md` is the single source of project instructions for AI models.
Update a rule in place when a decision changes. Keep it short.

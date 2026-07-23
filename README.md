# mianchuwang.github.io

Personal site. Content and UI are fully separated: posts are plain Markdown
files, and a small static frontend renders them in the browser (marked +
KaTeX + highlight.js — no build step, no framework).

## Writing a post

1. Add a Markdown file to `content/`, e.g. `content/my-post.md`:

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
   (`content/manifest.json`) and deploys automatically.

Add `draft: true` to the frontmatter to keep a post off the site.
See `content/technical-writing-template.md` for everything the renderer
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
content/           ← posts (Markdown + frontmatter); manifest.json is generated
content/profile.json       ← name, bio, and links shown in the home hero
content/publications.json  ← publication list shown on the home page
index.html         ← home page (post list)
post.html          ← article page (Markdown renderer, math, TOC)
assets/            ← styles and JS
scripts/           ← manifest generator
.github/workflows/ ← auto-deploy on push
```

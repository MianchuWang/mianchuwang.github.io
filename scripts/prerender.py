#!/usr/bin/env python3
"""Bake the homepage's JS-rendered content into index.html as static HTML.

The homepage normally renders profile / publications / tools / writing from
JSON in the browser, which leaves the raw HTML nearly empty — invisible to
crawlers and LLM agents that fetch without executing JS. This script mirrors
assets/home.js exactly (same markup, same classes) and writes the result
between <!--bake:*--> markers in index.html. home.js still runs and replaces
the baked content in place, so browsers see no difference.

Run before every push (it rebuilds writings/manifest.json first):

    python3 scripts/prerender.py
"""

import datetime
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
INDEX = ROOT / "index.html"


def load(rel):
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def format_date(iso):
    """Mirror formatDate in assets/md.js: en-US 'Aug 28, 2026'."""
    try:
        d = datetime.date.fromisoformat(iso)
    except (TypeError, ValueError):
        return iso or ""
    return f"{d.strftime('%b')} {d.day}, {d.year}"


# --- builders mirror the render* functions in assets/home.js ----------------

def bio_html(profile):
    return profile.get("bio", "")


def links_html(profile):
    out = []
    for link in profile.get("links", []):
        if not link.get("url"):
            out.append(f"<span>{link['label']}</span>")
        elif link["url"].startswith("mailto:"):
            out.append(f'<a href="{link["url"]}">{link["label"]}</a>')
        else:
            out.append(
                f'<a href="{link["url"]}" target="_blank" rel="noopener">{link["label"]}</a>'
            )
    return "".join(out)


def cv_html(profile):
    out = []
    for key, title in (("experience", "Experience"), ("education", "Education")):
        items = profile.get(key, [])
        if not items:
            continue
        entries = [f'<div class="hero-cv-title">{title}</div>']
        for item in items:
            role = item.get("role") or item.get("degree") or ""
            org = item.get("organization") or item.get("institution") or ""
            entries.append(
                f'<div class="hero-cv-item">\n'
                f'        <div class="hero-cv-role">{role}</div>\n'
                f'        <div class="hero-cv-org">{org} · {item.get("period", "")}</div></div>'
            )
        out.append('<div class="hero-cv-col">' + "\n      ".join(entries) + "</div>")
    return "".join(out)


def publications_html(pubs):
    lis = []
    for pub in pubs:
        title = (
            f'<a href="{pub["url"]}" target="_blank" rel="noopener">{pub["title"]}</a>'
            if pub.get("url")
            else pub["title"]
        )
        co = [a for a in pub.get("authors", []) if a != "Mianchu Wang"]
        with_str = f" · with {', '.join(co)}" if co else ""
        lis.append(
            f'<li>\n      <div class="pub-title">{title}</div>\n'
            f'      <div class="pub-meta"><span class="pub-venue-short">{pub["venueShort"]} '
            f'{pub["year"]}</span>{with_str}</div></li>'
        )
    return '<ul class="pub-list">' + "\n    ".join(lis) + "</ul>"


def tools_html(tools):
    lis = []
    for tool in tools:
        tid = f'<span class="tool-id">{tool["id"]}</span> ' if tool.get("id") else ""
        lis.append(
            f'<li>\n      <div class="pub-title"><a href="{tool["url"]}">{tool["title"]}</a></div>\n'
            f'      <div class="pub-meta">{tid}{tool.get("description", "")}</div></li>'
        )
    return '<ul class="pub-list">' + "\n    ".join(lis) + "</ul>"


def posts_html(posts):
    lis = []
    for p in posts:
        lang = f' lang="{p["lang"]}"' if p.get("lang") and p["lang"] != "en" else ""
        wid = f'<span class="tool-id">W{p["date"].replace("-", "")[2:]}</span>' if p.get("date") else ""
        special = {"wandb": " tag-wandb", "agent runbook": " tag-agent"}  # mirror tagClass() in md.js
        tags = "".join(
            f'<span class="tag{special.get(t.lower(), "")}">{t}</span>'
            for t in p.get("tags", [])
        )
        tags = f'<span class="post-tags">{wid}{tags}</span>'
        href = p.get("url") or f'post.html?p={p["slug"]}'
        external = ' target="_blank" rel="noopener"' if p.get("url") else ""
        lis.append(
            f'<li>\n      <a href="{href}"{external}{lang}>\n'
            f'        <span class="post-main">\n'
            f'          <span class="post-title">{p["title"]}</span>\n'
            f"          {tags}\n"
            f"        </span>\n"
            f'        <span class="post-date">{format_date(p["date"])}</span>\n'
            f"      </a></li>"
        )
    return '<ul class="post-list">' + "\n    ".join(lis) + "</ul>"


def sections_html(pubs, tools, posts):
    out = []
    if pubs:
        out.append('<h2 class="section-title" id="publications">Publications</h2>')
        out.append(publications_html(pubs))
    if tools:
        out.append('<h2 class="section-title" id="tool-set">Tool Set</h2>')
        out.append(tools_html(tools))
    if posts:
        out.append('<h2 class="section-title" id="writing">Writing</h2>')
        out.append(posts_html(posts))
    return "\n    ".join(out)


def bake(html, name, content):
    pattern = re.compile(f"(<!--bake:{name}-->).*?(<!--/bake:{name}-->)", re.S)
    if not pattern.search(html):
        sys.exit(f"marker bake:{name} not found in index.html")
    return pattern.sub(lambda m: m.group(1) + content + m.group(2), html)


SITE = "https://mianchuwang.github.io"


def llms_txt(profile, pubs, tools, posts):
    """Generate llms.txt — a markdown index for LLM agents, pointing articles
    at their raw .md files (post.html is an empty JS shell to a non-JS fetcher)."""
    bio_text = " ".join(re.sub(r"<[^>]+>", " ", profile.get("bio", "")).split())
    lines = [f"# {profile.get('name', 'Mianchu Wang')}", "", f"> {bio_text}", ""]
    if posts:
        lines += ["## Writing", ""]
        for p in posts:
            if p.get("url"):
                lines.append(f"- [{p['title']}]({p['url']}): {p.get('summary', '')}")
            else:
                lines.append(
                    f"- [{p['title']}]({SITE}/writings/{p['slug']}.md): {p.get('summary', '')} "
                    f"(raw markdown; human version at {SITE}/post.html?p={p['slug']})"
                )
        lines += [""]
    if pubs:
        lines += ["## Publications", ""]
        lines += [
            f"- [{pub['title']}]({pub.get('url', '')}): {pub['venueShort']} {pub['year']}"
            for pub in pubs
        ]
        lines += [""]
    if tools:
        lines += ["## Tools", ""]
        lines += [
            f"- [{t['title']}]({SITE}/{t['url']}): {t.get('description', '')}" for t in tools
        ]
        lines += [""]
    return "\n".join(lines)


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_manifest.py")], check=True)

    profile = load("profile/profile.json")
    pubs = load("profile/publications.json").get("publications", [])
    tools = load("profile/tools.json").get("tools", [])
    posts = load("writings/manifest.json").get("posts", [])

    html = INDEX.read_text(encoding="utf-8")
    html = bake(html, "bio", bio_html(profile))
    html = bake(html, "links", links_html(profile))
    html = bake(html, "cv", cv_html(profile))
    html = bake(html, "sections", sections_html(pubs, tools, posts))
    INDEX.write_text(html, encoding="utf-8")

    (ROOT / "llms.txt").write_text(llms_txt(profile, pubs, tools, posts), encoding="utf-8")

    text = re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>", "", html, flags=re.S))
    print(f"Baked index.html — {len(' '.join(text.split()))} chars of static text")
    print("Wrote llms.txt")


if __name__ == "__main__":
    main()

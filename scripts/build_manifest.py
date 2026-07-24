#!/usr/bin/env python3
"""Scan writings/*.md frontmatter and write writings/manifest.json.

Run manually for local preview; CI runs it automatically on deploy.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "writings"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(text: str) -> dict:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}
    meta = {}
    for line in m.group(1).splitlines():
        kv = re.match(r"^(\w+)\s*:\s*(.*)$", line)
        if not kv:
            continue
        key, value = kv.group(1), kv.group(2).strip()
        if value.startswith("[") and value.endswith("]"):
            meta[key] = [
                v.strip().strip("\"'") for v in value[1:-1].split(",") if v.strip()
            ]
        else:
            meta[key] = value.strip("\"'")
    return meta


def main() -> None:
    posts = []
    for path in sorted(CONTENT.glob("*.md")):
        meta = parse_frontmatter(path.read_text(encoding="utf-8"))
        if meta.get("draft") in ("true", "yes"):
            continue
        posts.append(
            {
                "slug": path.stem,
                "title": meta.get("title", path.stem),
                "date": meta.get("date", ""),
                "lang": meta.get("lang", "en"),
                "tags": meta.get("tags", []),
                "summary": meta.get("summary", ""),
            }
        )

    posts.sort(key=lambda p: p["date"], reverse=True)
    out = CONTENT / "manifest.json"
    out.write_text(
        json.dumps({"posts": posts}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out.relative_to(ROOT)} ({len(posts)} posts)")


if __name__ == "__main__":
    main()

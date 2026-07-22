import { formatDate, initTheme } from "./md.js";

initTheme();

const listEl = document.getElementById("post-list");

async function main() {
  let manifest;
  try {
    const res = await fetch("content/manifest.json", { cache: "no-cache" });
    manifest = await res.json();
  } catch {
    listEl.innerHTML = `<div class="empty-note">Could not load content/manifest.json — run <code>python3 scripts/build_manifest.py</code>.</div>`;
    return;
  }

  const posts = manifest.posts || [];
  if (posts.length === 0) {
    listEl.innerHTML = `<div class="empty-note">Nothing here yet.</div>`;
    return;
  }

  const ul = document.createElement("ul");
  ul.className = "post-list";
  for (const p of posts) {
    const li = document.createElement("li");
    const tags = (p.tags || [])
      .map((t) => `<span class="tag">${t}</span>`)
      .join("");
    li.innerHTML = `
      <a href="post.html?p=${encodeURIComponent(p.slug)}">
        <div class="post-title">${p.title}</div>
        <div class="post-meta"><span>${formatDate(p.date)}</span>${tags}</div>
        ${p.summary ? `<div class="post-summary">${p.summary}</div>` : ""}
      </a>`;
    ul.appendChild(li);
  }
  listEl.replaceChildren(ul);
}

main();

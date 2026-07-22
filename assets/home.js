import { formatDate, initTheme } from "./md.js";

initTheme();

const listEl = document.getElementById("post-list");

function renderList(posts) {
  const ul = document.createElement("ul");
  ul.className = "post-list";
  for (const p of posts) {
    const li = document.createElement("li");
    li.innerHTML = `
      <a href="post.html?p=${encodeURIComponent(p.slug)}"${p.lang && p.lang !== "en" ? ` lang="${p.lang}"` : ""}>
        <span class="post-title">${p.title}</span>
        <span class="post-date">${formatDate(p.date)}</span>
      </a>`;
    ul.appendChild(li);
  }
  return ul;
}

function renderPublications(pubs) {
  const ul = document.createElement("ul");
  ul.className = "pub-list";
  for (const pub of pubs) {
    const li = document.createElement("li");
    const title = pub.url
      ? `<a href="${pub.url}" target="_blank" rel="noopener">${pub.title}</a>`
      : pub.title;
    const coAuthors = pub.authors.filter((a) => a !== "Mianchu Wang");
    const withStr = coAuthors.length ? ` · with ${coAuthors.join(", ")}` : "";
    li.innerHTML = `
      <div class="pub-title">${title}</div>
      <div class="pub-meta"><span class="pub-venue-short">${pub.venueShort} ${pub.year}</span>${withStr}</div>`;
    ul.appendChild(li);
  }
  return ul;
}

function renderProfile(profile) {
  if (profile.name) {
    document.getElementById("hero-name").textContent = profile.name;
  }
  if (profile.bio) {
    document.getElementById("hero-bio").textContent = profile.bio;
  }
  const linksEl = document.getElementById("hero-links");
  for (const link of profile.links || []) {
    const a = document.createElement("a");
    a.href = link.url;
    a.textContent = link.label;
    if (!link.url.startsWith("mailto:")) {
      a.target = "_blank";
      a.rel = "noopener";
    }
    linksEl.appendChild(a);
  }
}

function renderHeroCv(profile) {
  const cvEl = document.getElementById("hero-cv");
  const columns = [
    { key: "experience", title: "Experience" },
    { key: "education", title: "Education" },
  ];
  for (const col of columns) {
    const items = profile[col.key] || [];
    if (items.length === 0) continue;
    const div = document.createElement("div");
    div.className = "hero-cv-col";
    div.innerHTML = `<div class="hero-cv-title">${col.title}</div>`;
    for (const item of items) {
      const entry = document.createElement("div");
      entry.className = "hero-cv-item";
      entry.innerHTML = `
        <div class="hero-cv-role">${item.role || item.degree || ""}</div>
        <div class="hero-cv-org">${item.organization || item.institution || ""} · ${item.period || ""}</div>`;
      div.appendChild(entry);
    }
    cvEl.appendChild(div);
  }
}

async function main() {
  let manifest = { posts: [] };
  let pubData = { publications: [] };
  let profile = {};
  try {
    const [manifestRes, pubRes, profileRes] = await Promise.all([
      fetch("content/manifest.json", { cache: "no-cache" }),
      fetch("content/publications.json", { cache: "no-cache" }),
      fetch("content/profile.json", { cache: "no-cache" }),
    ]);
    manifest = await manifestRes.json();
    if (pubRes.ok) pubData = await pubRes.json();
    if (profileRes.ok) profile = await profileRes.json();
    renderProfile(profile);
    renderHeroCv(profile);
  } catch {
    listEl.innerHTML = `<div class="empty-note">Could not load content/manifest.json — run <code>python3 scripts/build_manifest.py</code>.</div>`;
    return;
  }

  const posts = manifest.posts || [];
  const pubs = pubData.publications || [];
  if (posts.length === 0 && pubs.length === 0) {
    listEl.innerHTML = `<div class="empty-note">Nothing here yet.</div>`;
    return;
  }

  const frag = document.createDocumentFragment();
  if (posts.length > 0) {
    const h2 = document.createElement("h2");
    h2.className = "section-title";
    h2.textContent = "Technical Writing";
    frag.appendChild(h2);
    frag.appendChild(renderList(posts));
  }

  if (pubs.length > 0) {
    const h2 = document.createElement("h2");
    h2.className = "section-title";
    h2.textContent = "Publications";
    frag.appendChild(h2);
    frag.appendChild(renderPublications(pubs));
  }

  listEl.replaceChildren(frag);
}

main();

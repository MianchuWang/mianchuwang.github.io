/* Homepage renderer (index.html): profile / publications / tools / writing
   from JSON. scripts/prerender.py bakes IDENTICAL markup between the
   <!--bake:*--> markers — change one, change the other. Styles: home.css. */

import { formatDate, tagClass, articleId, initTheme, createScrollSpy } from "./site.js";

initTheme();

const sectionsEl = document.getElementById("sections"); // Publications / Writing / Toolbox

/* <ul class=className> with one <li> per item; itemHtml(item) is its markup.
   The *Html functions below are the ones prerender.py mirrors. */
function renderList(className, items, itemHtml) {
  const ul = document.createElement("ul");
  ul.className = className;
  for (const item of items) {
    const li = document.createElement("li");
    li.innerHTML = itemHtml(item);
    ul.appendChild(li);
  }
  return ul;
}

function postHtml(p) {
  const href = p.url || `post.html?p=${encodeURIComponent(p.slug)}`;
  const external = p.url ? ` target="_blank" rel="noopener"` : "";
  const lang = p.lang && p.lang !== "en" ? ` lang="${p.lang}"` : "";
  const id = p.date ? `<span class="tool-id">${articleId(p.date)}</span>` : "";
  const tags = (p.tags || []).map((t) => `<span class="${tagClass(t)}">${t}</span>`).join("");
  return `
      <a href="${href}"${external}${lang}>
        <span class="post-main">
          <span class="post-title">${p.title}</span>
          <span class="post-tags">${id}${tags}</span>
        </span>
        <span class="post-date">${formatDate(p.date)}</span>
      </a>`;
}

function toolHtml(tool) {
  const id = tool.id ? `<span class="tool-id">${tool.id}</span> ` : "";
  return `
      <div class="pub-title"><a href="${tool.url}">${tool.title}</a></div>
      <div class="pub-meta">${id}${tool.description || ""}</div>`;
}

function publicationHtml(pub) {
  const title = pub.url
    ? `<a href="${pub.url}" target="_blank" rel="noopener">${pub.title}</a>`
    : pub.title;
  const coAuthors = pub.authors.filter((a) => a !== "Mianchu Wang");
  const withStr = coAuthors.length ? ` · with ${coAuthors.join(", ")}` : "";
  return `
      <div class="pub-title">${title}</div>
      <div class="pub-meta"><span class="pub-venue-short">${pub.venueShort} ${pub.year}</span>${withStr}</div>`;
}

function renderProfile(profile) {
  if (profile.name) {
    document.getElementById("hero-name").textContent = profile.name;
  }
  if (profile.bio) {
    document.getElementById("hero-bio").innerHTML = profile.bio;
  }
  if (!profile.links?.length) return; // fetch failed or empty: keep the baked links
  const linksEl = document.getElementById("hero-links");
  linksEl.replaceChildren(); // drop prerendered copies before re-rendering
  for (const link of profile.links) {
    // Entries without a url render as plain text (e.g. spam-safe email).
    if (!link.url) {
      const span = document.createElement("span");
      span.textContent = link.label;
      linksEl.appendChild(span);
      continue;
    }
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
  if (!profile.experience?.length && !profile.education?.length) return; // keep the bake
  const cvEl = document.getElementById("hero-cv");
  cvEl.replaceChildren(); // drop prerendered copies before re-rendering
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

/* null when the file is missing; a network or JSON error rejects */
async function loadJson(url) {
  const res = await fetch(url, { cache: "no-cache" });
  return res.ok ? res.json() : null;
}

async function main() {
  let sources;
  try {
    sources = await Promise.all([
      loadJson("writings/manifest.json"),
      loadJson("profile/publications.json"),
      loadJson("profile/tools.json"),
      loadJson("profile/profile.json"),
    ]);
  } catch {
    // Keep the prerendered static content if the live fetch fails.
    if (!sectionsEl.querySelector(".section-title")) {
      sectionsEl.innerHTML = `<div class="empty-note">Could not load writings/manifest.json — run <code>python3 scripts/build_manifest.py</code>.</div>`;
    }
    return;
  }

  const [manifest, pubData, toolData, profile] = sources;
  renderProfile(profile || {});
  renderHeroCv(profile || {});
  // any source missing -> keep the prerendered sections rather than
  // re-rendering a page with sections silently dropped
  if (!manifest || !pubData || !toolData) return;

  const posts = manifest.posts || [];
  const pubs = pubData.publications || [];
  const tools = toolData.tools || [];
  if (posts.length === 0 && pubs.length === 0 && tools.length === 0) {
    sectionsEl.innerHTML = `<div class="empty-note">Nothing here yet.</div>`;
    return;
  }

  const frag = document.createDocumentFragment();
  const addSection = (id, title, list) => {
    const h2 = document.createElement("h2");
    h2.className = "section-title";
    h2.id = id;
    h2.textContent = title;
    frag.append(h2, list);
  };
  if (pubs.length > 0) addSection("publications", "Publications", renderList("pub-list", pubs, publicationHtml));
  if (posts.length > 0) addSection("writing", "Writing", renderList("post-list", posts, postHtml));
  if (tools.length > 0) addSection("toolbox", "Toolbox", renderList("pub-list", tools, toolHtml));

  sectionsEl.replaceChildren(frag);
}

/* Section index: the fixed left Index on wide screens and the top-nav links
   on narrow ones. Both work on static markup (nav, .home-toc, baked
   headings), so they must not depend on the JSON fetches succeeding. */
function initSectionIndex() {
  // The Index title has no hash, so its key is "": it owns the highlight
  // above the first section, and a click on it returns to the top.
  const links = new Map(
    [...document.querySelectorAll(".home-toc a")].map((a) => [a.hash.slice(1), a])
  );
  // look sections up per call: main() re-renders them over the baked copies
  const spy = createScrollSpy(links, (id) => document.getElementById(id));
  for (const a of document.querySelectorAll(".home-toc a, .nav-links a")) {
    a.addEventListener("click", (e) => {
      if (spy.jumpTo(a.hash.slice(1))) e.preventDefault();
    });
  }
}

initSectionIndex();
main();

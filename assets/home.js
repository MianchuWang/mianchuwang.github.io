/* Homepage renderer (index.html): profile / publications / tools / writing
   from JSON. scripts/prerender.py bakes IDENTICAL markup between the
   <!--bake:*--> markers — change one, change the other. Styles: home.css. */

import { formatDate, tagClass, initTheme } from "./md.js";

initTheme();

const listEl = document.getElementById("post-list");

function renderList(posts) {
  const ul = document.createElement("ul");
  ul.className = "post-list";
  for (const p of posts) {
    const li = document.createElement("li");
    const href = p.url || `post.html?p=${encodeURIComponent(p.slug)}`;
    const external = p.url ? ` target="_blank" rel="noopener"` : "";
    li.innerHTML = `
      <a href="${href}"${external}${p.lang && p.lang !== "en" ? ` lang="${p.lang}"` : ""}>
        <span class="post-main">
          <span class="post-title">${p.title}</span>
          <span class="post-tags">${p.date ? `<span class="tool-id">W${p.date.replaceAll("-", "").slice(2)}</span>` : ""}${(p.tags || []).map((t) => `<span class="${tagClass(t)}">${t}</span>`).join("")}</span>
        </span>
        <span class="post-date">${formatDate(p.date)}</span>
      </a>`;
    ul.appendChild(li);
  }
  return ul;
}

function renderTools(tools) {
  const ul = document.createElement("ul");
  ul.className = "pub-list";
  for (const tool of tools) {
    const li = document.createElement("li");
    const id = tool.id ? `<span class="tool-id">${tool.id}</span> ` : "";
    li.innerHTML = `
      <div class="pub-title"><a href="${tool.url}">${tool.title}</a></div>
      <div class="pub-meta">${id}${tool.description || ""}</div>`;
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
    document.getElementById("hero-bio").innerHTML = profile.bio;
  }
  if (!profile.links?.length) return; // fetch failed or empty: keep the baked links
  const linksEl = document.getElementById("hero-links");
  linksEl.replaceChildren(); // drop prerendered copies before re-rendering
  for (const link of profile.links || []) {
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

async function main() {
  let manifest = { posts: [] };
  let pubData = { publications: [] };
  let toolData = { tools: [] };
  let profile = {};
  try {
    const [manifestRes, pubRes, toolRes, profileRes] = await Promise.all([
      fetch("writings/manifest.json", { cache: "no-cache" }),
      fetch("profile/publications.json", { cache: "no-cache" }),
      fetch("profile/tools.json", { cache: "no-cache" }),
      fetch("profile/profile.json", { cache: "no-cache" }),
    ]);
    if (manifestRes.ok) manifest = await manifestRes.json();
    if (pubRes.ok) pubData = await pubRes.json();
    if (toolRes.ok) toolData = await toolRes.json();
    if (profileRes.ok) profile = await profileRes.json();
    renderProfile(profile);
    renderHeroCv(profile);
    // any source missing -> keep the prerendered sections rather than
    // re-rendering a page with sections silently dropped
    if (!manifestRes.ok || !pubRes.ok || !toolRes.ok) return;
  } catch {
    // Keep the prerendered static content if the live fetch fails.
    if (!listEl.querySelector(".section-title")) {
      listEl.innerHTML = `<div class="empty-note">Could not load writings/manifest.json — run <code>python3 scripts/build_manifest.py</code>.</div>`;
    }
    return;
  }

  const posts = manifest.posts || [];
  const pubs = pubData.publications || [];
  const tools = toolData.tools || [];
  if (posts.length === 0 && pubs.length === 0 && tools.length === 0) {
    listEl.innerHTML = `<div class="empty-note">Nothing here yet.</div>`;
    return;
  }

  const frag = document.createDocumentFragment();
  if (pubs.length > 0) {
    const h2 = document.createElement("h2");
    h2.className = "section-title";
    h2.id = "publications";
    h2.textContent = "Publications";
    frag.appendChild(h2);
    frag.appendChild(renderPublications(pubs));
  }

  if (posts.length > 0) {
    const h2 = document.createElement("h2");
    h2.className = "section-title";
    h2.id = "writing";
    h2.textContent = "Writing";
    frag.appendChild(h2);
    frag.appendChild(renderList(posts));
  }

  if (tools.length > 0) {
    const h2 = document.createElement("h2");
    h2.className = "section-title";
    h2.id = "tool-set";
    h2.textContent = "Tool Set";
    frag.appendChild(h2);
    frag.appendChild(renderTools(tools));
  }

  listEl.replaceChildren(frag);
}

/* Left index: navigate by scrollIntoView — Chromium can silently drop
   smooth fragment scrolls, and a same-hash re-click never scrolls. */
let spySet = null;     // assigned by spySections
let spyRefresh = null; // assigned by spySections
let spyLockUntil = 0;  // clicks pin the highlight briefly; geometry may disagree

function bindSectionLinks() {
  for (const a of document.querySelectorAll(".home-toc a, .nav-links a")) {
    a.addEventListener("click", (e) => {
      if (!a.hash) {
        // the Index title: back to the top, clean URL
        e.preventDefault();
        window.scrollTo({ top: 0, behavior: "instant" });
        history.replaceState(null, "", location.pathname);
        spySet?.("");
        spyLockUntil = Date.now() + 500;
        return;
      }
      const el = document.getElementById(a.hash.slice(1));
      if (!el) return;
      e.preventDefault();
      el.scrollIntoView({ behavior: "instant", block: "start" }); // instant: smooth hash-scrolls are unreliable and a same-hash re-click never fires
      history.replaceState(null, "", a.hash);
      spySet?.(a.hash.slice(1));
      spyLockUntil = Date.now() + 500;
    });
  }
}

/* Highlight the left index entry for the section in view.
   Scroll-scanline instead of IntersectionObserver: an instant jump
   teleports headings past an IO band without firing it, and at the page
   bottom the last heading may never enter the band at all. */
function spySections() {
  // The Index title participates with key "": it is the "above the first
  // section" state, so the bar rests on Index at the top of the page.
  const links = new Map(
    [...document.querySelectorAll(".home-toc a")].map((a) => [a.hash.slice(1), a])
  );
  if (links.size === 0) return;
  const ids = [...links.keys()];

  const setActive = (id) => {
    links.forEach((a) => a.classList.remove("active"));
    links.get(id)?.classList.add("active");
  };

  spySet = setActive;
  let deferred = 0;
  const onScroll = () => {
    const now = Date.now();
    if (now < spyLockUntil) {
      // a scroll landed inside the click-lock: re-check once it expires,
      // or the highlight would stay pinned wrong with no further events
      clearTimeout(deferred);
      deferred = setTimeout(onScroll, spyLockUntil - now + 20);
      return;
    }
    const atBottom = scrollY + innerHeight >= document.documentElement.scrollHeight - 2;
    if (atBottom) return setActive(ids[ids.length - 1]);
    const line = innerHeight * 0.3;
    let current = ids[0];
    for (const id of ids) {
      const el = document.getElementById(id);
      if (el && el.getBoundingClientRect().top <= line) current = id;
    }
    setActive(current);
  };
  spyRefresh = onScroll;
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();
}

// The index and its spy work on static markup (nav, .home-toc, baked
// headings) — they must not depend on the JSON fetches succeeding.
bindSectionLinks();
spySections();
main();

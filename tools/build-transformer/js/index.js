import { isSolved, solvedAt, clearAll } from "./store.js";

const groupsEl = document.getElementById("component-groups");
const fillEl = document.getElementById("progress-fill");
const textEl = document.getElementById("progress-text");

function since(iso) {
  const days = Math.floor((Date.now() - new Date(iso).getTime()) / 86400000);
  if (!Number.isFinite(days) || days < 0) return "";
  if (days === 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 30) return `${days}d ago`;
  return new Date(iso).toLocaleDateString();
}

function row(comp) {
  const a = document.createElement("a");
  a.className = "component-row" + (isSolved(comp.id) ? " solved" : "");
  a.href = `lab.html?c=${encodeURIComponent(comp.id)}`;
  const badge =
    comp.difficulty === "easy"
      ? ""
      : `<span class="badge badge-${comp.difficulty}">${comp.difficulty}</span>`;
  const when = solvedAt(comp.id);
  const solvedNote = when ? ` <span class="row-solved">&middot; solved ${since(when)}</span>` : "";
  const time = comp.minutes ? `<span class="row-time">~${comp.minutes}m</span>` : "";
  a.innerHTML = `
    <span class="row-status">${isSolved(comp.id) ? "&#10003;" : ""}</span>
    <span class="row-main">
      <span class="row-title">${comp.title}</span>
      <span class="row-tagline">${comp.tagline}${solvedNote}</span>
    </span>
    ${badge}
    ${time}`;
  return a;
}

function render(components) {
  const frag = document.createDocumentFragment();
  const groups = new Map();
  for (const comp of components) {
    if (!groups.has(comp.group)) groups.set(comp.group, []);
    groups.get(comp.group).push(comp);
  }
  for (const [group, comps] of groups) {
    const h2 = document.createElement("h2");
    h2.className = "group-title";
    h2.textContent = group;
    frag.appendChild(h2);
    const list = document.createElement("div");
    list.className = "component-list";
    comps.forEach((comp) => list.appendChild(row(comp)));
    frag.appendChild(list);
  }
  groupsEl.replaceChildren(frag);

  const solved = components.filter((c) => isSolved(c.id)).length;
  textEl.textContent = `${solved} / ${components.length} solved`;
  fillEl.style.width = `${(100 * solved) / components.length}%`;
}

async function main() {
  let components = [];
  try {
    const res = await fetch("data/components.json", { cache: "no-cache" });
    components = (await res.json()).components || [];
  } catch {
    groupsEl.innerHTML = `<p class="output-empty">Could not load the component list. Serve this
      folder over HTTP (for example <code>python3 -m http.server</code>) rather than opening the
      file directly.</p>`;
    return;
  }
  render(components);

  document.getElementById("reset-progress").addEventListener("click", () => {
    if (!confirm("Clear all saved code and progress for this tool?")) return;
    clearAll();
    render(components);
  });
}

main();

/* Reusable interactive SVG line charts — no dependencies.
 *
 * Embed anywhere (posts, tools, homepage) as:
 *
 *   <div class="chart" data-src="writings/figures/w260830.json" data-metric="acc">
 *     fallback text for no-JS readers
 *   </div>
 *
 * then call initCharts(rootEl). Data file schema:
 *
 *   { "metrics": { "<name>": { "title", "xLabel", "yLabel",
 *       "series": [ { "label", "points": [[x, y], ...] }, ... ] } } }
 *
 * All colors come from CSS variables (--chart-s1..s4 + site ink tokens), so
 * charts follow the light/dark theme with zero JS involvement.
 */

const NS = "http://www.w3.org/2000/svg";
const W = 860, H = 400;
const M = { t: 14, r: 96, b: 42, l: 58 };
const MAX_SERIES = 4;

export async function initCharts(root) {
  const els = [...root.querySelectorAll(".chart[data-src]")];
  const cache = new Map();
  for (const el of els) {
    const src = el.dataset.src;
    try {
      if (!cache.has(src)) {
        cache.set(src, fetch(src, { cache: "no-cache" }).then((r) => {
          if (!r.ok) throw new Error(r.status);
          return r.json();
        }));
      }
      const data = await cache.get(src);
      const metric = data.metrics[el.dataset.metric];
      if (!metric) throw new Error(`metric ${el.dataset.metric} not in ${src}`);
      render(el, metric);
    } catch (e) {
      console.warn("chart:", e); // keep the fallback content
    }
  }
}

function svgEl(name, attrs, parent) {
  const el = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}

/* 1-2-5 tick steps covering [min, max] with ~n ticks */
function ticks(min, max, n) {
  const span = max - min || 1;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
  const step = [1, 2, 5, 10].map((m) => m * step0).find((s) => span / s <= n);
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step / 1e6; v += step) {
    out.push(+v.toFixed(12));
  }
  return out;
}

const fmt = (v) => {
  if (Math.abs(v) >= 1000) return v.toLocaleString("en-US");
  return +v.toFixed(4) + "";
};

function render(el, metric) {
  const series = metric.series.slice(0, MAX_SERIES);
  const fallback = el.textContent.trim();
  el.textContent = "";
  el.classList.add("chart-ready");
  if (fallback) el.setAttribute("aria-label", fallback);
  el.setAttribute("role", "img");

  if (metric.title) {
    const t = document.createElement("div");
    t.className = "chart-title";
    t.textContent = metric.title;
    el.appendChild(t);
  }

  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const ys = series.flatMap((s) => s.points.map((p) => p[1]));
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  const pad = (yMax - yMin || 1) * 0.06;
  yMin -= pad; yMax += pad;

  const px = (x) => M.l + ((x - xMin) / (xMax - xMin || 1)) * (W - M.l - M.r);
  const py = (y) => H - M.b - ((y - yMin) / (yMax - yMin)) * (H - M.t - M.b);

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart-svg" }, el);

  // grid + tick labels (recessive)
  for (const y of ticks(yMin, yMax, 5)) {
    svgEl("line", { x1: M.l, x2: W - M.r, y1: py(y), y2: py(y), class: "chart-grid" }, svg);
    svgEl("text", { x: M.l - 8, y: py(y) + 4, "text-anchor": "end", class: "chart-tick" }, svg)
      .textContent = fmt(y);
  }
  for (const x of ticks(xMin, xMax, 6)) {
    svgEl("text", { x: px(x), y: H - M.b + 18, "text-anchor": "middle", class: "chart-tick" }, svg)
      .textContent = fmt(x);
  }
  if (metric.xLabel) {
    svgEl("text", { x: (M.l + W - M.r) / 2, y: H - 6, "text-anchor": "middle", class: "chart-axis-label" }, svg)
      .textContent = metric.xLabel;
  }
  if (metric.yLabel) {
    svgEl("text", {
      x: 14, y: (M.t + H - M.b) / 2, "text-anchor": "middle",
      transform: `rotate(-90 14 ${(M.t + H - M.b) / 2})`, class: "chart-axis-label",
    }, svg).textContent = metric.yLabel;
  }

  // series paths (+ markers when sparse)
  series.forEach((s, i) => {
    const d = s.points.map(([x, y], j) => `${j ? "L" : "M"}${px(x).toFixed(1)},${py(y).toFixed(1)}`).join("");
    svgEl("path", { d, class: `chart-line chart-c${i + 1}` }, svg);
    if (s.points.length <= 40) {
      for (const [x, y] of s.points) {
        svgEl("circle", { cx: px(x), cy: py(y), r: 3, class: `chart-dot chart-c${i + 1}` }, svg);
      }
    }
  });

  // direct labels at line ends, nudged apart (ink color, per-series mark implied by position)
  const ends = series
    .map((s, i) => ({ label: s.label, i, y: py(s.points[s.points.length - 1][1]) }))
    .sort((a, b) => a.y - b.y);
  for (let k = 1; k < ends.length; k++) {
    if (ends[k].y - ends[k - 1].y < 16) ends[k].y = ends[k - 1].y + 16;
  }
  for (const e of ends) {
    svgEl("circle", { cx: W - M.r + 8, cy: e.y - 4, r: 4, class: `chart-dot chart-c${e.i + 1}` }, svg);
    svgEl("text", { x: W - M.r + 16, y: e.y, class: "chart-end-label" }, svg).textContent = e.label;
  }

  // hover layer: crosshair + per-series highlight dots + tooltip
  const cross = svgEl("line", { y1: M.t, y2: H - M.b, class: "chart-crosshair", visibility: "hidden" }, svg);
  const hiDots = series.map((_, i) =>
    svgEl("circle", { r: 4.5, class: `chart-dot chart-hi chart-c${i + 1}`, visibility: "hidden" }, svg));
  const tip = document.createElement("div");
  tip.className = "chart-tip";
  tip.hidden = true;
  el.appendChild(tip);

  const hitbox = svgEl("rect", {
    x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, fill: "transparent",
  }, svg);

  const nearest = (s, x) => s.points.reduce((a, b) => (Math.abs(b[0] - x) < Math.abs(a[0] - x) ? b : a));

  hitbox.addEventListener("pointermove", (ev) => {
    const r = svg.getBoundingClientRect();
    const x = xMin + ((ev.clientX - r.left) * (W / r.width) - M.l) / (W - M.l - M.r) * (xMax - xMin);
    const hits = series.map((s) => nearest(s, x));
    const cx = px(hits[0][0]);
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx);
    cross.setAttribute("visibility", "visible");
    hits.forEach(([hx, hy], i) => {
      hiDots[i].setAttribute("cx", px(hx));
      hiDots[i].setAttribute("cy", py(hy));
      hiDots[i].setAttribute("visibility", "visible");
    });
    tip.innerHTML =
      `<div class="chart-tip-head">${metric.xLabel || "x"} ${fmt(hits[0][0])}</div>` +
      series.map((s, i) =>
        `<div class="chart-tip-row"><span class="chart-swatch chart-c${i + 1}"></span>` +
        `<span class="chart-tip-label">${s.label}</span><span class="chart-tip-val">${fmt(hits[i][1])}</span></div>`
      ).join("");
    tip.hidden = false;
    const scale = r.width / W;
    const svgTop = r.top - el.getBoundingClientRect().top;
    const flip = cx > (W - M.r + M.l) / 2;
    tip.style.left = flip ? "" : `${(cx + 14) * scale}px`;
    tip.style.right = flip ? `${(W - cx + 14) * scale}px` : "";
    tip.style.top = `${svgTop + (M.t + 8) * scale}px`;
  });
  hitbox.addEventListener("pointerleave", () => {
    cross.setAttribute("visibility", "hidden");
    hiDots.forEach((d) => d.setAttribute("visibility", "hidden"));
    tip.hidden = true;
  });

  // legend (>= 2 series) — swatch carries color, text wears ink
  if (series.length >= 2) {
    const legend = document.createElement("div");
    legend.className = "chart-legend";
    legend.innerHTML = series.map((s, i) =>
      `<span class="chart-legend-item"><span class="chart-swatch chart-c${i + 1}"></span>${s.label}</span>`
    ).join("");
    el.appendChild(legend);
  }

  // collapsed table view of the data
  const details = document.createElement("details");
  details.className = "chart-data";
  const head = `<tr><th>${metric.xLabel || "x"}</th>${series.map((s) => `<th>${s.label}</th>`).join("")}</tr>`;
  const xsAll = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
  const byX = series.map((s) => new Map(s.points));
  const rows = xsAll.map((x) =>
    `<tr><td>${fmt(x)}</td>${byX.map((m) => `<td>${m.has(x) ? fmt(m.get(x)) : ""}</td>`).join("")}</tr>`
  ).join("");
  details.innerHTML = `<summary>View data</summary><div class="chart-data-scroll"><table>${head}${rows}</table></div>`;
  el.appendChild(details);
}

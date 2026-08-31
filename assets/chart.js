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
 *       "options": { ... },                     // optional, see DEFAULTS
 *       "series": [ { "label", "points": [[x, y], ...] }, ... ] } } }
 *
 * COMPATIBILITY CONTRACT (how to extend without breaking old charts):
 *   - Every new capability is an option in DEFAULTS whose default reproduces
 *     today's behavior; charts opt in via "options" in the JSON or a
 *     data-opt-* attribute on the embed div (data-opt-markers="false").
 *   - Unknown fields in the data file and unknown options are ignored, so old
 *     renderers tolerate new data and the new renderer tolerates old data.
 *   - The render pipeline below is a list of stages sharing one ctx object;
 *     add features as new stages (or new branches inside one), never by
 *     changing what an existing stage draws by default.
 *   - Stable public surface: the embed API (div.chart + data-src/data-metric),
 *     the schema fields above, and the CSS class names (chart-*). Renaming any
 *     of these is a breaking change; don't.
 *
 * All colors come from CSS variables (--chart-s1..s4 + site ink tokens), so
 * charts follow the light/dark theme with zero JS involvement.
 */

const NS = "http://www.w3.org/2000/svg";

const DEFAULTS = {
  width: 860,           // viewBox size (rendered responsively)
  height: 400,
  margin: { t: 14, r: 96, b: 42, l: 58 },
  maxSeries: 4,         // palette slots available
  markers: "auto",      // "auto" (only when sparse) | true | false
  markersMax: 40,       // "auto" threshold: dots only if points <= this
  yTicks: 5,
  xTicks: 6,
  endLabels: true,      // direct labels at line ends
  legend: "auto",       // "auto" (>= 2 series) | true | false
  hover: true,          // crosshair + tooltip layer
  dataTable: true,      // collapsed "View data" table
  yPad: 0.06,           // headroom fraction above/below the data
};

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

/* ------------------------------------------------------------- helpers -- */

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

/* data-opt-* attributes on the embed div override the JSON's options */
function datasetOptions(el) {
  const out = {};
  for (const [k, v] of Object.entries(el.dataset)) {
    if (!k.startsWith("opt")) continue;
    const key = k[3].toLowerCase() + k.slice(4);
    out[key] = v === "true" ? true : v === "false" ? false : isNaN(+v) ? v : +v;
  }
  return out;
}

/* -------------------------------------------------------------- render -- */

function render(el, metric) {
  const opts = { ...DEFAULTS, ...(metric.options || {}), ...datasetOptions(el) };
  const series = metric.series.slice(0, opts.maxSeries);

  if (opts.endLabels) {
    // widen the right margin to fit the longest end label (13px, .chart-end-label)
    const meas = document.createElement("canvas").getContext("2d");
    meas.font = `13px ${getComputedStyle(el).fontFamily}`;
    const widest = Math.max(...series.map((s) => meas.measureText(s.label).width));
    opts.margin = { ...opts.margin, r: Math.max(opts.margin.r, Math.ceil(widest) + 24) };
  }

  const fallback = el.textContent.trim();
  el.textContent = "";
  el.classList.add("chart-ready");
  if (fallback) el.setAttribute("aria-label", fallback);
  el.setAttribute("role", "img");

  // shared context every stage reads from / writes to
  const ctx = { el, metric, series, opts, svg: null, px: null, py: null };
  for (const stage of [stageScales, stageTitle, stageAxes, stageSeries,
                       stageEndLabels, stageHover, stageLegend, stageDataTable]) {
    stage(ctx);
  }
}

function stageScales(ctx) {
  const { series, opts } = ctx;
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const ys = series.flatMap((s) => s.points.map((p) => p[1]));
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  const pad = (yMax - yMin || 1) * opts.yPad;
  yMin -= pad; yMax += pad;
  const { width: W, height: H, margin: M } = opts;
  ctx.bounds = { xMin, xMax, yMin, yMax };
  ctx.px = (x) => M.l + ((x - xMin) / (xMax - xMin || 1)) * (W - M.l - M.r);
  ctx.py = (y) => H - M.b - ((y - yMin) / (yMax - yMin)) * (H - M.t - M.b);
}

function stageTitle(ctx) {
  if (!ctx.metric.title) return;
  const t = document.createElement("div");
  t.className = "chart-title";
  const sub = ctx.metric.title.match(/^(E\d+\.\d+)\s*—\s*/); // "E1.1 — …" gets a sub-chip
  if (sub) {
    const chip = document.createElement("span");
    chip.className = "exp-subchip";
    chip.textContent = sub[1];
    t.appendChild(chip);
    t.appendChild(document.createTextNode(ctx.metric.title.slice(sub[0].length)));
  } else {
    t.textContent = ctx.metric.title;
  }
  ctx.el.appendChild(t);
}

function stageAxes(ctx) {
  const { opts, metric, px, py, bounds } = ctx;
  const { width: W, height: H, margin: M } = opts;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart-svg" }, ctx.el);
  ctx.svg = svg;

  for (const y of ticks(bounds.yMin, bounds.yMax, opts.yTicks)) {
    svgEl("line", { x1: M.l, x2: W - M.r, y1: py(y), y2: py(y), class: "chart-grid" }, svg);
    svgEl("text", { x: M.l - 8, y: py(y) + 4, "text-anchor": "end", class: "chart-tick" }, svg)
      .textContent = fmt(y);
  }
  for (const x of ticks(bounds.xMin, bounds.xMax, opts.xTicks)) {
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
}

function stageSeries(ctx) {
  const { series, opts, svg, px, py } = ctx;
  series.forEach((s, i) => {
    const d = s.points.map(([x, y], j) => `${j ? "L" : "M"}${px(x).toFixed(1)},${py(y).toFixed(1)}`).join("");
    svgEl("path", { d, class: `chart-line chart-c${i + 1}` }, svg);
    const dots = opts.markers === true ||
      (opts.markers === "auto" && s.points.length <= opts.markersMax);
    if (dots) {
      for (const [x, y] of s.points) {
        svgEl("circle", { cx: px(x), cy: py(y), r: 3, class: `chart-dot chart-c${i + 1}` }, svg);
      }
    }
  });
}

function stageEndLabels(ctx) {
  const { series, opts, svg, py } = ctx;
  if (!opts.endLabels) return;
  const { width: W, margin: M } = opts;
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
}

function stageHover(ctx) {
  const { el, metric, series, opts, svg, px, py, bounds } = ctx;
  if (!opts.hover) return;
  const { width: W, height: H, margin: M } = opts;

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
    const x = bounds.xMin +
      ((ev.clientX - r.left) * (W / r.width) - M.l) / (W - M.l - M.r) * (bounds.xMax - bounds.xMin);
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
}

function stageLegend(ctx) {
  const { el, series, opts } = ctx;
  const show = opts.legend === true || (opts.legend === "auto" && series.length >= 2);
  if (!show) return;
  const legend = document.createElement("div");
  legend.className = "chart-legend";
  legend.innerHTML = series.map((s, i) =>
    `<span class="chart-legend-item"><span class="chart-swatch chart-c${i + 1}"></span>${s.label}</span>`
  ).join("");
  el.appendChild(legend);
}

function stageDataTable(ctx) {
  const { el, metric, series, opts } = ctx;
  if (!opts.dataTable) return;
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

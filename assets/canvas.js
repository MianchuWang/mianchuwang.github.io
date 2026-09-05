/* Canvas mode for article pages: draw over the page with an Apple Pencil
   (or a mouse), red ink only, stroke-level eraser, nothing persisted —
   a refresh wipes the notes.

   Design: the <canvas> never receives pointer events (pointer-events:
   none). Instead we listen on the document: pen and mouse draw, fingers
   keep scrolling as usual. Strokes live in document coordinates; the
   canvas is only viewport-sized (fixed) and is redrawn with the scroll
   offset on every scroll frame — a document-sized canvas would exceed
   Safari's canvas memory limit on long articles and silently draw nothing.
   Styles: assets/article.css (.ink-*). */

const INK = "#e0433a";
const PEN_WIDTH = 2.5;
const ERASE_RADIUS = 14;

let on = false;
let tool = "pen";               // "pen" | "eraser"
let strokes = [];               // [{ points: [[x, y], ...] }]
let current = null;
let canvas, ctx, bar;
let scrollRaf = 0;
let swallowClick = false;   // a pointerdown we took for ink must not become a click

function ensureCanvas() {
  if (canvas) return;
  canvas = document.createElement("canvas");
  canvas.className = "ink-canvas";
  document.body.appendChild(canvas);
  ctx = canvas.getContext("2d");
  const onScroll = () => { if (on && !scrollRaf) scrollRaf = requestAnimationFrame(() => { scrollRaf = 0; redraw(); }); };
  addEventListener("resize", resize);
  addEventListener("scroll", onScroll, { passive: true });
  vv?.addEventListener("resize", resize);
  vv?.addEventListener("scroll", onScroll);
  resize();

  bar = document.createElement("div");
  bar.className = "ink-bar";
  bar.innerHTML =
    `<button data-tool="pen" class="active" title="Pen" aria-label="Pen"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/></svg></button>` +
    `<button data-tool="eraser" title="Eraser (removes whole strokes)" aria-label="Eraser"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21"/><path d="M22 21H7"/><path d="m5 11 9 9"/></svg></button>` +
    `<span class="ink-sep"></span>` +
    `<button data-tool="exit" title="Exit canvas mode" aria-label="Exit canvas mode"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg></button>`;
  bar.addEventListener("click", (e) => {
    const t = e.target.closest("button")?.dataset.tool;
    if (!t) return;
    if (t === "exit") return setMode(false);
    setTool(t);
  });
  document.body.appendChild(bar);
}

function setTool(t) {
  tool = t;
  for (const b of bar.querySelectorAll("button")) b.classList.toggle("active", b.dataset.tool === t);
  canvas.dataset.tool = t;
}

/* The visible region in page coordinates. Under pinch-zoom the visual
   viewport is smaller than, and offset inside, the layout viewport that
   position:fixed attaches to — so the canvas is sized and placed from the
   visual viewport, or the ink lands up-left of the pen (Safari, zoomed). */
const vv = window.visualViewport;
const view = () => vv
  ? { left: vv.pageLeft, top: vv.pageTop, w: vv.width, h: vv.height, ox: vv.offsetLeft, oy: vv.offsetTop, k: (devicePixelRatio || 1) * vv.scale }
  : { left: scrollX, top: scrollY, w: innerWidth, h: innerHeight, ox: 0, oy: 0, k: devicePixelRatio || 1 };

function resize() {
  if (!canvas) return;
  const v = view();
  canvas.width = Math.round(v.w * v.k);
  canvas.height = Math.round(v.h * v.k);
  canvas.style.width = v.w + "px";
  canvas.style.height = v.h + "px";
  redraw();
}

// page coordinates -> canvas pixels; also keeps the canvas over the visible region
function setTransform() {
  const v = view();
  canvas.style.transform = `translate(${v.ox}px, ${v.oy}px)`;
  ctx.setTransform(v.k, 0, 0, v.k, -v.left * v.k, -v.top * v.k);
}

// ?inkdebug — on-screen readout of the viewport numbers behind the mapping
const debug = location.search.includes("inkdebug") ? Object.assign(document.createElement("pre"), { className: "ink-debug", style: "position:fixed;left:8px;bottom:8px;z-index:70;font:11px/1.3 monospace;background:#0008;color:#fff;padding:4px 6px;margin:0;border-radius:4px;pointer-events:none" }) : null;
let lastPt = null;
function showDebug() {
  if (!debug) return;
  if (!debug.isConnected) document.body.appendChild(debug);
  const v = view();
  debug.textContent = `scale ${v.k / (devicePixelRatio || 1)}  off ${v.ox | 0},${v.oy | 0}  page ${v.left | 0},${v.top | 0}\nvv ${v.w | 0}x${v.h | 0}  inner ${innerWidth}x${innerHeight}  scroll ${scrollX | 0},${scrollY | 0}\nlast pt ${lastPt ? lastPt.map((n) => n | 0).join(",") : "-"}`;
}

function redraw() {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  setTransform();
  showDebug();
  ctx.strokeStyle = INK;
  ctx.lineWidth = PEN_WIDTH;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const s of strokes) drawStroke(s);
}

function drawStroke(s) {
  const p = s.points;
  if (p.length === 1) {
    ctx.beginPath();
    ctx.arc(p[0][0], p[0][1], PEN_WIDTH / 2, 0, Math.PI * 2);
    ctx.fillStyle = INK;
    ctx.fill();
    return;
  }
  ctx.beginPath();
  ctx.moveTo(p[0][0], p[0][1]);
  for (let i = 1; i < p.length; i++) ctx.lineTo(p[i][0], p[i][1]);
  ctx.stroke();
}

/* Only pens and mice draw; touch (fingers) is left alone so the page
   still scrolls. In canvas mode the pencil is a pen and nothing else:
   its touches never reach the page (no link taps, no selection). */
const draws = (e) => on && (e.pointerType === "pen" || e.pointerType === "mouse");

// squared distance from (x, y) to the segment a-b
function segDist2([ax, ay], [bx, by], x, y) {
  const dx = bx - ax, dy = by - ay;
  const len2 = dx * dx + dy * dy;
  const t = len2 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / len2)) : 0;
  const px = ax + t * dx - x, py = ay + t * dy - y;
  return px * px + py * py;
}

// remove every stroke that passes within ERASE_RADIUS of (x, y) —
// tested against the segments, not just the sampled points, so a fast
// stroke with sparse samples is still caught between them
function eraseAt(x, y) {
  const r2 = ERASE_RADIUS * ERASE_RADIUS;
  const hit = (s) => {
    const p = s.points;
    if (p.length === 1) return (p[0][0] - x) ** 2 + (p[0][1] - y) ** 2 <= r2;
    for (let i = 1; i < p.length; i++) if (segDist2(p[i - 1], p[i], x, y) <= r2) return true;
    return false;
  };
  const before = strokes.length;
  strokes = strokes.filter((s) => !hit(s));
  if (strokes.length !== before) redraw();
}

// Safari exposes no event for the Pencil's hardware double-tap, so we
// read a quick double tap on the page instead: two pen taps within
// TAP_MS, each moving less than TAP_MOVE px, toggle the tool. The dots
// the taps would leave are removed.
const TAP_MS = 450, TAP_MOVE = 4;
let lastTap = null;   // { t, x, y, stroke }

// pageX/Y is already in page coordinates in every browser, zoomed or not;
// synthetic (untrusted) events leave it unset, so those fall back to client + scroll
const pagePt = (e) => e.isTrusted ? [e.pageX, e.pageY] : [e.clientX + scrollX, e.clientY + scrollY];

function onDown(e) {
  swallowClick = false;
  if (!draws(e) || e.target.closest(".ink-bar, .site-nav button")) return;
  e.preventDefault();
  swallowClick = true;
  setTransform();
  const [x, y] = pagePt(e);
  lastPt = [x, y]; showDebug();
  if (tool === "eraser") {
    eraseAt(x, y);
    current = { points: [[x, y]], erasing: true };
    return;
  }
  current = { points: [[x, y]] };
  strokes.push(current);
  drawStroke(current);
}

function onMove(e) {
  if (!current || !draws(e) || e.buttons === 0) return;
  e.preventDefault();
  if (tool === "eraser") {
    const pt = pagePt(e);
    current.points.push(pt);   // recorded so a drag is not mistaken for a tap
    return eraseAt(...pt);
  }
  let events = e.getCoalescedEvents?.() ?? [];
  if (!events.length) events = [e];
  for (const ev of events) current.points.push(pagePt(ev));
  // incremental draw of the new tail
  const p = current.points;
  const from = Math.max(0, p.length - events.length - 1);
  setTransform();
  ctx.strokeStyle = INK; ctx.lineWidth = PEN_WIDTH; ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(p[from][0], p[from][1]);
  for (let i = from + 1; i < p.length; i++) ctx.lineTo(p[i][0], p[i][1]);
  ctx.stroke();
}

function onUp(e) {
  // the click for this press (if any) fires before this timer runs
  setTimeout(() => { swallowClick = false; }, 0);
  const s = current;
  current = null;
  if (!s || !draws(e)) return;
  const [x0, y0] = s.points[0];
  const moved = s.points.some(([x, y]) => Math.abs(x - x0) > TAP_MOVE || Math.abs(y - y0) > TAP_MOVE);
  if (moved) { lastTap = null; return; }
  const now = Date.now();
  if (lastTap && now - lastTap.t < TAP_MS && Math.abs(lastTap.x - x0) < 24 && Math.abs(lastTap.y - y0) < 24) {
    // second tap: drop both dots, toggle the tool
    strokes = strokes.filter((k) => k !== s && k !== lastTap.stroke);
    redraw();
    setTool(tool === "pen" ? "eraser" : "pen");
    lastTap = null;
  } else {
    lastTap = { t: now, x: x0, y: y0, stroke: s.erasing ? null : s };
  }
}

// A mouse press that drew ink must not also activate the link or button
// under it (Safari already drops the click for a prevented stylus touch;
// this makes the mouse behave the same). Finger taps are untouched.
function onClick(e) {
  if (!swallowClick) return;
  swallowClick = false;
  e.preventDefault();
  e.stopPropagation();
}

// Stylus touches are swallowed (no page panning, no link taps); fingers
// pass through so the page still scrolls. The toolbar stays tappable.
function onTouch(e) {
  if (!on) return;
  if (e.target.closest?.(".ink-bar, .site-nav button")) return;
  if ([...e.touches].some((t) => t.touchType === "stylus")) e.preventDefault();
}

function setMode(next) {
  on = next;
  ensureCanvas();
  document.documentElement.classList.toggle("ink-on", on);
  canvas.dataset.tool = tool;
  document.querySelector(".canvas-toggle")?.setAttribute("aria-pressed", String(on));
  if (on) resize();
}

export function initCanvasMode() {
  const btn = document.querySelector(".canvas-toggle");
  if (!btn) return;
  if (location.hostname === "localhost") window.__ink = { get tool() { return tool; }, get strokes() { return strokes; }, get lastTap() { return lastTap; } }; // dev hook
  btn.addEventListener("click", () => setMode(!on));
  document.addEventListener("pointerdown", onDown);
  document.addEventListener("pointermove", onMove);
  document.addEventListener("pointerup", onUp);
  document.addEventListener("pointercancel", onUp);
  document.addEventListener("click", onClick, true);
  document.addEventListener("touchstart", onTouch, { passive: false });
  document.addEventListener("touchmove", onTouch, { passive: false });
}

import { renderMathIn } from "../../../assets/math.js";
import { PythonRunner } from "./runner.js";
import { loadCode, saveCode, clearCode, isSolved, markSolved } from "./store.js";

const els = {
  title: document.getElementById("title"),
  difficulty: document.getElementById("difficulty"),
  timer: document.getElementById("timer"),
  brief: document.getElementById("brief"),
  status: document.getElementById("status"),
  output: document.getElementById("output"),
  runBtn: document.getElementById("run-btn"),
  resetBtn: document.getElementById("reset-btn"),
  prev: document.getElementById("prev-link"),
  next: document.getElementById("next-link"),
  textarea: document.getElementById("editor"),
};

const state = { comp: null, sources: null, editor: null, stdout: "", hasRun: false };

async function text(url) {
  const res = await fetch(url, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.text();
}

function setStatus(label, kind = "") {
  els.status.textContent = label;
  els.status.className = "status" + (kind ? ` status-${kind}` : "");
}

/* ---------- suggested-time countdown ----------
   Starts when the component opens. Purely informational: it goes quietly
   negative when the suggested time is up — no alerts — and freezes once
   the tests all pass. */

const timer = { deadline: 0, interval: null };

function renderTimer() {
  const left = Math.round((timer.deadline - Date.now()) / 1000);
  const abs = Math.abs(left);
  const mm = Math.floor(abs / 60);
  const ss = String(abs % 60).padStart(2, "0");
  els.timer.textContent = `${left < 0 ? "−" : ""}${mm}:${ss}`;
}

function startTimer(minutes) {
  if (!minutes) return;
  timer.deadline = Date.now() + minutes * 60000;
  els.timer.classList.remove("hidden");
  renderTimer();
  timer.interval = setInterval(renderTimer, 1000);
}

function freezeTimer() {
  if (timer.interval) {
    clearInterval(timer.interval);
    timer.interval = null;
  }
}

/* ---------- editor ---------- */

function makeEditor(initial) {
  if (window.CodeMirror) {
    const cm = window.CodeMirror.fromTextArea(els.textarea, {
      mode: "python",
      lineNumbers: true,
      indentUnit: 4,
      tabSize: 4,
      indentWithTabs: false,
      lineWrapping: false,
      viewportMargin: Infinity, // render everything: the editor grows with the code
      extraKeys: {
        Tab: (editor) => editor.execCommand("indentMore"),
        "Shift-Tab": (editor) => editor.execCommand("indentLess"),
        // In leading whitespace, Backspace deletes back to the previous tab
        // stop (a whole soft tab), mirroring what Tab inserts.
        Backspace: (editor) => {
          const sels = editor.listSelections();
          const cur = sels[0].head;
          const anchor = sels[0].anchor;
          if (sels.length !== 1 || cur.line !== anchor.line || cur.ch !== anchor.ch) {
            return window.CodeMirror.Pass;
          }
          const before = editor.getLine(cur.line).slice(0, cur.ch);
          if (cur.ch === 0 || /[^ ]/.test(before)) return window.CodeMirror.Pass;
          const unit = editor.getOption("indentUnit") || 4;
          const remove = ((cur.ch - 1) % unit) + 1;
          editor.replaceRange("", { line: cur.line, ch: cur.ch - remove }, cur);
        },
      },
    });
    cm.setValue(initial);
    return {
      getValue: () => cm.getValue(),
      setValue: (v) => cm.setValue(v),
      onChange: (fn) => cm.on("change", fn),
      focus: () => cm.focus(),
    };
  }
  // CodeMirror unavailable (offline, blocked CDN) — the plain textarea still works.
  els.textarea.value = initial;
  return {
    getValue: () => els.textarea.value,
    setValue: (v) => {
      els.textarea.value = v;
    },
    onChange: (fn) => els.textarea.addEventListener("input", fn),
    focus: () => els.textarea.focus(),
  };
}

/* ---------- output ---------- */

function renderStdout(parent) {
  if (!state.stdout.trim()) return;
  const box = document.createElement("div");
  box.className = "stdout";
  box.textContent = state.stdout.replace(/\n+$/, "");
  const label = document.createElement("div");
  label.className = "stdout-label";
  label.textContent = "Printed output";
  parent.appendChild(label);
  parent.appendChild(box);
}

function renderResults(results) {
  const frag = document.createDocumentFragment();
  const passed = results.filter((r) => r.ok).length;
  const allPassed = passed === results.length;

  const summary = document.createElement("div");
  summary.className = `output-summary ${allPassed ? "pass" : "fail"}`;
  summary.textContent = allPassed
    ? `All ${results.length} tests passed`
    : `${passed} of ${results.length} tests passed`;
  frag.appendChild(summary);

  for (const r of results) {
    const row = document.createElement("div");
    row.className = `case ${r.ok ? "pass" : "fail"}`;
    const mark = document.createElement("span");
    mark.className = "case-mark";
    mark.textContent = r.ok ? "✓" : "✗";
    const body = document.createElement("div");
    body.className = "case-body";
    const name = document.createElement("div");
    name.className = "case-name";
    name.textContent = r.name;
    body.appendChild(name);
    if (!r.ok && r.detail) {
      const detail = document.createElement("div");
      detail.className = "case-detail";
      detail.textContent = r.detail;
      body.appendChild(detail);
    }
    row.appendChild(mark);
    row.appendChild(body);
    frag.appendChild(row);
  }

  if (allPassed) {
    markSolved(state.comp.id);
    freezeTimer();
    const note = document.createElement("div");
    note.className = "solved-note";
    const next = state.comp.next;
    note.innerHTML = next
      ? `Solved. Next up: <a href="lab.html?c=${encodeURIComponent(next.id)}">${next.title}</a>.`
      : `Solved — that is the last component. <a href="./">Back to the list</a>.`;
    frag.appendChild(note);
  }

  renderStdout(frag);
  els.output.replaceChildren(frag);
}

function renderError(message) {
  const frag = document.createDocumentFragment();
  const summary = document.createElement("div");
  summary.className = "output-summary fail";
  summary.textContent = "The code did not run";
  const box = document.createElement("div");
  box.className = "traceback";
  box.textContent = message;
  frag.appendChild(summary);
  frag.appendChild(box);
  renderStdout(frag);
  els.output.replaceChildren(frag);
}

/* ---------- running ---------- */

const runner = new PythonRunner({
  onStatus: (label) => setStatus(label, "busy"),
  onStdout: (chunk) => {
    state.stdout += chunk.endsWith("\n") ? chunk : chunk + "\n";
  },
});

async function run() {
  // The disabled check also guards the keyboard shortcut: during the Pyodide
  // boot the button is disabled, and a second queued run would error out.
  if (runner.isBusy() || els.runBtn.disabled) return;
  els.runBtn.disabled = true;
  setStatus("Running…", "busy");
  state.stdout = "";
  els.output.innerHTML = `<p class="output-empty">Running…</p>`;

  const sources = { ...state.sources, code: state.editor.getValue() };
  try {
    const results = await runner.run(sources);
    renderResults(results);
    setStatus("Ready", "ready");
  } catch (err) {
    renderError(String((err && err.message) || err));
    setStatus(runner.ready ? "Ready" : "Python stopped — press Run to restart", runner.ready ? "ready" : "error");
  } finally {
    state.hasRun = true; // any completed run attempt unlocks the reference solution
    els.runBtn.disabled = false;
  }
}

/* ---------- page ---------- */

function wireNav(comp) {
  for (const [el, target, label] of [
    [els.prev, comp.prev, "← "],
    [els.next, comp.next, ""],
  ]) {
    if (!target) continue;
    el.classList.remove("disabled");
    el.href = `lab.html?c=${encodeURIComponent(target.id)}`;
    el.textContent = el === els.next ? `${target.title} →` : `${label}${target.title}`;
    el.title = target.title;
  }
}

function addSolutionToggle(id) {
  const details = document.createElement("details");
  details.innerHTML = "<summary>Reference solution</summary>";
  const body = document.createElement("pre");
  body.className = "solution-block";
  body.textContent = "Loading…";
  details.appendChild(body);
  details.addEventListener("toggle", async () => {
    if (!details.open) return;
    if (!state.hasRun) {
      body.textContent =
        "Locked — run the tests on your own attempt at least once, then reopen this.";
      return;
    }
    if (details.dataset.loaded) return;
    details.dataset.loaded = "1";
    try {
      body.textContent = await text(`data/${id}/solution.py`);
    } catch {
      body.textContent = "Could not load the solution.";
    }
  });
  els.brief.appendChild(details);
}

async function main() {
  const id = new URLSearchParams(location.search).get("c");
  let components = [];
  try {
    components = (await (await fetch("data/components.json", { cache: "no-cache" })).json())
      .components;
  } catch {
    els.brief.textContent = "Could not load the component list.";
    setStatus("Failed to load", "error");
    return;
  }

  const index = components.findIndex((c) => c.id === id);
  if (index === -1) {
    els.title.textContent = "Unknown component";
    els.brief.innerHTML = `<p>No component named <code>${id ?? ""}</code>. <a href="./">Pick one from the list</a>.</p>`;
    setStatus("Nothing to run", "error");
    return;
  }

  const comp = {
    ...components[index],
    prev: components[index - 1] || null,
    next: components[index + 1] || null,
  };
  state.comp = comp;

  document.title = `${comp.title} — Build a Transformer in 150 Minutes`;
  els.title.textContent = comp.title;
  if (comp.difficulty !== "easy") {
    els.difficulty.textContent = comp.difficulty;
    els.difficulty.className = `badge badge-${comp.difficulty}`;
  }
  wireNav(comp);
  // No countdown when revisiting a solved component — that is reference mode.
  if (!isSolved(comp.id)) startTimer(comp.minutes);

  const [brief, starter, tests, harness, runnerSrc, preamble] = await Promise.all([
    text(`data/${comp.id}/brief.html`),
    text(`data/${comp.id}/starter.py`),
    text(`data/${comp.id}/tests.py`),
    text("data/harness.py"),
    text("data/runner.py"),
    comp.preamble ? text(`data/${comp.id}/preamble.py`) : Promise.resolve(""),
  ]);

  els.brief.innerHTML = brief;
  renderMathIn(els.brief);
  addSolutionToggle(comp.id);
  state.sources = { harness, runner: runnerSrc, tests, preamble };

  state.hasRun = isSolved(comp.id); // already-solved components stay unlocked
  const saved = loadCode(comp.id);
  state.editor = makeEditor(saved ?? starter);
  state.editor.onChange(() => saveCode(comp.id, state.editor.getValue()));

  if (isSolved(comp.id)) {
    els.output.innerHTML = `<p class="output-empty">You have solved this one before — run again to check your current code.</p>`;
  }

  els.runBtn.addEventListener("click", run);
  els.resetBtn.addEventListener("click", () => {
    if (!confirm("Replace your code with the starter?")) return;
    clearCode(comp.id);
    state.editor.setValue(starter);
    state.editor.focus();
  });
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      run();
    }
  });

  try {
    await runner.boot();
    setStatus("Ready", "ready");
    els.runBtn.disabled = false;
  } catch (err) {
    setStatus("Python failed to load", "error");
    els.output.innerHTML = `<p class="output-empty">Python could not start: ${String(
      (err && err.message) || err
    )}. Pyodide is fetched from a CDN, so this page needs a network connection the first time.</p>`;
  }
}

main();

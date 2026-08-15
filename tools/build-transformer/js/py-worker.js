/* Pyodide lives in a worker so a runaway loop in the editor can be terminated
   without taking the page with it. Classic worker: importScripts, no modules. */

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/";

let pyodide = null;

function status(text) {
  self.postMessage({ type: "status", text });
}

async function boot() {
  status("Loading Python…");
  importScripts(PYODIDE_URL + "pyodide.js");
  pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

  status("Loading numpy…");
  await pyodide.loadPackage(["numpy", "micropip"]);

  status("Loading einops + jaxtyping…");
  const micropip = pyodide.pyimport("micropip");
  await micropip.install(["einops", "jaxtyping", "beartype"]);

  pyodide.setStdout({ batched: (text) => self.postMessage({ type: "stdout", text }) });
  pyodide.setStderr({ batched: (text) => self.postMessage({ type: "stdout", text }) });

  self.postMessage({ type: "ready" });
}

/* Drop the frames belonging to our own scaffolding so the traceback shows the
   user's code and nothing else. */
function cleanTraceback(message) {
  const lines = String(message).split("\n");
  const kept = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const frame = line.match(/^\s*File "([^"]+)", line/);
    if (frame) {
      const file = frame[1];
      if (file === "harness.py" || file === "runner.py" || file.startsWith("<")) {
        // Skip the frame and its source line, if any.
        if (lines[i + 1] && /^\s{4,}\S/.test(lines[i + 1]) && !/^\s*File "/.test(lines[i + 1])) i++;
        continue;
      }
    }
    kept.push(line);
  }
  return kept.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

async function run(id, sources) {
  const ns = pyodide.runPython("dict()");
  try {
    await pyodide.runPythonAsync(sources.harness, { globals: ns });
    const runSource = ns.get("run_source");
    try {
      if (sources.preamble) runSource(sources.preamble, "preamble.py");
      runSource(sources.code, "your_code.py");
      runSource(sources.tests, "tests.py");
      runSource(sources.runner, "runner.py");
      self.postMessage({ type: "result", id, results: JSON.parse(ns.get("RESULT")) });
    } finally {
      runSource.destroy();
    }
  } catch (err) {
    self.postMessage({ type: "run-error", id, error: cleanTraceback(err.message || err) });
  } finally {
    ns.destroy();
  }
}

self.onmessage = async (event) => {
  const msg = event.data;
  if (msg.type === "boot") {
    try {
      await boot();
    } catch (err) {
      self.postMessage({ type: "boot-error", error: String((err && err.message) || err) });
    }
    return;
  }
  if (msg.type === "run") {
    if (!pyodide) {
      self.postMessage({ type: "run-error", id: msg.id, error: "Python is not ready yet." });
      return;
    }
    await run(msg.id, msg.sources);
  }
};

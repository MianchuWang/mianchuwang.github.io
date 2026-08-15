// Main-thread side of the Python worker: boots it, runs one job at a time,
// and kills it if a run overruns (an infinite loop in the editor, usually).

const RUN_TIMEOUT_MS = 30000;

export class PythonRunner {
  constructor({ onStatus, onStdout }) {
    this.onStatus = onStatus || (() => {});
    this.onStdout = onStdout || (() => {});
    this.worker = null;
    this.ready = null;
    this.job = null;
    this.nextId = 1;
  }

  boot() {
    if (this.ready) return this.ready;

    this.worker = new Worker("js/py-worker.js?v=bt6");
    this.worker.onmessage = (event) => this._onMessage(event.data);
    this.worker.onerror = (event) => {
      const message = event.message || "the Python worker failed to start";
      if (this._bootReject) this._bootReject(new Error(message));
      this._failJob(message);
    };

    this.ready = new Promise((resolve, reject) => {
      this._bootResolve = resolve;
      this._bootReject = reject;
    });
    this.worker.postMessage({ type: "boot" });
    return this.ready;
  }

  /** Tear the worker down; the next run() boots a fresh one. */
  reset() {
    if (this.worker) this.worker.terminate();
    this.worker = null;
    this.ready = null;
    this.job = null;
  }

  isBusy() {
    return this.job !== null;
  }

  async run(sources) {
    await this.boot();
    if (this.job) throw new Error("a run is already in flight");

    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.reset();
        reject(
          new Error(
            `Timed out after ${RUN_TIMEOUT_MS / 1000}s — the run was stopped. ` +
              "An endless loop, or an array far larger than the tests use?"
          )
        );
      }, RUN_TIMEOUT_MS);
      this.job = { id, resolve, reject, timer };
      this.worker.postMessage({ type: "run", id, sources });
    });
  }

  _settle(fn, value) {
    const job = this.job;
    if (!job) return;
    clearTimeout(job.timer);
    this.job = null;
    job[fn](value);
  }

  _failJob(message) {
    this._settle("reject", new Error(message));
  }

  _onMessage(msg) {
    switch (msg.type) {
      case "status":
        this.onStatus(msg.text);
        break;
      case "stdout":
        this.onStdout(msg.text);
        break;
      case "ready":
        if (this._bootResolve) this._bootResolve();
        break;
      case "boot-error":
        if (this._bootReject) this._bootReject(new Error(msg.error));
        this._failJob(msg.error);
        break;
      case "result":
        if (this.job && this.job.id === msg.id) this._settle("resolve", msg.results);
        break;
      case "run-error":
        if (this.job && this.job.id === msg.id) this._failJob(msg.error);
        break;
    }
  }
}

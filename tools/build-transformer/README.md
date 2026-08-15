# Build a Transformer in 150 Minutes

An ML-coding practice tool: pick a transformer component, implement it in numpy and
einops in the browser, and run its test suite. The 150 minutes in the name is the sum
of the per-component `minutes` budgets in `data/components.json` — when adding or
removing components, keep the title (here, `index.html`, `lab.html`, `js/lab.js`,
and `profile/tools.json`) in sync with the new total. Python runs client-side through
[Pyodide](https://pyodide.org/) — no server, no accounts, no uploads. Code and progress
are kept in `localStorage`.

Shapes are part of the contract: every signature is annotated with
[jaxtyping](https://docs.kidger.site/jaxtyping/) (`Float[np.ndarray, "*batch d_in"]`)
and enforced at runtime by `@jaxtyped(typechecker=beartype)` — a wrong input or output
shape fails at the function boundary with a named-axis error message, before the value
tests even run. Deleting the decorator or the annotations is allowed (the value tests
still judge correctness); keeping them is the point of the exercise.

## Run

Serve the site root and open `/tools/build-transformer/`:

```bash
python3 -m http.server 8644
```

The page must be served over HTTP (it fetches its data files); opening `lab.html`
straight from disk will not work. The first load pulls Pyodide, numpy, einops,
jaxtyping and beartype from a CDN, so it needs a network connection — after that the
browser caches them.

## Layout

```
index.html          component list, progress
lab.html            problem + editor + test output
js/lab.js           page wiring
js/runner.js        main-thread side of the Python worker
js/py-worker.js     Pyodide in a web worker (so a runaway loop can be killed)
js/store.js         localStorage: saved code, solved marks
data/components.json  ordered list: id, group, title, tagline, difficulty, preamble?
data/harness.py     test helpers injected before everything else
data/runner.py      runs the registered cases, writes RESULT as JSON
data/<id>/          one folder per component
validate.py         dev check: every solution must pass its own tests
audit.py            dev check: known-wrong implementations must FAIL the tests
```

Each run executes, in one fresh namespace: `harness.py` → the component's `preamble.py`
(if any) → the editor's code (as `your_code.py`) → `tests.py` → `runner.py`. Tracebacks
therefore point at `your_code.py` with real line numbers.

## Adding a component

1. Create `data/<id>/` with four files (plus an optional fifth):

   - `brief.html` — the problem statement. Math is LaTeX, rendered by the site-wide
     KaTeX helper (`assets/math.js`): `$...$` inline, `$$...$$` display. Conventions:
     `<p class="formula">$$…$$</p>` for the main formula box, `<pre class="brief-code">`
     for pseudo-code (never rendered as math), and a
     `<details class="hint"><summary>Hint</summary>…</details>` block at the end.
   - `starter.py` — imports plus the jaxtyping-annotated signature (decorated with
     `@jaxtyped(typechecker=beartype)`) and a one-line docstring, body
     `raise NotImplementedError`. Shapes live in the annotations, not the docstring.
   - `solution.py` — the reference implementation.
   - `tests.py` — cases registered with the `@case("name")` decorator.
   - `preamble.py` — optional helpers defined before the user's code (e.g. `softmax` for
     the attention exercises), so an exercise can build on earlier ones.

2. Append an entry to `data/components.json` (`id`, `group`, `title`, `tagline`,
   `difficulty`: `easy` | `medium` | `hard`, and `"preamble": true` if you added one).
   Order in this file is the order on the page and the prev/next links.

3. Check it:

   ```bash
   python3 tools/build-transformer/validate.py <id>
   ```

   This runs the reference solution against the tests, and also asserts that the
   *starter* fails — a test suite that passes an unimplemented stub is testing nothing.

4. Add two or three plausible-but-wrong implementations of it to `WRONG` in
   `audit.py` (the classic mistakes for that component) and check they all fail:

   ```bash
   python3 tools/build-transformer/audit.py <id>
   ```

   If a wrong variant passes, the tests have a hole; add a test and keep the
   variant as a regression check.

### Writing tests

Helpers available in `tests.py` (from `data/harness.py`): `np`, `rng(seed)`,
`assert_close(got, want, tol=1e-6, label=...)`, `assert_shape`, `assert_finite`,
`assert_true(cond, msg)`, `as_array(x, label)`.

- Compute expected values inline with plain numpy rather than by calling the reference
  solution, so a test failure means the answer is wrong, not merely different.
- Prefer several small cases with names that say what broke ("the mean is not
  subtracted") over one big case — the names are what the learner reads first.
- Watch floating point in hand-written expectations: use binary-exact values
  (halves, quarters, eighths) when a comparison sits exactly on a threshold.

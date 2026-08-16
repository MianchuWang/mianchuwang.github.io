"""Test helpers, executed before the user's code so tests and solutions share them.

Everything defined here lands in the run's global namespace: `np`, the
assertion helpers, and `run_source` (used by the page to execute each stage
with a real filename so tracebacks stay readable).
"""

import numpy as np

# The run's shared namespace acts as the user's module: name it so runtime
# type errors read "your_code.linear", not "None.linear".
__name__ = "your_code"

CASES = []


def case(name):
    """Register a test case. Cases run in definition order."""

    def deco(fn):
        CASES.append((name, fn))
        return fn

    return deco


def rng(seed=0):
    return np.random.default_rng(seed)


def _preview(a, n=6):
    flat = np.asarray(a, dtype=np.float64).ravel()
    head = ", ".join(f"{v:.4f}" for v in flat[:n])
    tail = ", ..." if flat.size > n else ""
    return f"[{head}{tail}]"


def as_array(x, label="output"):
    if x is None:
        raise AssertionError(f"{label} is None — did you forget a `return`?")
    a = np.asarray(x)
    if a.dtype == object or a.dtype.kind not in "fiub":
        raise AssertionError(f"{label} is not a numeric array (got {type(x).__name__}).")
    return a.astype(np.float64)


def assert_shape(got, want, label="output"):
    a = as_array(got, label)
    want = tuple(want)
    if a.shape != want:
        raise AssertionError(f"{label} has shape {a.shape}, expected {want}.")


def assert_finite(got, label="output", hint=""):
    a = as_array(got, label)
    if not np.all(np.isfinite(a)):
        raise AssertionError(f"{label} contains NaN or inf{' — ' + hint if hint else ''}.")


def assert_close(got, want, tol=1e-6, label="output"):
    a = as_array(got, label)
    b = np.asarray(want, dtype=np.float64)
    if a.shape != b.shape:
        raise AssertionError(f"{label} has shape {a.shape}, expected {b.shape}.")
    if not np.all(np.isfinite(a)):
        raise AssertionError(f"{label} contains NaN or inf.")
    if not np.allclose(a, b, rtol=tol, atol=tol):
        diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        raise AssertionError(
            f"{label} differs from the expected value: max |diff| = {diff:.3e} "
            f"(tolerance {tol:g}).\n"
            f"    got      {_preview(a)}\n"
            f"    expected {_preview(b)}"
        )


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)


def assert_defined(name):
    if name not in globals():
        raise AssertionError(f"`{name}` is not defined — keep the function name unchanged.")
    if not callable(globals()[name]):
        raise AssertionError(f"`{name}` is not a function.")


def run_source(src, filename):
    """Exec a stage in the shared namespace, keeping source around for tracebacks."""
    import linecache

    lines = src.splitlines(True)
    linecache.cache[filename] = (len(src), None, lines, filename)
    exec(compile(src, filename, "exec"), globals())

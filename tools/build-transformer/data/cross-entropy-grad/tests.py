def _loss(logits, targets):
    z = logits - logits.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(z).sum(axis=-1))
    picked = np.take_along_axis(z, np.asarray(targets)[..., None], axis=-1)[..., 0]
    return float(np.mean(log_z - picked))


@case("shape matches the logits")
def _():
    r = rng(0)
    logits, targets = r.normal(size=(3, 7)), r.integers(0, 7, size=3)
    assert_shape(cross_entropy_grad(logits, targets), (3, 7))


@case("matches the finite-difference gradient")
def _():
    r = rng(1)
    logits = r.normal(size=(2, 5))
    targets = r.integers(0, 5, size=2)
    got = as_array(cross_entropy_grad(logits, targets))

    h = 1e-5
    num = np.zeros_like(logits)
    for idx in np.ndindex(*logits.shape):
        up, down = logits.copy(), logits.copy()
        up[idx] += h
        down[idx] -= h
        num[idx] = (_loss(up, targets) - _loss(down, targets)) / (2 * h)
    assert_close(got, num, tol=1e-5)


@case("every token's gradient row sums to zero")
def _():
    r = rng(2)
    logits, targets = r.normal(size=(4, 9)) * 3, r.integers(0, 9, size=4)
    out = as_array(cross_entropy_grad(logits, targets))
    assert_close(out.sum(axis=-1), np.zeros(4), tol=1e-9)


@case("the target entry is negative, the rest are positive")
def _():
    r = rng(3)
    logits, targets = r.normal(size=(3, 6)), r.integers(0, 6, size=3)
    out = as_array(cross_entropy_grad(logits, targets))
    for row, t in enumerate(targets):
        assert_true(out[row, t] < 0, f"row {row}: the target entry must be p - 1 < 0.")
        others = np.delete(out[row], t)
        assert_true(np.all(others > 0), f"row {row}: non-target entries must be p > 0.")


@case("divided by the TOTAL token count, batch and sequence together")
def _():
    r = rng(4)
    logits = r.normal(size=(2, 3, 5))
    targets = r.integers(0, 5, size=(2, 3))
    got = as_array(cross_entropy_grad(logits, targets))

    h = 1e-5
    idx = (1, 2, 3)
    up, down = logits.copy(), logits.copy()
    up[idx] += h
    down[idx] -= h
    want = (_loss(up, targets) - _loss(down, targets)) / (2 * h)
    assert_close(got[idx], want, tol=1e-5, label="one entry vs finite differences")


@case("stable for huge logits")
def _():
    logits = np.array([[1000.0, 1002.0, 998.0]])
    out = cross_entropy_grad(logits, np.array([0]))
    assert_finite(out)
    e = np.exp(np.array([-2.0, 0.0, -4.0]))
    p = e / e.sum()
    assert_close(out[0], p - np.array([1.0, 0.0, 0.0]))


@case("a perfectly confident correct prediction has ~zero gradient")
def _():
    logits = np.zeros((1, 4))
    logits[0, 2] = 60.0
    out = as_array(cross_entropy_grad(logits, np.array([2])))
    assert_close(out, np.zeros((1, 4)), tol=1e-9)

def _ref(logits, targets):
    z = logits - logits.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(z).sum(axis=-1))
    picked = np.take_along_axis(z, np.asarray(targets)[..., None], axis=-1)[..., 0]
    return float(np.mean(log_z - picked))


@case("uniform logits give log(vocab_size)")
def _():
    loss = cross_entropy(np.zeros((3, 5)), np.array([0, 1, 2]))
    assert_close(loss, np.log(5.0))


@case("matches the reference on random logits")
def _():
    r = rng(0)
    logits = r.normal(size=(4, 7)) * 2
    targets = r.integers(0, 7, size=4)
    assert_close(cross_entropy(logits, targets), _ref(logits, targets))


@case("averages over batch and sequence axes")
def _():
    r = rng(1)
    logits = r.normal(size=(2, 3, 9))
    targets = r.integers(0, 9, size=(2, 3))
    assert_close(cross_entropy(logits, targets), _ref(logits, targets))


@case("a confident correct prediction costs almost nothing")
def _():
    logits = np.zeros((1, 4))
    logits[0, 2] = 50.0
    assert_close(cross_entropy(logits, np.array([2])), 0.0, tol=1e-9)


@case("no overflow for huge logits")
def _():
    r = rng(2)
    logits = r.normal(size=(3, 6)) + 5000.0
    targets = r.integers(0, 6, size=3)
    loss = cross_entropy(logits, targets)
    assert_finite(loss, label="loss")
    assert_close(loss, _ref(logits, targets))


@case("no underflow for a very wrong prediction")
def _():
    logits = np.array([[0.0, 800.0]])
    loss = as_array(cross_entropy(logits, np.array([0])))
    assert_finite(loss, label="loss")
    assert_close(loss, 800.0, tol=1e-6)


@case("returns a scalar, not an array")
def _():
    r = rng(3)
    out = cross_entropy(r.normal(size=(5, 4)), r.integers(0, 4, size=5))
    assert_true(
        np.ndim(out) == 0,
        f"expected a scalar loss, got something with shape {np.shape(out)} — take the mean.",
    )

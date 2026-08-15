@case("normalised rows have zero mean and unit variance")
def _():
    x = rng(0).normal(size=(4, 8)) * 5 + 2
    out = as_array(layer_norm(x, np.ones(8), np.zeros(8)))
    assert_close(out.mean(axis=-1), np.zeros(4), tol=1e-6)
    assert_close(out.var(axis=-1), np.ones(4), tol=1e-4)


@case("gain and bias are applied")
def _():
    r = rng(1)
    x, g, b = r.normal(size=(3, 6)), r.normal(size=6), r.normal(size=6)
    want = g * (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) + b
    assert_close(layer_norm(x, g, b), want)


@case("statistics are per token, not over the batch")
def _():
    r = rng(2)
    x = r.normal(size=(2, 5, 7))
    g, b = np.ones(7), np.zeros(7)
    out = as_array(layer_norm(x, g, b))
    row = as_array(layer_norm(x[1, 3], g, b))
    assert_close(out[1, 3], row)


@case("eps lands inside the square root")
def _():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    g, b = np.ones(4), np.zeros(4)
    want = (x - x.mean()) / np.sqrt(x.var() + 0.5)
    assert_close(layer_norm(x, g, b, eps=0.5), want)


@case("a constant row does not blow up")
def _():
    out = layer_norm(np.full((2, 4), 3.0), np.ones(4), np.zeros(4))
    assert_finite(out)
    assert_close(out, np.zeros((2, 4)), tol=1e-3)

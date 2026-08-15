@case("matches the definition")
def _():
    r = rng(0)
    x, g = r.normal(size=(3, 6)), r.normal(size=6)
    rms = np.sqrt((x**2).mean(axis=-1, keepdims=True) + 1e-6)
    assert_close(rms_norm(x, g), g * x / rms)


@case("the mean is not subtracted")
def _():
    out = rms_norm(np.ones((2, 4)), np.ones(4))
    assert_close(out, np.ones((2, 4)), tol=1e-5)


@case("scale invariance: doubling x changes nothing")
def _():
    x, g = rng(1).normal(size=(2, 8)), np.ones(8)
    assert_close(rms_norm(2.0 * x, g), rms_norm(x, g), tol=1e-5)


@case("works with batch and sequence axes")
def _():
    r = rng(2)
    x, g = r.normal(size=(2, 5, 7)), r.normal(size=7)
    rms = np.sqrt((x**2).mean(axis=-1, keepdims=True) + 1e-6)
    assert_shape(rms_norm(x, g), (2, 5, 7))
    assert_close(rms_norm(x, g), g * x / rms)


@case("eps is used")
def _():
    x, g = np.zeros((1, 4)), np.ones(4)
    assert_finite(rms_norm(x, g))


@case("eps sits inside the square root")
def _():
    x, g = np.array([[3.0, 4.0]]), np.ones(2)
    # mean square = 12.5; a large eps makes sqrt(ms + eps) vs sqrt(ms) + eps visible
    assert_close(rms_norm(x, g, eps=1.0), x / np.sqrt(12.5 + 1.0))

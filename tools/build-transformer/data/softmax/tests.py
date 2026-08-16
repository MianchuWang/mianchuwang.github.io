@case("1-D vector")
def _():
    x = np.array([1.0, 2.0, 3.0])
    e = np.exp(x - x.max())
    assert_close(softmax(x), e / e.sum())


@case("rows sum to one")
def _():
    x = rng(0).normal(size=(4, 7)) * 3
    out = as_array(softmax(x))
    assert_close(out.sum(axis=-1), np.ones(4))


@case("axis argument is respected")
def _():
    x = rng(1).normal(size=(3, 5))
    e = np.exp(x - x.max(axis=0, keepdims=True))
    assert_close(softmax(x, axis=0), e / e.sum(axis=0, keepdims=True))


@case("stable for large logits")
def _():
    x = np.array([1000.0, 1000.0, 1001.0])
    out = softmax(x)
    assert_finite(out, hint="subtract the per-slice max before exponentiating")
    e = np.exp(x - 1001.0)
    assert_close(out, e / e.sum())


@case("rows at very different magnitudes are each stable")
def _():
    # A global max-subtraction (instead of per-row) sends the small row to 0/0.
    x = np.array([[1000.0, 1001.0], [-1000.0, -999.0]])
    out = as_array(softmax(x))
    assert_finite(out, hint="subtract the per-slice max before exponentiating")
    row = np.array([1.0, np.e]) / (1.0 + np.e)
    assert_close(out[0], row, label="row 0")
    assert_close(out[1], row, label="row 1")


@case("stable for very negative logits")
def _():
    out = softmax(np.array([-1200.0, -1200.0]))
    assert_finite(out, hint="subtract the per-slice max before exponentiating")
    assert_close(out, np.array([0.5, 0.5]))


@case("shifting the input changes nothing")
def _():
    x = rng(2).normal(size=(2, 6))
    assert_close(softmax(x + 50.0), softmax(x))

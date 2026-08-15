@case("1-D input")
def _():
    r = rng(0)
    x, W, b = r.normal(size=4), r.normal(size=(3, 4)), r.normal(size=3)
    assert_close(linear(x, W, b), W @ x + b)


@case("batched input keeps its leading axes")
def _():
    r = rng(1)
    x, W, b = r.normal(size=(2, 5, 4)), r.normal(size=(3, 4)), r.normal(size=3)
    assert_close(linear(x, W, b), np.einsum("bni,oi->bno", x, W) + b)


@case("no bias")
def _():
    r = rng(2)
    x, W = r.normal(size=(6, 4)), r.normal(size=(3, 4))
    assert_close(linear(x, W), x @ W.T)


@case("W is (d_out, d_in), not its transpose")
def _():
    x = np.array([1.0, 0.0, 0.0])
    W = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert_close(linear(x, W), np.array([1.0, 4.0]))


@case("bias is broadcast over the batch")
def _():
    x, W = np.zeros((2, 3)), np.zeros((4, 3))
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert_close(linear(x, W, b), np.tile(b, (2, 1)))

def _silu(z):
    return z / (1.0 + np.exp(-z))


def _params(seed, d_model=8, d_ff=16):
    r = rng(seed)
    return (
        r.normal(size=(d_ff, d_model)) * 0.3,
        r.normal(size=(d_model, d_ff)) * 0.3,
        r.normal(size=(d_ff, d_model)) * 0.3,
    )


@case("matches the definition")
def _():
    W1, W2, W3 = _params(0)
    x = rng(10).normal(size=(2, 3, 8))
    want = (_silu(x @ W1.T) * (x @ W3.T)) @ W2.T
    assert_close(swiglu(x, W1, W2, W3), want)


@case("output has d_model features")
def _():
    W1, W2, W3 = _params(1, d_model=6, d_ff=13)
    x = rng(11).normal(size=(4, 6))
    assert_shape(swiglu(x, W1, W2, W3), (4, 6))


@case("zero in, zero out")
def _():
    W1, W2, W3 = _params(2)
    assert_close(swiglu(np.zeros((2, 8)), W1, W2, W3), np.zeros((2, 8)))


@case("the gate is SiLU, not ReLU")
def _():
    one = np.ones((1, 1))
    got = as_array(swiglu(np.array([[-1.0]]), one, one, one))
    assert_close(got, np.array([[_silu(-1.0) * -1.0]]))


@case("W1 gates and W3 passes through, not the other way round")
def _():
    W1, W2, W3 = _params(3)
    x = rng(13).normal(size=(3, 8))
    swapped = (_silu(x @ W3.T) * (x @ W1.T)) @ W2.T
    got = as_array(swiglu(x, W1, W2, W3))
    assert_true(
        not np.allclose(got, swapped, atol=1e-8),
        "the output matches the swapped-projection result — W1 goes through SiLU "
        "and gates, W3 passes through.",
    )
    assert_close(got, (_silu(x @ W1.T) * (x @ W3.T)) @ W2.T)

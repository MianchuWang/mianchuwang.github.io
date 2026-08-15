def _params(seed, d=12, d_ff=24, scale=0.2):
    r = rng(seed)
    return {
        "g1": r.normal(size=d) * 0.1 + 1.0,
        "Wq": r.normal(size=(d, d)) * scale,
        "Wk": r.normal(size=(d, d)) * scale,
        "Wv": r.normal(size=(d, d)) * scale,
        "Wo": r.normal(size=(d, d)) * scale,
        "g2": r.normal(size=d) * 0.1 + 1.0,
        "W1": r.normal(size=(d_ff, d)) * scale,
        "W2": r.normal(size=(d, d_ff)) * scale,
        "W3": r.normal(size=(d_ff, d)) * scale,
    }


def _ref(x, p, h):
    n = x.shape[1]
    m = causal_mask(n, n)
    a = x + multihead_attention(rms_norm(x, p["g1"]), p["Wq"], p["Wk"], p["Wv"], p["Wo"], h, m)
    return a + swiglu(rms_norm(a, p["g2"]), p["W1"], p["W2"], p["W3"])


@case("shape is unchanged")
def _():
    x, p = rng(0).normal(size=(2, 6, 12)), _params(1)
    assert_shape(transformer_block(x, p, 4), (2, 6, 12))


@case("matches the reference wiring")
def _():
    x, p = rng(2).normal(size=(2, 6, 12)), _params(3)
    assert_close(transformer_block(x, p, 4), _ref(x, p, 4))


@case("all-zero weights leave x untouched (residuals present)")
def _():
    d, d_ff = 8, 16
    p = {
        "g1": np.ones(d),
        "Wq": np.zeros((d, d)),
        "Wk": np.zeros((d, d)),
        "Wv": np.zeros((d, d)),
        "Wo": np.zeros((d, d)),
        "g2": np.ones(d),
        "W1": np.zeros((d_ff, d)),
        "W2": np.zeros((d, d_ff)),
        "W3": np.zeros((d_ff, d)),
    }
    x = rng(4).normal(size=(2, 5, d))
    assert_close(transformer_block(x, p, 2), x)


@case("attention is causal")
def _():
    x, p = rng(5).normal(size=(1, 6, 12)), _params(6)
    x2 = x.copy()
    x2[:, 3:] += 5.0
    a = as_array(transformer_block(x, p, 4))
    b = as_array(transformer_block(x2, p, 4))
    assert_close(a[:, :3], b[:, :3], label="outputs at positions 0-2")


@case("the MLP branch reads the post-attention stream")
def _():
    # Feeding the MLP from x instead of h would change the result here.
    x, p = rng(7).normal(size=(2, 4, 12)), _params(8, scale=0.6)
    assert_close(transformer_block(x, p, 2), _ref(x, p, 2))


@case("works with a single head and a longer sequence")
def _():
    x, p = rng(9).normal(size=(1, 16, 12)), _params(10)
    assert_close(transformer_block(x, p, 1), _ref(x, p, 1))

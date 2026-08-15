def _weights(seed, d_model=6, d_k=4):
    r = rng(seed)
    return tuple(r.normal(size=(d_k, d_model)) * 0.4 for _ in range(3))


def _full_causal(X, Wq, Wk, Wv):
    """Reference: full attention over the whole sequence with a causal mask."""
    Q, K, V = X @ Wq.T, X @ Wk.T, X @ Wv.T
    n = X.shape[1]
    s = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])
    s = np.where(np.tril(np.ones((n, n), dtype=bool)), s, -np.inf)
    w = np.exp(s - s.max(axis=-1, keepdims=True))
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ V


@case("cache grows by exactly one position")
def _():
    Wq, Wk, Wv = _weights(0)
    x = rng(1).normal(size=(2, 6))
    k0, v0 = rng(2).normal(size=(2, 5, 4)), rng(3).normal(size=(2, 5, 4))
    out, k1, v1 = decode_step(x, k0, v0, Wq, Wk, Wv)
    assert_shape(out, (2, 4), label="out")
    assert_shape(k1, (2, 6, 4), label="k cache")
    assert_shape(v1, (2, 6, 4), label="v cache")


@case("the first token works with an empty cache")
def _():
    Wq, Wk, Wv = _weights(4)
    x = rng(5).normal(size=(2, 6))
    empty = np.zeros((2, 0, 4))
    out, k1, v1 = decode_step(x, empty, empty, Wq, Wk, Wv)
    # One key in the cache -> softmax weight 1 -> the output IS the new value.
    assert_close(out, x @ Wv.T, label="out (must equal v of the only token)")
    assert_close(k1[:, 0], x @ Wk.T, label="cached k")


@case("the old cache comes back unchanged as the prefix")
def _():
    Wq, Wk, Wv = _weights(6)
    x = rng(7).normal(size=(2, 6))
    k0, v0 = rng(8).normal(size=(2, 3, 4)), rng(9).normal(size=(2, 3, 4))
    _out, k1, v1 = decode_step(x, k0, v0, Wq, Wk, Wv)
    assert_close(k1[:, :3], k0, label="k cache prefix")
    assert_close(v1[:, :3], v0, label="v cache prefix")


@case("the query sees the new token too, not just the old cache")
def _():
    Wq, Wk, Wv = _weights(10)
    x = rng(11).normal(size=(1, 6))
    k0, v0 = rng(12).normal(size=(1, 2, 4)), rng(13).normal(size=(1, 2, 4))
    out, k1, v1 = decode_step(x, k0, v0, Wq, Wk, Wv)
    assert_close(out, attention((x @ Wq.T)[:, None], k1, v1)[:, 0], label="out")


@case("decoding one-by-one reproduces full causal attention")
def _():
    Wq, Wk, Wv = _weights(14)
    X = rng(15).normal(size=(2, 7, 6))
    want = _full_causal(X, Wq, Wk, Wv)

    k = np.zeros((2, 0, 4))
    v = np.zeros((2, 0, 4))
    outs = []
    for i in range(7):
        out, k, v = decode_step(X[:, i], k, v, Wq, Wk, Wv)
        outs.append(out)
    assert_close(np.stack(outs, axis=1), want, label="stacked decode outputs")


@case("the input caches are not mutated")
def _():
    Wq, Wk, Wv = _weights(16)
    x = rng(17).normal(size=(1, 6))
    k0 = rng(18).normal(size=(1, 4, 4))
    v0 = rng(19).normal(size=(1, 4, 4))
    k_snap, v_snap = k0.copy(), v0.copy()
    decode_step(x, k0, v0, Wq, Wk, Wv)
    assert_close(k0, k_snap, label="the original k cache")
    assert_close(v0, v_snap, label="the original v cache")

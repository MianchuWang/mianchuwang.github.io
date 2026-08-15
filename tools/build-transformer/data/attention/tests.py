def _ref(Q, K, V, mask=None):
    s = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    w = np.exp(s - s.max(axis=-1, keepdims=True))
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ V


@case("shapes: batch and head axes pass through")
def _():
    r = rng(0)
    Q, K, V = r.normal(size=(2, 3, 4, 8)), r.normal(size=(2, 3, 5, 8)), r.normal(size=(2, 3, 5, 6))
    assert_shape(attention(Q, K, V), (2, 3, 4, 6))
    assert_close(attention(Q, K, V), _ref(Q, K, V))


@case("uniform weights when the queries are zero")
def _():
    r = rng(1)
    Q, K, V = np.zeros((3, 4)), r.normal(size=(5, 4)), r.normal(size=(5, 6))
    assert_close(attention(Q, K, V), np.tile(V.mean(axis=0), (3, 1)))


@case("scaled by sqrt(d_k)")
def _():
    # Without the 1/sqrt(d_k), a large d_k makes the weights far too peaked.
    r = rng(2)
    Q, K, V = r.normal(size=(2, 64)), r.normal(size=(3, 64)), r.normal(size=(3, 5))
    assert_close(attention(Q, K, V), _ref(Q, K, V))


@case("a causal mask blocks the future")
def _():
    r = rng(3)
    Q, K, V = r.normal(size=(4, 8)), r.normal(size=(4, 8)), r.normal(size=(4, 6))
    m = np.tril(np.ones((4, 4), dtype=bool))
    out = as_array(attention(Q, K, V, m))
    assert_close(out, _ref(Q, K, V, m))
    assert_close(out[0], V[0], label="first row of the output")


@case("masked values never leak into the output")
def _():
    r = rng(4)
    Q, K, V = r.normal(size=(3, 8)), r.normal(size=(3, 8)), r.normal(size=(3, 6))
    m = np.tril(np.ones((3, 3), dtype=bool))
    V2 = V.copy()
    V2[1:] += 100.0
    a = as_array(attention(Q, K, V, m))
    b = as_array(attention(Q, K, V2, m))
    assert_close(
        a[0],
        b[0],
        label="the first row, which may only attend to key 0, changed when the masked "
        "values of V changed, so",
    )


@case("broadcast mask (1, n_q, n_k) over a batch")
def _():
    r = rng(5)
    Q, K, V = r.normal(size=(2, 4, 8)), r.normal(size=(2, 4, 8)), r.normal(size=(2, 4, 6))
    m = np.tril(np.ones((1, 4, 4), dtype=bool))
    assert_close(attention(Q, K, V, m), _ref(Q, K, V, m))

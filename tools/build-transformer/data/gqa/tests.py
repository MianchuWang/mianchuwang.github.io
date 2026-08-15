def _split(p, h):
    b, n, _ = p.shape
    return p.reshape(b, n, h, -1).transpose(0, 2, 1, 3)


def _ref(x, Wq, Wk, Wv, Wo, h, kvh, mask=None):
    b, n, d = x.shape
    dh = d // h
    q = _split(x @ Wq.T, h)
    k = _split(x @ Wk.T, kvh)
    v = _split(x @ Wv.T, kvh)
    k = np.repeat(k, h // kvh, axis=1)
    v = np.repeat(v, h // kvh, axis=1)
    s = q @ np.swapaxes(k, -1, -2) / np.sqrt(dh)
    if mask is not None:
        s = np.where(mask[None, None], s, -np.inf)
    w = np.exp(s - s.max(axis=-1, keepdims=True))
    w = w / w.sum(axis=-1, keepdims=True)
    o = (w @ v).transpose(0, 2, 1, 3).reshape(b, n, h * dh)
    return o @ Wo.T


def _setup(seed, h=4, kvh=2, b=2, n=5, d=16):
    r = rng(seed)
    dh = d // h
    x = r.normal(size=(b, n, d))
    Wq = r.normal(size=(h * dh, d)) * 0.2
    Wk = r.normal(size=(kvh * dh, d)) * 0.2
    Wv = r.normal(size=(kvh * dh, d)) * 0.2
    Wo = r.normal(size=(d, h * dh)) * 0.2
    return x, Wq, Wk, Wv, Wo


@case("output keeps the input shape")
def _():
    x, Wq, Wk, Wv, Wo = _setup(0)
    assert_shape(grouped_query_attention(x, Wq, Wk, Wv, Wo, 4, 2), (2, 5, 16))


@case("4 query heads over 2 KV heads")
def _():
    x, Wq, Wk, Wv, Wo = _setup(1)
    assert_close(
        grouped_query_attention(x, Wq, Wk, Wv, Wo, 4, 2), _ref(x, Wq, Wk, Wv, Wo, 4, 2)
    )


@case("num_kv_heads == num_heads reduces to MHA")
def _():
    x, Wq, Wk, Wv, Wo = _setup(2, h=4, kvh=4)
    assert_close(
        grouped_query_attention(x, Wq, Wk, Wv, Wo, 4, 4), _ref(x, Wq, Wk, Wv, Wo, 4, 4)
    )


@case("num_kv_heads == 1 is multi-query attention")
def _():
    x, Wq, Wk, Wv, Wo = _setup(3, h=4, kvh=1)
    assert_close(
        grouped_query_attention(x, Wq, Wk, Wv, Wo, 4, 1), _ref(x, Wq, Wk, Wv, Wo, 4, 1)
    )


@case("query head i pairs with KV head i // group")
def _():
    # Zero out the second KV head; the first group of query heads must not notice.
    x, Wq, Wk, Wv, Wo = _setup(4, h=4, kvh=2, d=16)
    Wo = np.eye(16)
    Wv2 = Wv.copy()
    Wv2[4:] = 0.0  # KV head 1's value slice
    a = as_array(grouped_query_attention(x, Wq, Wk, Wv, Wo, 4, 2))
    b = as_array(grouped_query_attention(x, Wq, Wk, Wv2, Wo, 4, 2))
    assert_close(a[..., :8], b[..., :8], label="query heads 0-1 (they share KV head 0)")
    assert_true(
        not np.allclose(a[..., 8:], b[..., 8:]),
        "query heads 2-3 should depend on KV head 1 — check the (kh g) order.",
    )


@case("mask is applied")
def _():
    x, Wq, Wk, Wv, Wo = _setup(5)
    m = np.tril(np.ones((5, 5), dtype=bool))
    assert_close(
        grouped_query_attention(x, Wq, Wk, Wv, Wo, 4, 2, m),
        _ref(x, Wq, Wk, Wv, Wo, 4, 2, m),
    )

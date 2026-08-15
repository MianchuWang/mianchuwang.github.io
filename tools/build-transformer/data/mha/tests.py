def _ref(x, Wq, Wk, Wv, Wo, h, mask=None):
    b, n, d = x.shape
    dh = d // h
    heads = []
    for w in (Wq, Wk, Wv):
        p = x @ w.T
        heads.append(p.reshape(b, n, h, dh).transpose(0, 2, 1, 3))
    q, k, v = heads
    s = q @ np.swapaxes(k, -1, -2) / np.sqrt(dh)
    if mask is not None:
        s = np.where(mask[None, None], s, -np.inf)
    w = np.exp(s - s.max(axis=-1, keepdims=True))
    w = w / w.sum(axis=-1, keepdims=True)
    o = (w @ v).transpose(0, 2, 1, 3).reshape(b, n, d)
    return o @ Wo.T


def _setup(seed, b=2, n=5, d=12):
    r = rng(seed)
    x = r.normal(size=(b, n, d))
    ws = [r.normal(size=(d, d)) * 0.2 for _ in range(4)]
    return (x, *ws)


@case("output keeps the input shape")
def _():
    x, Wq, Wk, Wv, Wo = _setup(0)
    assert_shape(multihead_attention(x, Wq, Wk, Wv, Wo, 4), (2, 5, 12))


@case("one head equals plain attention on the projections")
def _():
    x, Wq, Wk, Wv, Wo = _setup(1, b=1, n=4, d=8)
    assert_close(multihead_attention(x, Wq, Wk, Wv, Wo, 1), _ref(x, Wq, Wk, Wv, Wo, 1))


@case("four heads split the feature axis contiguously")
def _():
    x, Wq, Wk, Wv, Wo = _setup(2)
    assert_close(multihead_attention(x, Wq, Wk, Wv, Wo, 4), _ref(x, Wq, Wk, Wv, Wo, 4))


@case("heads really are independent")
def _():
    # Zeroing the second half of Wv must leave the first head's output untouched.
    x, Wq, Wk, Wv, Wo = _setup(3, d=8)
    Wo = np.eye(8)
    Wv2 = Wv.copy()
    Wv2[4:] = 0.0
    a = as_array(multihead_attention(x, Wq, Wk, Wv, Wo, 2))
    b = as_array(multihead_attention(x, Wq, Wk, Wv2, Wo, 2))
    assert_close(a[..., :4], b[..., :4], label="first head's slice of the output")


@case("the mask is applied per head")
def _():
    x, Wq, Wk, Wv, Wo = _setup(4)
    m = np.tril(np.ones((5, 5), dtype=bool))
    assert_close(
        multihead_attention(x, Wq, Wk, Wv, Wo, 4, m), _ref(x, Wq, Wk, Wv, Wo, 4, m)
    )


@case("with a causal mask the first token ignores later tokens")
def _():
    x, Wq, Wk, Wv, Wo = _setup(5)
    m = np.tril(np.ones((5, 5), dtype=bool))
    x2 = x.copy()
    x2[:, 1:] += 10.0
    a = as_array(multihead_attention(x, Wq, Wk, Wv, Wo, 4, m))
    b = as_array(multihead_attention(x2, Wq, Wk, Wv, Wo, 4, m))
    assert_close(a[:, 0], b[:, 0], label="output at position 0")

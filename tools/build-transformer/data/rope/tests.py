def _ref(x, positions, theta=10000.0):
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    out = np.empty_like(x)
    for t, p in enumerate(np.asarray(positions, dtype=np.float64)):
        for i in range(d // 2):
            ang = p * theta ** (-2.0 * i / d)
            c, s = np.cos(ang), np.sin(ang)
            a, b = x[..., t, 2 * i], x[..., t, 2 * i + 1]
            out[..., t, 2 * i] = a * c - b * s
            out[..., t, 2 * i + 1] = a * s + b * c
    return out


@case("shape is preserved")
def _():
    x = rng(0).normal(size=(2, 3, 6, 8))
    assert_shape(rope(x, np.arange(6)), (2, 3, 6, 8))


@case("position 0 is the identity")
def _():
    x = rng(1).normal(size=(1, 4))
    assert_close(rope(x, np.array([0])), x)


@case("matches the rotation, pair by pair")
def _():
    x = rng(2).normal(size=(2, 5, 8))
    pos = np.arange(5)
    assert_close(rope(x, pos), _ref(x, pos))


@case("positions may start at a KV-cache offset")
def _():
    x = rng(3).normal(size=(3, 8))
    pos = np.arange(3) + 17
    assert_close(rope(x, pos), _ref(x, pos))


@case("rotation preserves the norm")
def _():
    x = rng(4).normal(size=(6, 16))
    out = as_array(rope(x, np.arange(6)))
    assert_close(np.linalg.norm(out, axis=-1), np.linalg.norm(x, axis=-1))


@case("the dot product depends only on the relative position")
def _():
    r = rng(5)
    q, k = r.normal(size=(1, 8)), r.normal(size=(1, 8))
    near = float(np.sum(as_array(rope(q, np.array([3]))) * as_array(rope(k, np.array([5])))))
    far = float(np.sum(as_array(rope(q, np.array([23]))) * as_array(rope(k, np.array([25])))))
    assert_true(
        abs(near - far) < 1e-8,
        f"q·k at distance 2 should not depend on where the pair sits: {near:.6f} vs {far:.6f}.",
    )


@case("a custom base changes the angles")
def _():
    x = rng(6).normal(size=(4, 8))
    pos = np.arange(4)
    assert_close(rope(x, pos, theta=100.0), _ref(x, pos, theta=100.0))

def _ref(param, grad, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad**2
    mh = m / (1 - b1**t)
    vh = v / (1 - b2**t)
    param = param - lr * mh / (np.sqrt(vh) + eps) - lr * wd * param
    return param, m, v


@case("returns (param, m, v)")
def _():
    out = adamw_step(np.ones(3), np.ones(3), np.zeros(3), np.zeros(3), 1)
    assert_true(
        isinstance(out, tuple) and len(out) == 3,
        f"expected a 3-tuple (param, m, v), got {type(out).__name__}.",
    )


@case("first step from zero state")
def _():
    r = rng(0)
    p, g = r.normal(size=5), r.normal(size=5)
    m, v = np.zeros(5), np.zeros(5)
    got = adamw_step(p, g, m, v, 1, lr=0.1)
    want = _ref(p, g, m, v, 1, lr=0.1)
    assert_close(got[0], want[0], label="param")
    assert_close(got[1], want[1], label="m")
    assert_close(got[2], want[2], label="v")


@case("bias correction: step 1 moves by roughly lr in the sign of the gradient")
def _():
    p = np.zeros(4)
    g = np.array([1.0, -2.0, 0.5, -0.25])
    new_p = as_array(adamw_step(p, g, np.zeros(4), np.zeros(4), 1, lr=0.1, weight_decay=0.0)[0])
    assert_close(new_p, -0.1 * np.sign(g), tol=1e-6, label="param")


@case("three steps in a row")
def _():
    r = rng(1)
    p = r.normal(size=(2, 3))
    m, v = np.zeros((2, 3)), np.zeros((2, 3))
    rp, rm, rv = p.copy(), m.copy(), v.copy()
    for t in range(1, 4):
        g = r.normal(size=(2, 3))
        p, m, v = adamw_step(p, g, m, v, t, lr=0.05, weight_decay=0.1)
        rp, rm, rv = _ref(rp, g, rm, rv, t, lr=0.05, wd=0.1)
        assert_close(p, rp, label=f"param after step {t}")
        assert_close(m, rm, label=f"m after step {t}")
        assert_close(v, rv, label=f"v after step {t}")


@case("weight decay is decoupled from the gradient")
def _():
    p = np.array([2.0, -4.0])
    z = np.zeros(2)
    new_p = as_array(adamw_step(p, z, z, z, 1, lr=0.1, weight_decay=0.5)[0])
    assert_close(new_p, p - 0.1 * 0.5 * p, label="param")


@case("the stored moments are not bias-corrected")
def _():
    g = np.array([1.0, 2.0])
    z = np.zeros(2)
    _, m, v = adamw_step(np.zeros(2), g, z, z, 1)
    assert_close(m, 0.1 * g, label="m")
    assert_close(v, 0.001 * g**2, label="v")


@case("inputs are not modified in place")
def _():
    p = np.ones(3)
    m = np.full(3, 0.5)
    v = np.full(3, 0.25)
    adamw_step(p, np.ones(3), m, v, 2)
    assert_close(p, np.ones(3), label="the original param")
    assert_close(m, np.full(3, 0.5), label="the original m")
    assert_close(v, np.full(3, 0.25), label="the original v")

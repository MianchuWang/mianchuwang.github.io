def _global_norm(gs):
    return float(np.sqrt(sum(float((np.asarray(g) ** 2).sum()) for g in gs)))


@case("shapes and ordering survive")
def _():
    r = rng(0)
    gs = [r.normal(size=(2, 3)), r.normal(size=5), r.normal(size=(4, 1, 2))]
    out = clip_grad_norm(gs, 1.0)
    assert_true(len(out) == 3, f"expected 3 tensors back, got {len(out)}.")
    for got, want in zip(out, gs):
        assert_shape(got, want.shape)


@case("a small gradient is left alone")
def _():
    gs = [np.array([0.3, 0.4]), np.array([[0.1]])]
    out = clip_grad_norm(gs, 10.0)
    assert_close(out[0], gs[0], label="tensor 0")
    assert_close(out[1], gs[1], label="tensor 1")


@case("a large gradient is scaled down to max_norm")
def _():
    gs = [np.array([3.0, 4.0]), np.array([[12.0]])]  # norm 13
    out = clip_grad_norm(gs, 1.0)
    assert_close(_global_norm(out), 1.0, tol=1e-4, label="clipped global norm")


@case("the direction is preserved")
def _():
    gs = [np.array([3.0, 4.0]), np.array([[12.0]])]
    out = clip_grad_norm(gs, 2.0)
    scale = 2.0 / 13.0
    assert_close(out[0], gs[0] * scale, tol=1e-5, label="tensor 0")
    assert_close(out[1], gs[1] * scale, tol=1e-5, label="tensor 1")


@case("the norm is global, not per tensor")
def _():
    # Each tensor alone has norm 1; together the norm is sqrt(2) > 1.2.
    gs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    out = clip_grad_norm(gs, 1.2)
    assert_close(_global_norm(out), 1.2, tol=1e-4, label="clipped global norm")


@case("exactly at the threshold nothing changes")
def _():
    gs = [np.array([3.0, 4.0])]
    out = clip_grad_norm(gs, 5.0)
    assert_close(out[0], gs[0], tol=1e-6, label="tensor 0")


@case("the input list is not modified in place")
def _():
    gs = [np.array([30.0, 40.0])]
    clip_grad_norm(gs, 1.0)
    assert_close(gs[0], np.array([30.0, 40.0]), label="the original gradient")

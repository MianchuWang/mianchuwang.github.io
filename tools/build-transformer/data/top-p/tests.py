@case("keeps the smallest set reaching p")
def _():
    probs = np.array([0.5, 0.25, 0.125, 0.125])
    assert_close(top_p_filter(probs, 0.75), np.array([2 / 3, 1 / 3, 0.0, 0.0]))


@case("original positions are preserved")
def _():
    probs = np.array([0.125, 0.5, 0.25, 0.125])
    assert_close(top_p_filter(probs, 0.75), np.array([0.0, 2 / 3, 1 / 3, 0.0]))


@case("the top token is always kept")
def _():
    probs = np.array([0.5, 0.25, 0.125, 0.125])
    assert_close(top_p_filter(probs, 0.25), np.array([1.0, 0.0, 0.0, 0.0]))


@case("p = 1 keeps everything")
def _():
    probs = np.array([0.5, 0.25, 0.125, 0.125])
    assert_close(top_p_filter(probs, 1.0), probs)


@case("rows sum to one")
def _():
    r = rng(0)
    x = r.random(size=(3, 12))
    probs = x / x.sum(axis=-1, keepdims=True)
    out = as_array(top_p_filter(probs, 0.9))
    assert_close(out.sum(axis=-1), np.ones(3))


@case("batched rows are filtered independently")
def _():
    probs = np.array(
        [
            [0.5, 0.25, 0.125, 0.125],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    out = as_array(top_p_filter(probs, 0.6))
    assert_close(out[0], np.array([2 / 3, 1 / 3, 0.0, 0.0]), label="row 0")
    assert_close(out[1], np.array([1 / 3, 1 / 3, 1 / 3, 0.0]), label="row 1")


@case("dropped tokens are exactly zero")
def _():
    r = rng(1)
    x = r.random(size=(2, 20))
    probs = x / x.sum(axis=-1, keepdims=True)
    out = as_array(top_p_filter(probs, 0.5))
    assert_true(np.all(out >= 0.0), "probabilities cannot be negative.")
    assert_true(np.any(out == 0.0), "with p = 0.5 some tokens must be dropped outright.")

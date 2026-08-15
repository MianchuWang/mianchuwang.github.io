@case("square case is the lower triangle")
def _():
    m = as_array(causal_mask(4, 4))
    assert_close(m, np.tril(np.ones((4, 4))))


@case("returns a boolean array")
def _():
    m = causal_mask(3, 3)
    assert_true(
        isinstance(m, np.ndarray) and m.dtype == np.bool_,
        f"expected a boolean numpy array, got {type(m).__name__} "
        f"with dtype {getattr(m, 'dtype', '?')}.",
    )


@case("one new query against a full cache sees everything")
def _():
    assert_close(as_array(causal_mask(1, 5)), np.ones((1, 5)))


@case("offset diagonal when n_q < n_k")
def _():
    want = np.array(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    assert_close(as_array(causal_mask(2, 5)), want)


@case("a query never sees its own future")
def _():
    m = as_array(causal_mask(6, 6))
    assert_true(np.all(np.triu(m, 1) == 0), "positions above the diagonal must be False.")
    assert_true(np.all(np.diag(m) == 1), "every query must see itself.")

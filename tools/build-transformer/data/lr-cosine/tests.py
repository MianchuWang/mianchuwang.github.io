MAX, MIN, TW, TC = 1e-3, 1e-4, 100, 1000


def _lr(t):
    return lr_cosine_schedule(t, MAX, MIN, TW, TC)


@case("starts at zero")
def _():
    assert_close(_lr(0), 0.0)


@case("warmup is linear")
def _():
    assert_close(_lr(25), 0.25 * MAX)
    assert_close(_lr(50), 0.50 * MAX)


@case("peaks exactly at lr_max")
def _():
    assert_close(_lr(TW), MAX)


@case("halfway through the cosine phase sits at the midpoint")
def _():
    assert_close(_lr((TW + TC) // 2), MIN + 0.5 * (MAX - MIN))


@case("the decay is a cosine, not a straight line")
def _():
    # A quarter of the way in, cosine sits at (1 + cos(pi/4)) / 2 of the range.
    t = TW + (TC - TW) // 4
    assert_close(_lr(t), MIN + 0.5 * (1 + np.cos(np.pi * 0.25)) * (MAX - MIN))


@case("ends the cosine phase at lr_min")
def _():
    assert_close(_lr(TC), MIN)


@case("stays at lr_min afterwards")
def _():
    assert_close(_lr(TC + 1), MIN)
    # One full cosine period past T_c: an unclamped cosine is back at lr_max here.
    assert_close(_lr(TC + (TC - TW)), MIN)
    assert_close(_lr(10 * TC), MIN)


@case("decays monotonically after warmup")
def _():
    lrs = [_lr(t) for t in range(TW, TC + 1, 25)]
    assert_true(
        all(a >= b - 1e-15 for a, b in zip(lrs, lrs[1:])),
        "the cosine phase should never increase.",
    )


@case("returns a scalar")
def _():
    assert_true(np.ndim(_lr(500)) == 0, "expected a scalar learning rate.")

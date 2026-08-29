# The merge list the training exercise produces on the classic corpus.
MERGES = [("s", "t"), ("e", "st"), ("o", "w")]


@case("'lowest' encodes to ('l', 'ow', 'est')")
def _():
    got = bpe_encode(("l", "o", "w", "e", "s", "t"), MERGES)
    assert_true(
        got == ("l", "ow", "est"),
        f"got {got}; expected ('l', 'ow', 'est') — st, then est, then ow.",
    )


@case("'newest' encodes to ('n', 'e', 'w', 'est')")
def _():
    got = bpe_encode(("n", "e", "w", "e", "s", "t"), MERGES)
    assert_true(got == ("n", "e", "w", "est"), f"got {got}; expected ('n', 'e', 'w', 'est').")


@case("rank order beats left-to-right position")
def _():
    # ("a","b") sits first in the word, but ("b","c") has the better rank.
    got = bpe_encode(("a", "b", "c"), [("b", "c"), ("a", "b")])
    assert_true(
        got == ("a", "bc"),
        f"got {got}; expected ('a', 'bc') — merge the lowest-rank pair present, "
        "not the leftmost.",
    )


@case("a merge can enable an earlier-ranked merge")
def _():
    # ("ab","c") is rank 0 but only exists after ("a","b") fires: one pass over
    # the merge list misses it.
    got = bpe_encode(("a", "b", "c"), [("ab", "c"), ("a", "b")])
    assert_true(
        got == ("abc",),
        f"got {got}; expected ('abc',) — re-scan after every merge, a single pass "
        "over merges is not enough.",
    )


@case("all occurrences merge, not just the first")
def _():
    got = bpe_encode(("a", "b", "x", "a", "b"), [("a", "b")])
    assert_true(got == ("ab", "x", "ab"), f"got {got}; expected ('ab', 'x', 'ab').")


@case("overlapping runs merge left to right without overlap")
def _():
    got = bpe_encode(("a", "a", "a"), [("a", "a")])
    assert_true(got == ("aa", "a"), f"got {got}; expected ('aa', 'a').")


@case("unknown symbols pass through untouched")
def _():
    word = ("q", "z", "q")
    got = bpe_encode(word, MERGES)
    assert_true(got == word, f"got {got}; expected the word unchanged.")


@case("empty merge list and short words are safe")
def _():
    assert_true(bpe_encode(("h", "i"), []) == ("h", "i"), "no merges: word unchanged.")
    assert_true(bpe_encode(("x",), MERGES) == ("x",), "single symbol: nothing to merge.")

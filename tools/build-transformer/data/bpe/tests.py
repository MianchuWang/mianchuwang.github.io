# The classic Sennrich-style corpus used in CS336's tokenizer lecture.
CORPUS = {
    ("l", "o", "w"): 5,
    ("l", "o", "w", "e", "r"): 2,
    ("n", "e", "w", "e", "s", "t"): 6,
    ("w", "i", "d", "e", "s", "t"): 3,
}


@case("counts are weighted by word frequency")
def _():
    # Unweighted, ("y","z") wins 2-1; weighted, ("x","y") wins 3-2.
    words = {("x", "y"): 3, ("y", "z", "y", "z"): 1}
    pair, _merged = bpe_step(words)
    assert_true(
        pair == ("x", "y"),
        f'best pair is {pair}; expected ("x", "y") — pair counts must be weighted '
        "by each word's count.",
    )


@case("ties break to the lexicographically greater pair")
def _():
    words = {("a", "b"): 2, ("c", "d"): 2}
    pair, _merged = bpe_step(words)
    assert_true(pair == ("c", "d"), f'best pair is {pair}; expected ("c", "d") on a tie.')


@case("first merge on the classic corpus is ('s', 't')")
def _():
    # ("e","s") and ("s","t") both count 9; the tie goes to ("s","t").
    pair, merged = bpe_step(CORPUS)
    assert_true(pair == ("s", "t"), f"best pair is {pair}; expected ('s', 't').")
    assert_true(
        merged[("n", "e", "w", "e", "st")] == 6,
        "after the merge, newest should be ('n','e','w','e','st') with count 6.",
    )


@case("three steps: ('s','t'), then ('e','st'), then ('o','w')")
def _():
    words = CORPUS
    seen = []
    for _ in range(3):
        pair, words = bpe_step(words)
        seen.append(pair)
    assert_true(
        seen == [("s", "t"), ("e", "st"), ("o", "w")],
        f"merge sequence is {seen}; expected [('s','t'), ('e','st'), ('o','w')].",
    )
    assert_true(
        words[("l", "ow")] == 5 and words[("l", "ow", "e", "r")] == 2,
        "after three merges, low = ('l','ow'):5 and lower = ('l','ow','e','r'):2.",
    )


@case("adjacent occurrences merge left to right without overlap")
def _():
    words = {("a", "a", "a"): 2}
    pair, merged = bpe_step(words)
    assert_true(pair == ("a", "a"), f"best pair is {pair}; expected ('a', 'a').")
    assert_true(
        merged == {("aa", "a"): 2},
        f"merged words are {merged}; expected {{('aa', 'a'): 2}} — the scan must "
        "skip past a merge, not reuse its second symbol.",
    )


@case("only adjacent pairs merge; other occurrences of the symbols survive")
def _():
    words = {("a", "b"): 5, ("a", "c", "a", "b"): 1}
    _pair, merged = bpe_step(words)
    assert_true(
        merged == {("ab",): 5, ("a", "c", "ab"): 1},
        f"merged words are {merged}; the lone 'a' in ('a','c','a','b') must survive.",
    )


@case("every occurrence in a word merges, not just the first")
def _():
    words = {("a", "b", "c", "a", "b"): 1}
    pair, merged = bpe_step(words)
    assert_true(pair == ("a", "b"), f"best pair is {pair}; expected ('a', 'b').")
    assert_true(
        merged == {("ab", "c", "ab"): 1},
        f"merged words are {merged}; expected {{('ab', 'c', 'ab'): 1}} — the scan "
        "must continue after the first merge.",
    )


@case("words that collide after the merge add their counts")
def _():
    words = {("a", "b"): 2, ("ab",): 3}
    _pair, merged = bpe_step(words)
    assert_true(
        merged == {("ab",): 5},
        f"merged words are {merged}; expected {{('ab',): 5}} — colliding words sum.",
    )


@case("the input dict is not mutated")
def _():
    words = {("a", "b"): 1, ("b", "c"): 2}
    snapshot = dict(words)
    bpe_step(words)
    assert_true(words == snapshot, "bpe_step must return a new dict, not edit the input.")

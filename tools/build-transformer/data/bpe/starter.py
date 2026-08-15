from beartype import beartype


@beartype
def bpe_step(
    words: dict[tuple[str, ...], int],
) -> tuple[tuple[str, str], dict[tuple[str, ...], int]]:
    """One BPE training step: find the best pair, merge it in every word.

    Best = highest weighted count, ties broken by the lexicographically
    greater pair. Returns (best_pair, new_words) without mutating the input.
    """
    raise NotImplementedError

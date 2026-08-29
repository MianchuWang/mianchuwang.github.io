from beartype import beartype


@beartype
def bpe_encode(
    word: tuple[str, ...],
    merges: list[tuple[str, str]],
) -> tuple[str, ...]:
    """Tokenize one pre-split word by replaying learned merges, best rank first.

    merges is in learned order: earlier entries have higher priority. Repeat
    until no adjacent pair in the word appears in merges.
    """
    raise NotImplementedError

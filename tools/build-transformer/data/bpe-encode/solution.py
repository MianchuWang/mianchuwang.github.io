from beartype import beartype


@beartype
def bpe_encode(
    word: tuple[str, ...],
    merges: list[tuple[str, str]],
) -> tuple[str, ...]:
    """Tokenize one pre-split word by replaying learned merges, best rank first."""
    rank = {pair: i for i, pair in enumerate(merges)}
    symbols = list(word)
    while len(symbols) > 1:
        pairs = set(zip(symbols, symbols[1:]))
        best = min(pairs, key=lambda p: rank.get(p, float("inf")))
        if best not in rank:
            break
        out = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == best:
                out.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    return tuple(symbols)

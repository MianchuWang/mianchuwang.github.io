from beartype import beartype


@beartype
def bpe_step(
    words: dict[tuple[str, ...], int],
) -> tuple[tuple[str, str], dict[tuple[str, ...], int]]:
    """One BPE training step: find the best pair, merge it in every word."""
    counts: dict[tuple[str, str], int] = {}
    for word, freq in words.items():
        for pair in zip(word, word[1:]):
            counts[pair] = counts.get(pair, 0) + freq

    best = max(counts, key=lambda pair: (counts[pair], pair))

    merged: dict[tuple[str, ...], int] = {}
    for word, freq in words.items():
        out = []
        i = 0
        while i < len(word):
            if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                out.append(word[i] + word[i + 1])
                i += 2
            else:
                out.append(word[i])
                i += 1
        key = tuple(out)
        merged[key] = merged.get(key, 0) + freq

    return best, merged

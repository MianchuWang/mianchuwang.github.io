import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def top_p_filter(
    probs: Float[np.ndarray, "*batch vocab"],
    p: float,
) -> Float[np.ndarray, "*batch vocab"]:
    """Zero the tail outside the nucleus and renormalise; positions stay put."""
    order = np.argsort(-probs, axis=-1)
    ordered = np.take_along_axis(probs, order, axis=-1)
    # Cumulative mass *before* each token: keeps the one that crosses p, and
    # always keeps the most likely token (its prefix mass is 0).
    prefix = np.cumsum(ordered, axis=-1) - ordered

    keep = np.zeros_like(probs, dtype=bool)
    np.put_along_axis(keep, order, prefix < p, axis=-1)

    kept = np.where(keep, probs, 0.0)
    return kept / np.sum(kept, axis=-1, keepdims=True)

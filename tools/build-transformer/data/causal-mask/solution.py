import numpy as np
from beartype import beartype
from jaxtyping import Bool, jaxtyped


@jaxtyped(typechecker=beartype)
def causal_mask(n_q: int, n_k: int) -> Bool[np.ndarray, "n_q n_k"]:
    """True where a query may attend to a key; query i sits at position n_k - n_q + i."""
    q_pos = np.arange(n_q)[:, None] + (n_k - n_q)
    k_pos = np.arange(n_k)[None, :]
    return k_pos <= q_pos

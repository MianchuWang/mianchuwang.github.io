import numpy as np
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Bool, Float, jaxtyped

# `softmax(x, axis=-1)` is already defined for you.


@jaxtyped(typechecker=beartype)
def attention(
    Q: Float[np.ndarray, "*batch n_q d_k"],
    K: Float[np.ndarray, "*batch n_k d_k"],
    V: Float[np.ndarray, "*batch n_k d_v"],
    mask: Bool[np.ndarray, "*#batch n_q n_k"] | None = None,
) -> Float[np.ndarray, "*batch n_q d_v"]:
    """softmax(Q K^T / sqrt(d_k) + M) V, where M is 0 (True) or -inf (False)."""
    raise NotImplementedError

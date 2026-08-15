import numpy as np
from beartype import beartype
from einops import einsum
from jaxtyping import Bool, Float, jaxtyped


@jaxtyped(typechecker=beartype)
def attention(
    Q: Float[np.ndarray, "*batch n_q d_k"],
    K: Float[np.ndarray, "*batch n_k d_k"],
    V: Float[np.ndarray, "*batch n_k d_v"],
    mask: Bool[np.ndarray, "*#batch n_q n_k"] | None = None,
) -> Float[np.ndarray, "*batch n_q d_v"]:
    """softmax(Q K^T / sqrt(d_k) + M) V, where M is 0 (True) or -inf (False)."""
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return einsum(weights, V, "... q k, ... k v -> ... q v")

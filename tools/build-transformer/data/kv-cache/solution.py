import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def decode_step(
    x: Float[np.ndarray, "b d_model"],
    k_cache: Float[np.ndarray, "b t d_k"],
    v_cache: Float[np.ndarray, "b t d_k"],
    Wq: Float[np.ndarray, "d_k d_model"],
    Wk: Float[np.ndarray, "d_k d_model"],
    Wv: Float[np.ndarray, "d_k d_model"],
) -> tuple[
    Float[np.ndarray, "b d_k"],
    Float[np.ndarray, "b t+1 d_k"],
    Float[np.ndarray, "b t+1 d_k"],
]:
    """Attend one new token over the (updated) KV cache; single head, no mask."""
    q = x @ Wq.T
    k = x @ Wk.T
    v = x @ Wv.T

    k_cache = np.concatenate([k_cache, k[:, None, :]], axis=1)
    v_cache = np.concatenate([v_cache, v[:, None, :]], axis=1)

    out = attention(q[:, None, :], k_cache, v_cache)[:, 0]
    return out, k_cache, v_cache

import numpy as np
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, jaxtyped

# `softmax(x, axis=-1)` and `attention(Q, K, V, mask=None)` are already defined for you.


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
    raise NotImplementedError

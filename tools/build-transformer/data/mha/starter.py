import numpy as np
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Bool, Float, jaxtyped

# `softmax(x, axis=-1)` and `attention(Q, K, V, mask=None)` are already defined for you.


@jaxtyped(typechecker=beartype)
def multihead_attention(
    x: Float[np.ndarray, "b n d_model"],
    Wq: Float[np.ndarray, "d_model d_model"],
    Wk: Float[np.ndarray, "d_model d_model"],
    Wv: Float[np.ndarray, "d_model d_model"],
    Wo: Float[np.ndarray, "d_model d_model"],
    num_heads: int,
    mask: Bool[np.ndarray, "n n"] | None = None,
) -> Float[np.ndarray, "b n d_model"]:
    """Project, split into heads, attend, merge, project."""
    raise NotImplementedError

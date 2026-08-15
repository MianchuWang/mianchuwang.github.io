import numpy as np
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Bool, Float, jaxtyped

# `softmax(x, axis=-1)` and `attention(Q, K, V, mask=None)` are already defined for you.


@jaxtyped(typechecker=beartype)
def grouped_query_attention(
    x: Float[np.ndarray, "b n d_model"],
    Wq: Float[np.ndarray, "q_proj d_model"],
    Wk: Float[np.ndarray, "kv_proj d_model"],
    Wv: Float[np.ndarray, "kv_proj d_model"],
    Wo: Float[np.ndarray, "d_model q_proj"],
    num_heads: int,
    num_kv_heads: int,
    mask: Bool[np.ndarray, "n n"] | None = None,
) -> Float[np.ndarray, "b n d_model"]:
    """num_heads query heads share num_kv_heads K/V heads, group by group.

    q_proj = num_heads * d_head, kv_proj = num_kv_heads * d_head.
    """
    raise NotImplementedError

import numpy as np
from beartype import beartype
from einops import einsum, rearrange, repeat
from jaxtyping import Bool, Float, jaxtyped


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
    """num_heads query heads share num_kv_heads K/V heads, group by group."""
    q = rearrange(einsum(x, Wq, "b n d, o d -> b n o"), "b n (h d) -> b h n d", h=num_heads)
    k = rearrange(einsum(x, Wk, "b n d, o d -> b n o"), "b n (h d) -> b h n d", h=num_kv_heads)
    v = rearrange(einsum(x, Wv, "b n d, o d -> b n o"), "b n (h d) -> b h n d", h=num_kv_heads)

    group = num_heads // num_kv_heads
    k = repeat(k, "b kh n d -> b (kh g) n d", g=group)
    v = repeat(v, "b kh n d -> b (kh g) n d", g=group)

    if mask is not None:
        mask = rearrange(mask, "q k -> 1 1 q k")
    o = attention(q, k, v, mask)

    o = rearrange(o, "b h n d -> b n (h d)")
    return einsum(o, Wo, "b n d, o d -> b n o")

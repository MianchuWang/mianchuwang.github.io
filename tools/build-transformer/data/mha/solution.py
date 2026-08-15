import numpy as np
from beartype import beartype
from einops import einsum, rearrange
from jaxtyping import Bool, Float, jaxtyped


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
    q = einsum(x, Wq, "b n d, o d -> b n o")
    k = einsum(x, Wk, "b n d, o d -> b n o")
    v = einsum(x, Wv, "b n d, o d -> b n o")

    q = rearrange(q, "b n (h d) -> b h n d", h=num_heads)
    k = rearrange(k, "b n (h d) -> b h n d", h=num_heads)
    v = rearrange(v, "b n (h d) -> b h n d", h=num_heads)

    if mask is not None:
        mask = rearrange(mask, "q k -> 1 1 q k")
    o = attention(q, k, v, mask)

    o = rearrange(o, "b h n d -> b n (h d)")
    return einsum(o, Wo, "b n d, o d -> b n o")

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def transformer_block(
    x: Float[np.ndarray, "b n d_model"],
    p: dict[str, np.ndarray],
    num_heads: int,
) -> Float[np.ndarray, "b n d_model"]:
    """Pre-norm decoder layer with causal attention."""
    n = x.shape[1]
    mask = causal_mask(n, n)

    normed = rms_norm(x, p["g1"])
    h = x + multihead_attention(normed, p["Wq"], p["Wk"], p["Wv"], p["Wo"], num_heads, mask)

    return h + swiglu(rms_norm(h, p["g2"]), p["W1"], p["W2"], p["W3"])

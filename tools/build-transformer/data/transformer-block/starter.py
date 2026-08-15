import numpy as np
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, jaxtyped

# Already defined for you: rms_norm, softmax, attention, multihead_attention,
# swiglu, causal_mask.


@jaxtyped(typechecker=beartype)
def transformer_block(
    x: Float[np.ndarray, "b n d_model"],
    p: dict[str, np.ndarray],
    num_heads: int,
) -> Float[np.ndarray, "b n d_model"]:
    """Pre-norm decoder layer with causal attention.

    p: {"g1", "Wq", "Wk", "Wv", "Wo", "g2", "W1", "W2", "W3"}
    """
    raise NotImplementedError

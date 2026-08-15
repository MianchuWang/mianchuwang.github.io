import numpy as np
from beartype import beartype
from einops import einsum
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def linear(
    x: Float[np.ndarray, "*batch d_in"],
    W: Float[np.ndarray, "d_out d_in"],
    b: Float[np.ndarray, "d_out"] | None = None,
) -> Float[np.ndarray, "*batch d_out"]:
    """Affine map over the last axis: y = x W^T + b."""
    y = einsum(x, W, "... d_in, d_out d_in -> ... d_out")
    if b is not None:
        y = y + b
    return y

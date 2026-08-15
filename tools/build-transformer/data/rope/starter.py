import numpy as np
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, Int, jaxtyped


@jaxtyped(typechecker=beartype)
def rope(
    x: Float[np.ndarray, "*batch n d_head"],
    positions: Int[np.ndarray, "n"],
    theta: float = 10000.0,
) -> Float[np.ndarray, "*batch n d_head"]:
    """Rotate interleaved feature pairs (2i, 2i+1) by pos * theta^(-2i/d_head)."""
    raise NotImplementedError

import numpy as np
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def top_p_filter(
    probs: Float[np.ndarray, "*batch vocab"],
    p: float,
) -> Float[np.ndarray, "*batch vocab"]:
    """Zero the tail outside the nucleus and renormalise; positions stay put."""
    raise NotImplementedError

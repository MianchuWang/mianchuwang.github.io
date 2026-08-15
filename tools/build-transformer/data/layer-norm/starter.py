import numpy as np
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def layer_norm(
    x: Float[np.ndarray, "*batch d"],
    gamma: Float[np.ndarray, "d"],
    beta: Float[np.ndarray, "d"],
    eps: float = 1e-5,
) -> Float[np.ndarray, "*batch d"]:
    """Centre and rescale each vector over its last axis, then apply gain and bias."""
    raise NotImplementedError

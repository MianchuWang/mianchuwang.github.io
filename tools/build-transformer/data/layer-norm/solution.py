import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def layer_norm(
    x: Float[np.ndarray, "*batch d"],
    gamma: Float[np.ndarray, "d"],
    beta: Float[np.ndarray, "d"],
    eps: float = 1e-5,
) -> Float[np.ndarray, "*batch d"]:
    """Centre and rescale each vector over its last axis, then apply gain and bias."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

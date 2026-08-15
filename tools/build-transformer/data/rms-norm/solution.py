import numpy as np
from beartype import beartype
from einops import reduce
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def rms_norm(
    x: Float[np.ndarray, "*batch d"],
    gamma: Float[np.ndarray, "d"],
    eps: float = 1e-6,
) -> Float[np.ndarray, "*batch d"]:
    """Rescale each vector by its root-mean-square; no centring, no bias."""
    ms = reduce(x**2, "... d -> ... 1", "mean")
    return gamma * x / np.sqrt(ms + eps)

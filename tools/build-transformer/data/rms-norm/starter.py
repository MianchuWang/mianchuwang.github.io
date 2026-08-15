import numpy as np
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def rms_norm(
    x: Float[np.ndarray, "*batch d"],
    gamma: Float[np.ndarray, "d"],
    eps: float = 1e-6,
) -> Float[np.ndarray, "*batch d"]:
    """Rescale each vector by its root-mean-square; no centring, no bias."""
    raise NotImplementedError

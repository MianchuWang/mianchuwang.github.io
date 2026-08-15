import numpy as np
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def swiglu(
    x: Float[np.ndarray, "*batch d_model"],
    W1: Float[np.ndarray, "d_ff d_model"],
    W2: Float[np.ndarray, "d_model d_ff"],
    W3: Float[np.ndarray, "d_ff d_model"],
) -> Float[np.ndarray, "*batch d_model"]:
    """Gated MLP: (SiLU(x W1^T) * (x W3^T)) W2^T."""
    raise NotImplementedError

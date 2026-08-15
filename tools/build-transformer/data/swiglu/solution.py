import numpy as np
from beartype import beartype
from einops import einsum
from jaxtyping import Float, jaxtyped


def silu(z):
    return z / (1.0 + np.exp(-z))


@jaxtyped(typechecker=beartype)
def swiglu(
    x: Float[np.ndarray, "*batch d_model"],
    W1: Float[np.ndarray, "d_ff d_model"],
    W2: Float[np.ndarray, "d_model d_ff"],
    W3: Float[np.ndarray, "d_ff d_model"],
) -> Float[np.ndarray, "*batch d_model"]:
    """Gated MLP: (SiLU(x W1^T) * (x W3^T)) W2^T."""
    gate = einsum(x, W1, "... d, f d -> ... f")
    up = einsum(x, W3, "... d, f d -> ... f")
    return einsum(silu(gate) * up, W2, "... f, d f -> ... d")

import numpy as np
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, Int, jaxtyped


@jaxtyped(typechecker=beartype)
def rope(
    x: Float[np.ndarray, "*batch n d_head"],
    positions: Int[np.ndarray, "n"],
    theta: float = 10000.0,
) -> Float[np.ndarray, "*batch n d_head"]:
    """Rotate interleaved feature pairs (2i, 2i+1) by pos * theta^(-2i/d_head)."""
    d = x.shape[-1]
    pos = positions.astype(np.float64)

    freqs = theta ** (-2.0 * np.arange(d // 2) / d)  # (d/2,)
    angles = pos[:, None] * freqs[None, :]  # (n, d/2)
    cos, sin = np.cos(angles), np.sin(angles)

    pairs = rearrange(x, "... n (p two) -> ... n p two", two=2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    rotated = np.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)
    return rearrange(rotated, "... n p two -> ... n (p two)")

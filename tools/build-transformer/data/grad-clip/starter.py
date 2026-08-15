import numpy as np
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def clip_grad_norm(
    grads: list[Float[np.ndarray, "..."]],
    max_norm: float,
    eps: float = 1e-6,
) -> list[Float[np.ndarray, "..."]]:
    """Rescale the whole list so its global L2 norm is at most max_norm."""
    raise NotImplementedError

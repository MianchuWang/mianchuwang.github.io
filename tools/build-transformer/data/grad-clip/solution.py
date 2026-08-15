import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def clip_grad_norm(
    grads: list[Float[np.ndarray, "..."]],
    max_norm: float,
    eps: float = 1e-6,
) -> list[Float[np.ndarray, "..."]]:
    """Rescale the whole list so its global L2 norm is at most max_norm."""
    total = np.sqrt(sum(np.sum(g**2) for g in grads))
    if total <= max_norm:
        return [g.copy() for g in grads]
    scale = max_norm / (total + eps)
    return [g * scale for g in grads]

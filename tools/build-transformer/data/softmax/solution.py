import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def softmax(
    x: Float[np.ndarray, "*batch"],
    axis: int = -1,
) -> Float[np.ndarray, "*batch"]:
    """Normalise to a distribution along `axis`, without overflowing."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)

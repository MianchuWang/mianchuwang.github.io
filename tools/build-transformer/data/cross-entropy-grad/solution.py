import numpy as np
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped


@jaxtyped(typechecker=beartype)
def cross_entropy_grad(
    logits: Float[np.ndarray, "*batch vocab"],
    targets: Int[np.ndarray, "*batch"],
) -> Float[np.ndarray, "*batch vocab"]:
    """Gradient of the mean cross-entropy loss w.r.t. the logits."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(shifted)
    probs = e / np.sum(e, axis=-1, keepdims=True)

    onehot = np.zeros_like(probs)
    np.put_along_axis(onehot, targets[..., None], 1.0, axis=-1)

    return (probs - onehot) / targets.size

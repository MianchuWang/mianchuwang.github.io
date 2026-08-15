import numpy as np
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped


@jaxtyped(typechecker=beartype)
def cross_entropy(
    logits: Float[np.ndarray, "*batch vocab"],
    targets: Int[np.ndarray, "*batch"],
) -> float:
    """Mean of logsumexp(logits) - logits[target] over all leading axes."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_z = np.log(np.sum(np.exp(shifted), axis=-1))
    picked = np.take_along_axis(shifted, targets[..., None], axis=-1)[..., 0]
    return float(np.mean(log_z - picked))

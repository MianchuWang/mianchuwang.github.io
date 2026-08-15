import numpy as np
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, Int, jaxtyped


@jaxtyped(typechecker=beartype)
def cross_entropy(
    logits: Float[np.ndarray, "*batch vocab"],
    targets: Int[np.ndarray, "*batch"],
) -> float:
    """Mean of logsumexp(logits) - logits[target] over all leading axes."""
    raise NotImplementedError

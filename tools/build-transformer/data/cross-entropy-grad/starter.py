import numpy as np
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, Int, jaxtyped


@jaxtyped(typechecker=beartype)
def cross_entropy_grad(
    logits: Float[np.ndarray, "*batch vocab"],
    targets: Int[np.ndarray, "*batch"],
) -> Float[np.ndarray, "*batch vocab"]:
    """Gradient of the mean cross-entropy loss w.r.t. the logits."""
    raise NotImplementedError

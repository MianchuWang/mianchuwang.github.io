import numpy as np
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def softmax(
    x: Float[np.ndarray, "*batch"],
    axis: int = -1,
) -> Float[np.ndarray, "*batch"]:
    """Normalise to a distribution along `axis`, without overflowing."""
    raise NotImplementedError

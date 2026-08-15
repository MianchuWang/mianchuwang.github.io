import numpy as np
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import jaxtyped


@jaxtyped(typechecker=beartype)
def lr_cosine_schedule(
    t: int,
    lr_max: float,
    lr_min: float,
    warmup_iters: int,
    cosine_iters: int,
) -> float:
    """Linear warmup to lr_max, cosine decay to lr_min, then a constant floor."""
    raise NotImplementedError

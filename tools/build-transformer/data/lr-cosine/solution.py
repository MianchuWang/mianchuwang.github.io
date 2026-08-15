import numpy as np
from beartype import beartype
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
    if t < warmup_iters:
        return float(lr_max * t / warmup_iters)
    if t <= cosine_iters:
        progress = (t - warmup_iters) / (cosine_iters - warmup_iters)
        return float(lr_min + 0.5 * (1.0 + np.cos(np.pi * progress)) * (lr_max - lr_min))
    return float(lr_min)

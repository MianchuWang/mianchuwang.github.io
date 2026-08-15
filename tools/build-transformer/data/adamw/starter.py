import numpy as np
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def adamw_step(
    param: Float[np.ndarray, "*shape"],
    grad: Float[np.ndarray, "*shape"],
    m: Float[np.ndarray, "*shape"],
    v: Float[np.ndarray, "*shape"],
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[
    Float[np.ndarray, "*shape"],
    Float[np.ndarray, "*shape"],
    Float[np.ndarray, "*shape"],
]:
    """One AdamW step (t is 1-based); returns (param, m, v) without mutating the inputs."""
    raise NotImplementedError

import numpy as np
from beartype import beartype
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
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * grad**2

    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)

    param = param - lr * m_hat / (np.sqrt(v_hat) + eps) - lr * weight_decay * param
    return param, m, v

"""Provided for you — the components from the earlier exercises."""

import numpy as np
from einops import rearrange


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def rms_norm(x, gamma, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    ms = np.mean(x**2, axis=-1, keepdims=True)
    return gamma * x / np.sqrt(ms + eps)


def causal_mask(n_q, n_k):
    q_pos = np.arange(n_q)[:, None] + (n_k - n_q)
    return np.arange(n_k)[None, :] <= q_pos


def attention(Q, K, V, mask=None):
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    return softmax(scores, axis=-1) @ V


def multihead_attention(x, Wq, Wk, Wv, Wo, num_heads, mask=None):
    q, k, v = (rearrange(x @ w.T, "b n (h d) -> b h n d", h=num_heads) for w in (Wq, Wk, Wv))
    if mask is not None:
        mask = rearrange(np.asarray(mask), "q k -> 1 1 q k")
    o = rearrange(attention(q, k, v, mask), "b h n d -> b n (h d)")
    return o @ Wo.T


def swiglu(x, W1, W2, W3):
    gate = x @ W1.T
    silu = gate / (1.0 + np.exp(-gate))
    return (silu * (x @ W3.T)) @ W2.T

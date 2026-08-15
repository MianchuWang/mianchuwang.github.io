"""Provided for you — a stable softmax and scaled dot-product attention."""

import numpy as np


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def attention(Q, K, V, mask=None):
    """Q: (..., n_q, d_k), K: (..., n_k, d_k), V: (..., n_k, d_v) -> (..., n_q, d_v)"""
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    return softmax(scores, axis=-1) @ V

#!/usr/bin/env python3
"""Dev check: plausible-but-wrong implementations must FAIL each component's tests.

validate.py proves the reference solutions pass and the starters fail; this
script proves the test suites catch the classic mistakes. Every entry in WRONG
is a bug that shows up in real from-scratch implementations. If one of them
passes a component's whole test suite, that suite has a hole — add a test,
then keep the wrong variant here as a regression check.

    python3 tools/build-transformer/audit.py          # all components
    python3 tools/build-transformer/audit.py rope     # a subset

Needs numpy and einops in the current interpreter.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"

WRONG = {
    "bpe": [
        ("pair counts not weighted by word frequency", """
def bpe_step(words):
    counts = {}
    for word in words:
        for pair in zip(word, word[1:]):
            counts[pair] = counts.get(pair, 0) + 1
    best = max(counts, key=lambda p: (counts[p], p))
    merged = {}
    for word, freq in words.items():
        out, i = [], 0
        while i < len(word):
            if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                out.append(word[i] + word[i + 1]); i += 2
            else:
                out.append(word[i]); i += 1
        merged[tuple(out)] = merged.get(tuple(out), 0) + freq
    return best, merged
"""),
        ("ties broken toward the smaller pair", """
def bpe_step(words):
    counts = {}
    for word, freq in words.items():
        for pair in zip(word, word[1:]):
            counts[pair] = counts.get(pair, 0) + freq
    best = min((p for p in counts if counts[p] == max(counts.values())))
    merged = {}
    for word, freq in words.items():
        out, i = [], 0
        while i < len(word):
            if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                out.append(word[i] + word[i + 1]); i += 2
            else:
                out.append(word[i]); i += 1
        merged[tuple(out)] = merged.get(tuple(out), 0) + freq
    return best, merged
"""),
        ("overlapping merge (scan does not skip)", """
def bpe_step(words):
    counts = {}
    for word, freq in words.items():
        for pair in zip(word, word[1:]):
            counts[pair] = counts.get(pair, 0) + freq
    best = max(counts, key=lambda p: (counts[p], p))
    merged = {}
    for word, freq in words.items():
        out = []
        for i in range(len(word)):
            if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                out.append(word[i] + word[i + 1])
            elif i > 0 and (word[i - 1], word[i]) == best:
                continue
            else:
                out.append(word[i])
        merged[tuple(out)] = merged.get(tuple(out), 0) + freq
    return best, merged
"""),
        ("only the first occurrence per word merges", """
def bpe_step(words):
    counts = {}
    for word, freq in words.items():
        for pair in zip(word, word[1:]):
            counts[pair] = counts.get(pair, 0) + freq
    best = max(counts, key=lambda p: (counts[p], p))
    merged = {}
    for word, freq in words.items():
        out, i, done = [], 0, False
        while i < len(word):
            if not done and i + 1 < len(word) and (word[i], word[i + 1]) == best:
                out.append(word[i] + word[i + 1]); i += 2; done = True
            else:
                out.append(word[i]); i += 1
        merged[tuple(out)] = merged.get(tuple(out), 0) + freq
    return best, merged
"""),
    ],
    "kv-cache": [
        ("attends only over the old cache, not the new token", """
import numpy as np
def decode_step(x, k_cache, v_cache, Wq, Wk, Wv):
    q, k, v = x @ Wq.T, x @ Wk.T, x @ Wv.T
    out = attention(q[:, None, :], k_cache, v_cache)[:, 0]
    k_cache = np.concatenate([k_cache, k[:, None, :]], axis=1)
    v_cache = np.concatenate([v_cache, v[:, None, :]], axis=1)
    return out, k_cache, v_cache
"""),
        ("appends q to the cache instead of k", """
import numpy as np
def decode_step(x, k_cache, v_cache, Wq, Wk, Wv):
    q, v = x @ Wq.T, x @ Wv.T
    k_cache = np.concatenate([k_cache, q[:, None, :]], axis=1)
    v_cache = np.concatenate([v_cache, v[:, None, :]], axis=1)
    return attention(q[:, None, :], k_cache, v_cache)[:, 0], k_cache, v_cache
"""),
        ("no sqrt(d_k) scaling in the by-hand attention", """
import numpy as np
def decode_step(x, k_cache, v_cache, Wq, Wk, Wv):
    q, k, v = x @ Wq.T, x @ Wk.T, x @ Wv.T
    K = np.concatenate([k_cache, k[:, None, :]], axis=1)
    V = np.concatenate([v_cache, v[:, None, :]], axis=1)
    s = np.einsum("bd,btd->bt", q, K)
    w = np.exp(s - s.max(axis=-1, keepdims=True))
    w = w / w.sum(axis=-1, keepdims=True)
    return np.einsum("bt,btd->bd", w, V), K, V
"""),
    ],
    "cross-entropy-grad": [
        ("forgets to divide by the token count", """
import numpy as np
def cross_entropy_grad(logits, targets):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    onehot = np.zeros_like(p)
    np.put_along_axis(onehot, targets[..., None], 1.0, axis=-1)
    return p - onehot
"""),
        ("sign flipped (one-hot minus softmax)", """
import numpy as np
def cross_entropy_grad(logits, targets):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    onehot = np.zeros_like(p)
    np.put_along_axis(onehot, targets[..., None], 1.0, axis=-1)
    return (onehot - p) / targets.size
"""),
        ("unstable softmax inside", """
import numpy as np
def cross_entropy_grad(logits, targets):
    e = np.exp(logits)
    p = e / e.sum(axis=-1, keepdims=True)
    onehot = np.zeros_like(p)
    np.put_along_axis(onehot, targets[..., None], 1.0, axis=-1)
    return (p - onehot) / targets.size
"""),
        ("divides by the batch size only, not batch * seq", """
import numpy as np
def cross_entropy_grad(logits, targets):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    onehot = np.zeros_like(p)
    np.put_along_axis(onehot, targets[..., None], 1.0, axis=-1)
    return (p - onehot) / logits.shape[0]
"""),
    ],
    "bpe-encode": [
        ("one pass over the merge list misses re-enabled merges", """
def bpe_encode(word, merges):
    symbols = list(word)
    for pair in merges:
        out, i = [], 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                out.append(symbols[i] + symbols[i + 1]); i += 2
            else:
                out.append(symbols[i]); i += 1
        symbols = out
    return tuple(symbols)
"""),
        ("merges the leftmost rankable pair instead of the best-ranked", """
def bpe_encode(word, merges):
    rank = {p: i for i, p in enumerate(merges)}
    symbols = list(word)
    while True:
        for i in range(len(symbols) - 1):
            if (symbols[i], symbols[i + 1]) in rank:
                symbols[i:i + 2] = [symbols[i] + symbols[i + 1]]
                break
        else:
            break
    return tuple(symbols)
"""),
        ("missing infinity default lets unknown pairs win the min", """
def bpe_encode(word, merges):
    rank = {p: i for i, p in enumerate(merges)}
    symbols = list(word)
    while len(symbols) > 1:
        pairs = set(zip(symbols, symbols[1:]))
        best = min(pairs, key=lambda p: rank.get(p, 0))
        if best not in rank:
            break
        out, i = [], 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == best:
                out.append(symbols[i] + symbols[i + 1]); i += 2
            else:
                out.append(symbols[i]); i += 1
        symbols = out
    return tuple(symbols)
"""),
    ],
    "linear": [
        ("W treated as (d_in, d_out)", """
import numpy as np
def linear(x, W, b=None):
    y = x @ W
    return y if b is None else y + b
"""),
        ("bias ignored", """
import numpy as np
def linear(x, W, b=None):
    return x @ W.T
"""),
    ],
    "softmax": [
        ("no max subtraction", """
import numpy as np
def softmax(x, axis=-1):
    e = np.exp(np.asarray(x, dtype=np.float64))
    return e / np.sum(e, axis=axis, keepdims=True)
"""),
        ("global max instead of per-slice max", """
import numpy as np
def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / np.sum(e, axis=axis, keepdims=True)
"""),
        ("keepdims forgotten", """
import numpy as np
def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - np.max(x, axis=axis))
    return e / np.sum(e, axis=axis)
"""),
    ],
    "layer-norm": [
        ("eps outside the square root", """
import numpy as np
def layer_norm(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=-1, keepdims=True)
    return gamma * (x - mu) / (np.sqrt(x.var(axis=-1, keepdims=True)) + eps) + beta
"""),
        ("unbiased variance (ddof=1)", """
import numpy as np
def layer_norm(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True, ddof=1)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta
"""),
        ("statistics over the whole batch", """
import numpy as np
def layer_norm(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    return gamma * (x - x.mean()) / np.sqrt(x.var() + eps) + beta
"""),
    ],
    "rms-norm": [
        ("mean subtracted (layer-norm habit)", """
import numpy as np
def rms_norm(x, gamma, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    return gamma * x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)
"""),
        ("eps outside the square root", """
import numpy as np
def rms_norm(x, gamma, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    return gamma * x / (np.sqrt((x**2).mean(axis=-1, keepdims=True)) + eps)
"""),
        ("L2 norm instead of RMS", """
import numpy as np
def rms_norm(x, gamma, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    return gamma * x / np.sqrt((x**2).sum(axis=-1, keepdims=True) + eps)
"""),
    ],
    "swiglu": [
        ("W1 and W3 swapped", """
import numpy as np
def swiglu(x, W1, W2, W3):
    x = np.asarray(x, dtype=np.float64)
    gate = x @ W3.T
    return ((gate / (1.0 + np.exp(-gate))) * (x @ W1.T)) @ W2.T
"""),
        ("ReLU gate", """
import numpy as np
def swiglu(x, W1, W2, W3):
    x = np.asarray(x, dtype=np.float64)
    return (np.maximum(x @ W1.T, 0.0) * (x @ W3.T)) @ W2.T
"""),
        ("plain sigmoid gate (GLU, not SwiGLU)", """
import numpy as np
def swiglu(x, W1, W2, W3):
    x = np.asarray(x, dtype=np.float64)
    return ((1.0 / (1.0 + np.exp(-(x @ W1.T)))) * (x @ W3.T)) @ W2.T
"""),
    ],
    "attention": [
        ("no sqrt(d_k) scaling", """
import numpy as np
def attention(Q, K, V, mask=None):
    scores = Q @ np.swapaxes(K, -1, -2)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    return softmax(scores, axis=-1) @ V
"""),
        ("mask applied after the softmax", """
import numpy as np
def attention(Q, K, V, mask=None):
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(np.shape(Q)[-1])
    w = softmax(scores, axis=-1)
    if mask is not None:
        w = np.where(mask, w, 0.0)
    return w @ V
"""),
        ("scaled by d_k instead of sqrt(d_k)", """
import numpy as np
def attention(Q, K, V, mask=None):
    scores = Q @ np.swapaxes(K, -1, -2) / np.shape(Q)[-1]
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    return softmax(scores, axis=-1) @ V
"""),
    ],
    "causal-mask": [
        ("plain tril, offset ignored", """
import numpy as np
def causal_mask(n_q, n_k):
    return np.tril(np.ones((n_q, n_k), dtype=bool))
"""),
        ("strictly-lower triangle (self excluded)", """
import numpy as np
def causal_mask(n_q, n_k):
    return np.tril(np.ones((n_q, n_k), dtype=bool), k=n_k - n_q - 1)
"""),
    ],
    "mha": [
        ("reshape without moving the head axis", """
import numpy as np
def multihead_attention(x, Wq, Wk, Wv, Wo, num_heads, mask=None):
    b, n, d = np.shape(x)
    dh = d // num_heads
    q, k, v = ((x @ w.T).reshape(b, num_heads, n, dh) for w in (Wq, Wk, Wv))
    o = attention(q, k, v, None if mask is None else mask[None, None])
    return o.reshape(b, n, d) @ Wo.T
"""),
        ("features split as (d h) instead of (h d)", """
import numpy as np
from einops import rearrange
def multihead_attention(x, Wq, Wk, Wv, Wo, num_heads, mask=None):
    q, k, v = (rearrange(x @ w.T, "b n (d h) -> b h n d", h=num_heads) for w in (Wq, Wk, Wv))
    o = attention(q, k, v, None if mask is None else mask[None, None])
    return rearrange(o, "b h n d -> b n (d h)") @ Wo.T
"""),
        ("output projection skipped", """
import numpy as np
from einops import rearrange
def multihead_attention(x, Wq, Wk, Wv, Wo, num_heads, mask=None):
    q, k, v = (rearrange(x @ w.T, "b n (h d) -> b h n d", h=num_heads) for w in (Wq, Wk, Wv))
    o = attention(q, k, v, None if mask is None else mask[None, None])
    return rearrange(o, "b h n d -> b n (h d)")
"""),
    ],
    "gqa": [
        ("KV heads repeated in (g kh) order", """
import numpy as np
from einops import rearrange, repeat
def grouped_query_attention(x, Wq, Wk, Wv, Wo, num_heads, num_kv_heads, mask=None):
    q = rearrange(x @ Wq.T, "b n (h d) -> b h n d", h=num_heads)
    k = rearrange(x @ Wk.T, "b n (h d) -> b h n d", h=num_kv_heads)
    v = rearrange(x @ Wv.T, "b n (h d) -> b h n d", h=num_kv_heads)
    g = num_heads // num_kv_heads
    k = repeat(k, "b kh n d -> b (g kh) n d", g=g)
    v = repeat(v, "b kh n d -> b (g kh) n d", g=g)
    o = attention(q, k, v, None if mask is None else mask[None, None])
    return rearrange(o, "b h n d -> b n (h d)") @ Wo.T
"""),
        ("np.tile on the head axis (same interleaving bug)", """
import numpy as np
from einops import rearrange
def grouped_query_attention(x, Wq, Wk, Wv, Wo, num_heads, num_kv_heads, mask=None):
    q = rearrange(x @ Wq.T, "b n (h d) -> b h n d", h=num_heads)
    k = rearrange(x @ Wk.T, "b n (h d) -> b h n d", h=num_kv_heads)
    v = rearrange(x @ Wv.T, "b n (h d) -> b h n d", h=num_kv_heads)
    g = num_heads // num_kv_heads
    k, v = np.tile(k, (1, g, 1, 1)), np.tile(v, (1, g, 1, 1))
    o = attention(q, k, v, None if mask is None else mask[None, None])
    return rearrange(o, "b h n d -> b n (h d)") @ Wo.T
"""),
    ],
    "rope": [
        ("split-half pairing instead of interleaved", """
import numpy as np
def rope(x, positions, theta=10000.0):
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    pos = np.asarray(positions, dtype=np.float64)
    freqs = theta ** (-2.0 * np.arange(d // 2) / d)
    ang = pos[:, None] * freqs[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
"""),
        ("rotation angle sign flipped", """
import numpy as np
from einops import rearrange
def rope(x, positions, theta=10000.0):
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    pos = np.asarray(positions, dtype=np.float64)
    ang = pos[:, None] * theta ** (-2.0 * np.arange(d // 2) / d)[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    p = rearrange(x, "... n (p two) -> ... n p two", two=2)
    x0, x1 = p[..., 0], p[..., 1]
    out = np.stack([x0 * cos + x1 * sin, -x0 * sin + x1 * cos], axis=-1)
    return rearrange(out, "... n p two -> ... n (p two)")
"""),
        ("positions argument ignored", """
import numpy as np
from einops import rearrange
def rope(x, positions, theta=10000.0):
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    pos = np.arange(x.shape[-2], dtype=np.float64)
    ang = pos[:, None] * theta ** (-2.0 * np.arange(d // 2) / d)[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    p = rearrange(x, "... n (p two) -> ... n p two", two=2)
    x0, x1 = p[..., 0], p[..., 1]
    out = np.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)
    return rearrange(out, "... n p two -> ... n (p two)")
"""),
    ],
    "transformer-block": [
        ("post-norm wiring", """
import numpy as np
def transformer_block(x, p, num_heads):
    n = np.asarray(x).shape[1]
    m = causal_mask(n, n)
    h = rms_norm(x + multihead_attention(x, p["Wq"], p["Wk"], p["Wv"], p["Wo"], num_heads, m), p["g1"])
    return rms_norm(h + swiglu(h, p["W1"], p["W2"], p["W3"]), p["g2"])
"""),
        ("causal mask forgotten", """
import numpy as np
def transformer_block(x, p, num_heads):
    h = x + multihead_attention(rms_norm(x, p["g1"]), p["Wq"], p["Wk"], p["Wv"], p["Wo"], num_heads)
    return h + swiglu(rms_norm(h, p["g2"]), p["W1"], p["W2"], p["W3"])
"""),
        ("second residual branches from x", """
import numpy as np
def transformer_block(x, p, num_heads):
    n = np.asarray(x).shape[1]
    m = causal_mask(n, n)
    h = x + multihead_attention(rms_norm(x, p["g1"]), p["Wq"], p["Wk"], p["Wv"], p["Wo"], num_heads, m)
    return x + swiglu(rms_norm(h, p["g2"]), p["W1"], p["W2"], p["W3"])
"""),
    ],
    "cross-entropy": [
        ("log of a naive softmax", """
import numpy as np
def cross_entropy(logits, targets):
    logits = np.asarray(logits, dtype=np.float64)
    e = np.exp(logits)
    probs = e / e.sum(axis=-1, keepdims=True)
    picked = np.take_along_axis(probs, np.asarray(targets, dtype=np.int64)[..., None], axis=-1)
    return float(np.mean(-np.log(picked)))
"""),
        ("sum instead of mean", """
import numpy as np
def cross_entropy(logits, targets):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(z).sum(axis=-1))
    picked = np.take_along_axis(z, np.asarray(targets, dtype=np.int64)[..., None], axis=-1)[..., 0]
    return float(np.sum(log_z - picked))
"""),
        ("logsumexp without the shift", """
import numpy as np
def cross_entropy(logits, targets):
    z = np.asarray(logits, dtype=np.float64)
    log_z = np.log(np.exp(z).sum(axis=-1))
    picked = np.take_along_axis(z, np.asarray(targets, dtype=np.int64)[..., None], axis=-1)[..., 0]
    return float(np.mean(log_z - picked))
"""),
    ],
    "adamw": [
        ("weight decay folded into the gradient", """
import numpy as np
def adamw_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    param = np.asarray(param, dtype=np.float64)
    g = np.asarray(grad, dtype=np.float64) + weight_decay * param
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g**2
    mh, vh = m / (1 - beta1**t), v / (1 - beta2**t)
    return param - lr * mh / (np.sqrt(vh) + eps), m, v
"""),
        ("no bias correction", """
import numpy as np
def adamw_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    param = np.asarray(param, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    return param - lr * m / (np.sqrt(v) + eps) - lr * weight_decay * param, m, v
"""),
        ("bias-corrected moments stored back", """
import numpy as np
def adamw_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    param = np.asarray(param, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    m = (beta1 * m + (1 - beta1) * grad) / (1 - beta1**t)
    v = (beta2 * v + (1 - beta2) * grad**2) / (1 - beta2**t)
    return param - lr * m / (np.sqrt(v) + eps) - lr * weight_decay * param, m, v
"""),
        ("state mutated in place", """
import numpy as np
def adamw_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    grad = np.asarray(grad, dtype=np.float64)
    m *= beta1
    m += (1 - beta1) * grad
    v *= beta2
    v += (1 - beta2) * grad**2
    mh, vh = m / (1 - beta1**t), v / (1 - beta2**t)
    param -= lr * mh / (np.sqrt(vh) + eps) + lr * weight_decay * param
    return param, m, v
"""),
    ],
    "grad-clip": [
        ("per-tensor norms instead of one global norm", """
import numpy as np
def clip_grad_norm(grads, max_norm, eps=1e-6):
    out = []
    for g in grads:
        g = np.asarray(g, dtype=np.float64)
        n = np.sqrt((g**2).sum())
        out.append(g * (max_norm / (n + eps)) if n > max_norm else g.copy())
    return out
"""),
        ("always rescales, even under the cap", """
import numpy as np
def clip_grad_norm(grads, max_norm, eps=1e-6):
    grads = [np.asarray(g, dtype=np.float64) for g in grads]
    total = np.sqrt(sum((g**2).sum() for g in grads))
    return [g * (max_norm / (total + eps)) for g in grads]
"""),
    ],
    "lr-cosine": [
        ("cosine never clamped after cosine_iters", """
import numpy as np
def lr_cosine_schedule(t, lr_max, lr_min, warmup_iters, cosine_iters):
    if t < warmup_iters:
        return float(lr_max * t / warmup_iters)
    progress = (t - warmup_iters) / (cosine_iters - warmup_iters)
    return float(lr_min + 0.5 * (1 + np.cos(np.pi * progress)) * (lr_max - lr_min))
"""),
        ("cosine progress ignores the warmup offset", """
import numpy as np
def lr_cosine_schedule(t, lr_max, lr_min, warmup_iters, cosine_iters):
    if t < warmup_iters:
        return float(lr_max * t / warmup_iters)
    if t <= cosine_iters:
        return float(lr_min + 0.5 * (1 + np.cos(np.pi * t / cosine_iters)) * (lr_max - lr_min))
    return float(lr_min)
"""),
        ("linear decay instead of cosine", """
import numpy as np
def lr_cosine_schedule(t, lr_max, lr_min, warmup_iters, cosine_iters):
    if t < warmup_iters:
        return float(lr_max * t / warmup_iters)
    if t <= cosine_iters:
        frac = (t - warmup_iters) / (cosine_iters - warmup_iters)
        return float(lr_max + frac * (lr_min - lr_max))
    return float(lr_min)
"""),
    ],
    "top-p": [
        ("cumsum <= p (drops the crossing token)", """
import numpy as np
def top_p_filter(probs, p):
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-probs, axis=-1)
    ordered = np.take_along_axis(probs, order, axis=-1)
    keep_sorted = np.cumsum(ordered, axis=-1) <= p
    keep = np.zeros_like(probs, dtype=bool)
    np.put_along_axis(keep, order, keep_sorted, axis=-1)
    kept = np.where(keep, probs, 0.0)
    return kept / kept.sum(axis=-1, keepdims=True)
"""),
        ("no renormalisation", """
import numpy as np
def top_p_filter(probs, p):
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-probs, axis=-1)
    ordered = np.take_along_axis(probs, order, axis=-1)
    prefix = np.cumsum(ordered, axis=-1) - ordered
    keep = np.zeros_like(probs, dtype=bool)
    np.put_along_axis(keep, order, prefix < p, axis=-1)
    return np.where(keep, probs, 0.0)
"""),
        ("returns the row in sorted order", """
import numpy as np
def top_p_filter(probs, p):
    probs = np.asarray(probs, dtype=np.float64)
    ordered = -np.sort(-probs, axis=-1)
    prefix = np.cumsum(ordered, axis=-1) - ordered
    kept = np.where(prefix < p, ordered, 0.0)
    return kept / kept.sum(axis=-1, keepdims=True)
"""),
    ],
}


def read(path):
    return path.read_text(encoding="utf-8")


def run_with_code(comp, code):
    ns = {}
    exec(compile(read(DATA / "harness.py"), "harness.py", "exec"), ns)
    run_source = ns["run_source"]
    cdir = DATA / comp["id"]
    if comp.get("preamble"):
        run_source(read(cdir / "preamble.py"), "preamble.py")
    run_source(code, "your_code.py")
    run_source(read(cdir / "tests.py"), "tests.py")
    run_source(read(DATA / "runner.py"), "runner.py")
    return json.loads(ns["RESULT"])


def main(argv):
    components = json.loads(read(DATA / "components.json"))["components"]
    wanted = set(argv[1:])
    if wanted:
        components = [c for c in components if c["id"] in wanted]

    escapes = 0
    for comp in components:
        variants = WRONG.get(comp["id"], [])
        if not variants:
            print(f"WARN {comp['id']}: no wrong variants registered")
            continue
        caught = 0
        for label, code in variants:
            try:
                results = run_with_code(comp, code)
                passed = bool(results) and all(r["ok"] for r in results)
            except Exception:
                passed = False  # crashing at definition time counts as caught
            if passed:
                print(f'ESCAPE {comp["id"]}: "{label}" passes every test')
                escapes += 1
            else:
                caught += 1
        print(f"ok   {comp['id']:<18} {caught}/{len(variants)} wrong variants caught")

    print(f"\n{escapes} escape(s)")
    return 1 if escapes else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

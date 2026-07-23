---
title: Memory Profiling in Forward and Backward Propagation
date: 2026-07-23
tags: [Systems, Memory Profiling]
summary: Where GPU memory actually goes during a training step, and how to measure it.
---

In this article, we will firstly learn a standard transformer architecture and then analyze its memory profiling during the forward and backward propagation. We will find how the profiling is related to the input size, the model size, and other related design choices. This is very helpful for understanding advanced topics such as model parallelism.

## Standard Transformer LM

Stanford [CS336, *Language Modeling from Scratch*](https://cs336.stanford.edu/), is a highly recommended course in the community. Its assignment 1 introduces a standard transformer LM, which contains classical LM design choices. Let's understand this architecture by analysing the data flow. 

1. At the first step, we have a batch of token sequences in hand (shape $[b, s]$). They will be processed by an embedding layer, which maps the index to a latent state. The shape of the output $X$ is $[b, s, h]$. The memory counts from here. 

2. Now the data flows to the first Transformer block, including a multi-head self-attention layer and a SwiGLU feed-forward network. 

    - In the multi-head self-attention layer, the input $X$ from the previous layer is normalised by RMSNorm and then mapped to a query $Q = X W_Q$, where the shape of $W_Q$ is $[a, h, d_q]$ and the shape of the query $Q$ is $[b, s, a, d_q]$. I strongly recommend that the reader think of this matmul as an Einstein operation:
        ```python
            q = einops.einsum(x, wq, "b s h, a h dq -> b s a dq")
        ```
        Similarly, we have the key $K = X W_K$, and the value $V = X W_V$, where the shape of $W_K$ is $[a, h, d_k]$ and the shape of $W_V$ is $[a, h, d_v]$. The resulting key $K$ and value $V$ have shapes of $[b, s, a, d_k]$ and $[b, s, a, d_v]$, respectively. 

        There are two things we need to remember: (1) $d_q = d_k$ as there will be an inner product between $K$ and $Q$, and (2) the community usually sets $ad_v = h$ so that the output projection $W_O$ is square

        Next, the RoPE-rotated $Q$ and $K$ are used to compute the score matrix: 
        ```python
            scores = einops.einsum(q, k, "b sq a d, b sk a d -> b a sq sk") / math.sqrt(q.shape[-1])
        ```
        The shape of the resulting score matrix is $[b, a, s, s]$. After applying the causal mask and the SoftMax on the last dimension, we have the attention weights $P$ of the same shape.

        We multiply the attention weights with the values:
        ```python
            o = einops.einsum(p, v, "b a sq sk, b sk a dv -> b a sq dv")
        ```
        The output has a shape of $[b, a, s, d_v]$.

        The last step of the self-attention layer is to concatenate the head outputs along the feature axis, and project the result back to the model's hidden dimension $h$. We use a linear layer $W_O$ with shape $[h, h]$:
        ```python
            output = einops.rearrange(o, "b a s dv -> b s (a dv)")
            output = einops.einsum(output, wo, "b s hv, hv h -> b s h")
        ```
        The output of shape $[b, s, h]$ is added to the layer input, constructing a residual structure, and then sent to the SwiGLU layer.

    - The SwiGLU layer applies another RMSNorm to the input and sends the result to three feed-forward networks $W_1$, $W_2$, and $W_3$:
      $$
      \texttt{FFN}(X) = \texttt{SwiGLU}(X, W_1, W_2, W_3) = (\texttt{SiLU}(X W_1) \odot (X W_3)) W_2
      $$
      $W_1$ and $W_3$ share the same shape $[h, d_{ff}]$, whereas the shape of $W_2$ is $[d_{ff}, h]$. This output will then be added to the layer input. 
    
    - This transformer block has the same input and output size; the data need to pass $L$ transformer blocks. 

3. The transformer blocks produce hidden states $X$ of shape $[b, s, h]$. After the final RMSNorm layer, it will be projected to the logits $Z = X W_{vocab}$, where the shape of $W_{vocab}$ is $[h, n_{vocab}]$, and the shape of $Z$ is $[b, s, n_{vocab}]$. After applying SoftMax, we will know the probability distribution over the vocabulary at every position.

## Forward Propagation



## Backward Propagation

## Memory Snapshot in `PyTorch`

## References

1. Percy Liang and Tatsunori Hashimoto. *CS336: Language Modeling from Scratch*. Stanford University, 2025. [cs336.stanford.edu](https://cs336.stanford.edu/)

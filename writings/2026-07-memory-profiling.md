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

2. Now the data flows to the first Transformer block, including a multi-head self-attention layer and a GeLU feed-forward network. 

    - In the multi-head self-attention layer, the input $X$ from the previous layer is normalised by RMSNorm and then mapped to a query $Q = X W_Q$, where the shape of $W_Q$ is $[a, h, d_q]$ and the shape of the query $Q$ is $[b, s, a, d_q]$. I strongly recommend that the reader think of this matmul as an Einstein operation:
        ```python
            q = einops.einsum(x, wq, "b s h, a h dq -> b s a dq")
        ```
        Similarly, we have the key $K = X W_K$, and the value $V = X W_V$, where the shape of $W_K$ is $[a, h, d_k]$ and the shape of $W_V$ is $[a, h, d_v]$. The resulting key $K$ and value $V$ have shapes of $[b, s, a, d_k]$ and $[b, s, a, d_v]$, respectively. 

        There are two things we need to remember: (1) $d_q = d_k$ as there will be an inner product between $K$ and $Q$, and (2) the community usually sets $ad_v = h$ so that the later output projection $W_O$ is square.

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
        The output of shape $[b, s, h]$ is added to the layer input, constructing a residual structure, and then sent to the GeLU layer.

    - The GeLU layer applies another RMSNorm to the input and sends the result to two feed-forward networks $W_1$ and $W_2$:
      $$
      \texttt{FFN}(X) = \texttt{GeLU}(X W_1) W_2
      $$
      The shape of $W_1$ is $[h, d_{ff}]$, whereas the shape of $W_2$ is $[d_{ff}, h]$. This output will then be added to the layer input. 
    
    - This transformer block has the same input and output size; the data need to pass $L$ transformer blocks. 

3. The transformer blocks produce hidden states $X$ of shape $[b, s, h]$. After the final RMSNorm layer, it will be projected to the logits $Z = X W_{vocab}$, where the shape of $W_{vocab}$ is $[h, n_{vocab}]$, and the shape of $Z$ is $[b, s, n_{vocab}]$. After applying SoftMax, we will know the probability distribution over the vocabulary at every position.

## Model Parameters

## Propagation Activations

There is an equation [Korthikanti et al., 2023] we should keep in mind:

$$
bsh \left( 34 + 5\frac{as}{h} \right).
$$
This equation counts the bytes used to store the propagation activations. In this case, we use the BF16 or FP16 (2 bytes) to store each activation. The first term $34bsh$ comes from linear intermediates, such as $Q$, $K$, and $V$, whereas the second term $5bs^2a$, the only term that increases quadratically, comes from the intermediates related to the score matrix. 

Let's count the activations in the self-attention layers and the feed-forward networks. 

### Activations in feed-forward networks

The first step in a feed-forward network is the RMSNorm. When we derive the gradient of the RMSNorm (it needs some effort), we know that we need to store two kinds of activations: the root mean square in the denominator $\sqrt{\frac{1}{h}\sum^h_{i=1} X^2_i + \epsilon}$, and the input $X$. The first one is a constant for each token in the sequence batch, and thus has $bs$ activations or $2bs$ bytes to store. The second one is the latent states after the self-attention layer, and thus has $bsh$ activations and needs $2bsh$ bytes to store. Often, we omit the memory used for the root mean square because of two reasons: (1) it has a smaller scale than other activation counts, and (2) we do not need to save it as it can be re-computed during back-propagation. 

The next ones are the input of the GeLU layer, the activations from $X W_1$, and the activations from $\texttt{GeLU}(X W_1)$ (the output of the GeLU layer does not need to be stored, as the backward pass of $W_2$ only requires its input). They need $2bsh + 2bsd_{ff} + 2bsd_{ff}$ bytes to store. In most cases, we have $d_{ff} = 4 h$, and thus we need to allocate $18bsh$ bytes for the feed-forward activations. In addition, classical implementations such as Megatron-LM and GPT-2 incorporate dropout here and they need extra space $bsh$ to store binary numbers, indicating the dimensions that stop gradients. 

| Layers       | Activations | Bytes |
| :---:        | :---: | :---: |
| FFN Layer    | RMSNorm - input | 2bsh |
|              | FFN - input     | 2bsh |
|              | FFN - $W_1$     | 8bsh |
|              | FFN - GeLU      | 8bsh |
|              | FFN - dropout   | 1bsh |
|**FFN SUM**   |                 | 21bsh|

### Activations in self-attention layers



## Memory Snapshot in `PyTorch`

## References

1. Percy Liang and Tatsunori Hashimoto. *CS336: Language Modeling from Scratch*. Stanford University, 2025. [cs336.stanford.edu](https://cs336.stanford.edu/)

2. Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. *Reducing Activation Recomputation in Large Transformer Models*. In *Proceedings of Machine Learning and Systems (MLSys)*, 2023. [arxiv.org/abs/2205.05198](https://arxiv.org/abs/2205.05198)

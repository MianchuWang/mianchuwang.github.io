// CS336 Lecture 10 — Inference
// 20 essential questions curated from the basics and advances sets.
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture10"] = {
  title: "Lecture 10 — Inference",
  questions: [
    {
      type: "single",
      question: "What is the fundamental trade-off between latency and throughput in LLM serving?",
      options: [
        "Larger batch sizes improve throughput but increase per-request latency",
        "Larger batch sizes improve both latency and throughput without limit",
        "Throughput can only be improved by making each individual request faster",
        "Latency and throughput are independent, so there is no trade-off"
      ],
      correct: [0],
      explanation: "Batching lets one read of the model weights be shared across many requests, boosting hardware utilization and throughput. But each individual request now shares compute with others (and may wait to be scheduled), so its latency increases. Serving systems tune batch size to balance the two."
    },
    {
      type: "single",
      question: "Inference has two phases: prefill and decode. What characterizes the PREFILL phase?",
      options: [
        "All prompt tokens are processed in parallel, so it is typically compute-bound",
        "Tokens are produced one at a time, so it is typically memory-bound",
        "It reads the KV cache built during decode but never writes new entries",
        "It offloads the prompt to the CPU so the GPU is free for decoding"
      ],
      correct: [0],
      explanation: "Prefill processes the whole prompt at once — like a training forward pass — with large matmuls that saturate compute (compute-bound). It also WRITES the KV cache for all prompt tokens. Decode is the phase that generates one token at a time and is typically memory-bound."
    },
    {
      type: "single",
      question: "Why is the DECODE (generation) phase typically memory-bound rather than compute-bound?",
      options: [
        "Each step feeds one new token, so every weight is read from HBM for only a few FLOPs",
        "The KV cache must be rebuilt from scratch each step, wasting most of the compute budget",
        "The softmax dominates the runtime because it cannot run on the GPU's tensor cores",
        "Decode runs in higher precision than prefill, doubling the arithmetic cost per token"
      ],
      correct: [0],
      explanation: "At each decode step you multiply a single token's activation vector by the weight matrices. You still must move ALL the model weights (and the KV cache) from HBM into compute units, but you do only ~2 FLOPs per parameter loaded. The time is dominated by data movement, not arithmetic — the definition of memory-bound."
    },
    {
      type: "single",
      question: "What is 'arithmetic intensity'?",
      options: [
        "FLOPs performed per byte of data transferred from memory",
        "FLOPs performed per second at the accelerator's peak clock rate",
        "Bytes of HBM consumed per parameter of the model",
        "Matrix multiplications executed per transformer layer"
      ],
      correct: [0],
      explanation: "Arithmetic intensity $= \\frac{\\text{total FLOPs}}{\\text{total bytes moved}}$. Comparing a workload's intensity to the hardware's ratio of peak FLOP/s to memory bandwidth tells you whether the workload is compute-bound (high intensity) or memory-bound (low intensity)."
    },
    {
      type: "single",
      question: "An H100 delivers roughly 989 TFLOP/s (dense bf16) with about 3.35 TB/s of HBM bandwidth. What is its approximate accelerator intensity (FLOPs per byte), and what does it mean?",
      options: [
        "About 295 — workloads with arithmetic intensity below ~295 FLOPs/byte will be memory-bound on this GPU",
        "About 295 — workloads with arithmetic intensity below ~295 FLOPs/byte will be compute-bound on this GPU",
        "About 3.4 — the bandwidth-to-clock ratio, so nearly every workload ends up compute-bound",
        "About 989 — intensity equals the TFLOP rating, so only tiny workloads are memory-bound"
      ],
      correct: [0],
      explanation: "$\\frac{989 \\times 10^{12}}{3.35 \\times 10^{12}} \\approx 295$ FLOPs per byte. If your computation does fewer than ~295 FLOPs for every byte it moves, the memory system is the bottleneck (memory-bound); above that, compute is the bottleneck. Decode steps at small batch sizes have intensity near 1 — far below 295."
    },
    {
      type: "single",
      question: "For a matmul of $X$ ($B \\times D$) with $W$ ($D \\times F$), where $D$ and $F$ are much larger than $B$, the arithmetic intensity is approximately equal to what?",
      options: [
        "$B$, the number of rows (tokens) being processed",
        "$D$, the hidden dimension of the model",
        "$F$, the up-projection dimension of the MLP",
        "$D \\times F$, the parameter count of the weight matrix"
      ],
      correct: [0],
      explanation: "FLOPs $= 2BDF$; bytes moved $\\approx 2(BD + DF + BF)$, dominated by the weight-matrix term when $D, F \\gg B$. So intensity $\\approx \\frac{2BDF}{2DF} = B$. This is the key result of the lecture: intensity scales with the number of tokens processed per weight read — which is why batching matters."
    },
    {
      type: "single",
      question: "Why does increasing the batch size NOT fix the memory-bound nature of the ATTENTION computation during decoding?",
      options: [
        "Each sequence has its own KV cache, so attention reads scale linearly with batch size",
        "Attention is already compute-bound during decoding, so larger batches only add scheduling overhead",
        "The softmax cannot be batched across sequences",
        "Larger batches shorten each sequence's KV cache, reducing the useful FLOPs available"
      ],
      correct: [0],
      explanation: "Batching helps the MLP because ALL sequences share the same weight matrices — one read serves many tokens. But KV caches are per-sequence: doubling the batch doubles both attention FLOPs and KV bytes read. The lecture derives attention intensity $\\approx \\frac{S \\cdot T}{S + T}$ for cache length $S$ and $T$ new tokens — during generation $T = 1$, giving intensity below 1 with NO dependence on $B$. This makes the KV cache the central target of inference optimizations."
    },
    {
      type: "single",
      question: "Why is the KV cache needed at all during autoregressive generation?",
      options: [
        "Without it, generating each token recomputes the whole prefix, for $O(T^3)$ total instead of $O(T^2)$",
        "It stores the per-token gradients that generation needs for backpropagation through time",
        "It applies the causal mask that prevents the model from attending to future tokens during decoding",
        "It compresses past activations into a low-rank latent so they occupy fewer bytes"
      ],
      correct: [0],
      explanation: "Attention at step $t$ needs $K$ and $V$ for all previous tokens. One forward pass over a length-$T$ sequence costs $O(T^2)$ in attention, so naively re-running the model for each of $T$ generated tokens costs $O(T^3)$ total. Caching K/V makes each step incremental (just the new token's projections plus attention over the cache), bringing total generation back to $O(T^2)$."
    },
    {
      type: "single",
      question: "What is the key idea of Multi-head Latent Attention (MLA), used in DeepSeek v2?",
      options: [
        "Project each token's keys and values down to a small latent vector (e.g., 16384 to 512 dims) and cache only that",
        "Share one set of keys and values across a group of query heads (e.g., 8 KV heads serving 64 query heads)",
        "Restrict each head to a local window so the KV cache stops growing with sequence length",
        "Share the same keys and values across adjacent transformer layers instead of across heads"
      ],
      correct: [0],
      explanation: "MLA projects each key/value from $N \\cdot H = 16384$ dimensions down to a $C = 512$-dimensional latent, slashing the KV cache. One wrinkle: the compression is incompatible with RoPE, so DeepSeek v2 adds 64 dedicated RoPE dimensions ($512 + 64 = 576$ cached dims per token). Accuracy is competitive with — even slightly better than — full MHA. The distractors describe GQA, local attention, and cross-layer attention (CLA), the lecture's other KV-shrinking techniques."
    },
    {
      type: "single",
      question: "In practice, what usually limits the maximum batch size (concurrency) a serving GPU can sustain?",
      options: [
        "GPU memory: weights are fixed, so the growing KV cache decides how many sequences fit",
        "CPU scheduling: the host cannot dispatch more than a few dozen concurrent requests",
        "Compute: attention FLOPs grow quadratically with the number of sequences in the batch",
        "Network bandwidth between the serving GPU node and its downstream clients"
      ],
      correct: [0],
      explanation: "Weights occupy a fixed chunk of HBM; everything left goes to KV caches, which grow linearly in batch size and sequence length. Since larger batches are exactly what's needed for throughput, KV memory is the binding constraint — motivating KV-shrinking techniques (GQA, MLA, quantized KV, paging)."
    },
    {
      type: "single",
      question: "What is the key idea of CONTINUOUS batching (e.g., Orca)?",
      options: [
        "Iteration-level scheduling: finished sequences leave and new requests join after every decode step",
        "Padding every sequence in the batch to the same maximum length so the tensors stay rectangular",
        "Grouping requests that share a common prefix so their prefill can be computed only once",
        "Running each request's prefill and decode back-to-back on a dedicated stream to cut its latency"
      ],
      correct: [0],
      explanation: "Iteration-level scheduling means the batch composition changes every step: slots freed by finished sequences are refilled right away, keeping the GPU busy and reducing queueing delay. Orca pairs this with SELECTIVE batching to handle ragged (different-length) sequences: attention is computed per sequence, while all non-attention ops (MLP, projections) run on the tokens concatenated into one [sum-of-lengths x H] tensor. This is now standard in serving systems like vLLM."
    },
    {
      type: "single",
      question: "How does PagedAttention organize the KV cache?",
      options: [
        "In small fixed-size blocks that need not be contiguous, mapped through a per-request block table",
        "In one contiguous pre-allocated region per request, sized for the maximum possible sequence length",
        "Entirely in host CPU RAM, with blocks streamed to the GPU over PCIe on demand each step",
        "In a compressed low-rank form that is expanded on the fly during attention"
      ],
      correct: [0],
      explanation: "Borrowing the OS paging idea: fixed-size KV blocks are allocated on demand, and an indirection table maps each sequence's logical token positions to physical blocks. Waste shrinks to at most one partially-filled block per sequence, so far more sequences fit in memory. Blocks also enable prefix sharing across sequences (system prompts, parallel samples) with copy-on-write."
    },
    {
      type: "multi",
      question: "Which techniques are EXACT — they change how inference is executed but leave the model's output distribution unchanged? (Select all that apply)",
      options: [
        "Continuous batching",
        "PagedAttention",
        "Speculative decoding (with correct rejection sampling)",
        "Quantizing the weights to int4 with group-wise scales"
      ],
      correct: [0, 1, 2],
      explanation: "Batching/scheduling and KV memory layout don't change the math, and speculative decoding's modified rejection sampling provably preserves the target model's distribution. Quantization is LOSSY — it changes the weights and therefore the outputs (hopefully negligibly). The lecture's framing: systems tricks are exact; model modifications (quantization, pruning, architecture changes) are lossy."
    },
    {
      type: "single",
      question: "What is the core idea of speculative decoding?",
      options: [
        "A cheap draft model proposes several tokens; the target model verifies them all in one parallel pass",
        "The target model samples several candidate continuations and a reward model selects the best one",
        "The model emits every other token and a small model fills in the skipped gaps afterward",
        "A small model handles the whole prefill so the large model only performs the generation"
      ],
      correct: [0],
      explanation: "The insight: for a memory-bound target model, checking $K$ proposed tokens in one forward pass costs about the same as generating ONE token (same weight reads, like a mini-prefill). So let a small fast model draft $K$ tokens, then verify them in parallel — accepted tokens come almost for free."
    },
    {
      type: "single",
      question: "Why is verifying $K$ draft tokens with the target model roughly as cheap as generating a single token?",
      options: [
        "Decoding is memory-bound: the weight reads cost the same whether the pass covers 1 token or $K$",
        "The target model switches to a smaller vocabulary during the verification pass",
        "Verification runs only the final layer of the target model instead of the full stack",
        "The draft model shares its KV cache with the target model, so verification skips attention"
      ],
      correct: [0],
      explanation: "Processing $K$ tokens at once is a small parallel (prefill-like) computation. Since the bottleneck is streaming the weights from HBM — not FLOPs — the extra arithmetic for $K$ tokens is essentially free while intensity stays below the hardware's compute/memory ratio."
    },
    {
      type: "single",
      question: "In speculative sampling, the draft proposes token $x$ with probability $q(x)$; the target assigns $p(x)$. With what probability is the draft token ACCEPTED?",
      options: [
        "$\\min(1, p(x) / q(x))$",
        "$\\min(1, q(x) / p(x))$",
        "$p(x) \\cdot q(x)$",
        "$1$ whenever $p(x) > 0$"
      ],
      correct: [0],
      explanation: "Accept with probability $\\min(1, p(x)/q(x))$: tokens the target likes at least as much as the draft are always kept; tokens the draft over-proposes are kept proportionally less often. On rejection, resample from the residual distribution $\\mathrm{norm}(\\max(0, p - q))$. Together this yields samples EXACTLY from $p$ — speculative decoding is lossless."
    },
    {
      type: "single",
      question: "In Multi-Query Attention (MQA), what is shared across all query heads?",
      options: [
        "A single key head and a single value head",
        "The query projection weights of every layer",
        "The MLP weights within each transformer block",
        "The positional embeddings across all layers"
      ],
      correct: [0],
      explanation: "MQA keeps many query heads but only ONE K head and ONE V head. The KV cache shrinks by a factor of $n_{heads}$ (e.g., 32x), directly attacking the memory-bound attention bottleneck — at some cost to quality since all queries see the same K/V."
    },
    {
      type: "single",
      question: "How does Grouped-Query Attention (GQA) relate to MHA and MQA?",
      options: [
        "It interpolates: query heads are split into groups, each group sharing one KV head",
        "It inverts MQA: all KV heads are kept while the query heads are merged into one",
        "It is identical to MQA except that it is applied only to the first transformer layer",
        "It groups adjacent tokens into blocks so attention runs over fewer positions"
      ],
      correct: [0],
      explanation: "GQA with $g$ KV-head groups spans the spectrum: $g = n_{heads}$ is standard multi-head attention, $g = 1$ is MQA. Models like Llama-2-70B use e.g. 8 KV heads for 64 query heads — an 8x KV cache reduction with quality nearly matching full MHA. This is the mainstream choice in modern open models."
    },
    {
      type: "single",
      question: "Why does quantization speed up memory-bound inference?",
      options: [
        "Fewer bytes per parameter means fewer bytes streamed from HBM per decode step",
        "Lower-precision logits let the sampler draw tokens with fewer comparisons",
        "Rounding removes near-zero weights, shrinking the effective number of layers",
        "Smaller numbers shorten the sequence-length dimension of the KV cache"
      ],
      correct: [0],
      explanation: "When latency $\\approx \\frac{\\text{bytes}}{\\text{bandwidth}}$, halving bytes per weight (bf16 to int8) roughly halves weight-streaming time; int4 quarters it. It also frees HBM for larger batches or longer contexts. The KV cache can be quantized too, attacking the other big memory consumer."
    },
    {
      type: "multi",
      question: "Final review — which statements correctly summarize the lecture's big picture? (Select all that apply)",
      options: [
        "Generation is dominated by memory bandwidth, so most optimizations reduce bytes moved per token",
        "Systems-level techniques (continuous batching, PagedAttention, speculative decoding) don't change model outputs",
        "Model-level techniques (quantization, GQA/MLA, pruning, new architectures) trade some fidelity for efficiency",
        "Prefill and decode have identical performance characteristics, so no serving system treats them differently"
      ],
      correct: [0, 1, 2],
      explanation: "The unifying lens: arithmetic intensity. Decode is memory-bound, so wins come from moving fewer bytes (quantization, KV-cache reduction) or amortizing reads over more tokens (batching, speculative verification) — exactly split into lossless systems tricks and lossy model changes. Prefill (compute-bound) and decode (memory-bound) are OPPOSITE regimes, which is why systems schedule, chunk, or even disaggregate them differently."
    }
  ]
};

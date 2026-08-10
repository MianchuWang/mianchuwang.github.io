// CS336 Lecture 10 — Inference (Basics)
// 50 questions covering the core content of the lecture.
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture10-basics"] = {
  title: "Lecture 10 — Inference: Basics",
  questions: [
    // ---------- Motivation & metrics ----------
    {
      type: "multi",
      question: "Why does inference efficiency matter so much for modern LLMs? (Select all that apply)",
      options: [
        "Over a model's lifetime, total inference cost typically exceeds the one-time training cost",
        "Inference is used beyond chat: evaluation, batch data processing, and reinforcement learning rollouts all require generation",
        "Test-time compute (long chains of thought, agents) multiplies the number of generated tokens per query",
        "Inference requires computing gradients, which is more expensive than training"
      ],
      correct: [0, 1, 2],
      explanation: "Training is a one-time cost, but inference happens every time anyone uses the model — so lifetime inference cost dominates. Generation also shows up in evals, batch processing, and RL sampling, and test-time compute scaling (reasoning models, agents) further multiplies token counts. Inference does NOT compute gradients — that is exactly why it is cheaper per token than training (roughly 2 FLOPs/param/token vs. 6 for training)."
    },
    {
      type: "single",
      question: "What does Time To First Token (TTFT) measure?",
      options: [
        "The time from sending a request until the first output token is produced",
        "The average time between consecutive generated tokens",
        "The total time to generate the full response",
        "The time to load model weights onto the GPU"
      ],
      correct: [0],
      explanation: "TTFT is the delay before the user sees the first token — dominated by the prefill phase, which must process the entire prompt. It matters most for interactive applications. Time between consecutive tokens is per-token latency (inter-token latency), a different metric."
    },
    {
      type: "single",
      question: "Which metric matters MOST for offline batch data processing (e.g., generating synthetic data at scale)?",
      options: [
        "Throughput (total tokens per second across all requests)",
        "Time to first token",
        "Per-token latency for a single stream",
        "Model load time"
      ],
      correct: [0],
      explanation: "For offline/batch workloads no human is waiting on each token, so total throughput (and hence cost per token) is what matters. Latency and TTFT matter for interactive uses like chatbots and code completion."
    },
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
      question: "Compared to training, why is inference in some ways HARDER to do efficiently, even though it uses fewer FLOPs per token?",
      options: [
        "Generation is sequential — each token depends on the previous one — so you cannot parallelize across time steps of a single sequence",
        "Inference must store optimizer states for every request",
        "Inference requires higher numerical precision than training",
        "Unlike training, inference cannot use batching, so the hardware always runs at batch size 1"
      ],
      correct: [0],
      explanation: "In training, the full sequence is known, so all positions are processed in parallel with big efficient matmuls. In autoregressive generation, token t+1 depends on token t, forcing sequential steps with small (matrix-vector-like) computations that utilize hardware poorly. There are no optimizer states or gradients at inference, and precision is typically lower, not higher."
    },

    // ---------- Prefill vs decode ----------
    {
      type: "single",
      question: "Inference has two phases: prefill and decode. What characterizes the PREFILL phase?",
      options: [
        "All prompt tokens are known and processed in parallel, so it is typically compute-bound",
        "Tokens are generated one at a time, so it is typically memory-bound",
        "It only reads the KV cache but never writes it",
        "It runs on the CPU while decode runs on the GPU"
      ],
      correct: [0],
      explanation: "Prefill processes the whole prompt at once — like a training forward pass — with large matmuls that saturate compute (compute-bound). It also WRITES the KV cache for all prompt tokens. Decode is the phase that generates one token at a time and is typically memory-bound."
    },
    {
      type: "single",
      question: "Why is the DECODE (generation) phase typically memory-bound rather than compute-bound?",
      options: [
        "Each step processes just one new token per sequence, so the arithmetic looks like matrix-vector products: every parameter is read from memory but used for very few FLOPs",
        "The KV cache must be recomputed from scratch at every step, wasting most of the compute budget",
        "The softmax operation dominates the runtime because it cannot use tensor cores",
        "Decode uses higher precision arithmetic than prefill"
      ],
      correct: [0],
      explanation: "At each decode step you multiply a single token's activation vector by the weight matrices. You still must move ALL the model weights (and the KV cache) from HBM into compute units, but you do only ~2 FLOPs per parameter loaded. The time is dominated by data movement, not arithmetic — the definition of memory-bound."
    },
    {
      type: "single",
      question: "During generation, roughly what determines the time per decoding step for a memory-bound workload?",
      options: [
        "(Bytes of parameters + bytes of KV cache that must be read) ÷ memory bandwidth",
        "Total FLOPs ÷ peak FLOP/s of the accelerator",
        "The number of CUDA kernels launched",
        "Vocabulary size ÷ clock speed"
      ],
      correct: [0],
      explanation: "When memory-bound, arithmetic is effectively free and each step's latency is the time to stream the needed bytes (all model weights plus the sequence's KV cache) through memory bandwidth. The FLOPs-based estimate would apply only if the workload were compute-bound."
    },
    // ---------- Arithmetic intensity ----------
    {
      type: "single",
      question: "What is 'arithmetic intensity'?",
      options: [
        "FLOPs performed per byte of data transferred from memory",
        "FLOPs performed per second by the accelerator",
        "Bytes of memory used per model parameter",
        "The number of matrix multiplications per layer"
      ],
      correct: [0],
      explanation: "Arithmetic intensity = total FLOPs ÷ total bytes moved. Comparing a workload's intensity to the hardware's ratio of peak FLOP/s to memory bandwidth tells you whether the workload is compute-bound (high intensity) or memory-bound (low intensity)."
    },
    {
      type: "single",
      question: "An H100 delivers roughly 989 TFLOP/s (dense bf16) with about 3.35 TB/s of HBM bandwidth. What is its approximate accelerator intensity (FLOPs per byte), and what does it mean?",
      options: [
        "About 295 — workloads with arithmetic intensity below ~295 FLOPs/byte will be memory-bound on this GPU",
        "About 295 — workloads with intensity below ~295 will be compute-bound",
        "About 3.4 — almost every workload is compute-bound",
        "About 989 — the intensity equals the TFLOP rating"
      ],
      correct: [0],
      explanation: "989e12 / 3.35e12 ≈ 295 FLOPs per byte. If your computation does fewer than ~295 FLOPs for every byte it moves, the memory system is the bottleneck (memory-bound); above that, compute is the bottleneck. Decode steps at small batch sizes have intensity near 1 — far below 295."
    },
    {
      type: "single",
      question: "For a matmul X(B×D) @ W(D×F) with large D and F, the arithmetic intensity is approximately equal to what?",
      options: [
        "B, the number of rows (tokens) being processed",
        "D, the model dimension",
        "F, the feed-forward dimension",
        "D × F, the parameter count of the matrix"
      ],
      correct: [0],
      explanation: "FLOPs = 2BDF; bytes moved ≈ 2(BD + DF + BF), which is dominated by the weight matrix DF when D,F ≫ B. So intensity ≈ 2BDF / 2DF = B. This is the key result of the lecture: intensity scales with the number of tokens processed per weight read — which is why batching matters."
    },
    {
      type: "single",
      question: "What is the arithmetic intensity of a matrix–VECTOR multiply (i.e., decoding with batch size 1), and what does it imply?",
      options: [
        "About 1 FLOP/byte — hopelessly memory-bound; the GPU's compute units sit mostly idle",
        "About 300 FLOPs/byte — right at the compute/memory boundary",
        "It is infinite, since no memory needs to be read",
        "About 100 FLOPs/byte — mildly compute-bound"
      ],
      correct: [0],
      explanation: "With B = 1, intensity ≈ B = 1: for every ~2 bytes of weight loaded you do ~2 FLOPs. On hardware needing ~295 FLOPs/byte to saturate compute, utilization is under 1%. This is why single-stream generation is dominated entirely by memory bandwidth."
    },
    {
      type: "single",
      question: "How can the MLP (feed-forward) layers be made compute-bound during generation?",
      options: [
        "Increase the batch size — process many sequences' tokens at once so each weight read is amortized over more FLOPs",
        "Increase the sequence length of a single request",
        "Use a larger vocabulary",
        "Switch from bf16 to fp32 weights"
      ],
      correct: [0],
      explanation: "MLP intensity ≈ number of tokens in the batch, and during decode each sequence contributes one token per step — so you need on the order of B ≈ 300 concurrent sequences to reach the H100's ~295 FLOPs/byte threshold. Longer sequences don't help the MLP (still one new token per step), and fp32 makes memory traffic worse."
    },
    {
      type: "single",
      question: "Why does increasing the batch size NOT fix the memory-bound nature of the ATTENTION computation during decoding?",
      options: [
        "Each sequence has its own KV cache, so attention reads scale linearly with batch size — there is no sharing to amortize, and intensity stays roughly constant",
        "Attention is already compute-bound during decoding, so larger batches only add scheduling overhead",
        "The softmax cannot be batched across sequences",
        "Larger batches shorten each sequence's KV cache, reducing the useful FLOPs available"
      ],
      correct: [0],
      explanation: "Batching helps the MLP because ALL sequences share the same weight matrices — one read serves many tokens. But KV caches are per-sequence: doubling the batch doubles both attention FLOPs and KV bytes read. The lecture derives attention intensity ≈ S·T/(S+T) for cache length S and T new tokens — during generation T=1, giving intensity below 1 with NO dependence on B. This makes the KV cache the central target of inference optimizations."
    },
    // ---------- KV cache ----------
    {
      type: "single",
      question: "What exactly is stored in the KV cache?",
      options: [
        "The key and value vectors of every previous token, for each layer and each KV head",
        "The model's weight matrices in compressed form",
        "The logits of all previously generated tokens",
        "The query vectors of all previous tokens"
      ],
      correct: [0],
      explanation: "To attend over the past without recomputing it, each generated token's keys and values are cached per layer per KV head. Queries do not need caching — only the CURRENT token's query is ever used; past queries are never needed again."
    },
    {
      type: "single",
      question: "Why is the KV cache needed at all during autoregressive generation?",
      options: [
        "Without it, each new token would require recomputing keys and values for the entire prefix, raising the total cost of generating T tokens from O(T²) to O(T³)",
        "It stores gradients needed for backpropagation",
        "It prevents the model from attending to future tokens",
        "It reduces the memory footprint of generation by compressing past activations into fewer bytes"
      ],
      correct: [0],
      explanation: "Attention at step t needs K and V for all previous tokens. One forward pass over a length-T sequence costs O(T²) in attention, so naively re-running the model for each of T generated tokens costs O(T³) total. Caching K/V makes each step incremental (just the new token's projections plus attention over the cache), bringing total generation back to O(T²)."
    },
    {
      type: "single",
      question: "Per token, the KV cache size in bytes equals: 2 × n_layers × n_kv_heads × head_dim × bytes_per_element. What does the leading factor of 2 account for?",
      options: [
        "Storing both keys AND values",
        "Storing the token in two precisions",
        "Double-buffering for speed",
        "The forward and backward passes"
      ],
      correct: [0],
      explanation: "One copy of the vector for K and one for V at every layer and KV head. Example: 32 layers × 32 heads × head_dim 128 in bf16 (2 bytes) gives 2×32×32×128×2 ≈ 0.5 MB per token — so a 4K-token context costs ~2 GB per sequence."
    },
    {
      type: "multi",
      question: "Which factors increase total KV cache memory? (Select all that apply)",
      options: [
        "Longer sequence lengths",
        "More concurrent sequences (larger batch)",
        "More layers or more KV heads",
        "A larger vocabulary"
      ],
      correct: [0, 1, 2],
      explanation: "KV cache = batch × seq_len × 2 × layers × kv_heads × head_dim × bytes. Vocabulary size affects the embedding/output matrices, not the KV cache. The batch and sequence-length terms are why long-context, high-concurrency serving is memory-hungry — KV can exceed the weights themselves."
    },
    {
      type: "single",
      question: "In practice, what usually limits the maximum batch size (concurrency) a serving GPU can sustain?",
      options: [
        "GPU memory: weights are fixed, so the growing per-sequence KV cache determines how many sequences fit",
        "The operating system's process limit",
        "The number of CUDA cores",
        "Network bandwidth to the client"
      ],
      correct: [0],
      explanation: "Weights occupy a fixed chunk of HBM; everything left goes to KV caches, which grow linearly in batch size and sequence length. Since larger batches are exactly what's needed for throughput, KV memory is the binding constraint — motivating KV-shrinking techniques (GQA, MLA, quantized KV, paging)."
    },

    // ---------- Continuous batching ----------
    {
      type: "single",
      question: "What is the main problem with naive STATIC batching of generation requests?",
      options: [
        "Sequences finish at different times, so short requests sit idle waiting for the longest one, and new requests must wait for the whole batch to complete",
        "It requires each request to have exactly the same prompt",
        "It cannot run on more than one GPU",
        "It makes generation non-deterministic"
      ],
      correct: [0],
      explanation: "In static batching, the batch is formed once and runs until all sequences finish. Completed sequences waste their slots (padding), and arriving requests queue up. Utilization drops badly when output lengths vary — which they always do."
    },
    {
      type: "single",
      question: "What is the key idea of CONTINUOUS batching (e.g., Orca)?",
      options: [
        "Schedule at the granularity of individual iterations: after each decoding step, finished sequences leave the batch and newly arrived requests join immediately",
        "Pad all sequences in the batch to the same maximum length so the tensors stay rectangular",
        "Batch only requests that share the same prompt",
        "Run prefill and decode for each request back-to-back on a dedicated stream to minimize its latency"
      ],
      correct: [0],
      explanation: "Iteration-level scheduling means the batch composition changes every step: slots freed by finished sequences are refilled right away, keeping the GPU busy and reducing queueing delay. Orca pairs this with SELECTIVE batching to handle ragged (different-length) sequences: attention is computed per sequence, while all non-attention ops (MLP, projections) run on the tokens concatenated into one [sum-of-lengths × H] tensor. This is now standard in serving systems like vLLM."
    },
    // ---------- PagedAttention / vLLM ----------
    {
      type: "single",
      question: "What memory problem motivated PagedAttention (vLLM)?",
      options: [
        "Pre-allocating contiguous KV cache buffers for the maximum possible length wastes huge amounts of memory through internal and external fragmentation",
        "The KV cache was stored on disk, which is too slow",
        "Model weights were duplicated once per request",
        "Attention scores overflowed in fp16"
      ],
      correct: [0],
      explanation: "Earlier systems reserved a contiguous max-length region per request. Most requests are far shorter than the max (internal fragmentation), and variable-sized contiguous chunks leave unusable gaps (external fragmentation) — prior systems wasted a large fraction of KV memory, directly limiting batch size and throughput."
    },
    {
      type: "single",
      question: "How does PagedAttention organize the KV cache?",
      options: [
        "In small fixed-size blocks (pages) that need not be contiguous, with a per-request block table mapping logical positions to physical blocks — like virtual memory in an OS",
        "In one contiguous pre-allocated region per request, sized for the maximum possible sequence length",
        "Entirely in CPU RAM, streamed to the GPU on demand",
        "In a compressed low-rank form that is expanded on the fly during attention"
      ],
      correct: [0],
      explanation: "Borrowing the OS paging idea: fixed-size KV blocks are allocated on demand, and an indirection table maps each sequence's logical token positions to physical blocks. Waste shrinks to at most one partially-filled block per sequence, so far more sequences fit in memory."
    },
    {
      type: "single",
      question: "Besides reducing fragmentation, what sharing capability does paging the KV cache enable?",
      options: [
        "Multiple sequences can share the physical KV blocks of a common prefix (e.g., a system prompt or parallel samples), using copy-on-write when they diverge",
        "Two different models can share the same weights",
        "Requests can share sampled tokens to finish faster",
        "The GPU can share memory with the network card"
      ],
      correct: [0],
      explanation: "With block-level indirection, identical prefixes (shared system prompts, beam search, n parallel samples from one prompt) map to the same physical blocks; a block is copied only when a sequence writes divergent tokens into it (copy-on-write). This saves both memory and prefill compute."
    },
    {
      type: "multi",
      question: "Which techniques are EXACT — they change how inference is executed but leave the model's output distribution unchanged? (Select all that apply)",
      options: [
        "Continuous batching",
        "PagedAttention",
        "Speculative decoding (with correct rejection sampling)",
        "Quantizing weights to int4"
      ],
      correct: [0, 1, 2],
      explanation: "Batching/scheduling and KV memory layout don't change the math, and speculative decoding's modified rejection sampling provably preserves the target model's distribution. Quantization is LOSSY — it changes the weights and therefore the outputs (hopefully negligibly). The lecture's framing: systems tricks are exact; model modifications (quantization, pruning, architecture changes) are lossy."
    },

    // ---------- Speculative decoding ----------
    {
      type: "single",
      question: "What is the core idea of speculative decoding?",
      options: [
        "A cheap draft model proposes several tokens sequentially; the expensive target model then verifies them all in a single parallel pass, accepting a prefix of them",
        "The target model generates several candidate continuations in parallel and a reward model picks the best one",
        "The model skips every other token and a small model fills in the gaps afterward",
        "The prompt is prefilled by a small model so the large model only handles generation"
      ],
      correct: [0],
      explanation: "The insight: for a memory-bound target model, checking K proposed tokens in one forward pass costs about the same as generating ONE token (same weight reads, like a mini-prefill). So let a small fast model draft K tokens, then verify them in parallel — accepted tokens come almost for free."
    },
    {
      type: "single",
      question: "Why is verifying K draft tokens with the target model roughly as cheap as generating a single token?",
      options: [
        "Decoding is memory-bound: the dominant cost is reading the weights, which is the same whether the forward pass processes 1 token or K tokens in parallel",
        "The target model uses a smaller vocabulary during verification",
        "Verification skips the attention layers",
        "The draft model shares its KV cache with the target model"
      ],
      correct: [0],
      explanation: "Processing K tokens at once is a small parallel (prefill-like) computation. Since the bottleneck is streaming the weights from HBM — not FLOPs — the extra arithmetic for K tokens is essentially free while intensity is below the hardware's compute/memory ratio."
    },
    {
      type: "single",
      question: "In speculative sampling, the draft proposes token x with probability q(x); the target assigns p(x). With what probability is the draft token ACCEPTED?",
      options: [
        "min(1, p(x) / q(x))",
        "min(1, q(x) / p(x))",
        "p(x) × q(x)",
        "Always accepted if p(x) > 0"
      ],
      correct: [0],
      explanation: "Accept with probability min(1, p(x)/q(x)): tokens the target likes at least as much as the draft are always kept; tokens the draft over-proposes are kept proportionally less often. On rejection, resample from the residual distribution norm(max(0, p − q)). Together this yields samples EXACTLY from p — speculative decoding is lossless."
    },
    {
      type: "single",
      question: "When a draft token is REJECTED in speculative sampling, what happens?",
      options: [
        "A replacement token is sampled from the normalized residual distribution max(0, p − q), and the remaining draft tokens are discarded",
        "The entire response is restarted from scratch",
        "The draft token is kept anyway but flagged",
        "The target model is fine-tuned on the error"
      ],
      correct: [0],
      explanation: "Sampling the replacement from norm(max(0, p(x) − q(x))) is exactly what makes the combined procedure equivalent to sampling from the target distribution p. All draft tokens after the rejection point are thrown away, and drafting resumes from the corrected position."
    },
    {
      type: "single",
      question: "What property makes a GOOD draft model for speculative decoding?",
      options: [
        "Much cheaper per token than the target while matching its distribution closely enough for a high acceptance rate",
        "Being larger than the target model to guarantee quality",
        "Having a completely different vocabulary from the target",
        "Producing deliberately random tokens to explore alternatives"
      ],
      correct: [0],
      explanation: "Speedup ≈ (accepted tokens per verification) balanced against draft cost. You want the draft cheap (e.g., a small model from the same family, or lightweight heads on the target's own hidden states as in Medusa/EAGLE) yet aligned with the target so most proposals are accepted. A mismatched or random draft would be rejected constantly, giving no speedup."
    },
    {
      type: "single",
      question: "In which regime does speculative decoding provide the LEAST benefit?",
      options: [
        "Very large batch sizes where decoding is already close to compute-bound",
        "Single-user, latency-sensitive generation at batch size 1",
        "Long generations from a memory-bound target model",
        "Greedy decoding from a 70B model with a 1B draft"
      ],
      correct: [0],
      explanation: "Speculation exploits idle compute during memory-bound decode. At large batch sizes the GPU is already compute-saturated, so verifying extra draft tokens costs real compute time and the advantage vanishes. It shines at small batch / low latency settings."
    },
    // ---------- MQA / GQA / MLA and attention variants ----------
    {
      type: "single",
      question: "In Multi-Query Attention (MQA), what is shared across all query heads?",
      options: [
        "A single key head and a single value head",
        "The query projections",
        "The MLP weights",
        "The positional embeddings"
      ],
      correct: [0],
      explanation: "MQA keeps many query heads but only ONE K head and ONE V head. The KV cache shrinks by a factor of n_heads (e.g., 32×), directly attacking the memory-bound attention bottleneck — at some cost to quality since all queries see the same K/V."
    },
    {
      type: "single",
      question: "How does Grouped-Query Attention (GQA) relate to MHA and MQA?",
      options: [
        "It interpolates: query heads are split into groups, each group sharing one KV head — fewer KV heads than MHA, more than MQA",
        "It removes query heads and keeps all KV heads",
        "It is identical to MQA but applied only to the first layer",
        "It groups tokens rather than heads"
      ],
      correct: [0],
      explanation: "GQA with g KV-head groups spans the spectrum: g = n_heads is standard multi-head attention, g = 1 is MQA. Models like Llama-2-70B use e.g. 8 KV heads for 64 query heads — an 8× KV cache reduction with quality nearly matching full MHA. This is the mainstream choice in modern open models."
    },
    {
      type: "single",
      question: "By reducing the number of KV heads, what does GQA/MQA do to the arithmetic intensity of attention during decoding?",
      options: [
        "Increases it — the same attention FLOPs are performed per query head while far fewer KV bytes are read, moving attention closer to compute-bound",
        "Decreases it — fewer heads means fewer FLOPs per byte",
        "Leaves it unchanged, since intensity only depends on batch size",
        "Makes it infinite, eliminating memory traffic entirely"
      ],
      correct: [0],
      explanation: "With n query heads sharing each KV head, each loaded K/V byte now serves n query heads' worth of attention arithmetic. KV memory traffic drops by the grouping factor while useful FLOPs stay, raising intensity by roughly that factor — a rare win on the otherwise stubborn attention bottleneck."
    },
    {
      type: "single",
      question: "What is the key idea of Multi-head Latent Attention (MLA), used in DeepSeek-V2/V3?",
      options: [
        "Compress keys and values into a single low-rank latent vector per token; cache only the small latent and expand to full K/V via learned up-projections",
        "Group query heads so that several of them share one full-rank KV head, reducing the number of cached heads",
        "Store the KV cache in int4 precision and dequantize it during attention",
        "Use one attention head for the entire model"
      ],
      correct: [0],
      explanation: "MLA projects each token down to a compact latent (e.g., 512 dims instead of tens of thousands across heads) and caches only that. Full-rank K/V are reconstructed by up-projection when needed (and the up-projections can be absorbed into other matmuls). It cuts KV cache dramatically while preserving many distinct heads — DeepSeek reports quality better than GQA at far smaller cache."
    },
    {
      type: "single",
      question: "Why does MLA need a 'decoupled' RoPE design?",
      options: [
        "RoPE's position-dependent rotation doesn't commute with the low-rank down/up-projections, so a separate small set of dimensions carries positional information via RoPE while the compressed latent stays position-independent",
        "RoPE requires fp32 precision which MLA forbids",
        "RoPE only works with exactly 64 attention heads",
        "MLA removes positions entirely, so RoPE is discarded"
      ],
      correct: [0],
      explanation: "If keys were RoPE-rotated, the rotation (which depends on each token's position) would sit between the latent and the up-projection, preventing the weight-absorption trick that makes MLA efficient. DeepSeek's fix: keep a few extra 'rope' dimensions per key/query that carry position, concatenated with the position-free compressed part."
    },
    {
      type: "multi",
      question: "Which techniques reduce KV cache memory? (Select all that apply)",
      options: [
        "Grouped-Query Attention (fewer KV heads)",
        "Sliding-window / local attention in some layers (bounded cache per layer)",
        "Sharing KV caches across adjacent layers (cross-layer attention, as in Character.AI's stack)",
        "Increasing the model's hidden dimension"
      ],
      correct: [0, 1, 2],
      explanation: "GQA cuts the head dimension of the cache; sliding-window layers cap cached tokens at the window size (e.g., interleaving local:global layers, as in Character.AI or Gemma-style stacks); cross-layer KV sharing divides the layer factor. Increasing hidden size GROWS the cache. These multiply: production stacks combine them for order-of-magnitude reductions."
    },
    {
      type: "single",
      question: "What is the drawback of using ONLY sliding-window (local) attention in every layer?",
      options: [
        "The model loses direct access to context beyond the window, hurting long-range recall, so designs interleave local layers with occasional full/global attention layers",
        "The KV cache grows faster than with full attention",
        "It cannot be parallelized during prefill",
        "It requires a separate GPU per window"
      ],
      correct: [0],
      explanation: "A fixed window bounds the KV cache but limits the receptive field per layer; purely local models struggle with long-range dependencies. Hybrid stacks (e.g., mostly local layers plus periodic global layers, sometimes with cross-layer KV sharing) keep most of the memory savings while retaining long-context ability."
    },

    // ---------- Quantization ----------
    {
      type: "single",
      question: "Why does quantization speed up memory-bound inference?",
      options: [
        "Fewer bytes per parameter means fewer bytes streamed from memory per decode step — and time per token is proportional to bytes moved",
        "Lower precision numbers are easier to sample from",
        "It reduces the number of layers in the model",
        "It shortens the KV cache sequence length"
      ],
      correct: [0],
      explanation: "When latency ≈ bytes/bandwidth, halving bytes per weight (bf16→int8) roughly halves weight-streaming time; int4 quarters it. It also frees HBM for larger batches or longer contexts. The KV cache can be quantized too, attacking the other big memory consumer."
    },
    {
      type: "single",
      question: "How many bytes per parameter do fp32, bf16/fp16, int8/fp8, and int4 use respectively?",
      options: [
        "4, 2, 1, 0.5",
        "8, 4, 2, 1",
        "4, 2, 2, 1",
        "32, 16, 8, 4"
      ],
      correct: [0],
      explanation: "fp32 = 4 bytes, bf16/fp16 = 2, int8/fp8 = 1, int4 = half a byte (two weights per byte). A 70B-parameter model goes from 280 GB (fp32) to 140 (bf16) to 70 (int8) to 35 GB (int4) — the difference between needing multiple GPUs and fitting on one."
    },
    {
      type: "single",
      question: "What problem do OUTLIER features cause when naively quantizing large LLMs to int8, and how did LLM.int8() address it?",
      options: [
        "A few activation dimensions have magnitudes far larger than the rest, wrecking the quantization scale; LLM.int8() computes those outlier dimensions in higher precision while quantizing the rest",
        "Outliers make the model too small to store; the fix is padding",
        "Outliers occur only in the tokenizer and are removed",
        "Outliers cause overflow in the KV cache index"
      ],
      correct: [0],
      explanation: "Beyond ~6B parameters, systematic large-magnitude outlier dimensions emerge in activations. A single per-tensor scale must stretch to cover them, crushing the resolution for normal values. LLM.int8() uses mixed-precision decomposition: outlier columns stay in fp16, everything else runs in int8 — recovering full quality."
    },
    {
      type: "single",
      question: "What does 'activation-aware' mean in AWQ (Activation-aware Weight Quantization)?",
      options: [
        "Weights that are multiplied by large activations matter most, so AWQ identifies these salient channels via activation statistics and protects them with per-channel scaling before quantizing",
        "The activations themselves are quantized instead of the weights",
        "Quantization is only applied while the model is inactive",
        "Activation functions are replaced with linear ones"
      ],
      correct: [0],
      explanation: "AWQ observes that a small fraction (~1%) of weight channels — those seeing large activation magnitudes — dominate quality. Rather than keeping them in fp16 (hardware-unfriendly), it rescales channels so the salient ones lose less precision under int quantization, achieving strong 4-bit weight-only quantization."
    },
    {
      type: "single",
      question: "Which statement about quantization is TRUE?",
      options: [
        "It is a lossy technique: outputs can change, and aggressive low-bit quantization can measurably degrade quality, so the accuracy–efficiency trade-off must be evaluated",
        "It is exact, like PagedAttention, and never changes outputs",
        "It only reduces disk storage, not inference speed, because GPUs compute in fp32 internally",
        "It speeds up prefill more than decode, since prefill is the memory-bound phase"
      ],
      correct: [0],
      explanation: "Rounding weights changes the function the model computes. Well-designed 8-bit (and often 4-bit weight-only) schemes lose little on benchmarks, but degradation grows as bits shrink, and post-training quantization (GPTQ, AWQ) exists precisely to minimize that loss. This contrasts with the exact systems-level techniques."
    },

    // ---------- Pruning & distillation ----------
    {
      type: "single",
      question: "What is model pruning in the context of inference efficiency?",
      options: [
        "Removing parts of a trained network — layers, attention heads, or hidden dimensions — to get a smaller, faster model",
        "Deleting training data that the model memorized",
        "Truncating the input prompt to fit the context window",
        "Dynamically skipping some transformer layers at inference time depending on the input"
      ],
      correct: [0],
      explanation: "Structured pruning drops whole architectural components (depth: layers; width: heads, MLP hidden units, embedding channels), chosen via importance scores computed on calibration data. Unlike quantization it shrinks FLOPs as well as memory — but on its own it damages quality, which is why it's paired with distillation."
    },
    {
      type: "single",
      question: "In the prune-then-distill recipe (e.g., NVIDIA's Minitron), what role does knowledge distillation play?",
      options: [
        "After pruning damages the model, the pruned student is retrained to match the original teacher's outputs, recovering most quality at a fraction of from-scratch training cost",
        "Distillation selects which weights to prune",
        "Distillation converts the model to int4",
        "Distillation increases the model's parameter count back to the original"
      ],
      correct: [0],
      explanation: "The teacher (original model) supervises the pruned student via its logits/intermediate states — a much richer signal than raw next-token labels. Minitron showed a 15B model pruned to 8B and distilled with ~40× fewer tokens than training from scratch can match or beat similarly-sized models."
    },
    {
      type: "single",
      question: "Compared to quantization, what is a distinctive ADVANTAGE of pruning+distillation?",
      options: [
        "It reduces the actual FLOPs and parameter count, speeding up both memory- AND compute-bound regimes (quantization mainly cuts bytes moved)",
        "It is exact and never changes model outputs",
        "It requires no additional training",
        "It only works on vision models"
      ],
      correct: [0],
      explanation: "A pruned 8B model simply does less work than the 15B original — fewer FLOPs, less memory, smaller KV cache — helping even when compute-bound (large-batch serving, prefill). The cost: it needs a (modest) distillation training run and, like quantization, it is lossy."
    },

    // ---------- Alternative architectures ----------
    {
      type: "single",
      question: "Why are state-space models (e.g., Mamba) attractive for inference?",
      options: [
        "They maintain a constant-size recurrent state instead of a KV cache that grows with sequence length, so per-token generation cost and memory stay O(1) in context length",
        "They eliminate matrix multiplications entirely",
        "They generate all tokens of the response simultaneously",
        "They compress the KV cache into fewer heads, like an extreme form of GQA"
      ],
      correct: [0],
      explanation: "Transformers pay O(t) memory and attention reads per token at step t (growing KV cache); SSMs compress history into a fixed-size state, making decode cost flat in sequence length. Trade-off: a fixed-size state can lose information full attention would keep (e.g., precise long-range recall), so quality on recall-heavy tasks can suffer."
    },
    {
      type: "single",
      question: "How does LINEAR attention achieve constant-memory generation?",
      options: [
        "It replaces softmax with a kernel feature map, allowing the (key × value) statistics to be accumulated into a fixed-size running state so attention can be computed recurrently",
        "It keeps the softmax but sparsifies the attention matrix so only a constant number of entries are computed per step",
        "It stores the KV cache in a linked list to avoid fragmentation",
        "It quantizes the attention scores to 1 bit"
      ],
      correct: [0],
      explanation: "Without the softmax nonlinearity, sum_j φ(k_j)v_jᵀ can be pre-accumulated into a single matrix-valued state updated per token, so each new query needs only that fixed-size state — no per-token cache. Like SSMs, this trades expressiveness for O(1) inference cost, and hybrid attention+recurrent stacks try to get both."
    },
    {
      type: "single",
      question: "How do DIFFUSION language models differ from autoregressive generation at inference time?",
      options: [
        "They generate/refine all token positions in parallel over a series of denoising steps, rather than producing tokens strictly left-to-right",
        "They still generate left-to-right but use a small draft model to propose several tokens at once",
        "They generate one character at a time instead of one token",
        "They require a KV cache twice the normal size to store both noisy and clean tokens"
      ],
      correct: [0],
      explanation: "Diffusion LMs start from noise (or masks) over the whole sequence and iteratively refine every position at once — sidestepping the sequential bottleneck of autoregression and enabling very high token throughput if few denoising steps suffice. The open question is matching autoregressive quality at comparable compute."
    },
    {
      type: "multi",
      question: "Final review — which statements correctly summarize the lecture's big picture? (Select all that apply)",
      options: [
        "Generation is dominated by memory bandwidth, so most inference optimizations reduce bytes moved (weights and KV cache) per token",
        "Systems-level techniques (continuous batching, PagedAttention, speculative decoding) speed things up without changing model outputs",
        "Model-level techniques (quantization, GQA/MLA, pruning, alternative architectures) trade some fidelity or retraining effort for large efficiency gains",
        "Prefill and decode have identical performance characteristics, so no serving system treats them differently"
      ],
      correct: [0, 1, 2],
      explanation: "The unifying lens: arithmetic intensity. Decode is memory-bound, so wins come from moving fewer bytes (quantization, KV-cache reduction) or amortizing reads over more tokens (batching, speculative verification) — exactly split into lossless systems tricks and lossy model changes. Prefill (compute-bound) and decode (memory-bound) are OPPOSITE regimes, which is why systems schedule, chunk, or even disaggregate them differently."
    }
  ]
};

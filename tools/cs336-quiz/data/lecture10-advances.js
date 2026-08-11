// CS336 Lecture 10 — Inference (Advances)
// 50 interview-level questions emphasizing quantitative reasoning and deep understanding.
// Questions that rely on knowledge beyond the lecture state the necessary background inline.
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture10-advances"] = {
  title: "Lecture 10 — Inference: Advances",
  questions: [
    // ---------- Roofline & performance math ----------
    {
      type: "single",
      question: "Compute the arithmetic intensity of X(B×D) @ W(D×F) in bf16 with B=64, D=4096, F=16384. (FLOPs = 2BDF; bytes = 2(BD + DF + BF).) Which is closest?",
      options: [
        "≈ 63 FLOPs/byte",
        "≈ 295 FLOPs/byte — it always matches the accelerator's ratio",
        "≈ 4096 FLOPs/byte — intensity scales with D",
        "≈ 2 FLOPs/byte — intensity of any matmul is a constant"
      ],
      correct: [0],
      explanation: "FLOPs = 2·64·4096·16384 ≈ 8.59e9. Bytes = 2(64·4096 + 4096·16384 + 64·16384) ≈ 1.37e8, dominated by the 4096×16384 weight matrix. Intensity ≈ 8.59e9 / 1.37e8 ≈ 62.8 ≈ B. The exact value is slightly below B because the activation terms add a little memory traffic. This confirms the lecture's rule of thumb: intensity ≈ tokens processed per weight read."
    },
    {
      type: "single",
      question: "A workload has arithmetic intensity 100 FLOPs/byte on an H100 (989 TFLOP/s peak, 3.35 TB/s bandwidth, ratio ≈ 295). What is the maximum FLOP rate it can achieve, per the roofline model?",
      options: [
        "≈ 335 TFLOP/s (bandwidth × intensity = 3.35e12 × 100), about 34% of peak",
        "989 TFLOP/s — the peak is always achievable with sufficiently well-tuned kernels",
        "≈ 100 TFLOP/s — the FLOP rate equals the intensity",
        "≈ 3.35 TFLOP/s — the FLOP rate equals the bandwidth"
      ],
      correct: [0],
      explanation: "Below the ridge point (intensity < 295), the roofline is the slanted memory line: achievable FLOP/s = bandwidth × intensity = 3.35 TB/s × 100 FLOP/byte = 335 TFLOP/s, i.e. ~34% utilization. No amount of kernel tuning can beat this without changing the intensity itself — you must either move fewer bytes or do more work per byte."
    },
    {
      type: "single",
      question: "A 13B-parameter model in bf16 decodes a single stream (B=1) at its memory-bandwidth limit on an H100 (3.35 TB/s). Roughly what Model FLOPs Utilization (MFU) does this represent? (Per-token forward FLOPs ≈ 2 × params; peak 989 TFLOP/s.)",
      options: [
        "≈ 0.3% — only ~3.4 TFLOP/s of the 989 available is used",
        "≈ 34% — the same as any memory-bound workload",
        "≈ 50% — typical for well-optimized decode",
        "≈ 3% — an order of magnitude higher"
      ],
      correct: [0],
      explanation: "Weights = 26 GB, so each step takes ≥ 26e9/3.35e12 ≈ 7.8 ms → ~129 tokens/s. FLOPs consumed: 129 × 2×13e9 ≈ 3.4 TFLOP/s, versus 989 TFLOP/s peak → MFU ≈ 0.34%. This shocking number is the core motivation for batching, speculative decoding, and quantization: at B=1 more than 99.5% of the GPU's compute sits idle."
    },
    {
      type: "single",
      question: "Estimate the best-case single-stream decode speed of a 70B model with int4 weights (0.5 bytes/param) on an H100 (3.35 TB/s), ignoring KV cache and activations.",
      options: [
        "≈ 95 tokens/s (35 GB per step → ~10.4 ms/token)",
        "≈ 24 tokens/s (140 GB per step)",
        "≈ 380 tokens/s (8.75 GB per step)",
        "≈ 6 tokens/s — 70B models cannot run on one GPU"
      ],
      correct: [0],
      explanation: "int4 weights: 70e9 × 0.5 = 35 GB. Per-token time ≈ 35e9/3.35e12 ≈ 10.4 ms → ~96 tokens/s. In bf16 the same model would be 140 GB — it wouldn't even fit in 80 GB of HBM, let alone stream once per token. Quantization here is not a 10% optimization; it decides whether single-GPU serving is possible at all."
    },
    {
      type: "single",
      question: "With bf16 weights, the MLP becomes compute-bound on an H100 at roughly B ≈ 295 tokens. If you switch to int4 weights (0.5 bytes) while computing in bf16, roughly where does the crossover batch size move?",
      options: [
        "B ≈ 74 — intensity per weight byte is 4× higher, so the crossover drops 4×",
        "B ≈ 295 — weight quantization doesn't change where the compute crossover sits",
        "B ≈ 1180 — quantization raises the crossover 4×",
        "B ≈ 1 — int4 makes everything compute-bound"
      ],
      correct: [0],
      explanation: "Intensity ≈ FLOPs/bytes = 2BDF/(0.5·DF·bytes-scale) ≈ 4B when weights shrink from 2 bytes to 0.5. Setting 4B = 295 gives B ≈ 74. Insight: quantization doesn't just speed up small-batch decode — it lets you reach compute-bound operation (peak throughput) at 4× smaller batch, which matters when KV memory limits how large a batch you can form."
    },
    {
      type: "single",
      question: "Once decoding is fully compute-bound, what is the theoretical throughput ceiling for a 13B dense model on one H100 (989 TFLOP/s, 2 FLOPs per parameter per token)?",
      options: [
        "≈ 38,000 tokens/s (989e12 / 2.6e10)",
        "≈ 128 tokens/s — the same as the memory-bound limit",
        "≈ 989,000 tokens/s (FLOP rate ÷ 1e9)",
        "≈ 3,800 tokens/s"
      ],
      correct: [0],
      explanation: "Per token: 2 × 13e9 = 2.6e10 FLOPs. Ceiling = 989e12 / 2.6e10 ≈ 38k tokens/s (aggregate, across the whole batch). Compare with ~129 tokens/s at B=1: batching offers up to ~300× more throughput on identical hardware — which is why serving economics are dominated by how large a batch the KV cache lets you form."
    },
    {
      type: "single",
      question: "Estimate the minimum TTFT for a 2,000-token prompt on a 13B model on one H100, assuming prefill is compute-bound (989 TFLOP/s, forward FLOPs ≈ 2 × params × tokens).",
      options: [
        "≈ 50 ms (2 × 13e9 × 2000 ≈ 5.2e13 FLOPs ÷ 989e12)",
        "≈ 8 ms — one weight-streaming pass, like a decode step",
        "≈ 2 seconds — prefill processes tokens sequentially",
        "≈ 500 ms"
      ],
      correct: [0],
      explanation: "Prefill FLOPs = 2 × 13e9 × 2000 ≈ 5.2e13; at 989 TFLOP/s that's ≈ 53 ms. Note the contrast with decode: prefill cost scales with prompt length but runs at near-peak FLOPs (compute-bound), while decode runs at ~0.3% MFU. This asymmetry — the same model, two regimes — drives systems designs like chunked prefill and prefill/decode disaggregation."
    },
    {
      type: "single",
      question: "The lecture derives attention arithmetic intensity ≈ S·T/(S+T) for T query tokens attending over S cached tokens. During speculative-decoding verification of K=8 draft tokens with a cache of S=4096, what is the attention intensity, and what does it show?",
      options: [
        "≈ 8 (i.e., ≈ K, since S ≫ K)",
        "≈ 2048 (S/2) — verification behaves like a full prefill",
        "≈ 1 — attention intensity never changes",
        "≈ 512 (S/K)"
      ],
      correct: [0],
      explanation: "S·T/(S+T) = 4096·8/4104 ≈ 7.98 ≈ K. Normal decode has T=1 → intensity <1; verifying K tokens raises it to ≈K. So speculative decoding improves the arithmetic intensity of BOTH the weight reads (K tokens per weight pass) and the KV cache reads (K queries per cache pass) — the whole memory-bound bottleneck, not just the MLP."
    },
    {
      type: "single",
      question: "For memory-bound decode below the compute crossover, how do per-step latency and total throughput behave as batch size B grows (weights dominate the traffic)?",
      options: [
        "Latency stays roughly flat (same weight bytes per step) while throughput grows ≈ linearly in B",
        "Latency grows linearly with B while total throughput stays flat at every batch size",
        "Both latency and throughput stay constant until memory runs out",
        "Latency halves every time B doubles"
      ],
      correct: [0],
      explanation: "While memory-bound, a step's time ≈ (weight bytes)/(bandwidth) regardless of B — the extra FLOPs are free — so B tokens emerge per step and throughput ∝ B at almost no latency cost. Past the crossover (B ≈ 295 on H100 bf16), compute time ∝ B takes over: latency rises linearly and throughput flattens at the compute roof. (In practice growing per-sequence KV reads also add a B-proportional term earlier.)"
    },
    {
      type: "single",
      question: "Background — Little's law: for a stable queueing system, average concurrency L = arrival rate λ × average time in system W. A serving deployment handles 10 requests/s with an average end-to-end latency of 5 s. How many requests are in flight on average, and why does this matter for KV memory?",
      options: [
        "L = 50 concurrent requests",
        "L = 2 concurrent requests — λ ÷ W",
        "L = 15 concurrent requests — λ + W",
        "L = 0.5 concurrent requests — W ÷ λ"
      ],
      correct: [0],
      explanation: "L = λW = 10 × 5 = 50. Each in-flight request holds its KV blocks, so KV provisioning ≈ concurrency × per-sequence KV size. The law also exposes a feedback loop: if latency rises (e.g., from queueing), concurrency rises at the same arrival rate, consuming more KV memory, which can force preemptions that raise latency further."
    },

    // ---------- KV cache math & attention variants ----------
    {
      type: "single",
      question: "Llama-2-13B: 40 layers, 40 KV heads, head_dim 128, bf16. How much KV cache does ONE token require (all layers)?",
      options: [
        "≈ 0.8 MB (2 × 40 × 40 × 128 × 2 bytes)",
        "≈ 0.4 MB — only keys are cached",
        "≈ 8 MB",
        "≈ 20 KB"
      ],
      correct: [0],
      explanation: "2 (K and V) × 40 layers × 40 heads × 128 dims × 2 bytes = 819,200 bytes ≈ 0.8 MB per token. A 1,024-token sequence therefore holds ≈ 840 MB of KV — nearly a gigabyte per sequence. This is the number that makes everything else in the lecture (GQA, MLA, paging, quantized KV) necessary."
    },
    {
      type: "single",
      question: "Continuing: Llama-2-13B weights ≈ 26 GB (bf16) on an 80 GB H100, sequences of length 1024 at ≈ 0.84 GB KV each. Roughly what is the maximum batch size, ignoring activations?",
      options: [
        "≈ 64 sequences ((80 − 26) / 0.84)",
        "≈ 300 sequences — enough to reach the compute crossover",
        "≈ 8 sequences",
        "≈ 1000 sequences"
      ],
      correct: [0],
      explanation: "(80 − 26) GB / 0.84 GB ≈ 64. Note the punchline: the compute-bound crossover needs B ≈ 295, but memory caps you at ~64 — the KV cache prevents you from ever reaching peak throughput. This exact gap is why KV-reduction techniques translate directly into throughput: 8× smaller KV (GQA) → ~8× larger batch → ~8× closer to the compute roof."
    },
    {
      type: "single",
      question: "Per decode step, weight traffic is fixed (26 GB for 13B bf16) while KV traffic is B × S × 0.82 MB. At B = 64, beyond roughly what context length S does KV traffic EXCEED weight traffic?",
      options: [
        "S ≈ 500 tokens (26e9 / (64 × 0.82e6))",
        "S ≈ 32,000 tokens",
        "S ≈ 5 tokens",
        "KV traffic can never exceed weight traffic"
      ],
      correct: [0],
      explanation: "Solve 64 × S × 0.82e6 = 26e9 → S ≈ 495. So at realistic batch sizes, the KV cache dominates memory traffic beyond only ~500 tokens of context! For long-context serving the model weights are a minority of the bytes moved — reading everyone's KV is the real cost, which is why per-step latency degrades as conversations grow."
    },
    {
      type: "single",
      question: "Llama-2-70B uses GQA with 64 query heads and 8 KV heads. Relative to full MHA, what happens to (a) KV cache size and (b) the arithmetic intensity of reading the cache during decode?",
      options: [
        "(a) 8× smaller; (b) ≈ 8× higher",
        "(a) 8× smaller; (b) unchanged — intensity only depends on batch size",
        "(a) 64× smaller; (b) 64× higher",
        "(a) unchanged; (b) 8× higher"
      ],
      correct: [0],
      explanation: "KV heads drop 64→8, an 8× cache reduction. And since all 64 query heads still attend — 8 queries sharing each KV head — the attention FLOPs per KV byte rise ~8×. GQA is thus a double win on the two binding constraints: KV memory capacity (bigger batches fit) and KV bandwidth (each step reads fewer bytes per sequence)."
    },
    {
      type: "single",
      question: "DeepSeek-V2's MLA: 128 heads × head_dim 128, so full K plus V would be 2 × 16,384 = 32,768 dims per token per layer. MLA caches a 512-dim latent plus 64 decoupled-RoPE dims (576 total). Roughly what is the KV cache reduction?",
      options: [
        "≈ 57× (32,768 / 576)",
        "≈ 28× (16,384 / 576)",
        "≈ 4× — comparable to GQA with 8 groups",
        "≈ 320×"
      ],
      correct: [0],
      explanation: "Both K and V are reconstructed from the SAME shared latent, so the 576 cached dims replace all 32,768: ≈ 57×. (If you forgot that one latent serves both K and V, you'd get 28× — a common slip.) This is far beyond what GQA achieves at similar quality, which is why MLA lets DeepSeek serve long contexts cheaply while keeping 128 distinct heads' expressiveness."
    },
    {
      type: "single",
      question: "MLA never materializes full per-head keys during decode. Which linear-algebra fact enables this 'weight absorption'?",
      options: [
        "qᵀk = (W_Q h)ᵀ(W_UK c) = hᵀ(W_Qᵀ W_UK)c: the two projections collapse into one precomputed matrix",
        "Softmax is linear over the value dimension, so it can be applied directly to the compressed latent",
        "The K and V matrices are orthogonal, so their product is the identity",
        "RoPE rotations cancel between queries and keys"
      ],
      correct: [0],
      explanation: "Since the score is a bilinear form, W_Qᵀ W_UK can be merged offline into a single matrix; at decode time you compute hᵀ(merged)c directly against the cached latent — full 16k-dim keys never exist in memory. This is exactly what position-dependent RoPE would break if applied to the compressed dims (the rotation sits between the two matrices and varies per token), hence the decoupled-RoPE side channel."
    },
    {
      type: "single",
      question: "Background — Cross-Layer Attention (CLA) shares one KV cache between groups of adjacent layers. A model combines GQA (8× fewer KV heads) with 2-way CLA. What total KV reduction do you expect, and why?",
      options: [
        "≈ 16× — the savings act on independent factors (heads × layers) and multiply",
        "≈ 10× — the savings add together (8 + 2)",
        "≈ 8× — CLA's benefit is already subsumed by GQA's",
        "≈ 4× — combining them causes destructive interference between the two"
      ],
      correct: [0],
      explanation: "KV cache = 2 × layers × kv_heads × head_dim × bytes × tokens. GQA divides the kv_heads factor by 8; CLA divides the effective layers factor by 2; the factors are orthogonal, so 8 × 2 = 16×. Production stacks (e.g., Character.AI's) chain GQA + CLA + local attention + quantized KV precisely because multiplicative savings compound into order-of-magnitude reductions."
    },
    {
      type: "single",
      question: "A model interleaves local and global attention 5:1 (like Character.AI): 5 of every 6 layers use a sliding window W=1024, 1 of 6 uses full attention. At context S=8192, what is the approximate KV-traffic/memory reduction versus all-global?",
      options: [
        "≈ 3.7× — average per-layer cache is (5×1024 + 8192)/6 ≈ 2219 vs 8192 tokens",
        "≈ 6× — five of every six layers are local",
        "≈ 8× — the sliding window is 8× smaller than the full context length here",
        "≈ 1.2× — the local layers barely reduce anything in practice"
      ],
      correct: [0],
      explanation: "Per-layer average = (5·1024 + 1·8192)/6 ≈ 2219 tokens vs 8192 for all-global → ≈ 3.7×. Note the structure of the answer: the single global layer DOMINATES the remaining cost (8192 of the 13312 token-slots). The reduction improves with longer S but saturates at 6/1 only if the global layer is also bounded — which is why global layers are the next target (e.g., sharing them cross-layer)."
    },
    {
      type: "single",
      question: "Background — StreamingLLM observed that naively evicting the OLDEST tokens from the KV cache (pure sliding window) catastrophically degrades fluency, but keeping the first ~4 tokens plus the recent window works well. What is the accepted explanation?",
      options: [
        "Attention 'sinks': models park excess softmax probability mass on the initial tokens",
        "The first tokens contain a compressed summary of the entire document",
        "Position 0 stores the model's language identity, without which it cannot decode",
        "The first tokens are needed to recompute RoPE phases for all later tokens"
      ],
      correct: [0],
      explanation: "Attention weights must sum to 1 even when a head has 'nothing to say', and trained models systematically dump that residual mass on the earliest positions (attention sinks). Remove them and the mass redistributes, shifting all attention outputs off-distribution. Keeping ~4 sink tokens + a rolling window gives near-full quality with O(W) cache — an elegant example of interpretability informing systems design."
    },
    {
      type: "single",
      question: "Mistral-style sliding-window attention uses a ROLLING BUFFER KV cache: token at position p writes to slot p mod W. What property does this give the serving system?",
      options: [
        "KV memory per sequence is capped at W entries per layer regardless of how long generation runs",
        "It makes attention exact over the full history, since old tokens are compressed before overwrite",
        "It removes the need for position embeddings",
        "It allows the cache to be shared between different requests"
      ],
      correct: [0],
      explanation: "The buffer is a fixed-size circular array: constant memory, constant per-step cost, trivially predictable allocation — a serving system's dream, at the cost of the layer only seeing the last W tokens (mitigated by stacking layers: L layers of window W give an effective receptive field up to ~L×W, plus interleaved global layers in hybrid designs)."
    },

    // ---------- Speculative decoding, deeper ----------
    {
      type: "single",
      question: "Speculative decoding with K=4 draft tokens; assume each is accepted independently with probability α=0.8, a rejection yields one residual-sampled token, and full acceptance yields one bonus token. Expected tokens per target pass = Σᵢ₌₀..K αⁱ. What is it?",
      options: [
        "≈ 3.4 tokens (1 + 0.8 + 0.64 + 0.512 + 0.41)",
        "≈ 4.0 tokens (α × 5)",
        "≈ 5.0 tokens — all drafts plus the bonus",
        "≈ 1.6 tokens"
      ],
      correct: [0],
      explanation: "E = (1 − α^(K+1))/(1 − α) = (1 − 0.328)/0.2 ≈ 3.36. So a target pass that would have produced 1 token now produces ~3.4 — a ~3.4× decode speedup if drafting is free, and (3.4)/(1 + K·c) if the draft costs c target-forward-equivalents per token. The geometric structure explains why speedups saturate: later draft positions survive only with probability α^i."
    },
    {
      type: "single",
      question: "For one draft position, the probability that the draft token is accepted (draft q, target p, accept w.p. min(1, p/q)) equals Σₓ min(p(x), q(x)). What is this quantity in terms of a standard distance?",
      options: [
        "1 − TV(p, q), where TV is the total variation distance",
        "The KL divergence KL(q‖p)",
        "The cosine similarity between the logit vectors",
        "1/2 + the Wasserstein distance"
      ],
      correct: [0],
      explanation: "E[accept] = Σₓ q(x)·min(1, p/q) = Σₓ min(p, q) = 1 − TV(p, q). This identity makes the design goal precise: distill or choose the draft to minimize total variation to the target — not to be 'small and good in general'. It also predicts acceptance degrades on hard tokens (where the target is confident in something the draft isn't)."
    },
    {
      type: "single",
      question: "Why does increasing the number of draft tokens K eventually yield DIMINISHING or negative returns, even with a fast draft model?",
      options: [
        "The probability that position i is even reached decays as αⁱ, while drafting cost grows linearly in K",
        "Verification cost grows quadratically with K, because each additional draft token must attend to every other draft token in the batch",
        "Longer drafts violate the exactness guarantee",
        "The KV cache must be cleared after each rejection"
      ],
      correct: [0],
      explanation: "Expected extra tokens from raising K to K+1 is α^(K+1) — exponentially small — while draft cost (and verification width, once compute stops being free) grows linearly. Optimal K is finite and depends on α and the draft:target cost ratio; production systems typically use K ≈ 3–8. Exactness is never affected by K, and verification width is limited by compute, not a hard constant."
    },
    {
      type: "single",
      question: "With greedy decoding (temperature 0), what does speculative decoding's accept/reject rule reduce to?",
      options: [
        "Accept a draft token iff it equals the target's argmax; on mismatch, emit the target's argmax",
        "Accept every draft token, since greedy sampling has no randomness",
        "It cannot be used at temperature 0 at all, because rejection sampling requires randomness",
        "Accept iff the draft's argmax probability exceeds 0.5"
      ],
      correct: [0],
      explanation: "At temperature 0, p is a point mass on the argmax x*. min(1, p/q) is 1 for x* and 0 for everything else, and the residual max(0, p−q) is a point mass on x* too. So the procedure deterministically reproduces the target's greedy output — a clean special case showing the exactness guarantee concretely (and the easiest setting in which to unit-test an implementation)."
    },
    {
      type: "single",
      question: "Background — Medusa accelerates decoding WITHOUT a separate draft model. How?",
      options: [
        "It adds several extra lightweight decoding heads on the target's last hidden state, each predicting a token 2, 3, … steps ahead; candidate continuations are verified with tree attention in one pass",
        "It searches the existing context for n-grams matching the recent output and replays their continuations as draft candidates, verified in a single target pass",
        "It runs the first few transformer layers of the target as the draft, exiting early to propose tokens, then completes the remaining layers only for verification",
        "It runs the target twice at half precision and reconciles the outputs"
      ],
      correct: [0],
      explanation: "Medusa heads read the same trunk feature and cheaply guess several future tokens (with top-k options each), forming a candidate TREE verified in a single target pass via a tree-structured attention mask. No second model to deploy or keep distribution-matched; the trade-off is heads condition only on the current hidden state — not on each other's guesses — limiting acceptance on hard continuations. (n-gram replay exists too — 'prompt lookup decoding' — but that isn't Medusa.)"
    },
    {
      type: "single",
      question: "Background — EAGLE improves on Medusa's acceptance rates. What is its key idea?",
      options: [
        "Autoregress in FEATURE space: a small head predicts the target's next top-layer hidden state, then reuses the target's own output embedding for token distributions",
        "It fine-tunes the draft model with reinforcement learning to directly maximize the acceptance rate of its proposals against the frozen target",
        "It replaces the token-level accept/reject test with a cosine-similarity threshold between draft and target hidden states, trading exactness for speed",
        "It verifies tokens with a second, larger target model"
      ],
      correct: [0],
      explanation: "EAGLE's insight: the target's top-layer features are much more predictable than tokens (they evolve smoothly), and feature-level drafting is autoregressive — each drafted step conditions on the previous drafted feature AND the actually-sampled token, resolving sampling ambiguity that Medusa's independent heads can't. Result: markedly higher acceptance at similar draft cost, making it a strong open-source default."
    },
    {
      type: "single",
      question: "In speculative sampling, why must the replacement token after a rejection be drawn from the RESIDUAL distribution norm(max(0, p − q)) rather than simply from the target p itself?",
      options: [
        "Accepted tokens already contribute min(p, q); only the residual tops each token up to exactly p(x)",
        "Sampling the replacement from p would require an extra forward pass of the target model",
        "The residual is cheaper to compute than the full distribution",
        "It doesn't matter — both choices give exact samples"
      ],
      correct: [0],
      explanation: "Work the algebra: P(output x) = q(x)·min(1, p/q) + P(reject)·r(x) where the first term is min(p,q)(x). For this to equal p(x), the correction must supply exactly p(x) − min(p,q)(x) = max(0, p−q)(x) — the residual, normalized by P(reject) = TV(p,q). Substituting r = p instead leaves the distribution provably biased. This is THE classic correctness question for speculative decoding implementations."
    },

    // ---------- Quantization, deeper ----------
    {
      type: "single",
      question: "GPTQ, AWQ, and LLM.int8() are all post-training quantization methods. What specifically characterizes GPTQ?",
      options: [
        "Layer-by-layer weight quantization minimizing ‖WX − ŴX‖² on calibration data, compensating remaining weights via Hessian information",
        "Keeping outlier activation channels in fp16 and computing the rest in int8 via mixed-precision decomposition",
        "Scaling salient weight channels based on activation magnitudes before uniform quantization",
        "Simulating quantization during pretraining so the model learns robust weights"
      ],
      correct: [0],
      explanation: "GPTQ (from OBQ/OBS lineage) treats each layer as a least-squares problem: quantize one weight (column) at a time and optimally update the not-yet-quantized weights to absorb the error, using the Hessian of the layer's inputs. Option 2 describes LLM.int8(), option 3 AWQ, option 4 quantization-aware training. Knowing WHICH error each method minimizes — weight error vs. layer-output error vs. runtime outlier handling — is the real distinction."
    },
    {
      type: "single",
      question: "Background — fp8 comes in two formats: E4M3 (4 exponent, 3 mantissa bits; range ≈ ±448) and E5M2 (5 exponent, 2 mantissa; range ≈ ±57,344). Which is preferred for inference weights/activations, and why?",
      options: [
        "E4M3 — forward-pass values have modest range, so the extra mantissa bit (precision) is worth more than range",
        "E5M2 — inference always needs the maximum dynamic range to survive activation outliers in every layer",
        "They are interchangeable since both are 8 bits",
        "Neither — fp8 cannot represent negative activations"
      ],
      correct: [0],
      explanation: "The bit budget is a range-vs-precision trade. Weights and forward activations are well-scaled (especially with per-tensor/per-channel scales), so E4M3's extra mantissa bit halves relative rounding error where it counts. Gradients span wild magnitudes, needing E5M2's range. This is why mixed-format fp8 training uses E4M3 forward / E5M2 backward, and inference standardizes on E4M3."
    },
    {
      type: "single",
      question: "Compare W4A16 (int4 weights, bf16 activations/compute) with W8A8 (int8 weights AND activations, int8 tensor cores). Which statement is correct for serving?",
      options: [
        "W4A16 excels at memory-bound decode but computes in bf16; W8A8 also accelerates compute-bound prefill via int8 tensor cores",
        "W8A8 is strictly better than W4A16 in every serving regime, since quantizing more tensors always wins",
        "W4A16 accelerates prefill more than decode, because prefill reads the most weight bytes",
        "Neither affects throughput; both only save storage"
      ],
      correct: [0],
      explanation: "Weight-only quantization shrinks bytes moved — decisive when memory-bound — but the matmuls still run at bf16 speed, so prefill/large-batch gains are minimal. Quantizing activations too unlocks int8 tensor cores (≈2× bf16 FLOPs), helping exactly where weight-only doesn't. Hence the practical split: latency-focused local inference favors W4A16; high-throughput serving invests in W8A8/fp8 despite activations being harder to quantize."
    },
    {
      type: "multi",
      question: "Which statements about quantizing the KV CACHE (rather than weights) are true? (Select all that apply)",
      options: [
        "Going bf16 → int8/fp8 halves KV memory, roughly doubling the batch size that fits",
        "It reduces the per-step KV bandwidth cost, which dominates at long contexts and large batches",
        "Keys are typically harder to quantize than values, motivating per-channel key scales",
        "It dramatically accelerates prefill, because prefill repeatedly re-reads the whole KV cache"
      ],
      correct: [0, 1, 2],
      explanation: "KV quantization attacks both KV problems: capacity (bigger batches) and bandwidth (cheaper long-context steps). Empirically keys show strong per-channel outliers (values are smoother), so schemes like KIVI quantize keys per-channel and values per-token. The last option is false: prefill WRITES the cache once and is compute-bound — KV quantization barely affects it; decode is where the wins are."
    },
    {
      type: "single",
      question: "int4 weight quantization almost always uses GROUP-WISE scales (e.g., one fp16 scale per 128 weights) rather than per-tensor or per-channel. Why is this the sweet spot?",
      options: [
        "With only 16 levels, scales must track local statistics; ~128-weight groups cost just ~0.1 bits/weight",
        "Per-tensor scales would be more accurate but are too slow to compute during inference",
        "Group-wise scaling is required by the CUDA compiler",
        "It allows different weight groups to switch bit-widths dynamically at runtime"
      ],
      correct: [0],
      explanation: "The coarser the scale granularity, the more one outlier stretches the range for all weights it shares a scale with — catastrophic when you have only 16 levels. Per-128 groups localize that damage: one fp16 scale per 128 int4 weights ≈ 16/128 ≈ 0.125 extra bits/weight, and contiguous groups map cleanly onto GPU dequant kernels. It's a pure accuracy-per-byte trade, and ~64–128 empirically wins."
    },
    {
      type: "single",
      question: "Why are ACTIVATIONS fundamentally harder to quantize than weights in LLMs?",
      options: [
        "Activations are input-dependent",
        "Activations are stored in a format GPUs cannot address",
        "Activations contain gradients, which require full precision",
        "Weights are larger tensors, so they tolerate more error"
      ],
      correct: [0],
      explanation: "Weights sit still: you can spend hours offline finding optimal scales/clips per group (GPTQ, AWQ). Activations only exist at runtime, differ per input, and — the killer — a few hidden channels systematically run 10–100× hotter than the rest (the LLM.int8() outlier phenomenon). Every W8A8 scheme is essentially a strategy for those channels: mixed precision, smoothing/rotating them into weights (SmoothQuant, QuaRot), or per-token dynamic scales."
    },

    // ---------- Systems, deeper ----------
    {
      type: "single",
      question: "Background — 'chunked prefill' splits a long prompt's prefill into pieces processed across multiple iterations, mixed into batches with ongoing decode steps. What problem does this solve, and what does it trade?",
      options: [
        "It stops long prefills from monopolizing the GPU and stalling running decodes, at slight TTFT cost",
        "It reduces the total FLOPs of prefill, because tokens within each chunk only need to attend to that chunk rather than the whole prompt",
        "It eliminates the need for a KV cache during prefill",
        "It converts prefill from compute-bound to memory-bound, which is faster"
      ],
      correct: [0],
      explanation: "A 10k-token prefill at ~50ms+ would freeze every other user's token stream if run monolithically. Chunking (e.g., 512-token pieces) bounds each iteration's compute so decodes continue smoothly — smoothing P99 inter-token latency at slight TTFT/efficiency cost. It also fills a scheduling gap: each mixed batch pairs compute-heavy prefill chunks with memory-heavy decodes, using both resources at once."
    },
    {
      type: "single",
      question: "Background — disaggregated serving (DistServe, Mooncake) runs prefill and decode on SEPARATE GPU pools, shipping the KV cache between them. What is the core justification?",
      options: [
        "Prefill and decode have conflicting optimal batch sizes, parallelism strategies, and scaling needs; separation removes interference between the two phases",
        "It halves total KV memory: the prefill pool keeps the prompt's KV while the decode pool only stores KV for newly generated tokens",
        "It lets prefill run in int4 while decode runs in bf16",
        "It is required for models that do not fit on a single GPU"
      ],
      correct: [0],
      explanation: "Colocated phases fight: prefills stall decodes (ITL spikes), decodes delay prefills (TTFT misses), and the phases even prefer different tensor-parallel degrees. Disaggregation turns one bad multi-objective scheduler into two clean single-objective ones, at the cost of streaming ~GB-scale KV per request across NVLink/RDMA — viable because transfer time is small versus decode duration. The 'goodput per GPU under both SLOs' metric is what it optimizes."
    },
    {
      type: "single",
      question: "Background — CUDA graphs let you capture a sequence of GPU kernel launches and replay it with a single call. Why do they matter disproportionately for small-batch DECODE?",
      options: [
        "A decode step launches hundreds of tiny kernels, each finishing in microseconds at small batch",
        "They fuse the step's kernels into one large kernel so intermediate activations stay in registers, sharply reducing memory traffic — which is what decode is bound by",
        "CUDA graphs increase effective memory bandwidth by pinning the weight pages in HBM",
        "They allow the CPU to compute attention in parallel with the GPU"
      ],
      correct: [0],
      explanation: "At B=1, many decode kernels run ~1–10 µs while each launch costs ~5–10 µs of CPU time — the GPU can be idle half the step, an overhead invisible in FLOP or bandwidth accounting. Capturing the whole step as a graph replays it with near-zero CPU involvement; vLLM ships this ('CUDA graph mode') precisely for small-batch decoding. The catch: graphs need static shapes, hence padding/bucketing batch sizes."
    },
    {
      type: "single",
      question: "Background — FlashDecoding extends FlashAttention for generation. Standard FlashAttention parallelizes across queries — why is that a problem during decode, and what does FlashDecoding do?",
      options: [
        "Decode has ONE query per sequence, so parallelism is only batch × heads",
        "It streams the KV cache through shared memory in larger tiles, raising the arithmetic intensity of decode attention above the compute roofline",
        "FlashAttention cannot read a paged KV cache; FlashDecoding defragments it into contiguous form first",
        "It skips softmax during decode, since a single query needs no normalization"
      ],
      correct: [0],
      explanation: "With B=1, one head's decode attention offers a single query row — under FlashAttention's query-parallel mapping, most of the GPU's SMs idle while one block scans a huge KV. Splitting over the KV axis (split-K style) creates parallel work; the online-softmax trick (tracking each chunk's max and sum) makes the final merge exact. Same math, restructured for the degenerate-parallelism regime — a classic GPU-programming interview topic."
    },
    {
      type: "single",
      question: "Background — SGLang's RadixAttention keeps a radix tree (trie) over token prefixes of RECENT requests, mapping them to retained KV blocks. What does this add beyond vLLM-style within-batch sharing?",
      options: [
        "Cross-request prefix reuse: a new request whose prompt shares a prefix with any cached earlier request skips prefill for the shared part",
        "It compresses the KV cache using low-rank factorization along tree branches",
        "It predicts likely future requests from historical traffic and speculatively prefills their prompts during idle cycles, discarding mispredictions",
        "It shares KV between different MODELS that use the same tokenizer"
      ],
      correct: [0],
      explanation: "PagedAttention's copy-on-write shares KV among sequences alive at the same time (e.g., n samples of one prompt). RadixAttention generalizes across time: the trie matches any new prompt against all retained prefixes, reusing KV and cutting TTFT dramatically for template-heavy workloads (agents, multi-turn chat, evals with shared few-shot preambles). It's exact — reused KV is identical to what prefill would recompute — with cache-hit rate becoming a first-class serving metric."
    },
    {
      type: "single",
      question: "When vLLM runs out of KV memory mid-flight, it must preempt a running sequence, later restoring it by either SWAPPING its KV blocks to CPU RAM or RECOMPUTING them from tokens. When is recomputation typically preferred?",
      options: [
        "For short-to-moderate contexts: one compute-bound prefill pass on-GPU often beats paging gigabytes back over PCIe",
        "Never — recomputation produces subtly different results than the saved cache, corrupting generation",
        "Always — CPU memory cannot hold KV blocks",
        "Only when the sequence used greedy decoding"
      ],
      correct: [0],
      explanation: "It's a bandwidth race: re-prefill runs at ~TFLOP/s effective rates on-GPU, while swap-in crawls through PCIe (~tens of GB/s, shared). For a 1–2k-token context, recompute takes tens of ms; swapping ~1 GB back takes comparable or worse, plus pinned-CPU-memory pressure. Both restore a numerically equivalent cache (recomputation reruns prefill on the same tokens), so the choice is essentially a cost model — and the crossover point shifts with context length and interconnect."
    },
    {
      type: "single",
      question: "PagedAttention needs a block size (tokens per KV block, e.g., 16). What is the trade-off in choosing it?",
      options: [
        "Small blocks minimize internal fragmentation but inflate block-table overhead and hurt access efficiency; large blocks do the reverse",
        "Larger blocks improve model accuracy, because attention can read longer contiguous spans of context per fetch",
        "Block size must equal the attention head dimension",
        "Small blocks are only useful for beam search"
      ],
      correct: [0],
      explanation: "It's the same page-size dilemma as in operating systems. Waste per sequence averages ~half a block (×batch, ×it matters); tiny blocks fix that but mean more table entries, more pointer-chasing per attention read, and worse coalescing. vLLM's default of 16 tokens sits where fragmentation is small relative to sequence length yet gather efficiency remains high — and the analogy to OS paging is exactly the mental model to invoke."
    },
    {
      type: "single",
      question: "For LATENCY (not just memory capacity), why does tensor parallelism (TP) across 8 GPUs speed up memory-bound decode, and why is pipeline parallelism (PP) unhelpful for the same goal?",
      options: [
        "TP shards every weight matrix, so each GPU streams ~1/8 of the bytes per step in parallel",
        "TP caches the KV on all GPUs redundantly, which is faster",
        "PP is always better for low-latency decode because it requires no inter-GPU communication",
        "TP reduces the FLOPs per token by 8×"
      ],
      correct: [0],
      explanation: "Memory-bound step time ≈ bytes-per-GPU ÷ per-GPU bandwidth: TP divides the numerator, aggregating 8 GPUs' bandwidth on one token — that's why frontier-scale low-latency serving runs TP=8 within a node (2 all-reduces per layer make it interconnect-hungry, hence NVLink). PP leaves each token's serial path length unchanged; it helps fit the model and boosts aggregate throughput, but a single stream decodes no faster. KV also shards naturally under TP (by heads — GQA's KV-head count constrains the TP degree)."
    },
    {
      type: "single",
      question: "Background — users observe that the SAME prompt at temperature 0 can yield different outputs from a production serving endpoint across the day. What is the standard systems explanation?",
      options: [
        "Floating-point addition is non-associative, and continuous batching changes which requests are batched together",
        "Serving frameworks internally clamp temperature to a small epsilon like 1e-5 rather than exactly zero, so sampling remains slightly stochastic",
        "The model's weights are re-quantized periodically as calibration data shifts, changing outputs",
        "Servers add random noise to logits to prevent prompt-extraction attacks"
      ],
      correct: [0],
      explanation: "Greedy decoding is deterministic only if the logits are bit-identical. (a+b)+c ≠ a+(b+c) in floating point, and your request's batch companions — which you don't control — change tensor shapes, kernel selection (including split-K strategies), and summation orders. A near-tie argmax flips once, and autoregression amplifies it into a different continuation. True determinism requires batch-invariant kernels at a real throughput cost, which most production endpoints don't pay."
    },

    // ---------- MoE & alternative architectures, deeper ----------
    {
      type: "single",
      question: "Background — Mixtral 8×7B routes each token to 2 of 8 FFN experts per layer (~13B of ~47B params active per token). At batch size 1, how does memory traffic per decode step compare to a dense 47B model, and what happens as batch size grows?",
      options: [
        "At B=1 only the ~2 routed experts per layer (plus attention weights) are read",
        "All 47B parameters must be read every step, because the routing decision is only known after each layer's expert weights have already been loaded",
        "MoE traffic is independent of batch size, like attention",
        "At large B the router collapses to one expert, minimizing traffic"
      ],
      correct: [0],
      explanation: "MoE's decode economics are batch-dependent in a way dense models aren't: B=1 streams only active experts (fast, ~13B-like); moderate B is the WORST regime — nearly all experts get touched by some token, so you pay 47B of reads for few tokens each (poor per-expert intensity); very large B amortizes every expert across many tokens and MoE's FLOP savings shine again. Serving MoE well means engineering around that middle valley (expert-parallel sharding, capacity limits)."
    },
    {
      type: "single",
      question: "The lecture notes S4-style state-space models struggled with ASSOCIATIVE RECALL (retrieving a value seen earlier, keyed by context). What did Mamba change to address this, and why does it help?",
      options: [
        "Its SSM parameters (input/forget dynamics) became functions of the current input",
        "It made the state grow linearly with sequence length, like a KV cache",
        "It interleaves full softmax-attention layers between SSM blocks, letting attention handle recall while the recurrence handles everything else",
        "It replaced the recurrence with a lookup table of past tokens"
      ],
      correct: [0],
      explanation: "A linear time-invariant system convolves all history with one fixed kernel — it cannot 'store THIS token's value under THIS key on demand'. Making the transition/input matrices input-dependent turns the recurrence into a content-controlled read-write memory, recovering induction-head-like recall while keeping O(1) state. The price: the convolutional training trick no longer applies, requiring Mamba's custom parallel scan — a good example of an inference win demanding new training machinery."
    },
    {
      type: "single",
      question: "Jamba interleaves Transformer attention and Mamba layers at 1:7 (one attention layer per 8). Why keep ANY attention layers, given Mamba's O(1) inference state?",
      options: [
        "A few attention layers restore capabilities fixed-size states handle poorly",
        "Mamba layers cannot be stacked more than 8 deep due to vanishing gradients",
        "Attention layers are needed to compute position embeddings for Mamba",
        "The 1:7 ratio is required for tensor-parallel sharding"
      ],
      correct: [0],
      explanation: "Needle-in-a-haystack retrieval and verbatim copying are where pure SSMs measurably lag: a compressed state cannot guarantee recovering an arbitrary earlier token, but even one attention layer with full-context KV can. The hybrid buys back those capabilities for 1/8 of the KV cost (Jamba further constrains it), landing on a far better quality-per-KV-byte Pareto point than either extreme — the same logic as local:global interleaving, one level up."
    },
    {
      type: "single",
      question: "BASED (mentioned in the lecture) combines LINEAR attention with a small window of exact LOCAL attention. What failure mode of each component does the other cover?",
      options: [
        "Linear attention gives cheap but 'fuzzy' global memory; exact local attention covers precise recent-token recall",
        "Linear attention handles short contexts and local attention handles long ones",
        "Local attention exists mainly to correct the numerical instability of linear attention's kernel feature map",
        "The two components are redundant; BASED uses both only for training stability"
      ],
      correct: [0],
      explanation: "It's an accuracy-decomposition argument: most precise-recall queries target either the recent window (local attention answers these exactly and cheaply) or benefit from coarse global aggregation (the linear-attention state suffices). Neither piece alone matches full attention on recall benchmarks; the sum nearly does, at constant inference memory. This 'exact-near + fuzzy-far' design recurs across the efficient-architecture literature."
    },
    {
      type: "single",
      question: "A diffusion LM generates a T-token response in n full-sequence denoising passes; an autoregressive (AR) model uses T sequential steps with an incremental KV cache. When does diffusion actually win on latency, given similar model sizes?",
      options: [
        "When n ≪ T and there is spare compute: each diffusion pass is compute-bound, while AR decode steps are memory-bound",
        "Whenever n < T, regardless of hardware utilization — fewer forward passes always means lower wall-clock latency",
        "Only when T is very small, since diffusion cost grows with T²",
        "Never — diffusion models cannot generate coherent text"
      ],
      correct: [0],
      explanation: "Count both axes: AR pays T memory-bound steps (weights streamed T times); diffusion pays n compute-bound passes over all T positions (weights streamed ~n times, but FLOPs ≈ n·T·2N). At B=1 the GPU's idle compute makes n=10-ish passes far faster wall-clock than T=500 sequential streams — hence the dramatic Inception Labs coding demos. At serving-scale batch, AR is already compute-efficient and diffusion's extra FLOPs become a real cost. Same roofline lens as everything else in this lecture."
    },
    {
      type: "single",
      question: "Compare long-context inference memory: a Transformer's KV cache at S tokens vs. a Mamba-style model's recurrent state. Which statement is quantitatively right?",
      options: [
        "Transformer KV grows linearly in S (≈80 GB at 100k tokens for a 13B-class MHA model), while the SSM state stays a fixed size",
        "Both grow linearly with S, but the SSM's per-token constant is far smaller than the KV cache's per-token cost",
        "The SSM state grows with S while the Transformer's KV cache stays fixed once allocated",
        "They are equal at all context lengths by construction"
      ],
      correct: [0],
      explanation: "The KV cache is a verbatim record: O(S) by definition (0.82 MB/token × 100k ≈ 82 GB — more than an H100 for ONE sequence). An SSM state is O(1): layers × state_dim × d_model-ish, hundreds of MB at 7B scale, whether S is 1k or 1M. That's the whole bet: pay with lossy compression (recall risk) to make context length free at inference. GQA/MLA/local attention are intermediate points on this same memory-vs-fidelity dial."
    },

    // ---------- Serving metrics & scenario ----------
    {
      type: "single",
      question: "Background — serving SLOs are usually stated as percentiles (e.g., 'P99 TTFT < 1s'), and DistServe defines GOODPUT as the rate of requests meeting ALL latency SLOs per GPU. Why are mean-based metrics considered insufficient?",
      options: [
        "Latency distributions in LLM serving are heavy-tailed",
        "Means are harder to compute than percentiles in streaming systems",
        "P99 is always 99× the mean, so either can be derived from the other",
        "Goodput equals throughput divided by GPU count, making means redundant"
      ],
      correct: [0],
      explanation: "Classic example: crank the batch size — throughput (tokens/s) rises, but occasional monolithic prefills freeze decodes, P99 inter-token latency explodes, and requests violating SLOs are wasted work from the product's perspective. Optimizing 'goodput under P99 SLOs' rather than raw tokens/s is what motivates chunked prefill, disaggregation, and admission control. Interviewers probe exactly this throughput-vs-tail distinction."
    },
    {
      type: "multi",
      question: "You serve an interactive coding assistant: prompts of ~10k tokens (mostly a shared repository context), short answers, strict TTFT and smooth inter-token latency. Which techniques directly target THIS workload's bottlenecks? (Select all that apply)",
      options: [
        "Prefix caching (radix-tree KV reuse)",
        "Chunked prefill or prefill/decode disaggregation",
        "Speculative decoding",
        "Maximizing batch size to 512 with monolithic (unchunked) prefills — maximum throughput always minimizes latency"
      ],
      correct: [0, 1, 2],
      explanation: "Diagnose first: the pain is (a) repeated 10k-token prefills → prefix caching eliminates the repeated part entirely (biggest single win, and exact); (b) prefill/decode interference → chunking or disaggregation protects ITL; (c) decode speed at modest batch → speculation helps precisely in this memory-bound, latency-sensitive regime. Option 4 optimizes raw throughput at the direct EXPENSE of both SLOs — monolithic 10k prefills at huge batch are exactly how you destroy P99. Matching techniques to a workload's actual bottleneck is the real interview skill this lecture builds."
    }
  ]
};

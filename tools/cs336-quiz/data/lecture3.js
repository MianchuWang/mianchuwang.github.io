// CS336 Lecture 3 — Architectures and hyperparameters
// 20 questions covering: pre-norm vs post-norm and double-norm variants, RMSNorm vs
// LayerNorm and wall-clock cost of small ops, gated activations (SwiGLU/GeGLU) and the
// 8/3·d convention, rotary embeddings, hyperparameter regularities (d_ff ratio, aspect
// ratio, head dims, vocab sizes), weight decay in pretraining, stability tricks
// (z-loss, QK-norm, logit soft-capping), GQA/MQA and the KV cache, sliding-window
// attention, parallel layers, and near-consensus architectural choices.
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture3"] = {
  title: "Lecture 3 — Architectures & Hyperparameters",
  questions: [
    // ---------- Norm placement ----------
    {
      type: "single",
      question: "Why does pre-norm (LayerNorm before each sublayer) train more stably than the original post-norm transformer?",
      options: [
        "It normalizes attention logits directly, so the softmax never saturates during long training runs",
        "The residual stream stays an identity path, so gradients reach early layers unattenuated",
        "It cuts parameter count in every block, giving a smoother and better-conditioned loss surface",
        "It makes each block's forward pass linear, so gradient magnitudes are constant across depth"
      ],
      correct: [1],
      explanation: "With pre-norm, the norm sits on the branch rather than on the residual stream, so the identity path from loss to every layer is never rescaled — gradient magnitudes stay well-behaved even in deep stacks. Post-norm inserts a norm into that path after every block, which distorts gradients and is a key reason post-norm models diverge without care. Normalizing attention logits is a different trick (QK-norm), not what pre-norm does."
    },
    {
      type: "single",
      question: "How does RMSNorm differ from LayerNorm?",
      options: [
        "It normalizes across the batch dimension instead of the feature dimension",
        "It adds a learned bias so activations are re-centered after scaling",
        "It drops mean subtraction and the bias term, rescaling only by the root-mean-square",
        "It replaces per-token statistics with a single fixed scale shared across all positions"
      ],
      correct: [2],
      explanation: "RMSNorm keeps only the RMS rescaling and a learned gain — no mean-centering, no bias — yet models like T5 and Llama show quality on par with LayerNorm. Fewer operations and fewer memory movements make it cheaper in practice. Normalizing across the batch dimension describes BatchNorm, which is not used in modern LMs."
    },
    {
      type: "multi",
      question: "Which statements about gated activations (SwiGLU, GeGLU) are accurate? (Select all that apply)",
      options: [
        "They add an elementwise multiplicative gate to the FFN's up-projection",
        "Shazeer's comparison found modest but consistent gains over ReLU/GELU FFNs",
        "Most recent strong models (Llama, PaLM, and others) use a gated variant",
        "At a fixed $d_{\\text{ff}}$, the extra gate matrix halves the FFN's parameter count",
        "Their main benefit is eliminating dead neurons, an effect proven in Shazeer's paper"
      ],
      correct: [0, 1, 2],
      explanation: "A GLU-style FFN computes an extra linear projection and multiplies it elementwise against the activated path — a gate. The empirical story is 'helps a little, basically always', which is why gated units are near-universal in recent models. At fixed $d_{\\text{ff}}$ the third matrix INCREASES parameters (hence $d_{\\text{ff}}$ is shrunk to compensate), and no result proves anything about dead neurons."
    },
    {
      type: "single",
      question: "Why do gated FFNs conventionally use $d_{\\text{ff}} = \\frac{8}{3}d_{\\text{model}}$ instead of the usual $4d_{\\text{model}}$?",
      options: [
        "$\\frac{8}{3}$ is the largest ratio whose activations fit in GPU memory at scale",
        "The ratio maximizes tensor-core utilization on modern GPU accelerators",
        "Sweeps found $\\frac{8}{3}$ minimizes loss regardless of the parameter budget",
        "Gating adds a third weight matrix, so $d_{\\text{ff}}$ shrinks to keep parameters matched"
      ],
      correct: [3],
      explanation: "A standard FFN has two $d \\times d_{\\text{ff}}$ matrices ($2 \\cdot 4d^2 = 8d^2$ params); a gated FFN has three, so setting $d_{\\text{ff}} = \\frac{8}{3}d$ gives $3 \\cdot \\frac{8}{3}d^2 = 8d^2$ — the same budget. It is a parameter-matching convention, not a hardware or loss-optimality result. That keeps comparisons between gated and non-gated FFNs fair."
    },
    // ---------- Positions ----------
    {
      type: "single",
      question: "What is the core idea of rotary position embeddings (RoPE)?",
      options: [
        "Rotate query/key coordinate pairs by angles set by position, making attention relative",
        "Add a learned vector for each absolute position to the token embeddings at the input",
        "Add a scalar bias to each attention logit that decays with the distance between tokens",
        "Concatenate sinusoidal position features onto the value vectors in every attention layer"
      ],
      correct: [0],
      explanation: "RoPE rotates 2D slices of q and k by angles proportional to position; because rotations preserve inner products up to the angle difference, q·k depends only on the relative offset between tokens. Learned absolute vectors are the GPT-2/OPT approach RoPE replaced, and distance-decaying logit biases describe ALiBi/T5-style biases, not RoPE."
    },
    {
      type: "multi",
      question: "Which statements about norm placement in transformers are true? (Select all that apply)",
      options: [
        "The original 2017 transformer used post-norm",
        "Pre-norm largely removes the need for delicate learning-rate warmup",
        "'Double norm' models (e.g., Grok, Gemma 2) add norms before and after blocks, still off the residual path",
        "Post-norm has become the standard again in today's largest LMs"
      ],
      correct: [0, 1, 2],
      explanation: "Vaswani et al. placed the norm after the residual add; later work showed pre-norm gives well-scaled gradients from initialization, so training works without heavy warmup. Newer 'double norm' variants add a second norm around the sublayer while keeping the residual stream itself untouched — preserving pre-norm's key property. Post-norm has not returned as the default; the residual-path-free placement is one of the field's clearest consensus points."
    },
    // ---------- Wall-clock cost of small ops ----------
    {
      type: "single",
      question: "Norm layers are a negligible share of a transformer's FLOPs. Why do they still matter for wall-clock speed?",
      options: [
        "At large scale, norm layers actually come to dominate the total FLOP count",
        "They silently force the surrounding computation to run in FP64 precision",
        "They sit on the residual path and prevent the model from being tensor-parallelized",
        "They run memory-bandwidth-bound with kernel overhead, so cost far exceeds FLOP share"
      ],
      correct: [3],
      explanation: "FLOPs concentrate in matmuls, but norms, activations, and other pointwise ops run near memory bandwidth, not peak compute, and add kernel launches and synchronization. That is why swapping LayerNorm for the cheaper RMSNorm produces real runtime wins despite a rounding-error FLOP difference. Counting FLOPs alone would predict no benefit at all."
    },
    // ---------- Hyperparameter regularities ----------
    {
      type: "single",
      question: "For a standard (non-gated) FFN, what is the conventional $d_{\\text{ff}}$ to $d_{\\text{model}}$ ratio?",
      options: [
        "$d_{\\text{ff}} = 2\\,d_{\\text{model}}$",
        "$d_{\\text{ff}} = 4\\,d_{\\text{model}}$",
        "$d_{\\text{ff}} = 8\\,d_{\\text{model}}$",
        "$d_{\\text{ff}} = d_{\\text{model}}$"
      ],
      correct: [1],
      explanation: "The 4x expansion dates to the original transformer and has been carried through GPT-2, GPT-3, and most successors; gated variants use roughly $\\frac{8}{3}d_{\\text{model}}$ to match parameters. Ratios far outside this band exist but are rare exceptions, not the convention."
    },
    {
      type: "multi",
      question: "Which hyperparameter regularities hold across most modern LMs? (Select all that apply)",
      options: [
        "$d_{\\text{ff}}/d_{\\text{model}}$ near 4 (about $\\frac{8}{3}$ with gating)",
        "$n_{\\text{heads}} \\times d_{\\text{head}}$ approximately equal to $d_{\\text{model}}$",
        "Aspect ratio $d_{\\text{model}}/n_{\\text{layer}}$ roughly in the 100–200 range",
        "Depth strictly dominating width whenever the parameter count is held fixed",
        "Vocabulary size growing in direct proportion to the total parameter count"
      ],
      correct: [0, 1, 2],
      explanation: "These three ratios are remarkably consistent across model families — most designs sit near a 4x FFN expansion, heads that tile $d_{\\text{model}}$, and aspect ratios around 100–200. Depth vs width is fairly flat near the optimum (systems constraints like pipelining often decide it), and vocab size tracks the data mix and tokenizer, not parameter count."
    },
    {
      type: "single",
      question: "What vocabulary sizes are typical for monolingual versus multilingual models?",
      options: [
        "About 1k–5k for all models, keeping the output softmax and embedding tables cheap",
        "About 500k–1M for all models, so common word forms get dedicated tokens",
        "Roughly 30–50k monolingual; roughly 100–250k multilingual/production systems",
        "Exactly 65,536 across the board, chosen for alignment with GPU memory layouts"
      ],
      correct: [2],
      explanation: "GPT-2/GPT-3-style English models sit near 50k tokens (T5 and Llama near 32k), while multilingual and production systems — mT5 (250k), PaLM (256k), GPT-4 (~100k), Command A, DeepSeek, Qwen — use 100k–250k so non-English text is not shredded into tiny fragments. Monolingual vocabs don't need to be huge, but multilingual ones do."
    },
    {
      type: "single",
      question: "What is the best account of what weight decay does in LM pretraining?",
      options: [
        "It interacts with the learning-rate schedule, improving optimization late in training",
        "It mainly prevents overfitting, since LMs repeat their corpus for many epochs",
        "It has no measurable effect on final loss and survives purely by convention",
        "It exists mainly to keep BF16 weight magnitudes from overflowing during training"
      ],
      correct: [0],
      explanation: "Modern LMs see roughly one epoch of a huge corpus, so the classic overfitting story barely applies; instead, weight decay's effect shows up through its coupling with learning-rate decay, yielding lower TRAINING loss by the end of the schedule. It is an optimization-dynamics knob here, not a regularizer in the textbook sense."
    },
    // ---------- Stability tricks ----------
    {
      type: "multi",
      question: "Which statements about training-stability tricks are correct? (Select all that apply)",
      options: [
        "Z-loss penalizes $\\log^2 Z$, pushing the output softmax normalizer toward 1",
        "QK-norm applies a LayerNorm/RMSNorm to queries and keys before the attention logits",
        "Logit soft-capping bounds logits at a maximum value via a tanh",
        "These tricks target softmaxes, a common source of divergence",
        "Adopting all three lets a model drop learning-rate warmup with no divergence risk"
      ],
      correct: [0, 1, 2, 3],
      explanation: "Softmaxes misbehave because of exponentials and division: z-loss (pioneered by PaLM, used by Baichuan 2, DCLM, OLMo 2) keeps the output normalizer near 1, QK-norm (from vision/multimodal models, now in Gemma 2, DCLM, OLMo 2) normalizes q and k before the attention softmax, and soft-capping squashes logits through a tanh — though the slides note it may cost performance. None of them substitutes for warmup, which addresses early-training dynamics more broadly."
    },
    // ---------- Attention variants ----------
    {
      type: "single",
      question: "In sliding-window attention (e.g., Mistral), each layer attends only to a local window. How is long-range information still captured?",
      options: [
        "Tokens outside the window are summarized into a single memory token",
        "Stacking layers extends the effective context, window by window",
        "The KV cache secretly retains every past token for all layers",
        "The attention window is widened exponentially in the later layers"
      ],
      correct: [1],
      explanation: "The slides note that depth extends the effective context: information hops one window per layer, so deep stacks reach far beyond a single window. The current standard refinement interleaves full and local attention — Cohere Command A makes every 4th layer full attention (NoPE for long-range, RoPE + SWA for short-range), and Llama 4 and Gemma do SWA plus full-attention RoPE layers."
    },
    {
      type: "single",
      question: "What is the primary motivation for multi-query and grouped-query attention (MQA/GQA)?",
      options: [
        "Reducing the training FLOPs spent in the attention matrix multiplies",
        "Removing the need for explicit positional information in attention",
        "Shrinking the KV cache, easing the memory bandwidth of decoding",
        "Improving quality by sharing gradients across the query heads"
      ],
      correct: [2],
      explanation: "Autoregressive generation is memory-bound: every new token re-reads the whole KV cache, so arithmetic intensity is terrible. Sharing K/V across query heads cuts the cache by the group factor, directly speeding up inference. Attention FLOPs are essentially unchanged — the win is bandwidth, not compute."
    },
    {
      type: "multi",
      question: "Which statements about MQA/GQA are true? (Select all that apply)",
      options: [
        "MQA shares a single key/value head across all query heads",
        "GQA interpolates, giving several query heads per KV head and recovering most quality",
        "The KV-cache savings matter mainly at inference, not for training loss",
        "GQA reduces total attention FLOPs by the same factor as the group size"
      ],
      correct: [0, 1, 2],
      explanation: "MQA is the extreme point (one KV head); GQA picks an intermediate KV-head count and empirically sits near full multi-head quality while keeping most of the cache savings. The savings are in memory traffic and cache size — the q·k and attention·v computations still involve all query heads, so FLOPs barely change."
    },
    {
      type: "single",
      question: "Where in the network is RoPE actually applied?",
      options: [
        "To the token embeddings once, just before the first transformer layer",
        "To queries, keys, and values in every attention layer",
        "To queries and keys in every attention layer, but not to values",
        "Only in the final layer's attention block, right before the output head"
      ],
      correct: [2],
      explanation: "The rotation must hit exactly the vectors whose inner product forms the attention logits — q and k — in every layer; rotating v would scramble the content that gets mixed. This is unlike sinusoidal or learned absolute embeddings, which are added to the input once and then fade through the residual stream."
    },
    // ---------- Consensus vs variation ----------
    {
      type: "single",
      question: "Why do some models (GPT-J, PaLM, GPT-NeoX) run attention and the FFN in parallel instead of serially?",
      options: [
        "Ablations show parallel blocks reach lower loss at matched compute",
        "It halves the number of residual connections needed per block",
        "It lets the block skip the second LayerNorm and its parameters",
        "The shared LayerNorm and fusable matmuls give a systems win"
      ],
      correct: [3],
      explanation: "In a parallel block both sublayers read the same input, so one LayerNorm can be shared and the input-side matrix multiplies can be fused — a wall-clock win, first used in GPT-J and later in PaLM and recent models like Cohere Command A and Falcon 2 11B. The slides note there are no extremely serious ablations showing a quality advantage; the motivation is compute, not loss."
    },
    {
      type: "single",
      question: "What did the original post-norm transformer typically require to train successfully?",
      options: [
        "Careful learning-rate warmup to avoid divergence early in training",
        "No warmup at all, since its gradients were naturally bounded",
        "Much larger batch sizes than pre-norm models need",
        "Freezing the embedding layer for the first part of training"
      ],
      correct: [0],
      explanation: "Post-norm's norm-on-the-residual-path gives badly scaled gradients at initialization, so training needed a slow warmup ramp (and even then could be fragile at depth). Pre-norm was introduced largely because it trains reliably with little or no warmup — the opposite of option two's claim."
    },
    {
      type: "multi",
      question: "Which choices are near-consensus in today's 'standard' LM architecture? (Select all that apply)",
      options: [
        "A gated SwiGLU-family activation in the FFN",
        "Dropping bias terms from the linear layers",
        "Rotary position embeddings",
        "ReLU with $d_{\\text{ff}} = 4\\,d_{\\text{model}}$, as in the original transformer",
        "Sinusoidal position embeddings added at the input"
      ],
      correct: [0, 1, 2],
      explanation: "The modern recipe — visible across Llama-family and most recent open models — is bias-free linear layers, a gated activation, and RoPE, alongside RMSNorm and pre-norm residuals. Plain ReLU FFNs and input-added sinusoidal embeddings are the 2017 defaults that this consensus displaced."
    },
    {
      type: "single",
      question: "Which model is a famous outlier to the $d_{\\text{ff}} \\approx 4\\,d_{\\text{model}}$ convention?",
      options: [
        "GPT-3, with a ratio near 1",
        "T5 (11B), with a ratio of 64",
        "Llama 2, with a ratio of 32",
        "BERT, with a ratio of 16"
      ],
      correct: [1],
      explanation: "T5's 11B configuration paired $d_{\\text{model}} = 1024$ with $d_{\\text{ff}} = 65{,}536$ — a 64x expansion — and still trained to strong results, showing these conventions are soft preferences rather than hard requirements. GPT-3, Llama 2, and BERT all sit at or near the standard 4x (Llama's gated FFN near $\\frac{8}{3}d$)."
    }
  ]
};

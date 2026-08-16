// CS336 Lecture 2 — PyTorch & Resource Accounting
// 20 questions covering: floating-point formats (fp32/fp16/bf16/fp8), strides,
// views vs copies, einops, parameter initialization, FLOPs accounting (2mnp,
// 6ND), MFU, training memory, mixed-precision training, optimizers, and
// training best practices (pinned memory, memmap, checkpointing, seeding).
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture2"] = {
  title: "Lecture 2 — PyTorch & Resource Accounting",
  questions: [
    // ---------- Floating-point formats ----------
    {
      type: "single",
      question: "What is PyTorch's default floating-point dtype, and how much memory does each element take?",
      options: [
        "float16, at 2 bytes per element",
        "float32, at 4 bytes per element",
        "float64, at 8 bytes per element",
        "bfloat16, at 2 bytes per element"
      ],
      correct: [1],
      explanation: "PyTorch defaults to float32 (single precision): 1 sign bit, 8 exponent bits, 23 mantissa bits, 4 bytes per element. It is the safe baseline for numerics, but for large models memory and bandwidth pressure push training toward half-precision formats."
    },
    {
      type: "single",
      question: "What is the main danger of training a model entirely in fp16?",
      options: [
        "Modern tensor cores cannot run matrix multiplies on fp16 inputs",
        "Its narrow exponent range makes small values underflow to zero",
        "It stores each element in 4 bytes, so no memory is saved over fp32",
        "Its 10 mantissa bits round large integers, corrupting token indices"
      ],
      correct: [1],
      explanation: "fp16 has only 5 exponent bits, so its dynamic range is small: the lecture shows torch.tensor([1e-8], dtype=torch.float16) silently underflowing to zero. When that happens to gradients or activations during training, you get instability."
    },
    {
      type: "multi",
      question: "Which statements about bfloat16 (bf16) are true? (Select all that apply)",
      options: [
        "Its 8 exponent bits give roughly the same dynamic range as fp32",
        "It has fewer mantissa bits than fp16, so less precision per value",
        "It stores each element in 4 bytes, exactly matching fp32's footprint",
        "It keeps fp32's full precision while halving the exponent's bit width"
      ],
      correct: [0, 1],
      explanation: "bf16, developed by Google Brain in 2018, uses the same 2 bytes as fp16 but keeps fp32's dynamic range — the lecture shows 1e-8 surviving in bf16 where fp16 underflows. The catch is worse resolution (only 7 mantissa bits), which matters less for deep learning."
    },
    {
      type: "single",
      question: "Which statement about fp8 is accurate?",
      options: [
        "H100s support two one-byte variants of it, E4M3 and E5M2",
        "Its E5M2 variant has a wider dynamic range than bf16",
        "PyTorch makes it the default dtype on H100-class hardware",
        "It was standardized in 2016 for graphics rather than ML use"
      ],
      correct: [0],
      explanation: "FP8 was standardized in 2022, motivated by machine learning workloads. H100s support two variants — E4M3 (range $[-448, 448]$) and E5M2 (range $[-57344, 57344]$) — squeezing a float into a single byte at the cost of precision that must be managed carefully during training."
    },

    // ---------- Parameter initialization ----------
    {
      type: "single",
      question: "When initializing a weight matrix with i.i.d. normal entries, why does the lecture rescale them by $1/\\sqrt{d_{\\text{in}}}$ (up to a constant, Xavier initialization)?",
      options: [
        "It keeps the output scale constant in $d_{\\text{in}}$, avoiding value blowup",
        "It keeps the weight matrix approximately orthogonal during training",
        "It makes the loss surface convex so SGD converges in fewer steps",
        "It clips every weight into $[-3, 3]$ so no single entry dominates"
      ],
      correct: [0],
      explanation: "Without rescaling, each output element scales as $\\sqrt{d_{\\text{in}}}$, and large activations can make gradients blow up and training unstable. Dividing by $\\sqrt{d_{\\text{in}}}$ makes the output scale invariant to the input dimension; for extra safety the lecture also truncates the normal to $[-3, 3]$ to rule out outliers."
    },
    {
      type: "single",
      question: "A contiguous $4 \\times 3$ float32 matrix is stored row-major. What are its strides, in elements?",
      options: [
        "$(3, 1)$",
        "$(1, 3)$",
        "$(4, 1)$",
        "$(12, 4)$"
      ],
      correct: [0],
      explanation: "Moving down one row skips a full row of 3 elements, and moving one column skips 1 element, so the strides are $(3, 1)$. Element $(r, c)$ lives at flat index $r \\cdot \\text{stride}(0) + c \\cdot \\text{stride}(1)$; transposing just swaps the strides without touching the data."
    },
    {
      type: "multi",
      question: "Which operations return a view that shares storage with the original tensor, rather than a copy? (Select all that apply)",
      options: [
        "Transposing with x.transpose(0, 1)",
        "Slicing, such as x[0] or x[:, 1]",
        "Calling x.contiguous() on a non-contiguous tensor",
        "Calling x.view(...) on a contiguous tensor",
        "Elementwise arithmetic such as x + 1"
      ],
      correct: [0, 1, 3],
      explanation: "Transpose, slicing, and view only rewrite shape and stride metadata, so they are free and mutations show through to the original. contiguous() on a non-contiguous tensor must materialize a fresh laid-out copy, and arithmetic like x + 1 allocates a new result tensor."
    },

    // ---------- Einops ----------
    {
      type: "single",
      question: "In einsum(x, y, \"batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2\"), what happens to the hidden dimension?",
      options: [
        "It is broadcast, appearing unchanged in the output",
        "It raises an error unless reduced beforehand",
        "It is summed over, because it is not named in the output",
        "It is flattened together with seq2 into one dimension"
      ],
      correct: [2],
      explanation: "Einsum is generalized matrix multiplication with good bookkeeping: any input dimension not named in the output pattern is summed over. This replaces error-prone code like x @ y.transpose(-2, -1), and \"...\" can stand in for any number of broadcast dimensions."
    },

    // ---------- FLOPs accounting ----------
    {
      type: "single",
      question: "Multiplying an $m \\times n$ matrix by an $n \\times p$ matrix costs how many FLOPs?",
      options: [
        "$mnp$",
        "$2(mn + np)$",
        "$2mnp$",
        "$mnp^2$"
      ],
      correct: [2],
      explanation: "Each of the $m \\cdot p$ output entries needs $n$ multiplications and $n$ additions, giving $2mnp$ FLOPs total. This single rule dominates deep learning cost accounting, since matrix multiplies dwarf every other operation in a large model."
    },
    {
      type: "single",
      question: "For a model with $N$ parameters processing $D$ tokens, the forward pass costs approximately how many FLOPs?",
      options: [
        "$ND$",
        "$4ND$",
        "$6ND$",
        "$2ND$"
      ],
      correct: [3],
      explanation: "Each parameter participates in roughly one multiply and one add per token, giving about $2 \\cdot (\\#\\text{tokens}) \\cdot (\\#\\text{parameters})$ FLOPs for the forward pass. The lecture derives this for a linear model and notes it generalizes to Transformers to a first-order approximation."
    },
    {
      type: "multi",
      question: "Which statements about the FLOPs of one training step are true? (Select all that apply)",
      options: [
        "The backward pass costs about twice the forward pass",
        "The backward pass is nearly free because activations are cached",
        "Total training compute is approximately $6ND$ FLOPs",
        "Backward is 2× forward because gradients are computed w.r.t. both inputs and weights"
      ],
      correct: [0, 2, 3],
      explanation: "Each layer's backward pass performs two matmuls — one for the gradient w.r.t. the layer input (to keep backpropagating) and one w.r.t. the weights — so backward costs about $4ND$ versus forward's $2ND$. Cached activations save recomputation but not these matmuls, and the total lands at the famous $6ND$."
    },
    {
      type: "single",
      question: "Training a 70B-parameter model on 15T tokens requires roughly how many total FLOPs?",
      options: [
        "$1.05 \\times 10^{24}$",
        "$2.1 \\times 10^{24}$",
        "$6.3 \\times 10^{24}$",
        "$4.2 \\times 10^{24}$"
      ],
      correct: [2],
      explanation: "Use $6ND$: $6 \\times (70 \\times 10^{9}) \\times (15 \\times 10^{12}) = 6.3 \\times 10^{24}$ FLOPs. The lecture's napkin math continues: at 50% MFU on 1024 H100s, this pins down the number of training days before committing any hardware."
    },

    // ---------- MFU ----------
    {
      type: "single",
      question: "What is Model FLOPs Utilization (MFU)?",
      options: [
        "Achieved useful FLOP/s divided by the hardware's promised peak FLOP/s",
        "The fraction of GPU memory bandwidth consumed by matrix multiplies",
        "The percentage of wall-clock training time spent inside matmul kernels",
        "The ratio of forward-pass FLOPs to combined forward-backward FLOPs"
      ],
      correct: [0],
      explanation: "MFU divides the FLOP rate your model actually achieves (counting only the useful model FLOPs) by the hardware's promised peak, ignoring communication and overhead. It measures how much of the silicon you paid for is doing real model math rather than waiting on memory or communication."
    },
    {
      type: "single",
      question: "What is a typical, realistically achievable MFU for large-model training?",
      options: [
        "Well-tuned jobs routinely reach 99%",
        "Around 50% is considered quite good",
        "Anything above 5% is considered excellent",
        "Above 100%, thanks to tensor cores"
      ],
      correct: [1],
      explanation: "Real training is limited by memory bandwidth, communication, and non-matmul overhead, so peak FLOP/s is never reached. The lecture calls an MFU of at least 0.5 quite good — and it will be higher when matmuls dominate the workload."
    },

    // ---------- Training memory ----------
    {
      type: "multi",
      question: "When training with Adam, which items contribute to GPU memory use? (Select all that apply)",
      options: [
        "The compiled CUDA kernels, at 4 bytes per parameter",
        "Model parameters and their gradients",
        "Adam's first and second moment estimates ($m$ and $v$)",
        "A full duplicate of the training dataset in GPU memory",
        "Activations saved from the forward pass for backpropagation"
      ],
      correct: [1, 2, 4],
      explanation: "Training memory = parameters + gradients + optimizer state + activations. Adam keeps two extra tensors the size of the model ($m$ and $v$), and activations must be stashed during forward so backward can use them; the dataset streams in per batch and kernels are negligible."
    },
    {
      type: "single",
      question: "Training fully in fp32 with AdamW, how many bytes per parameter do parameters, gradients, and optimizer state require (excluding activations)?",
      options: [
        "8 bytes",
        "12 bytes",
        "16 bytes",
        "20 bytes"
      ],
      correct: [2],
      explanation: "Each parameter needs 4 bytes for the weight, 4 for its gradient, and 4 each for the two optimizer states: $4 + 4 + 4 + 4 = 16$ bytes total. The lecture uses this for napkin math — 8 H100s give 640 GB, so naively the largest trainable model is about 40B parameters, before counting activations."
    },
    {
      type: "single",
      question: "One matrix in the feedforward layer of GPT-3 ($49{,}152 \\times 12{,}288$) stored in float32 takes roughly how much memory?",
      options: [
        "0.6 GB",
        "1.2 GB",
        "2.3 GB",
        "4.6 GB"
      ],
      correct: [2],
      explanation: "Memory = number of elements × bytes per element: $49{,}152 \\times 12{,}288 \\approx 6 \\times 10^{8}$ values at 4 bytes each $\\approx 2.3$ GB — and that is a single matrix in a single layer. This is why dtype matters: the same matrix in bfloat16 would take half the memory."
    },

    // ---------- Mixed precision ----------
    {
      type: "multi",
      question: "Which practices are part of the mixed-precision training recipe from the lecture? (Select all that apply)",
      options: [
        "Run the forward pass (activations) in bf16 or fp8",
        "Keep parameters and gradients in float32",
        "Store the optimizer state in fp8 to save memory",
        "Use PyTorch's automatic mixed precision (AMP) library to automate dtype choices"
      ],
      correct: [0, 1, 3],
      explanation: "The concrete plan: default to float32, but use cheap formats like bfloat16 or fp8 where possible — the forward pass activations — while keeping the numerically fragile parameters and gradients in float32. PyTorch's AMP automates this, and NVIDIA's Transformer Engine brings fp8 to linear layers; fp8 optimizer state is not part of the recipe."
    },

    // ---------- Optimizers ----------
    {
      type: "single",
      question: "According to the lecture's optimizer family tree, how is Adam constructed?",
      options: [
        "Adam = AdaGrad + momentum",
        "Adam = SGD + second-moment normalization only",
        "Adam = momentum + learning-rate warmup",
        "Adam = RMSProp + momentum"
      ],
      correct: [3],
      explanation: "The lecture builds the family incrementally: momentum is SGD plus exponential averaging of gradients; AdaGrad is SGD plus normalization by accumulated squared gradients; RMSProp replaces that accumulation with an exponential average; and Adam is RMSProp plus momentum."
    },

    // ---------- Training best practices ----------
    {
      type: "multi",
      question: "Which statements about the lecture's training best practices are true? (Select all that apply)",
      options: [
        "Pinned CPU memory enables asynchronous copies of batches to the GPU",
        "np.memmap lazily loads only the accessed parts of a huge dataset",
        "Checkpoints should save both the model and optimizer state",
        "Setting torch's random seed alone makes all randomness reproducible"
      ],
      correct: [0, 1, 2],
      explanation: "Pinning memory lets you copy a batch to the GPU with non_blocking=True while fetching the next batch on the CPU, and memmap avoids loading a multi-terabyte token array into RAM. Long training runs will crash, so checkpoints must capture model and optimizer state — and reproducibility requires seeding torch, NumPy, and Python's random module, not torch alone."
    }
  ]
};

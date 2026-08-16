// Lecture manifest — one 20-question quiz per non-guest lecture of CS336
// (Spring 2025). To add or revise a lecture:
//   1. Create data/lecture<N>.js that sets window.QUIZ_DATA["lecture<N>"] = { title, questions }
//   2. Keep the entry here in sync (topics shown under the lecture name).
window.LECTURES = [
  {
    id: "lecture1",
    tag: "Lecture 1",
    title: "Overview & Tokenization",
    topics: ["Course philosophy", "Efficiency framing", "Bytes & Unicode", "BPE", "Vocab trade-offs"],
    questionCount: 20,
    file: "data/lecture1.js"
  },
  {
    id: "lecture2",
    tag: "Lecture 2",
    title: "PyTorch & Resource Accounting",
    topics: ["fp16/bf16/fp8", "Matmul FLOPs", "6ND & MFU", "Memory accounting", "Mixed precision"],
    questionCount: 20,
    file: "data/lecture2.js"
  },
  {
    id: "lecture3",
    tag: "Lecture 3",
    title: "Architectures & Hyperparameters",
    topics: ["Pre-norm & RMSNorm", "SwiGLU", "RoPE", "Hyperparameter ratios", "Stability tricks", "GQA"],
    questionCount: 20,
    file: "data/lecture3.js"
  },
  {
    id: "lecture4",
    tag: "Lecture 4",
    title: "Mixture of Experts",
    topics: ["Sparse FLOPs", "Routing & balancing", "DeepSeek MoE & MLA", "Fine-tuning MoEs", "Upcycling"],
    questionCount: 20,
    file: "data/lecture4.js"
  },
  {
    id: "lecture5",
    tag: "Lecture 5",
    title: "GPUs",
    topics: ["SMs & warps", "Memory hierarchy", "Roofline", "Tensor cores", "FlashAttention"],
    questionCount: 20,
    file: "data/lecture5.js"
  },
  {
    id: "lecture6",
    tag: "Lecture 6",
    title: "Kernels & Triton",
    topics: ["Benchmarking & profiling", "Kernel fusion", "Triton model", "Online softmax", "FlashAttention"],
    questionCount: 20,
    file: "data/lecture6.js"
  },
  {
    id: "lecture7",
    tag: "Lecture 7",
    title: "Parallelism I",
    topics: ["Collectives", "ZeRO 1/2/3", "FSDP", "Tensor & pipeline parallel", "3D rules of thumb"],
    questionCount: 20,
    file: "data/lecture7.js"
  },
  {
    id: "lecture8",
    tag: "Lecture 8",
    title: "Parallelism II",
    topics: ["Collectives in code", "NCCL & NVLink", "Data & tensor parallel", "Pipeline & micro-batches", "Parallelism axes"],
    questionCount: 20,
    file: "data/lecture8.js"
  },
  {
    id: "lecture9",
    tag: "Lecture 9",
    title: "Scaling Laws I",
    topics: ["Power laws", "Data scaling theory", "Critical batch size", "Kaplan vs Chinchilla", "IsoFLOP"],
    questionCount: 20,
    file: "data/lecture9.js"
  },
  {
    id: "lecture10",
    tag: "Lecture 10",
    title: "Inference",
    topics: ["Arithmetic intensity", "KV cache", "PagedAttention", "Speculative decoding", "Quantization"],
    questionCount: 20,
    file: "data/lecture10.js"
  },
  {
    id: "lecture11",
    tag: "Lecture 11",
    title: "Scaling Laws II",
    topics: ["MiniCPM & muP", "WSD schedules", "Chinchilla in practice", "Batch & LR scaling", "muP conditions"],
    questionCount: 20,
    file: "data/lecture11.js"
  },
  {
    id: "lecture12",
    tag: "Lecture 12",
    title: "Evaluation",
    topics: ["Perplexity", "MMLU & GPQA", "Arena & LM judges", "Agentic evals", "Contamination"],
    questionCount: 20,
    file: "data/lecture12.js"
  },
  {
    id: "lecture13",
    tag: "Lecture 13",
    title: "Data I",
    topics: ["Common Crawl & WARC", "C4 & quality rules", "Model-based filtering", "Copyright", "Mid-training"],
    questionCount: 20,
    file: "data/lecture13.js"
  },
  {
    id: "lecture14",
    tag: "Lecture 14",
    title: "Data II",
    topics: ["KenLM & CCNet", "fastText & DSIR", "Bloom filters", "MinHash & LSH", "phi-1 & synthetic"],
    questionCount: 20,
    file: "data/lecture14.js"
  },
  {
    id: "lecture15",
    tag: "Lecture 15",
    title: "Alignment: SFT & RLHF",
    topics: ["SFT datasets", "Midtraining recipes", "Reward models & RLHF", "DPO", "Length & style effects"],
    questionCount: 20,
    file: "data/lecture15.js"
  },
  {
    id: "lecture16",
    tag: "Lecture 16",
    title: "Alignment: RL",
    topics: ["Verifiable rewards", "Policy gradient", "PPO & GRPO", "R1 & K1.5", "Overoptimization"],
    questionCount: 20,
    file: "data/lecture16.js"
  },
  {
    id: "lecture17",
    tag: "Lecture 17",
    title: "Alignment: RL II",
    topics: ["LM as MDP", "Policy gradient derivation", "Baselines & advantage", "GRPO", "Sorting walkthrough"],
    questionCount: 20,
    file: "data/lecture17.js"
  }
];

// Lecture manifest. To add a new lecture:
//   1. Create data/<id>.js that sets window.QUIZ_DATA["<id>"] = { title, questions }
//   2. Add an entry here.
window.LECTURES = [
  {
    id: "lecture10-basics",
    tag: "Lecture 10",
    title: "Inference — Basics",
    description: "Arithmetic intensity, prefill vs. decode, KV cache, batching, PagedAttention, speculative decoding, MQA/GQA/MLA, quantization, pruning, and alternative architectures.",
    questionCount: 50,
    file: "data/lecture10-basics.js"
  },
  {
    id: "lecture10-advances",
    tag: "Lecture 10",
    title: "Inference — Advances",
    description: "Interview-level questions: roofline math, KV cache sizing, MLA weight absorption, speculative decoding theory, GPTQ/fp8/KV quantization, chunked prefill, disaggregation, FlashDecoding, MoE serving, and SLO-driven design.",
    questionCount: 50,
    file: "data/lecture10-advances.js"
  }
];

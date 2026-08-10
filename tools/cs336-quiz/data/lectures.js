// Lecture manifest. To add a new lecture:
//   1. Create data/<id>.js that sets window.QUIZ_DATA["<id>"] = { title, questions }
//   2. Add an entry here. Quizzes sharing a `tag` are grouped into one row;
//      `topics` (shown under the lecture name) is taken from the first entry.
window.LECTURES = [
  {
    id: "lecture10-basics",
    tag: "Lecture 10",
    title: "Inference — Basics",
    topics: ["Arithmetic intensity", "KV cache", "PagedAttention", "Speculative decoding", "Quantization"],
    questionCount: 50,
    file: "data/lecture10-basics.js"
  },
  {
    id: "lecture10-advances",
    tag: "Lecture 10",
    title: "Inference — Advances",
    questionCount: 50,
    file: "data/lecture10-advances.js"
  }
];

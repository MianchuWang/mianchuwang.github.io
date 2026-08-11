// Lecture manifest. To add a new lecture:
//   1. Create data/<id>.js that sets window.QUIZ_DATA["<id>"] = { title, questions }
//   2. Add an entry here. Quizzes sharing a `tag` are grouped into one row;
//      `topics` (shown under the lecture name) is taken from the first entry.
window.LECTURES = [
  {
    id: "lecture9-basics",
    tag: "Lecture 9",
    title: "Scaling Laws I — Basics",
    topics: ["Power laws", "Data scaling theory", "Critical batch size", "Kaplan vs Chinchilla", "IsoFLOP"],
    questionCount: 50,
    file: "data/lecture9-basics.js"
  },
  {
    id: "lecture9-advances",
    tag: "Lecture 9",
    title: "Scaling Laws I — Advances",
    comingSoon: true
  },
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
  },
  {
    id: "lecture11-basics",
    tag: "Lecture 11",
    title: "Scaling Laws II — Basics",
    topics: ["MiniCPM & muP", "WSD schedules", "Chinchilla in practice", "Step Law", "Optimizer scaling"],
    questionCount: 50,
    file: "data/lecture11-basics.js"
  },
  {
    id: "lecture11-advances",
    tag: "Lecture 11",
    title: "Scaling Laws II — Advances",
    comingSoon: true
  }
];

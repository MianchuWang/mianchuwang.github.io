# CS336 Quiz (T260810)

A self-testing site for the Stanford CS336 (Language Modeling from Scratch, Spring 2025)
lecture series: one quiz of **20 key questions** per non-guest lecture (lectures 1–16).
Pure HTML/CSS/JS — no build step, no dependencies. Results (last and best score per
lecture) are saved in the browser's `localStorage` under `cq:result:<id>`.

## Run

Part of the personal site — serve the site root and open `/tools/cs336-quiz/`:

```bash
python3 -m http.server 8644
```

Then open <http://localhost:8644/tools/cs336-quiz/>.

## Features

- One question at a time, single- and multiple-answer questions
- The explanation is always shown after checking, whether right or wrong
- Progress bar, final score with previous-best comparison, and a review list of missed
  questions with explanations
- Per-lecture results on the landing page (best score + when last taken), with reset
- LaTeX math in questions/options/explanations — `$...$` inline, `$$...$$` display —
  rendered by the site-wide KaTeX helper (`assets/math.js`)

## Revising a lecture

1. Edit `data/lecture<N>.js` (each sets `window.QUIZ_DATA["lecture<N>"]`):

   ```js
   window.QUIZ_DATA = window.QUIZ_DATA || {};
   window.QUIZ_DATA["lecture<N>"] = {
     title: "Lecture N — Topic",
     questions: [
       {
         type: "single",            // or "multi"
         question: "…?",
         options: ["A…", "B…", "C…", "D…"],
         correct: [0],              // indices of correct options
         explanation: "Why this is the answer."
       }
     ]
   };
   ```

2. Keep the matching entry in `data/lectures.js` in sync (`topics`, `questionCount`),
   and bump `DATA_VERSION` in `js/quiz.js` so browsers refetch the data.

Convention: exactly 20 questions per lecture — the essential knowledge, not exhaustive
coverage. Questions may carry an `image` (path under `data/figures/`) and
`imageCredit`.

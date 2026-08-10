# CS336 Quiz

A simple, reusable self-testing site for the Stanford CS336 (Language Modeling from Scratch) lecture series. Pure HTML/CSS/JS — no build step, no dependencies.

## Run

Part of the personal site — serve the site root and open
`/tools/cs336-quiz/`:

```bash
python3 -m http.server 8642 --directory /Users/mianchu/Desktop/CS336-Quiz
```

Then open <http://localhost:8642/tools/cs336-quiz/>.

## Features

- One question at a time, single- and multiple-answer questions
- The explanation is always shown after checking, whether right or wrong
- Progress bar, final score, and a review list of missed questions with explanations
- Retry without reloading

## Adding a new lecture

1. Create `data/<id>.js` (copy `data/lecture10-basics.js` as a template):

   ```js
   window.QUIZ_DATA = window.QUIZ_DATA || {};
   window.QUIZ_DATA["<id>"] = {
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

2. Register it in `data/lectures.js` by adding an entry with the same `id` and `file: "data/<id>.js"`.

That's it — the card appears on the landing page and `quiz.html?lecture=<id>` serves it.

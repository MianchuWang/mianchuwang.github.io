// Generic quiz engine. Loads a lecture's question file based on ?lecture=<id>,
// renders one question at a time, and always shows the explanation after checking.
// Answered questions can be revisited: navigation restores the locked state.
// Question, option, and explanation text may contain $...$ / $$...$$ math.
import { renderMathIn } from "../../../assets/math.js";

(function () {
  const params = new URLSearchParams(window.location.search);
  const lectureId = params.get("lecture");
  const manifest = (window.LECTURES || []).find(l => l.id === lectureId);

  const titleEl = document.getElementById("quiz-title");
  const questionArea = document.getElementById("question-area");
  const resultArea = document.getElementById("result-area");
  const progressFill = document.getElementById("progress-fill");
  const progressText = document.getElementById("progress-text");

  if (!manifest) {
    titleEl.textContent = "Lecture not found";
    questionArea.innerHTML = '<p>Unknown lecture. <a href="index.html">Back to all lectures</a>.</p>';
    return;
  }

  if (manifest.comingSoon || !manifest.file) {
    titleEl.textContent = manifest.title;
    questionArea.innerHTML = '<p>This quiz is coming soon. <a href="index.html">Back to all lectures</a>.</p>';
    return;
  }

  // Dynamically load the lecture's data file, then start.
  const DATA_VERSION = "study8"; // bump when question data changes
  const script = document.createElement("script");
  script.src = manifest.file + "?v=" + DATA_VERSION;
  script.onload = init;
  script.onerror = () => {
    titleEl.textContent = "Failed to load quiz data";
  };
  document.body.appendChild(script);

  let quiz, questions;
  let current = 0;
  const selections = []; // per-question Set of selected ORIGINAL option indices
  const answered = [];   // per-question boolean: has been checked
  const results = [];    // per-question boolean: was correct
  const orders = [];     // per-question display order (shuffled original indices)

  const LETTERS = ["A", "B", "C", "D", "E", "F"];

  function shuffled(n) {
    const arr = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  function init() {
    quiz = window.QUIZ_DATA[lectureId];
    questions = quiz.questions;
    titleEl.textContent = quiz.title;
    document.title = quiz.title + " · CS336 Learning Tools";
    document.addEventListener("keydown", onKeydown);
    renderQuestion();
  }

  // Keyboard flow: 1-4 / a-d toggle an answer, Enter checks or advances,
  // arrow keys move between questions.
  function onKeydown(e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (!resultArea.classList.contains("hidden")) return;
    const keyMap = { "1": 0, "2": 1, "3": 2, "4": 3, "a": 0, "b": 1, "c": 2, "d": 3 };
    const k = e.key.toLowerCase();
    if (k in keyMap) {
      const opts = questionArea.querySelectorAll(".option");
      if (keyMap[k] < opts.length) {
        opts[keyMap[k]].click();
        e.preventDefault();
      }
    } else if (e.key === "Enter" || e.key === "Return" || e.keyCode === 13) {
      const btns = [...document.querySelectorAll(".quiz-actions .btn-primary")];
      const active = btns.find(b => !b.classList.contains("hidden") && !b.disabled);
      if (active) {
        active.click();
        e.preventDefault();
      }
    } else if (e.key === "ArrowLeft" && current > 0) {
      goTo(current - 1);
      e.preventDefault();
    } else if (e.key === "ArrowRight" && answered[current] && current < questions.length - 1) {
      goTo(current + 1);
      e.preventDefault();
    }
  }

  function goTo(index) {
    current = index;
    renderQuestion();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function updateProgress() {
    const done = answered.filter(Boolean).length;
    progressFill.style.width = (100 * done / questions.length) + "%";
    progressText.textContent = "Question " + (current + 1) + " of " + questions.length;
  }

  function setEqual(a, b) {
    if (a.size !== b.length) return false;
    return b.every(x => a.has(x));
  }

  function renderQuestion() {
    updateProgress();
    const q = questions[current];
    // Reuse this attempt's shuffle and selections so revisiting is stable.
    const order = orders[current] || (orders[current] = shuffled(q.options.length));
    const sel = selections[current] || (selections[current] = new Set());
    const isAnswered = !!answered[current];

    const card = document.createElement("div");
    card.className = "question-card";

    const typeLabel = q.type === "multi" ? "Multiple answers — select all that apply" : "Single answer";
    let head = '<span class="question-type">' + typeLabel + "</span>" +
      '<div class="question-text">' + escapeHtml(q.question) + "</div>";
    if (q.image) {
      head += '<figure class="question-figure"><img src="' + q.image + '" alt="Question figure">' +
        (q.imageCredit ? "<figcaption>" + escapeHtml(q.imageCredit) + "</figcaption>" : "") +
        "</figure>";
    }
    card.innerHTML = head;

    const optionsWrap = document.createElement("div");
    optionsWrap.className = "options";

    order.forEach((i, pos) => {
      const opt = q.options[i];
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "option";
      btn.dataset.index = i;
      btn.innerHTML = '<span class="option-letter">' + LETTERS[pos] + "</span><span>" + escapeHtml(opt) + "</span>";
      if (!isAnswered && sel.has(i)) btn.classList.add("selected");
      btn.addEventListener("click", () => {
        if (answered[current]) return;
        if (q.type === "single") {
          sel.clear();
          optionsWrap.querySelectorAll(".option").forEach(o => o.classList.remove("selected"));
          sel.add(i);
          btn.classList.add("selected");
        } else {
          if (sel.has(i)) { sel.delete(i); btn.classList.remove("selected"); }
          else { sel.add(i); btn.classList.add("selected"); }
        }
        checkBtn.disabled = sel.size === 0;
      });
      optionsWrap.appendChild(btn);
    });
    card.appendChild(optionsWrap);

    const explanationEl = document.createElement("div");
    explanationEl.className = "explanation hidden";
    card.appendChild(explanationEl);

    const actions = document.createElement("div");
    actions.className = "quiz-actions";

    const prevBtn = document.createElement("button");
    prevBtn.className = "btn btn-secondary";
    prevBtn.textContent = "Previous";
    prevBtn.addEventListener("click", () => goTo(current - 1));

    const checkBtn = document.createElement("button");
    checkBtn.className = "btn btn-primary";
    checkBtn.textContent = "Check answer";
    checkBtn.disabled = sel.size === 0;

    const nextBtn = document.createElement("button");
    nextBtn.className = "btn btn-primary hidden";
    nextBtn.textContent = current === questions.length - 1 ? "See results" : "Next question";

    function reveal(scroll) {
      optionsWrap.querySelectorAll(".option").forEach(o => {
        const i = Number(o.dataset.index);
        o.classList.add("locked");
        o.classList.remove("selected");
        const letterEl = o.querySelector(".option-letter");
        if (q.correct.includes(i)) {
          o.classList.add("reveal-correct");
          letterEl.textContent = "✓"; // non-color status cue
        } else if (sel.has(i)) {
          o.classList.add("reveal-incorrect");
          letterEl.textContent = "✗";
        }
      });

      const isCorrect = results[current];
      const correctLetters = q.correct
        .map(i => order.indexOf(i))
        .sort((a, b) => a - b)
        .map(pos => LETTERS[pos])
        .join(", ");
      explanationEl.classList.remove("hidden");
      explanationEl.classList.add(isCorrect ? "correct" : "incorrect");
      explanationEl.innerHTML =
        '<div class="verdict">' + (isCorrect ? "✓ Correct" : "✗ Not quite — correct answer: " + correctLetters) + "</div>" +
        "<div>" + escapeHtml(q.explanation) + "</div>";

      checkBtn.classList.add("hidden");
      nextBtn.classList.remove("hidden");
      renderMathIn(explanationEl);
      if (scroll) explanationEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    checkBtn.addEventListener("click", () => {
      if (sel.size === 0 || answered[current]) return;
      answered[current] = true;
      results[current] = setEqual(sel, q.correct);
      updateProgress();
      reveal(true);
    });

    nextBtn.addEventListener("click", () => {
      if (current === questions.length - 1) {
        renderResults();
      } else {
        goTo(current + 1);
      }
    });

    if (current > 0) actions.appendChild(prevBtn);
    actions.appendChild(checkBtn);
    actions.appendChild(nextBtn);
    card.appendChild(actions);

    questionArea.innerHTML = "";
    questionArea.appendChild(card);

    if (isAnswered) reveal(false);
    renderMathIn(card);
  }

  // Results live in localStorage so the landing page can show best/last per lecture.
  function saveResult(score, total) {
    try {
      const key = "cq:result:" + lectureId;
      const prev = JSON.parse(localStorage.getItem(key) || "null");
      const entry = { score: score, total: total, date: new Date().toISOString() };
      const best = prev && prev.best && prev.best.score >= score ? prev.best : entry;
      localStorage.setItem(key, JSON.stringify({ last: entry, best: best }));
      return prev ? prev.best : null;
    } catch (e) {
      return null;
    }
  }

  function renderResults() {
    const total = questions.length;
    const score = results.filter(Boolean).length;
    const pct = Math.round(100 * score / total);
    const prevBest = saveResult(score, total);

    let message;
    if (pct >= 90) message = "Excellent — you have a strong grasp of this lecture.";
    else if (pct >= 70) message = "Good work — review the missed questions below.";
    else if (pct >= 50) message = "A solid start — worth rewatching the tricky sections.";
    else message = "Keep going — go through the explanations below and try again.";

    questionArea.classList.add("hidden");
    resultArea.classList.remove("hidden");
    progressFill.style.width = "100%";
    progressText.textContent = "Complete";

    const bestNote = prevBest
      ? '<div class="result-best">Previous best: ' + prevBest.score + " / " + prevBest.total +
        (score > prevBest.score ? " — new best!" : "") + "</div>"
      : "";

    let html =
      '<div class="result-card">' +
      '<div class="result-label">Your score</div>' +
      '<div class="result-score">' + score + " / " + total + "</div>" +
      '<div class="result-message">' + message + "</div>" +
      bestNote +
      '<div class="quiz-actions" style="justify-content:center; margin-top:24px;">' +
      '<a class="btn btn-secondary" href="index.html" style="text-decoration:none;">All lectures</a>' +
      '<button class="btn btn-primary" id="retry-btn">Retry quiz</button>' +
      "</div></div>";

    const missed = questions.map((q, i) => ({ q, i })).filter(({ i }) => !results[i]);
    if (missed.length > 0) {
      html += '<div class="review-section"><h2>Review: questions you missed (' + missed.length + ")</h2>";
      missed.forEach(({ q, i }) => {
        const optText = idxs => idxs.map(x => escapeHtml(q.options[x])).join(" &middot; ") || "—";
        const yourAns = optText([...selections[i]].sort((a, b) => a - b));
        const correctAns = optText(q.correct);
        html +=
          '<div class="review-item">' +
          '<div class="rq">Q' + (i + 1) + ". " + escapeHtml(q.question) + "</div>" +
          '<div class="ra">Your answer: ' + yourAns + "</div>" +
          '<div class="ra">Correct answer: ' + correctAns + "</div>" +
          '<div class="re">' + escapeHtml(q.explanation) + "</div>" +
          "</div>";
      });
      html += "</div>";
    }

    resultArea.innerHTML = html;
    renderMathIn(resultArea);
    document.getElementById("retry-btn").addEventListener("click", () => {
      current = 0;
      results.length = 0;
      selections.length = 0;
      answered.length = 0;
      orders.length = 0;
      resultArea.classList.add("hidden");
      resultArea.innerHTML = "";
      questionArea.classList.remove("hidden");
      renderQuestion();
      window.scrollTo({ top: 0 });
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
})();

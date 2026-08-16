// CS336 Lecture 12 — Evaluation
// 20 questions covering: perplexity (definition, GPT-2 zero-shot, why it's still
// useful), knowledge benchmarks (MMLU, MMLU-Pro, GPQA, Humanity's Last Exam),
// instruction-following evals (Chatbot Arena, IFEval, AlpacaEval, WildBench),
// agent benchmarks (SWE-bench, CyBench), pure reasoning (ARC-AGI), safety
// (capability vs propensity, dual-use, GCG jailbreaks), realism (quizzing vs
// asking), validity (train-test overlap, dataset quality), and methods vs models.
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture12"] = {
  title: "Lecture 12 — Evaluation",
  questions: [
    // ---------- Perplexity ----------
    {
      type: "single",
      question: "What is the perplexity of a language model on a held-out text?",
      options: [
        "The fraction of held-out tokens the model predicts incorrectly",
        "The sum of the log-probabilities the model assigns to each token",
        "$\\exp$ of the average per-token negative log-likelihood",
        "The KL divergence between the model and the true distribution"
      ],
      correct: [2],
      explanation: "A language model is a probability distribution $p(x)$ over token sequences, and perplexity $(1/p(D))^{1/|D|}$ measures whether $p$ assigns high probability to a dataset $D$ — equivalently, $\\exp$ of the mean negative log-likelihood per token. Pre-training minimizes perplexity on the training set, so the obvious evaluation is to measure it on a test set. Lower is better, and a perfect model $p = t$ would reach the entropy $H(t)$ of the true distribution."
    },
    {
      type: "single",
      question: "GPT-2 was trained on WebText and evaluated zero-shot on standard perplexity datasets. What was the outcome?",
      options: [
        "It set state of the art on every dataset, including One Billion Words",
        "Transfer helped on small datasets but not on One Billion Words",
        "Zero-shot perplexity could not be computed without fine-tuning first",
        "It matched dataset-trained LSTMs on all datasets but never beat them"
      ],
      correct: [1],
      explanation: "Unlike earlier papers that trained and tested on splits of the same dataset, GPT-2 trained on 40GB of Reddit-linked web text and evaluated zero-shot — an out-of-distribution evaluation. Transfer from broad training data helps where in-domain data is scarce, but on the large One Billion Word benchmark, in-domain training still won."
    },
    {
      type: "multi",
      question: "Papers shifted toward downstream task accuracy after GPT-2/GPT-3. Which reasons does the lecture give for perplexity still being useful? (Select all that apply)",
      options: [
        "It directly measures usefulness on downstream tasks such as coding and chat",
        "It is smoother than 0/1 task accuracy, which helps when fitting scaling laws",
        "It is universal, capturing nuances that task accuracy might miss entirely",
        "Conditional perplexity on downstream task data can also be measured"
      ],
      correct: [1, 2, 3],
      explanation: "Perplexity is a smooth, graded signal (unlike 0/1 task accuracy), which is why scaling laws are fit on it, and it is universal — the same reason we use it for training. One can even measure conditional perplexity on downstream task data. What it does not do is directly measure downstream usefulness, which is exactly why the field added task benchmarks."
    },

    // ---------- Knowledge benchmarks ----------
    {
      type: "single",
      question: "According to the lecture, what does MMLU really test?",
      options: [
        "Multi-turn instruction following in realistic chat settings",
        "Reasoning isolated from linguistic and world knowledge",
        "Deep language understanding, as its full name suggests",
        "Knowledge across 57 subjects, from math to US history to law"
      ],
      correct: [3],
      explanation: "Despite being named 'Massive Multitask Language Understanding', MMLU is really about testing knowledge, not language understanding. Its multiple-choice questions across 57 subjects were collected by graduate and undergraduate students from freely available online sources, and it was originally evaluated on GPT-3 with few-shot prompting."
    },
    {
      type: "single",
      question: "How does MMLU-Pro differ from the original MMLU?",
      options: [
        "Noisy or trivial questions were removed and choices expanded from 4 to 10",
        "Questions were rewritten by PhD contractors to make them Google-proof",
        "Multimodal questions were added along with short-answer grading",
        "Synthetic constraints were added that can be checked automatically"
      ],
      correct: [0],
      explanation: "MMLU-Pro removed noisy and trivial questions and expanded the four answer choices to ten, and models are evaluated using chain of thought to give them more of a chance. Accuracy drops by 16% to 33% relative to MMLU, so the benchmark is much less saturated."
    },
    {
      type: "single",
      question: "Why is GPQA described as 'Google-proof'?",
      options: [
        "Its questions are withheld entirely from the public internet",
        "It was released after every frontier model's training cutoff",
        "Non-experts reach only about 34% even with Google access",
        "Search engines are blocked during official evaluation runs"
      ],
      correct: [2],
      explanation: "GPQA's questions were written by 61 PhD contractors recruited from Upwork. PhD experts achieve 65% accuracy, but non-experts reach only 34% despite 30 minutes with Google access — searching does not rescue you. For calibration, GPT-4 achieved 39% at the time."
    },
    {
      type: "multi",
      question: "Which statements about Humanity's Last Exam (HLE) are accurate? (Select all that apply)",
      options: [
        "It contains 2500 questions spanning many subjects, including multimodal ones",
        "Question creators shared a $500K prize pool and received co-authorship",
        "Submitted questions were filtered by frontier LLMs plus multiple review stages",
        "It is exclusively multiple-choice, with no short-answer or multimodal items"
      ],
      correct: [0, 1, 2],
      explanation: "HLE crowdsourced 2500 hard questions — multimodal, many subjects — by offering a $500K prize pool and co-authorship to question creators. Submissions passed through frontier-LLM filtering (questions current models can already answer are discarded) and multiple stages of human review. The format mixes multiple-choice with short-answer, not multiple-choice alone."
    },

    // ---------- Instruction-following evals ----------
    {
      type: "single",
      question: "How does Chatbot Arena produce its model rankings?",
      options: [
        "Expert annotators score each model's answers on a fixed multi-criteria rubric",
        "Models are ranked by their average accuracy across a suite of static benchmarks",
        "Users vote between two anonymized models, and pairwise wins feed Elo scores",
        "A judge model assigns each response a quality score from 1 to 10 on a rubric"
      ],
      correct: [2],
      explanation: "A random person from the internet types a prompt, receives responses from two random anonymized models, and rates which is better; Elo scores are computed from the pairwise comparisons. Its distinctive features are live (not static) inputs and the ability to accommodate new models as they appear."
    },
    {
      type: "single",
      question: "What is the key design idea of IFEval?",
      options: [
        "A judge model scores how well each response follows the instruction",
        "Synthetic constraints added to instructions are verified automatically",
        "Instructions are sampled live from real user traffic each week",
        "Responses are graded by running unit tests in a sandboxed environment"
      ],
      correct: [1],
      explanation: "IFEval augments instructions with simple synthetic constraints (formatting, counts, etc.) that a program can check mechanically — but the semantics of the response are not verified. The lecture notes the instructions are fairly simple and the constraints a bit artificial, trading realism for automatic verifiability."
    },
    {
      type: "single",
      question: "How does AlpacaEval score a model on its 805 instructions?",
      options: [
        "Win rate against GPT-4 preview, judged by GPT-4 preview",
        "Elo ratings computed from live pairwise votes by users",
        "Automatic verification of synthetic formatting constraints",
        "Human experts grading each response against a fixed rubric"
      ],
      correct: [0],
      explanation: "AlpacaEval collects 805 instructions from various sources and reports a win rate against GPT-4 preview as judged by GPT-4 preview itself. The lecture flags the potential bias in this setup: the judge is the same model whose outputs form the baseline, so its verdicts may systematically favor familiar styles."
    },
    {
      type: "multi",
      question: "Which statements about WildBench are accurate? (Select all that apply)",
      options: [
        "Its 1024 examples were sourced from one million real human-chatbot conversations",
        "Responses are scored by running unit tests against reference outputs",
        "GPT-4 turbo judges responses using a checklist, like chain-of-thought for judging",
        "Its scores correlate 0.95 with Chatbot Arena rankings"
      ],
      correct: [0, 2, 3],
      explanation: "WildBench selects 1024 examples from a corpus of one million real human-chatbot conversations, making its prompts far more realistic than hand-written instructions. A GPT-4 turbo judge scores responses with a checklist — analogous to chain-of-thought for judging — and the benchmark correlates 0.95 with Chatbot Arena, the de facto sanity check for new benchmarks. No unit tests are involved."
    },

    // ---------- Agent benchmarks ----------
    {
      type: "single",
      question: "In SWE-bench, what must a model (or agent) do for an instance to count as resolved?",
      options: [
        "Answer multiple-choice questions about a repository's structure",
        "Generate a standalone function that a judge model deems correct",
        "Reproduce the reference developer's patch exactly, token for token",
        "Produce a patch to the repository that makes the unit tests pass"
      ],
      correct: [3],
      explanation: "SWE-bench contains 2294 tasks across 12 Python repositories: given the codebase and an issue description, the agent must submit a PR, and the evaluation metric is unit tests. It exemplifies agent evaluation — a language model plus agent scaffolding using tools and iterating over time, scored by outcome rather than text similarity."
    },
    {
      type: "single",
      question: "How does CyBench estimate the difficulty of its 40 Capture the Flag tasks?",
      options: [
        "By the number of competition teams that failed to solve each task",
        "By the first-solve time from the original competitions",
        "By a difficulty rating assigned by GPT-4",
        "By the length of the reference exploit's code"
      ],
      correct: [1],
      explanation: "CyBench comprises 40 Capture the Flag cybersecurity tasks and uses first-solve time — how long the fastest human team took in the original competition — as a measure of task difficulty. This gives a natural, human-grounded difficulty scale for an agentic benchmark."
    },

    // ---------- Pure reasoning ----------
    {
      type: "single",
      question: "What is ARC-AGI, introduced by Francois Chollet in 2019, designed to measure?",
      options: [
        "Knowledge across dozens of academic subjects",
        "Instruction following under synthetic constraints",
        "Reasoning isolated from linguistic and world knowledge",
        "Tool use and iteration over long time horizons"
      ],
      correct: [2],
      explanation: "All the other tasks in the lecture require linguistic and world knowledge; ARC-AGI's abstract grid puzzles try to isolate reasoning from knowledge, on the argument that reasoning captures a purer form of intelligence than memorizing facts. ARC-AGI-2 is a harder successor that current models still struggle with."
    },

    // ---------- Safety ----------
    {
      type: "multi",
      question: "Which claims about safety evaluation does the lecture make? (Select all that apply)",
      options: [
        "For API models, propensity matters — whether the model will comply with harm",
        "For open-weight models, capability matters, since safety can be fine-tuned away",
        "Safety is purely a matter of refusal and is therefore always at odds with capability",
        "CyBench is dual-use: a safety evaluation that is arguably a capability evaluation"
      ],
      correct: [0, 1, 3],
      explanation: "Two aspects of a model reduce safety: capability and propensity. API models can be capable yet refuse, so propensity matters; open-weight models can have safety fine-tuned away, so capability matters. Safety is not simply refusal opposed to capability — reducing hallucinations in a medical setting makes a system both more capable and safer — and capable cybersecurity agents illustrate dual-use."
    },
    {
      type: "single",
      question: "What notable property of Greedy Coordinate Gradient (GCG) jailbreak prompts does the lecture highlight?",
      options: [
        "They require white-box gradient access to the target model at attack time",
        "Optimized on open models like Llama, they transfer to closed models like GPT-4",
        "They only succeed against models that never received safety training",
        "They must be hand-crafted by expert red-teamers over many iterations"
      ],
      correct: [1],
      explanation: "GCG automatically optimizes adversarial prompts to bypass the refusal behavior that safety training instills. Strikingly, prompts optimized against open-weight models such as Llama transfer to closed models such as GPT-4, so an attacker never needs white-box access to the deployed system."
    },

    // ---------- Realism ----------
    {
      type: "single",
      question: "The lecture contrasts 'quizzing' and 'asking' prompts. Why is asking considered more realistic?",
      options: [
        "Asking prompts are easier to verify with automatic metrics",
        "Quizzing prompts contain too much garbage from live traffic",
        "Asking prompts are always multi-turn and adapted to the model",
        "The user doesn't know the answer, so the response has real value"
      ],
      correct: [3],
      explanation: "Quizzing means the user already knows the answer and is testing the system, like a standardized exam — the mode of most benchmarks such as MMLU. Asking means the user genuinely wants the answer, which is how models create value in practice; efforts like Clio (analyzing real usage) and MedHELM (clinical tasks from clinicians rather than exams) push evaluation toward this realism."
    },

    // ---------- Validity ----------
    {
      type: "multi",
      question: "Which statements about evaluation validity does the lecture support? (Select all that apply)",
      options: [
        "Modern LMs come with well-defined train-test splits, as in the ImageNet era",
        "Train-test overlap can be inferred from a model by exploiting exchangeability of data points",
        "Model providers should report train-test overlap as a community norm",
        "Efforts like SWE-bench Verified and Platinum benchmarks fix noisy test examples"
      ],
      correct: [1, 2, 3],
      explanation: "Pre-foundation-model benchmarks like ImageNet and SQuAD had well-defined splits, but today models train on the internet without disclosing their data. The lecture's two routes are inferring overlap from the model itself (exploiting exchangeability of data points) and encouraging reporting norms; separately, dataset-quality efforts like SWE-bench Verified and Platinum benchmarks clean up erroneous test items."
    },

    // ---------- Methods vs models ----------
    {
      type: "single",
      question: "What do the nanogpt speedrun and DataComp-LM have in common, according to the lecture?",
      options: [
        "They evaluate methods under fixed rules rather than final models",
        "They rank competing models by live human preference votes",
        "They measure raw capability with knowledge-heavy questions",
        "They are safety evaluations run by government safety institutes"
      ],
      correct: [0],
      explanation: "Most evaluation today targets models or systems, where anything goes; these are exceptions that evaluate methods. The nanogpt speedrun fixes the data and measures compute time to reach a target validation loss, while DataComp-LM fixes the training pipeline and varies the dataset. Evaluating methods encourages algorithmic innovation, whereas evaluating models serves downstream users."
    },

    // ---------- Takeaways ----------
    {
      type: "multi",
      question: "Which conclusions capture the lecture's takeaways about evaluation? (Select all that apply)",
      options: [
        "There is no one true evaluation; choose it based on what you're trying to measure",
        "Always look at the individual instances and the model's predictions",
        "MMLU accuracy is sufficient on its own, since it spans 57 subjects",
        "Clearly state the rules of the game: are you evaluating methods or models/systems?"
      ],
      correct: [0, 1, 3],
      explanation: "The lecture closes with four takeaways: there is no one true evaluation — it depends on the question you're answering; always inspect individual instances and predictions rather than trusting aggregates; consider many aspects (capabilities, safety, costs, realism); and clearly state the rules of the game. No single benchmark, MMLU included, settles which model is 'best'."
    }
  ]
};

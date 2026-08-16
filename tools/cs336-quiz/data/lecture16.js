// CS336 Lecture 16 — Alignment: RL
// 20 questions covering: why RLVR (RLHF overoptimization, mode collapse), policy
// gradients (REINFORCE, baselines, advantages), PPO, GRPO and its biases (length,
// std division), KL regularization, case studies (DeepSeek R1/R1-Zero, Kimi K1.5,
// Qwen 3), distillation, on-/off-policy trade-offs.
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture16"] = {
  title: "Lecture 16 — Alignment: RL",
  questions: [
    // ---------- Verifiable rewards ----------
    {
      type: "single",
      question: "Why do verifiable rewards (checking a math answer, running unit tests) sidestep the reward hacking that plagues learned reward models?",
      options: [
        "They make the reward signal dense at every token instead of sparse at the end",
        "They guarantee the policy converges to a unique global optimum of the objective",
        "The reward comes from a program, not a learned model that can be exploited",
        "They remove the need for any exploration by the policy during training"
      ],
      correct: [2],
      explanation: "A learned reward model is an imperfect proxy that the policy can over-optimize — finding outputs the model scores highly but humans would not. A programmatic checker (exact answer match, test suite) cannot be fooled by stylistic tricks in the same way, though it can still be gamed if the checker itself is loose. That is why RLVR anchors training in math and code domains."
    },
    // ---------- Policy gradients: REINFORCE ----------
    {
      type: "single",
      question: "In REINFORCE, what does the policy gradient estimator multiply the reward (or return) by?",
      options: [
        "The gradient of the sampled trajectory's log-probability, $\\nabla_\\theta \\log \\pi_\\theta(a|s)$",
        "The gradient of the value function's prediction, $\\nabla_\\theta V_\\theta(s)$, at each state",
        "The KL divergence between the current policy and the frozen reference policy",
        "The squared temporal-difference error of the critic network at each timestep"
      ],
      correct: [0],
      explanation: "REINFORCE uses the score-function (log-derivative) trick: $\\nabla_\\theta E[R] = E[R \\cdot \\nabla_\\theta \\log \\pi_\\theta]$. Sampled trajectories with high reward have their log-probability pushed up, and low-reward trajectories pushed down. It requires no differentiable reward — only the ability to sample and score."
    },
    {
      type: "single",
      question: "Subtracting a baseline $b$ from the return in REINFORCE, i.e., using $(R - b) \\cdot \\nabla \\log \\pi$, is standard practice. What does it change?",
      options: [
        "It reduces bias but increases the variance of the gradient estimate",
        "It makes the estimator biased toward safer, lower-variance actions",
        "It reduces variance while leaving the expected gradient unbiased",
        "It converts the on-policy estimator into an off-policy estimator"
      ],
      correct: [2],
      explanation: "Because $E[b \\cdot \\nabla \\log \\pi] = b \\cdot \\nabla E[1] = 0$ for any action-independent baseline, subtracting $b$ does not change the expected gradient — no bias is introduced. But it can dramatically shrink variance: raw returns are all-positive and noisy, so without a baseline every sampled response gets pushed up, just by different amounts. The centered quantity $R - b$ is called the advantage."
    },
    // ---------- Why RLVR: RLHF failure modes ----------
    {
      type: "multi",
      question: "Why does the lecture pivot from RLHF toward RL on verifiable rewards? (Select all that apply)",
      options: [
        "Overoptimization: pushing a learned reward up eventually hurts true output quality",
        "Mode collapse: models lose calibration and stop being good probabilistic models",
        "In verifiable domains, RL can optimize exactly the quantity we actually care about",
        "PPO fundamentally cannot be implemented for autoregressive language models"
      ],
      correct: [0, 1, 2],
      explanation: "The lecture's RLHF recap flags two persistent problems: optimizing a learned (human or LM) preference reward overfits past a point — reward keeps rising while true quality drops — and RLHF'd models lose their calibration. RLVR is the escape hatch: in domains like math and code the checker measures exactly what we want, so RL can be scaled out without the proxy-reward ceiling. PPO for LMs is entirely implementable — the lecture walks through AlpacaFarm's."
    },
    // ---------- PPO ----------
    {
      type: "single",
      question: "PPO's clipped surrogate objective takes $\\min(rA, \\text{clip}(r, 1-\\epsilon, 1+\\epsilon)A)$, where $r$ is the importance ratio. What is this clipping for?",
      options: [
        "To normalize the advantages to zero mean and unit variance within each batch",
        "To cut the incentive to move the policy far from the one that generated the rollouts",
        "To bound the reward magnitude so the learned value network cannot diverge",
        "To encourage exploration by flattening the policy's action distribution"
      ],
      correct: [1],
      explanation: "Once the ratio $\\pi_{\\text{new}}/\\pi_{\\text{old}}$ leaves the $[1-\\epsilon, 1+\\epsilon]$ band in the direction the advantage favors, the clipped term's gradient vanishes, so the update gains nothing from pushing further. This keeps each update in a trust region around the sampling policy, where the importance-sampled surrogate is still a reliable estimate of true performance."
    },
    {
      type: "single",
      question: "In PPO, the importance ratio $r_t$ is defined as which quantity?",
      options: [
        "$\\pi_{\\text{new}}(a_t|s_t)$ divided by the old sampling policy's probability",
        "The advantage at step $t$ divided by the group-average advantage of the batch",
        "The reward at step $t$ divided by the discounted return-to-go from that step",
        "$\\pi_{\\text{new}}(a_t|s_t)$ divided by the frozen reference (SFT) policy's probability"
      ],
      correct: [0],
      explanation: "The ratio reweights samples drawn from the old (behavior) policy so they estimate the new policy's objective — standard importance sampling. Note the denominator is the policy that generated the rollouts, not the frozen reference model; the reference model only appears in the separate KL penalty term."
    },
    {
      type: "single",
      question: "What role does the learned value function (critic) play in PPO for language models?",
      options: [
        "It replaces the learned reward model as the source of the training signal",
        "It predicts the end-of-sequence token to decide when generation should stop",
        "It clips the policy ratio whenever an update step grows too large",
        "It gives a per-state baseline for advantages, cutting gradient variance"
      ],
      correct: [3],
      explanation: "The critic predicts expected return from each state (prefix), and advantages are computed as return minus this prediction (often via GAE). This is the baseline idea made state-dependent. The cost is real: the value network is typically as large as the policy itself, doubling memory and adding a whole second model to train — which is exactly what GRPO later removes."
    },
    // ---------- GRPO ----------
    {
      type: "multi",
      question: "Which statements correctly describe GRPO (Group Relative Policy Optimization)? (Select all that apply)",
      options: [
        "It samples a group of responses per prompt and scores each one",
        "The baseline is the group's mean reward rather than a learned value network",
        "It was the RL algorithm used to train DeepSeek-R1",
        "It requires a critic network twice the size of the policy"
      ],
      correct: [0, 1, 2],
      explanation: "GRPO's move is to get the baseline statistically instead of parametrically: sample $G$ responses to the same prompt, and use the group's mean reward as the baseline (advantage = reward minus group mean, typically divided by the group std). This deletes the value network entirely — the opposite of option four — and it is the algorithm behind DeepSeek's R1 reasoning models."
    },
    {
      type: "single",
      question: "In GRPO, how is the advantage of a single sampled response computed?",
      options: [
        "Its reward minus a learned value network's prediction for the prompt",
        "Its reward minus its group's mean reward, usually divided by the group std",
        "Its log-probability under the reference policy multiplied by its reward",
        "Its percentile rank within the group, divided by the total group size"
      ],
      correct: [1],
      explanation: "$A_i = \\frac{r_i - \\bar{r}}{\\sigma_r}$, the z-score of the response's reward within its group, in the standard formulation. Responses better than their siblings get positive advantage, worse ones negative — a purely relative, per-prompt signal. The std normalization equalizes update magnitude across prompts, though follow-up work (e.g., Dr. GRPO) argues it introduces its own difficulty bias."
    },
    {
      type: "single",
      question: "The literature identifies a length bias in GRPO's original formulation. What causes it?",
      options: [
        "Each response's summed token loss is divided by its own token length",
        "The group mean baseline systematically favors the shortest response",
        "The KL penalty grows linearly with the number of generated tokens",
        "Unit-test rewards take longer to compute for longer programs"
      ],
      correct: [0],
      explanation: "Dividing each sequence's summed token loss by its own length means tokens in long responses get smaller per-token gradients. For negatively-rewarded responses this penalizes long wrong answers less than short wrong ones — a systematic pressure toward longer incorrect outputs. Corrections such as Dr. GRPO remove the per-response length normalization (and the std division) to fix this."
    },
    {
      type: "single",
      question: "Dr. GRPO also criticizes GRPO's division of the advantage by the group's reward std, $\\sigma_r$. What is the objection?",
      options: [
        "It is an invalid baseline that upweights very easy or very hard questions",
        "It doubles memory cost by requiring a second value network per group",
        "It flips the sign of the advantage for below-average responses",
        "It makes gradients vanish for any group where all rewards are positive"
      ],
      correct: [0],
      explanation: "Subtracting the group mean is a legitimate baseline, but dividing by $\\sigma_r$ is a per-group rescaling that does not preserve the gradient's unbiasedness. Groups with low reward variance — prompts that are nearly always solved or nearly always failed — get their updates amplified, a difficulty bias. Dr. GRPO drops the std division (and the length normalization), landing close to REINFORCE with a leave-one-out baseline."
    },
    // ---------- KL regularization ----------
    {
      type: "multi",
      question: "Why do RLHF pipelines add a KL penalty against a frozen reference policy? (Select all that apply)",
      options: [
        "To keep the policy from drifting far from the reference and losing fluency",
        "To limit over-optimization of an imperfect learned reward model",
        "To make sampled trajectories exactly reproducible across training runs",
        "To eliminate the need for a baseline in the gradient estimator"
      ],
      correct: [0, 1],
      explanation: "The KL term anchors the policy near its initialization: it preserves the language modeling behaviors RL does not directly reward, and it bounds how far the policy can wander into regions where the reward model's scores are unreliable. In PPO implementations like AlpacaFarm it is applied as a per-token penalty, with the full reward added at the last token. It has nothing to do with determinism or with variance baselines."
    },
    {
      type: "single",
      question: "Kimi K1.5 adds a per-batch length reward on top of correctness. What behavior does it incentivize?",
      options: [
        "Longer chains of thought whenever the final answer is correct",
        "Every response in the batch converging to one fixed target length",
        "Incorrect answers padded out with additional retry attempts",
        "Short correct answers, with longer rollouts in a group scoring lower"
      ],
      correct: [3],
      explanation: "Kimi's length reward assigns a value in $[0.5, -0.5]$ based on where a response's length falls within its group's range, so longer sequences score lower: correct answers are pushed to be short, and incorrect ones shorter than the center of the rollout range. Because it can hurt performance, it is only enabled later in training. Unlike GRPO, Kimi's objective lacks the length-normalization bias, so this is deliberate CoT compression rather than a bias fix."
    },
    // ---------- Reward design & hacking ----------
    {
      type: "single",
      question: "DeepSeek-R1-Zero's reward combined an accuracy reward with a format reward. What does the format reward check?",
      options: [
        "That the total response length stays under a fixed token budget",
        "That the reasoning appears inside designated thinking tags before the answer",
        "That the response never repeats any sentence it has already generated",
        "That the model's log-probabilities stay close to the reference policy's"
      ],
      correct: [1],
      explanation: "Rule-based rewards in R1-Zero were deliberately simple: is the final answer correct (exact match / checker), and does the output follow the required template separating thinking from the answer. The format reward makes answers mechanically extractable and structures the chain of thought without a learned judge."
    },
    {
      type: "multi",
      question: "How does the full DeepSeek-R1 pipeline differ from the pure-RL R1-Zero? (Select all that apply)",
      options: [
        "It replaces GRPO with MCTS-guided tree search over reasoning steps during RL",
        "It starts from a long-CoT reasoning SFT initialization instead of the base model",
        "It adds a language-consistency reward to stop the CoT from mixing languages",
        "A later stage adds non-verifiable tasks through the usual SFT/RLHF pipeline"
      ],
      correct: [1, 2, 3],
      explanation: "R1 = SFT initialization on long-CoT data (claimed benefit: interpretability), the same GRPO RL step plus a language-consistency loss (pure RL naturally drifts into mixed languages), and then a standard post-training stage: SFT on 600k reasoning samples judged by V3 plus 200k non-reasoning samples, followed by RLHF for non-verifiable tasks. MCTS — like PRMs — is listed in the paper's unsuccessful-attempts section, not the recipe."
    },
    // ---------- Emergent effects at scale ----------
    {
      type: "single",
      question: "What striking behavior emerged during DeepSeek-R1-Zero's RL training, without being explicitly taught?",
      options: [
        "Responses grew longer, with re-checking and 'aha' self-correction appearing",
        "Responses became steadily shorter as the policy compressed its reasoning",
        "The model learned to refuse problems it judged too difficult to attempt",
        "The model memorized the training answers and stopped generalizing entirely"
      ],
      correct: [0],
      explanation: "With only outcome-based verifiable rewards, response length climbed throughout training and behaviors like re-checking and explicit self-correction ('wait... let me reevaluate') emerged — the DeepSeek-R1 paper's 'aha moment'. The lecture adds a caveat from follow-up analysis (Dr. GRPO): the length growth may partly come from GRPO's biased length-normalized objective, and base models already show some 'aha'-style phrases before any RL."
    },
    {
      type: "single",
      question: "How did DeepSeek transfer R1's reasoning ability to much smaller models (e.g., 1.5B–70B)?",
      options: [
        "By running the same large-scale RL recipe independently on each small model",
        "By averaging the weights of R1 with each of the smaller base models",
        "By pruning and quantizing R1 progressively down to each target size",
        "By fine-tuning the small models on reasoning traces generated by R1"
      ],
      correct: [3],
      explanation: "Distillation here is plain SFT: R1 generates ~800K CoT traces, and smaller Qwen 2.5 (and Llama) models are fine-tuned on them — no RL on the student at all. Notably, this outperforms running RL directly on the small models: discovering the reasoning patterns needs scale, but imitating them once discovered does not."
    },
    // ---------- On-policy vs off-policy ----------
    {
      type: "single",
      question: "Why does policy-gradient RL keep regenerating fresh samples from the current policy, while SFT and DPO train on a fixed dataset?",
      options: [
        "Fixed datasets are too small to give stable policy-gradient estimates",
        "The gradient is an expectation under the current policy's own samples",
        "RL requires each training example to be seen exactly once to converge",
        "SFT and DPO also silently regenerate their datasets at every epoch"
      ],
      correct: [1],
      explanation: "REINFORCE-style objectives are expectations over the policy's OWN sampling distribution; once parameters update, old samples come from the wrong distribution and the estimator no longer points along the true gradient. SFT and DPO instead minimize losses on a fixed offline dataset, so they can loop over it. This is why RL training alternates rollout generation with updates — and why the lecture stresses RL infrastructure: on-policy means slow inference, framework switching, and uneven long-CoT batches."
    },
    // ---------- Kimi K1.5 recipe ----------
    {
      type: "multi",
      question: "Which of these describe Kimi K1.5's data curation and training recipe? (Select all that apply)",
      options: [
        "Multiple-choice and true/false problems are excluded to avoid false-positive rewards",
        "Only problems the model fails to solve with best-of-8 sampling are kept",
        "A curriculum samples problems in proportion to one minus their success rate",
        "A process reward model scores every intermediate step of the reasoning chain"
      ],
      correct: [0, 1, 2],
      explanation: "Kimi filters aggressively: guessable formats (multiple choice, true/false) are dropped because a lucky guess yields a false-positive reward, and only problems hard enough to fail best-of-8 survive. Training then follows an easy-to-hard curriculum, sampling proportional to (1 − success rate) so solved problems fade out. There is no per-step process supervision; Kimi's objective comes from a DPO-style derivation, optimized with a squared-loss surrogate and a baselined policy gradient."
    },
    // ---------- Qwen 3 ----------
    {
      type: "multi",
      question: "Which statements describe Qwen 3's reasoning training? (Select all that apply)",
      options: [
        "The reasoning RL stage runs GRPO on only about 4,000 heavily filtered examples",
        "Filtering removes problems the model already solves without a chain of thought",
        "Thinking-mode fusion mixes thinking and non-thinking data with control tags",
        "General-purpose RLHF runs before the reasoning RL stage in the pipeline"
      ],
      correct: [0, 1, 2],
      explanation: "Qwen 3 is the low-data RLVR case study: after difficulty filtering (best-of-n, like Kimi), removing problems solvable without CoT, deduplicating against validation data, and manual CoT quality checks, GRPO runs on just 3,995 examples. Thinking-mode fusion then teaches length control — mixing tagged thinking/non-thinking data with an early-stopping string. The pipeline order matches R1: reasoning RL first, then general RLHF (which slightly degrades math/STEM), then distillation."
    }
  ]
};

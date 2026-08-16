// CS336 Lecture 17 — Alignment: RL II (policy gradient deep dive)
// 20 questions covering: the MDP setup for LMs, the policy gradient derivation,
// baselines and the two-state worked example, advantage functions, GRPO's
// group baseline, the sorting-task walkthrough (rewards, toy model, delta and
// loss modes, freezing p_old, KL estimator), and the training experiments.
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture17"] = {
  title: "Lecture 17 — Alignment: RL II",
  questions: [
    // ---------- RL setup for language models ----------
    {
      type: "single",
      question: "In the RL formulation for language models, what are the state and the action?",
      options: [
        "State: the model's hidden activations at each layer; action: emit the full response at once",
        "State: the prompt plus the response generated so far; action: generate the next token",
        "State: the prompt embedding only, fixed for the rollout; action: choose a reward function",
        "State: the full dataset of training prompts; action: update the model's parameters"
      ],
      correct: [1],
      explanation: "The state $s$ is the prompt plus tokens generated so far, and each action $a$ generates the next token; the policy $\\pi(a \\mid s)$ is just a fine-tuned language model. A rollout is $s \\to a \\to \\dots \\to a \\to R$, and the objective is to maximize expected reward $\\mathbb{E}[R]$ over prompts and responses."
    },
    {
      type: "single",
      question: "What is distinctive about the transition probabilities $T(s' \\mid s, a)$ in the LM setting?",
      options: [
        "They are stochastic as in robotics, which makes planning impossible",
        "They must be learned from data using a separate world model",
        "They are deterministic: $s' = s + a$, appending the action token",
        "They are undefined, since language has no meaningful state notion"
      ],
      correct: [2],
      explanation: "Transitions are deterministic, $s' = s + a$ — the new state is just the old state with the generated token appended. Unlike robotics, this enables planning and test-time compute, and since states are 'made up' rather than physical, there is a lot of flexibility."
    },
    // ---------- Policy gradient derivation ----------
    {
      type: "single",
      question: "Which identity turns $\\nabla \\mathbb{E}[R] = \\int p(s) \\, \\nabla \\pi(a \\mid s) \\, R(s, a)$ into an expectation you can sample?",
      options: [
        "$\\nabla \\pi(a \\mid s) = \\pi(a \\mid s) \\, \\nabla \\log \\pi(a \\mid s)$, the log-derivative trick",
        "$\\nabla \\pi(a \\mid s) = \\nabla \\log \\pi(a \\mid s)$, since gradients commute with logs",
        "Jensen's inequality, $\\mathbb{E}[\\log R] \\le \\log \\mathbb{E}[R]$, applied to the reward",
        "Integration by parts applied over the continuous space of responses"
      ],
      correct: [0],
      explanation: "The log-derivative trick rewrites $\\nabla \\pi$ as $\\pi \\, \\nabla \\log \\pi$, so the gradient becomes $\\mathbb{E}[\\nabla \\log \\pi(a \\mid s) \\, R(s, a)]$ — an expectation under the policy itself. The naive algorithm then samples a prompt $s$, samples a response $a \\sim \\pi(a \\mid s)$, and updates with $\\nabla \\log \\pi(a \\mid s)$ weighted by $R(s, a)$."
    },
    {
      type: "multi",
      question: "With binary rewards $R(s, a) \\in \\{0, 1\\}$, which statements describe the naive policy gradient? (Select all that apply)",
      options: [
        "It only performs parameter updates on correct responses",
        "It is like SFT, but the dataset shifts as the policy changes",
        "Sparse rewards make the gradient estimate high-variance",
        "It requires a learned critic network in order to run at all",
        "It cannot be computed without pairwise preference data"
      ],
      correct: [0, 1, 2],
      explanation: "When rewards are 0/1, $\\nabla \\log \\pi \\cdot R$ is nonzero only for correct responses, so it looks like SFT on a self-generated, ever-changing dataset. Because few responses earn reward 1, the estimate is noisy — in contrast, RLHF's learned reward models (from pairwise preferences) are more continuous."
    },
    // ---------- Baselines ----------
    {
      type: "single",
      question: "The two-state example ($s_1$: rewards 11 and 9; $s_2$: rewards 0 and 2) motivates baselines. What problem does it show?",
      options: [
        "Rewards must be rescaled to binary before the gradient works",
        "Deterministic transitions make the policy gradient vanish",
        "The optimal action in each state changes as training goes on",
        "Reward 9 for a worse action outweighs reward 2 for a better one"
      ],
      correct: [3],
      explanation: "In $s_1$ the reward-9 action $a_2$ is worse than $a_1$ (reward 11), while in $s_2$ the reward-2 action $a_2$ is the best available — yet $9 > 2$, so raw rewards push probability the wrong way. Comparing rewards within a state, not across states, is what a baseline $b(s)$ accomplishes."
    },
    {
      type: "single",
      question: "With baselines $b(s_1) = 10$ and $b(s_2) = 1$, the spread (std) of the update terms in the two-state example drops from about 5.3 to roughly what?",
      options: [
        "About 0",
        "About 1.15",
        "About 5.3",
        "About 10.6"
      ],
      correct: [1],
      explanation: "The raw rewards $[11, 9, 0, 2]$ have std $\\approx 5.32$; after subtracting the per-state baselines they become $[1, -1, -1, 1]$ with std $\\approx 1.15$. Same expected gradient, far lower variance — the whole point of baselining. (The baseline reduces variance greatly but does not eliminate it.)"
    },
    {
      type: "single",
      question: "Why does subtracting a baseline $b(s)$ not bias the objective being optimized?",
      options: [
        "Because $b(s)$ has zero mean under any policy, so nothing changes",
        "Because the baseline is clipped to the range $[1 - \\epsilon, 1 + \\epsilon]$",
        "Because the objective shifts by $\\mathbb{E}[b(s)]$, independent of $\\pi$",
        "Because $b(s)$ tracks the reward exactly and cancels it out"
      ],
      correct: [2],
      explanation: "Maximizing $\\mathbb{E}[R - b(s)]$ is just maximizing $\\mathbb{E}[R]$ shifted by $\\mathbb{E}[b(s)]$, which does not depend on $\\pi$. So the gradient of the baselined objective, $\\nabla \\log \\pi(a \\mid s)(R(s, a) - b(s))$, is still an unbiased estimate of $\\nabla \\mathbb{E}[R]$, only with (hopefully) lower variance."
    },
    {
      type: "single",
      question: "The optimal baseline $b^*(s)$ is difficult to compute. What heuristic does the lecture use instead?",
      options: [
        "The mean reward $b(s) = \\mathbb{E}[R \\mid s]$, estimated from samples",
        "The maximum reward observed among sampled responses for the state",
        "A learned critic network trained with temporal-difference targets",
        "The KL divergence between the policy and a frozen reference model"
      ],
      correct: [0],
      explanation: "The optimal $b^*(s)$ weights rewards by squared gradient magnitudes (for one-parameter models), which is impractical. The heuristic is the mean reward $\\mathbb{E}[R \\mid s]$ — still hard to compute exactly, so in practice it must be estimated from sampled responses."
    },
    // ---------- Advantage functions ----------
    {
      type: "multi",
      question: "Which statements about advantage functions are correct in this lecture's setting? (Select all that apply)",
      options: [
        "The advantage is defined as $A(s, a) = Q(s, a) - V(s)$",
        "With outcome rewards over the whole response, $Q(s, a)$ equals $R(s, a)$",
        "Choosing $b(s) = \\mathbb{E}[R \\mid s]$ makes the baselined reward the advantage",
        "$V(s)$ is the maximum reward achievable from state $s$ by any action",
        "Advantages cannot be defined without a discount factor $\\gamma$"
      ],
      correct: [0, 1, 2],
      explanation: "$A(s, a) = Q(s, a) - V(s)$ measures how much better action $a$ is than expected from $s$. Since $a$ denotes the entire response and rewards are outcome-based, $Q$ and $R$ coincide — so subtracting the mean-reward baseline $V(s) = \\mathbb{E}[R \\mid s]$ turns the baselined reward $R - b(s)$ exactly into the advantage."
    },
    // ---------- GRPO ----------
    {
      type: "single",
      question: "Why does the language-model setting make GRPO's removal of the critic natural?",
      options: [
        "Rewards arrive densely per token, so no value estimate is ever needed",
        "The deterministic transition model gives all values in closed form",
        "The mean reward of a group of responses per prompt estimates $b(s)$",
        "The critic is simply folded into a larger learned reward model"
      ],
      correct: [2],
      explanation: "GRPO simplifies PPO by dropping the learned value function and exploiting the group structure of LM training: you sample several responses for the same prompt, and their mean reward provides a natural baseline $b(s)$. No separate critic network needs to be trained or served."
    },
    // ---------- Sorting task walkthrough ----------
    {
      type: "single",
      question: "For prompt $[3, 1, 0, 2]$ and response $[0, 3, 1, 2]$, what do the two sorting rewards give?",
      options: [
        "Distance reward 4, inclusion+ordering reward 7",
        "Distance reward 1, inclusion+ordering reward 6",
        "Distance reward 6, inclusion+ordering reward 1",
        "Distance reward 0, inclusion+ordering reward 0"
      ],
      correct: [1],
      explanation: "The ground truth is sorted(prompt) $= [0, 1, 2, 3]$; only position 0 matches, so the distance reward is 1. The inclusion+ordering reward gives 4 (every prompt token appears) plus 2 (adjacent pairs $(0,3)$ and $(1,2)$ are ordered) $= 6$ — more partial credit, which shapes learning better."
    },
    {
      type: "multi",
      question: "Which simplifications does the toy sorting model make? (Select all that apply)",
      options: [
        "Fixed prompt and response lengths",
        "Each response position is decoded independently, not autoregressively",
        "Separate per-position encode and decode matrices capture position",
        "A full transformer with causal attention over the whole sequence",
        "Rewards are backpropagated directly through the generation step"
      ],
      correct: [0, 1, 2],
      explanation: "The model embeds the prompt, applies a per-position encoding matrix to collapse it into one vector, then per-position decoding matrices produce independent logits for each response slot (sharing input/output embeddings). No autoregression, no attention — small enough to watch RL dynamics clearly."
    },
    // ---------- Deltas ----------
    {
      type: "multi",
      question: "Which statements correctly describe the delta (advantage-like) modes implemented in the walkthrough? (Select all that apply)",
      options: [
        "centered_rewards subtracts each prompt's group mean reward",
        "normalized_rewards also divides by the group's standard deviation",
        "max_rewards zeroes every reward that is not the group maximum",
        "rewards mode subtracts a learned value function's prediction",
        "centered_rewards clips each reward to the binary set $\\{0, 1\\}$"
      ],
      correct: [0, 1, 2],
      explanation: "The four modes are: raw rewards; centered rewards (subtract the per-prompt group mean); normalized rewards (also divide by the group std plus $10^{-5}$); and max rewards (keep only the group's best). The first three trace the path from naive REINFORCE to the GRPO-style group advantage."
    },
    // ---------- Freezing p_old ----------
    {
      type: "single",
      question: "In the freezing demo, what goes wrong if $p_{\\text{old}}$ in the ratio $p / p_{\\text{old}}$ is computed without torch.no_grad()?",
      options: [
        "The ratio is identically 1, so the gradient is zero instead of nonzero",
        "The gradient is exactly doubled, since both terms contribute to it",
        "backward() raises an error about reusing the computation graph",
        "Nothing — autograd detaches older model checkpoints automatically"
      ],
      correct: [0],
      explanation: "If $p_{\\text{old}}$ is built from the same live parameters, $p / p_{\\text{old}}$ collapses to the constant 1 and backward yields zero gradient. Wrapping the old-model computation in torch.no_grad() treats $p_{\\text{old}}$ as a constant, recovering the intended nonzero gradient through $p$."
    },
    // ---------- Loss modes ----------
    {
      type: "single",
      question: "How does the walkthrough's clipped loss combine the ratio and the delta $\\delta$?",
      options: [
        "Deltas clamped to $[-\\epsilon, \\epsilon]$, then multiplied by the log probs",
        "The max of the clipped and the unclipped terms, favoring larger updates",
        "Gradients clipped by a global norm threshold set at $\\epsilon = 0.01$",
        "The min of ratio times $\\delta$ and clamped-ratio times $\\delta$, $\\epsilon = 0.01$"
      ],
      correct: [3],
      explanation: "The ratio $\\exp(\\log \\pi - \\log \\pi_{\\text{old}})$ multiplies $\\delta$ in an unclipped term, the ratio clamped to $[1 - \\epsilon, 1 + \\epsilon]$ does so in a clipped term, and the loss is the negated mean of the elementwise minimum. This PPO-style pessimism keeps updates from exploiting large probability ratios."
    },
    // ---------- KL penalty ----------
    {
      type: "single",
      question: "The lecture estimates $\\mathrm{KL}(p \\, \\| \\, q)$ via $\\mathbb{E}_p[q/p - \\log(q/p) - 1]$. Why is adding $q/p - 1$ to $-\\log(q/p)$ legitimate?",
      options: [
        "Because KL divergence is symmetric, so $p$ and $q$ can be swapped",
        "Because $\\mathbb{E}_p[q/p] = 1$, so the extra terms have zero mean",
        "Because $\\log(q/p)$ is bounded above by 1 for any distributions",
        "Because the estimator is exact per sample, needing no expectation"
      ],
      correct: [1],
      explanation: "$\\mathrm{KL}(p \\, \\| \\, q) = \\mathbb{E}_p[-\\log(q/p)]$, and since $\\mathbb{E}_p[q/p] = 1$ you can add $q/p - 1$ without changing the expectation. The resulting per-sample estimator $x - \\log x - 1$ (with $x = q/p$) is always nonnegative, giving a lower-variance, better-behaved estimate."
    },
    {
      type: "single",
      question: "When running with a KL penalty, how does the training loop handle the reference model?",
      options: [
        "It is fixed at initialization and never changed for the entire run",
        "It is updated every gradient step with an exponential moving average",
        "It is re-cloned from the current model every few epochs, then frozen",
        "It is loaded from a separately pretrained checkpoint each epoch"
      ],
      correct: [2],
      explanation: "Every compute_ref_model_period epochs, the loop clones the current model as the new frozen reference and computes reference log probs under no_grad. The lecture's motivation: a KL penalty lets you RL a new capability into a model without forgetting its original capabilities."
    },
    // ---------- Experiments ----------
    {
      type: "multi",
      question: "What did the sorting experiments show? (Select all that apply)",
      options: [
        "Raw-reward updates failed to learn sorting well, even on train prompts",
        "Centering gives below-average responses a negative gradient update",
        "When all of a prompt's responses tie in reward, centered updates vanish",
        "Normalizing by the standard deviation dramatically beat plain centering",
        "Every configuration reached perfect reward with default hyperparameters"
      ],
      correct: [0, 1, 2],
      explanation: "Raw rewards learned poorly; centered rewards helped because worse-than-average responses get pushed down and all-tied groups produce no update. Std normalization made little difference — and variants like Dr. GRPO drop it anyway to avoid length bias. Even so, runs still got stuck in local optima: RL is not trivial."
    },
    {
      type: "single",
      question: "Why does the training walkthrough need importance ratios (old vs. current log probs) at all?",
      options: [
        "The ratios are what make the surrogate loss differentiable",
        "The ratios implement the KL penalty to the reference model",
        "The ratios normalize rewards across prompts of varying difficulty",
        "Responses are reused for several gradient steps, drifting off-policy"
      ],
      correct: [3],
      explanation: "Each epoch generates responses once, freezes old_log_probs under no_grad, then takes num_steps_per_epoch gradient steps on those same responses. After the first step the current policy differs from the sampling policy, so the ratio $p / p_{\\text{old}}$ (and clipping) corrects for this mild off-policy reuse."
    },
    // ---------- Summary ----------
    {
      type: "multi",
      question: "Which takeaways appear in the lecture's summary? (Select all that apply)",
      options: [
        "Reinforcement learning is the key to surpassing human abilities",
        "If you can measure it, you can optimize it",
        "RL systems beat pretraining in complexity: inference plus many models",
        "Policy gradient needs no variance reduction once rewards are verifiable",
        "RL is operationally simpler than SFT since no labels are ever needed"
      ],
      correct: [0, 1, 2],
      explanation: "The summary: RL is how models surpass human abilities, and anything measurable is optimizable. The policy gradient framework is conceptually clear — it just needs baselines to reduce variance — but the systems side (serving rollouts, juggling policy, old, and reference models) is much heavier than pretraining."
    }
  ]
};

// CS336 Lecture 15 — Alignment: supervised finetuning and RLHF
// 20 questions covering: the SFT-then-RLHF pipeline, instruction-tuning datasets
// (FLAN, OpenAssistant, Alpaca), style/length effects, hallucination from tail
// knowledge, safety-tuning, midtraining, why optimize (costs, G-V gap), RLHF data
// collection (annotators, GPT-4 feedback), reward models and overoptimization,
// PPO, DPO and variants, and mode collapse.
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture15"] = {
  title: "Lecture 15 — Alignment: SFT & RLHF",
  questions: [
    // ---------- SFT data ----------
    {
      type: "single",
      question: "The lecture pulled random examples from FLAN, Alpaca, and OpenAssistant. Which characterization of the three datasets is right?",
      options: [
        "FLAN: long human answers with references; Alpaca: templated NLP tasks; OpenAssistant: short model-generated answers",
        "FLAN: short model-generated answers; Alpaca: long human answers with references; OpenAssistant: templated NLP tasks",
        "FLAN: templated NLP tasks; Alpaca: short model-generated answers; OpenAssistant: long human answers with references",
        "FLAN: templated NLP tasks; Alpaca: long human answers with references; OpenAssistant: short model-generated answers"
      ],
      correct: [2],
      explanation: "FLAN is templated NLP tasks (classification, summarization, subject lines); Alpaca is short model-generated instruction/response pairs; OpenAssistant is long, human-written answers with citations and references. Their differences in length, style, and citation behavior motivate the question of what SFT data actually teaches the model."
    },
    {
      type: "single",
      question: "In the standard modern pipeline, where does alignment (SFT and RLHF) sit relative to pretraining?",
      options: [
        "It replaces pretraining when instruction data is abundant",
        "It runs interleaved with pretraining on a mixed objective",
        "It is a post-training stage applied after pretraining produces a base model",
        "It is applied only at inference time via prompting, with no weight updates"
      ],
      correct: [2],
      explanation: "The standard approach, per InstructGPT [Ouyang 2022], is imitation (SFT) followed by reinforcement (RLHF) on top of a pretrained base model. Pretraining builds broad capabilities from raw text; post-training uses far less data and compute but enables much tighter control over outputs."
    },
    {
      type: "single",
      question: "To scale up instruction tuning without catastrophic forgetting, what \"midtraining\" (two-phase training) recipe does the lecture describe?",
      options: [
        "Mix instruction data into pretraining, then do a short final instruction-tuning round",
        "Alternate one epoch of pretraining and one of instruction tuning throughout",
        "Freeze the base model and train adapter layers on instruction data only",
        "Distill an instruction-tuned teacher into the base model during pretraining"
      ],
      correct: [0],
      explanation: "If you have lots of instruction data and compute, you can mix instruction-tuning data into the pretraining mixture and finish with a short, actual instruction-tuning round. This scales instruction tuning without catastrophic forgetting; the recipe is common knowledge at LLM companies but was mainly publicized by models like MiniCPM and JetMoE."
    },
    {
      type: "single",
      question: "Which summary matches the lecture's takeaways about SFT data?",
      options: [
        "SFT is the main stage where models acquire new factual knowledge about the world",
        "SFT works best at extracting existing pretraining behaviors, not adding new ones",
        "SFT only helps once the dataset reaches millions of high-quality examples",
        "SFT data quality barely matters as long as the chat format stays consistent"
      ],
      correct: [1],
      explanation: "The lecture concludes instruction tuning works best when it surfaces behaviors the base model already has, not when it tries to add new ones — adding facts the model doesn't know can hurt even when they are correct. Small amounts of the right behaviors (safety, instruction-following, style) go a long way, though a long tail benefits from more data."
    },
    {
      type: "single",
      question: "What does the lecture show about safety-tuning with small amounts of data?",
      options: [
        "Safety behavior cannot be meaningfully changed without a full RLHF pipeline",
        "Safety data must make up over half of the SFT mixture to have any effect",
        "Safety tuning uniformly degrades performance across capability benchmarks",
        "About 500 safety examples improve safety a lot; the challenge is over-refusals"
      ],
      correct: [3],
      explanation: "Adding about 500 safety examples to Alpaca-style instruction tuning makes models follow safety guidelines dramatically better (evaluated on hate speech and Anthropic HH prompts). The real difficulty is balance: pushing safety behavior too hard produces over-refusals of benign requests."
    },
    {
      type: "single",
      question: "Why can SFT actively teach a model to hallucinate?",
      options: [
        "Gradient descent on small datasets always increases output variance and randomness",
        "Targets stating facts it doesn't know teach it to assert facts confidently anyway",
        "Chat templates inject formatting tokens that corrupt the model's factual recall",
        "Small finetuning datasets overwrite factual knowledge learned during pretraining"
      ],
      correct: [1],
      explanation: "The folklore [Schulman 2023], supported by Gekhman et al.: fine-tuning on facts the model doesn't know teaches it to produce confident specifics regardless, which generalizes to fabrication. The lecture's takeaways: avoid fine-tuning on tail knowledge even when that is the use case, and RL-style correctness feedback could in principle help."
    },

    // ---------- Style and length ----------
    {
      type: "multi",
      question: "Which statements about style and length effects are correct? (Select all that apply)",
      options: [
        "Both human and GPT-based preference evaluations show strong length effects",
        "Length increase is a very significant outcome of RLHF training itself",
        "These style factors mostly do not move standard capability benchmarks",
        "Longer responses are genuinely better, so length preference is not a bias"
      ],
      correct: [0, 1, 2],
      explanation: "Dubois et al. show raters — human and GPT alike — strongly prefer longer responses, and Chen et al. and Singhal et al. show RLHF itself inflates response length. Yet these factors are mostly irrelevant to standard benchmark performance, a warning that preference-evaluation wins can reflect style rather than capability."
    },

    // ---------- Why optimize ----------
    {
      type: "multi",
      question: "Why might you prefer RLHF-style optimization over collecting more SFT data? (Select all that apply)",
      options: [
        "Pairwise feedback is far cheaper to annotate than expert demonstrations",
        "Some tasks are much easier for experts to verify than to solve",
        "Optimization injects new factual knowledge that SFT cannot",
        "People don't always write the responses they would themselves prefer"
      ],
      correct: [0, 1, 3],
      explanation: "The lecture's cost breakdown puts pairwise feedback at a small fraction of demonstration cost, and verification is often much easier than generation. The generation-validation gap (Zhang et al. on summarization) shows annotators' own writing is not what they rate highest, so imitating demonstrations has a ceiling that scalar feedback does not. Neither stage is a knowledge-injection mechanism."
    },

    // ---------- RLHF data ----------
    {
      type: "single",
      question: "What does the lecture (citing Santurkar+ 2023) say about who annotates RLHF data?",
      options: [
        "Annotator identity washes out once enough preference data is collected",
        "Only annotator training protocols, not demographics, affect the model",
        "The annotator pool's distribution can significantly shift model behaviors",
        "Models ignore annotator disagreement because labels are majority-voted"
      ],
      correct: [2],
      explanation: "The demographics and opinions of the annotator pool measurably shift model behavior, and Hosking, Blunsom, and Bartolo show individual annotators matter a lot too. Crowdsourcing adds further complications: getting verifiable high-quality annotators, getting correctness actually checked, and screening for annotators quietly using GPT-4."
    },
    {
      type: "single",
      question: "What did the lecture report about using GPT-4 to provide pairwise feedback?",
      options: [
        "It agrees with humans at near inter-annotator agreement levels",
        "It agrees with human annotators only barely above chance",
        "It costs more than human annotation once run at scale",
        "It refuses to compare responses for most realistic prompts"
      ],
      correct: [0],
      explanation: "GPT-4 is a surprisingly good pairwise judge: agreement near human inter-annotator levels and near-perfect rank correlation at the system level. AI feedback is thus the cheap end of the cost-quality spectrum and widely used — e.g., UltraFeedback in Zephyr, OLMo, and Tulu 3 — and Constitutional AI (Bai et al.) pushes further with self-generated critiques."
    },

    // ---------- Reward models ----------
    {
      type: "single",
      question: "Under the Bradley-Terry model with reward function $r$, what is $P(A \\succ B)$, the probability that response A is preferred to response B?",
      options: [
        "$\\sigma(r(A) - r(B))$",
        "$r(A) / (r(A) + r(B))$, rewards positive",
        "$\\exp(-(r(A) - r(B))^2)$",
        "$\\max(0, r(A) - r(B))$"
      ],
      correct: [0],
      explanation: "Bradley-Terry posits $P(A \\succ B) = \\sigma(r(A) - r(B))$: the probability of preferring A grows with the reward gap. This is the objective from Stiennon et al.'s \"Learning to summarize from human feedback\" used to train reward models by maximum likelihood on preference pairs — and the same objective DPO reuses on its implied reward."
    },
    {
      type: "multi",
      question: "Which statements about reward hacking and overoptimization are correct? (Select all that apply)",
      options: [
        "The learned reward is a proxy, so optimizing it hard exploits its errors (Goodhart's law)",
        "Proxy reward can keep rising while true human preference for outputs falls",
        "Reward hacking is fully solved by making the reward model larger than the policy",
        "Overoptimization worsens as the policy moves further from the data the RM was trained on"
      ],
      correct: [0, 1, 3],
      explanation: "The reward model only approximates human judgment on its training distribution; strong optimization finds off-distribution outputs where the proxy is wrong, so proxy reward climbs while gold preference drops. The lecture shows this across many RLHF-style optimizers, for human preferences and noisy LM preferences alike (though not for a noiseless LM judge). Bigger RMs delay but do not eliminate it."
    },
    {
      type: "single",
      question: "Why does RLHF include a KL penalty between the policy and the reference (SFT) model?",
      options: [
        "To speed up convergence by raising the effective learning rate",
        "To keep the policy near the reference, limiting reward hacking",
        "To make the reward model's scores sum to one across responses",
        "To guarantee the policy's responses stay shorter than the reference's"
      ],
      correct: [1],
      explanation: "The objective is roughly $\\mathbb{E}[r(x, y)] - \\beta\\,\\mathrm{KL}(\\pi \\,\\|\\, \\pi_{\\text{ref}})$, as in the InstructGPT formulation. Without the KL term, the policy drifts far from the reference to exploit reward-model errors, producing degenerate text that scores well. The penalty anchors the policy where the RM is trustworthy and language quality is preserved."
    },

    // ---------- PPO ----------
    {
      type: "single",
      question: "In the lecture's sketch of PPO's lineage, what problem do TRPO and PPO address, and how does PPO do it?",
      options: [
        "Reward sparsity; PPO adds shaping bonuses to intermediate tokens",
        "Sample inefficiency; PPO trains from a replay buffer of stale rollouts",
        "Credit assignment; PPO backpropagates directly through the reward model",
        "High variance of vanilla policy gradients; PPO clips the policy probability ratios"
      ],
      correct: [3],
      explanation: "The naive policy gradient $\\nabla_\\theta \\mathbb{E}_{p_\\theta}[R(z)] = \\mathbb{E}_{p_\\theta}[R(z) \\nabla_\\theta \\log p_\\theta(z)]$ has variance too high to be practical. TRPO linearizes the problem around the current policy to take trust-region steps; PPO approximates this more simply by clipping the probability ratios at some epsilon, giving stable, conservative updates."
    },
    {
      type: "multi",
      question: "Which components are typically involved in a PPO-based RLHF setup? (Select all that apply)",
      options: [
        "The policy model being trained",
        "A frozen reference model for computing the KL penalty",
        "A learned reward model scoring sampled responses",
        "A retrieval index over the pretraining corpus"
      ],
      correct: [0, 1, 2],
      explanation: "Classic PPO RLHF juggles several models: the trainable policy, a frozen reference copy for the KL term, the reward model, and usually a value/critic model. This is exactly what DPO's pitch targets — getting rid of the reward model and all the on-policy machinery (rollouts, outer loops). Retrieval is not part of the algorithm."
    },

    // ---------- DPO ----------
    {
      type: "single",
      question: "What is the key idea of Direct Preference Optimization (DPO)?",
      options: [
        "Train the reward model and policy jointly with alternating gradient steps",
        "Replace human preference labels with the model's own self-generated rewards",
        "Distill a PPO-trained teacher policy into a smaller student model",
        "Solve the RLHF objective in closed form, giving a supervised loss on preference pairs"
      ],
      correct: [3],
      explanation: "Under a nonparametric assumption, the KL-regularized RLHF objective has a closed-form optimal policy, which can be solved for an \"implied reward\" expressed via policy/reference log-ratios. Plugging that into the Stiennon pairwise objective gives a supervised classification-style loss on preference pairs — no reward model, no rollouts, no outer RL loop."
    },
    {
      type: "single",
      question: "In the DPO loss, what does the $\\beta$ parameter control?",
      options: [
        "How strongly the implicit KL constraint ties the policy to the reference",
        "The learning-rate warmup schedule used during preference finetuning",
        "The fraction of preference pairs whose labels are flipped for augmentation",
        "The sampling temperature applied when generating responses at inference"
      ],
      correct: [0],
      explanation: "$\\beta$ is inherited from the KL coefficient in the underlying RLHF objective and scales the policy/reference log-ratios inside the sigmoid. The resulting update is a positive gradient on the chosen response and a negative gradient on the rejected one, weighted by the implied reward model's prediction error."
    },
    {
      type: "multi",
      question: "Which are fair statements in the DPO vs PPO comparison? (Select all that apply)",
      options: [
        "DPO drops the reward model and on-policy rollouts, making it far simpler to run",
        "In the AlpacaFarm comparison, DPO matched PPO with far less pain",
        "DPO provably dominates PPO in theory, making further PPO experiments obsolete",
        "PPO sometimes still wins; RL comparisons are highly contingent on experimental setup"
      ],
      correct: [0, 1, 3],
      explanation: "DPO is \"RLHF without tears\": in the lecture's AlpacaFarm comparison it matched PPO on simulated preferences, and most top open-source RLHF models are DPO'd. Yet other careful studies find PPO as good or better — RL-related empirical results depend heavily on setup. Tulu 3 also highlights variants like SimPO (no reference model) and length-normalized DPO."
    },

    // ---------- Side effects ----------
    {
      type: "single",
      question: "Why does the lecture say RLHF-trained models should no longer be treated as probabilistic models?",
      options: [
        "Their effective vocabulary shrinks sharply during preference tuning",
        "Sampling multiple responses from them becomes impossible after RL",
        "Mode collapse means their output probabilities are no longer calibrated",
        "The KL penalty forces their output distribution toward uniformity"
      ],
      correct: [2],
      explanation: "RLHF optimizes a reward rather than fitting a distribution, collapsing entropy onto preferred modes. The resulting model loses the calibration base models have — output probabilities no longer track correctness by default — so it should be treated as a policy, not a model of a distribution."
    },
    {
      type: "multi",
      question: "The lecture lists ideas people tried for using pairwise feedback without on-policy RL. Which are among them? (Select all that apply)",
      options: [
        "SFT on both responses, prepending a [GOOD] or [BAD] control token",
        "Rerunning pretraining from scratch with the preference pairs mixed in",
        "Fine-tuning only on the preferred response of each pair",
        "Training a reward model and taking the best of ~1024 sampled outputs"
      ],
      correct: [0, 2, 3],
      explanation: "Before DPO, natural attempts included conditioning on control tokens, simply imitating the chosen responses, and reward-model-based selection — training on RM-preferred outputs or best-of-n sampling. These avoid rollouts and outer loops but do not directly optimize the KL-constrained RLHF objective the way PPO or DPO does."
    }
  ]
};

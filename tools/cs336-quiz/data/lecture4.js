// CS336 Lecture 4 — Mixture of experts
// 20 questions covering: the core MoE idea (params vs FLOPs), token-choice
// top-k routing (softmax before vs after top-k), load balancing (Switch aux
// loss, DeepSeek-V3 loss-free bias), fine-grained + shared experts, systems
// (expert parallelism, batch-level token dropping), training issues
// (non-differentiability, z-loss, FP32 router, fine-tuning overfit,
// upcycling), MLA, and canonical models (Switch, Mixtral, DeepSeek-V3).
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture4"] = {
  title: "Lecture 4 — Mixture of Experts",
  questions: [
    // ---------- Core idea ----------
    {
      type: "single",
      question: "In a standard MoE transformer, which component of the dense architecture is replaced by the mixture of experts?",
      options: [
        "The self-attention layer, replaced by many attention heads chosen per token",
        "The feed-forward network, replaced by many expert FFNs of which few fire",
        "The embedding matrix, replaced by per-domain embedding tables chosen per input",
        "The final softmax, replaced by an ensemble of output heads averaged per token"
      ],
      correct: [1],
      explanation: "The canonical MoE design keeps attention dense and swaps the FFN for a set of expert FFNs plus a selector layer. Since the FFN holds most of a transformer's parameters and FLOPs, sparsifying it gives the biggest win. Routing attention heads instead exists (ModuleFormer, JetMoE) but the lecture flags it as much less common."
    },
    {
      type: "single",
      question: "What is the key economic property that makes MoE attractive compared to a dense model?",
      options: [
        "It removes the need for load balancing across GPUs",
        "It makes attention cost linear in sequence length",
        "It decouples total parameter count from FLOPs per token",
        "It reduces the memory needed to store the model"
      ],
      correct: [2],
      explanation: "A dense model's compute per token grows with its parameter count. In an MoE, only the activated experts run, so you can increase the number of experts without affecting FLOPs per token. Memory actually goes UP — all experts must be stored — which is the price paid for the compute decoupling."
    },
    {
      type: "multi",
      question: "Why are MoEs getting popular, according to the lecture? (Select all that apply)",
      options: [
        "At the same FLOPs, more parameters does better",
        "MoEs train to a given quality faster than dense equivalents",
        "Sparse routing makes the gradient estimates exactly unbiased",
        "Experts parallelize naturally across many devices"
      ],
      correct: [0, 1, 3],
      explanation: "The lecture's opening case: same-FLOP models with more (sparse) parameters do better, MoEs reach a target loss faster than dense equivalents (e.g. the OLMoE comparison), they are highly competitive among open models, and they shard cleanly across devices. Routing does not make gradients unbiased — the discrete top-$k$ choice is a genuine training complication."
    },
    // ---------- Routing ----------
    {
      type: "single",
      question: "In standard token-choice top-$k$ routing, how are a token's experts selected?",
      options: [
        "A router scores every expert per token and the $k$ highest scorers are used",
        "Each expert scores all tokens and claims the ones it prefers most",
        "Experts are assigned round-robin by token position to guarantee balance",
        "A hash of the token id deterministically maps it to $k$ experts"
      ],
      correct: [0],
      explanation: "A learned gate (a logistic regressor on the token's hidden state) scores each expert, and the token is dispatched to the top-$k$; expert outputs are combined weighted by the gate values. Hashing is a common baseline, and expert-choice and optimization-based routing exist — but almost all MoEs use learned token-choice top-$k$."
    },
    {
      type: "single",
      question: "What values of $k$ are typical for token-choice top-$k$ routing in practice?",
      options: [
        "Small values like 1, 2, or 4, as in Switch, Mixtral, and DBRX",
        "Around half the expert count, so most of the model stays active",
        "A schedule growing from 1 to the full expert count over training",
        "A per-token value chosen by a second, smaller learned network"
      ],
      correct: [0],
      explanation: "Sparsity is the point: the lecture's list has Switch at $k=1$, GShard, Grok, and Mixtral at $k=2$, and Qwen and DBRX at $k=4$. Fine-grained designs use larger $k$ over many small experts — DeepSeek-V3 activates 8 of 256 routed experts. Larger $k$ means more FLOPs per token, so $k$ stays small relative to the expert count."
    },
    {
      type: "multi",
      question: "Sparse gating decisions are not differentiable. Which solutions does the lecture discuss? (Select all that apply)",
      options: [
        "Reinforcement learning (e.g. REINFORCE) to optimize gating policies",
        "Exact gradients by marginalizing over all possible expert assignments",
        "Stochastic perturbations, such as Gaussian noise on routing scores",
        "Heuristic balancing losses — the approach actually used in practice"
      ],
      correct: [0, 2, 3],
      explanation: "The lecture lists three fixes: RL, stochastic perturbations, and heuristic balancing losses. REINFORCE works but gradient variance and complexity keep it from being a clear win; noise-based approaches (Shazeer's Gaussian perturbations, Fedus's multiplicative jitter) were later dropped by Zoph et al. In practice everyone just trains the heuristic top-$k$ system with balancing losses. Marginalizing over all assignments would destroy the sparsity that makes MoEs cheap."
    },
    {
      type: "single",
      question: "Why can't we just train a top-$k$ router with plain gradient descent and no extra tricks?",
      options: [
        "The router weights are integers and cannot receive gradient updates",
        "The softmax saturates, giving vanishing gradients for all experts at once",
        "The top-$k$ selection is discrete, so unchosen experts get no gradient",
        "Backpropagating through several experts at once is numerically unstable"
      ],
      correct: [2],
      explanation: "Top-$k$ is a hard, non-differentiable decision: an expert that wasn't selected gets no gradient signal about whether it should have been. Gradients only flow through the gate weights of chosen experts. This is the central tension the lecture highlights — sparsity is needed for efficiency, but sparse gating decisions are not differentiable."
    },
    // ---------- Load balancing ----------
    {
      type: "single",
      question: "Sparse MoEs tend to overfit when fine-tuned on smaller datasets. What remedies does the lecture mention?",
      options: [
        "Add more experts so each sees a smaller slice of the fine-tuning data",
        "Raise the capacity factor so no tokens are dropped during fine-tuning",
        "Fine-tune only the non-MoE parameters, or use very large SFT datasets",
        "Freeze the router and reinitialize the expert weights before fine-tuning"
      ],
      correct: [2],
      explanation: "Zoph et al. found sparse models overfit small fine-tuning sets and got good results updating only the dense, non-MoE MLPs. DeepSeek instead sidesteps the problem with scale, fine-tuning on about 1.4M SFT examples. Both remedies accept that abundant sparse capacity is easy to overfit."
    },
    {
      type: "single",
      question: "The Switch Transformer auxiliary balance loss has the form $\\alpha \\, N \\sum_i f_i P_i$. What are $f_i$ and $P_i$?",
      options: [
        "$f_i$ is expert $i$'s share of total FLOPs; $P_i$ is its share of the parameters",
        "$f_i$ is the fraction of tokens routed to expert $i$; $P_i$ is its mean router probability",
        "$f_i$ is expert $i$'s failure rate on the batch; $P_i$ is its precision on held-out data",
        "$f_i$ is expert $i$'s dropout rate during training; $P_i$ is its capacity factor"
      ],
      correct: [1],
      explanation: "$f_i$ counts the actual (hard, non-differentiable) share of the batch's tokens sent to expert $i$, while $P_i$ is the average (soft, differentiable) softmax probability assigned to it. The derivative with respect to $p_i(x)$ is proportional to how often expert $i$ wins, so more frequent use means stronger downweighting — exactly the pressure toward even use that systems efficiency requires."
    },
    {
      type: "single",
      question: "How does DeepSeek-V3 achieve balanced routing without a large auxiliary balance loss?",
      options: [
        "A per-expert bias on the top-$k$ selection scores, tuned online by expert load",
        "A hash of the token id assigns experts, so balance holds by construction",
        "Expert weights are reshuffled across devices whenever load becomes skewed",
        "A cap on the router's learning rate prevents any expert from dominating"
      ],
      correct: [0],
      explanation: "Each expert gets a bias term added to its affinity score only when selecting the top-$k$ — overloaded experts have their bias decreased and underloaded ones increased via online learning. The bias does not affect the gating weights used to combine outputs. DeepSeek calls this 'auxiliary-loss-free balancing', though the lecture notes it is not fully aux-loss-free: a small sequence-wise auxiliary loss remains."
    },
    {
      type: "multi",
      question: "What goes wrong if expert load is left badly imbalanced? (Select all that apply)",
      options: [
        "Some experts receive almost no tokens and effectively die, wasting capacity",
        "In expert parallelism, the device hosting popular experts becomes a straggler",
        "The router's softmax outputs stop summing to one, corrupting the gate weights",
        "With fixed expert capacity, overflow tokens at popular experts get dropped"
      ],
      correct: [0, 1, 3],
      explanation: "Imbalance wastes parameters (dead experts), wastes hardware (hot devices bottleneck each step while others idle), and — under fixed capacity buffers — silently drops the tokens that overflow a popular expert, hurting quality. The softmax itself always normalizes fine; the problem is where its mass concentrates, not its arithmetic."
    },
    // ---------- Fine-grained & shared experts ----------
    {
      type: "single",
      question: "DeepSeekMoE's 'fine-grained expert segmentation' splits each expert into $m$ smaller ones and activates $m$ times as many. Why?",
      options: [
        "Smaller experts fit in on-chip cache, cutting memory bandwidth pressure",
        "Tokens stay on their local device, reducing all-to-all communication traffic",
        "Even load becomes automatic, making the auxiliary balance loss unnecessary",
        "Far more expert combinations per token, enabling specialized decomposition"
      ],
      correct: [3],
      explanation: "Holding parameters and FLOPs fixed, slicing experts finer explodes the number of possible expert combinations — e.g. choosing 8 of 256 rather than 2 of 8 — so each small expert can specialize narrowly. The lecture's ablations back this up: DeepSeek finds more, finer experts help, and OLMoE likewise attributes its gains to fine-grained experts."
    },
    {
      type: "single",
      question: "What is the role of the shared expert(s) in DeepSeekMoE-style architectures?",
      options: [
        "They serve as a fallback path for tokens that overflow expert capacity",
        "Always-on experts absorb common knowledge, letting routed experts specialize",
        "They are frozen copies of the dense checkpoint, acting as a regularizer",
        "They process rare tokens whose hidden states the router cannot classify"
      ],
      correct: [1],
      explanation: "A shared expert processes every token, bypassing the router entirely, so common knowledge concentrates there instead of being duplicated across routed experts. The idea (originally from DeepSpeed-MoE) is used by DeepSeek and Qwen; DeepSeek-V3 keeps 1 shared expert beside 256 routed. The evidence is mixed, though: DeepSeek's ablations show shared experts help, while OLMoE found gains from fine-grained experts but none from shared ones."
    },
    // ---------- Systems ----------
    {
      type: "single",
      question: "Why do MoEs lend themselves to parallelism across many devices?",
      options: [
        "The router can be replicated, so the devices never need to communicate",
        "Experts share most of their weights, so devices need little synchronization",
        "Attention can be computed on one device and skipped on all the others",
        "Each expert FFN fits on one device, making experts a natural sharding unit"
      ],
      correct: [3],
      explanation: "Expert parallelism places different expert FFNs on different devices, adding an extra axis of parallelism beyond data, tensor, and pipeline. The complexity is that tokens must be exchanged across devices to reach their assigned experts, and routing imbalance shows up as idle hardware. Libraries like MegaBlocks, used in many open MoEs, handle the resulting irregular sparse matrix multiplications efficiently."
    },
    {
      type: "single",
      question: "There was speculation that GPT-4's nondeterminism comes from being an MoE. What mechanism could make an MoE's outputs depend on other users' requests?",
      options: [
        "The router adds Gaussian noise to its logits at inference time for robustness",
        "Experts are randomly reassigned to different devices for each incoming request",
        "Batch-level token dropping, where other queries overflow an expert and drop your token",
        "The gating softmax temperature adapts to the statistics of each serving batch"
      ],
      correct: [2],
      explanation: "Expert capacity is enforced per batch: when an expert fills up, overflow tokens are dropped and skip that expert, passing through via the residual stream. Since serving batches mix queries from different users, whether your token overflows depends on what everyone else sent — so other people's queries can drop your token."
    },
    {
      type: "multi",
      question: "DeepSeek-V3 pairs its MoE with multi-head latent attention (MLA). Which statements about MLA are correct? (Select all that apply)",
      options: [
        "MLA replaces standard attention with a routed mixture of expert attention heads",
        "Q, K, and V are expressed as functions of a lower-dimensional latent activation",
        "KV caching only needs to store the small latent, cutting cache memory",
        "RoPE conflicts with MLA's caching trick; a few non-latent rotated key dimensions fix it"
      ],
      correct: [1, 2, 3],
      explanation: "MLA compresses the hidden state into a low-dimensional latent from which Q, K, V are reconstructed; at inference only the latent needs caching, since the key up-projection can be merged into the query projection. Rotating keys breaks that merge, so RoPE is applied only to a few dedicated non-latent key dimensions. MLA is attention compression — it has nothing to do with routing experts."
    },
    // ---------- Training stability & upcycling ----------
    {
      type: "single",
      question: "The router z-loss (ST-MoE) penalizes the squared log-sum-exp of router logits. What problem does it address?",
      options: [
        "Dead experts caused by winner-take-all routing dynamics early in training",
        "Tokens silently dropped when popular experts exceed their capacity factor",
        "Slow convergence of the shared expert relative to the routed experts",
        "Instability from large router logits amplifying roundoff error in the softmax"
      ],
      correct: [3],
      explanation: "The router's exponentiated softmax is disproportionately sensitive to large logits, and in low precision small roundoff errors there can destabilize the whole model. Zoph et al.'s fix is to keep the router in float32, sometimes with the auxiliary z-loss shrinking logit magnitudes; their ablation shows loss diverging when the z-loss is removed. It complements — not replaces — the balance loss."
    },
    {
      type: "single",
      question: "What is 'upcycling' in the context of MoE training?",
      options: [
        "Distilling a trained MoE back into a small dense model for cheap serving",
        "Initializing an MoE's experts from a pre-trained dense model's checkpoint",
        "Reusing an older model's tokenizer and embeddings while retraining the rest",
        "Adding experts progressively during training as the loss begins to plateau"
      ],
      correct: [1],
      explanation: "Upcycling uses a pre-trained dense LM to initialize the MoE — its FFN seeds the experts — letting training start from a strong solution instead of scratch. The lecture's examples: MiniCPM's simple upcycled MoE (8 experts, top-2) gains over its base with ~520B training tokens, and Qwen's MoE — initialized from the dense Qwen 1.8B with 60 experts, top-4, and 4 shared — was one of the first confirmed upcycling successes."
    },
    // ---------- Canonical models ----------
    {
      type: "multi",
      question: "Which statements about Mixtral's MoE design are correct? (Select all that apply)",
      options: [
        "Each layer has 8 expert FFNs and each token is routed to 2 of them",
        "Like DBRX and DeepSeek-V3, it applies the softmax after top-$k$ selection",
        "It adds always-on shared experts alongside its 8 routed experts",
        "It uses expert-choice routing, where experts select their preferred tokens",
        "It is effectively 8 independent 7B models voting on each token"
      ],
      correct: [0, 1],
      explanation: "Per the lecture's table, Mixtral has 8 routed experts with 2 active and no shared experts. It sits in the family that softmaxes over just the selected top-$k$ scores (with DBRX and DeepSeek-V3), whereas DeepSeek v1-2, Grok, and Qwen softmax before selecting. It is one token-choice MoE model — not expert-choice, and not an ensemble of separate 7B models."
    },
    {
      type: "multi",
      question: "Which statements about DeepSeek-V3's MoE design are correct? (Select all that apply)",
      options: [
        "Each MoE layer routes every token to 8 of its 256 fine-grained experts",
        "A single always-on shared expert per MoE layer processes every token",
        "It balances load with a large Switch-style auxiliary loss as its primary mechanism",
        "About 37B of its 671B total parameters are activated per token",
        "It abandons fine-grained experts in favor of a few large ones per layer"
      ],
      correct: [0, 1, 3],
      explanation: "DeepSeek-V3 is the endpoint of the v1-v2-v3 evolution: 256 small routed experts (top-8, scored with a sigmoid and softmaxed after selection, plus top-M device routing) and 1 shared expert per layer, 671B total with 37B active. Its primary balancing mechanism is the aux-loss-free per-expert bias — only a small sequence-wise auxiliary loss remains — unlike v1-2, which used Switch-style expert and device balance losses."
    },
  ]
};

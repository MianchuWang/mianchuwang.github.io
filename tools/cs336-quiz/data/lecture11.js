// CS336 Lecture 11 — Scaling Laws II (curated 20 essential questions)
// Audited against the Spring 2025 slides: muP & hyperparameter transfer
// (CerebrasGPT, MiniCPM, replication study), WSD schedules, Chinchilla-style
// analysis in practice (MiniCPM, DeepSeek IsoFLOP, Llama 3, Hunyuan), and
// batch/LR scaling.
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture11"] = {
  title: "Lecture 11 — Scaling Laws II",
  questions: [
    {
      type: "single",
      question: "At a high level, how do the MiniCPM and DeepSeek scaling recipes differ?",
      options: [
        "MiniCPM relies on muP transfer; DeepSeek directly fits power laws for batch and LR",
        "MiniCPM trains only mixture-of-experts models; DeepSeek's analysis covers only dense models",
        "MiniCPM skips scaling analysis and reuses Llama 2's settings; DeepSeek relies on muP transfer",
        "MiniCPM tunes hyperparameters on the final full-scale model; DeepSeek never tunes them at all"
      ],
      correct: [0],
      explanation: "These are the lecture's two detailed case studies. MiniCPM: apply muP so LR (and friends) stay stable across widths, then use a WSD schedule for cheap Chinchilla fitting. DeepSeek: no muP — instead run small-scale sweeps and fit η_opt and B_opt as power laws of compute, plus IsoFLOP analysis for model sizing. Both arrive at strong models, showing there is more than one workable recipe."
    },
    {
      type: "single",
      question: "What is the purpose of adopting muP in MiniCPM's training recipe?",
      options: [
        "To keep optimal hyperparameters (especially LR) stable as width scales",
        "To reduce the memory footprint of optimizer states at large width",
        "To make the model converge in fewer epochs on small datasets",
        "To eliminate the need for a learning rate schedule entirely"
      ],
      correct: [0],
      explanation: "muP (maximum update parametrization) rescales initializations, learning rates, and output multipliers so that the optimal hyperparameters found on a small model remain (nearly) optimal at larger widths. MiniCPM tunes on tiny models (scale_emb=12, scale_depth=1.4, init_std=0.1, lr=0.01) and transfers those settings up — avoiding expensive tuning at full scale."
    },
    {
      type: "single",
      question: "This figure shows MiniCPM's loss versus learning rate for models from 0.04B to 2.1B after applying muP. What is the key takeaway?",
      image: "data/figures/lec11-mup-lr-stable.png",
      imageCredit: "Figure from MiniCPM (Hu et al., 2024), via CS336 Lecture 11",
      options: [
        "The minima line up at nearly the same LR across model sizes",
        "Larger models require much larger learning rates to converge",
        "The loss is insensitive to learning rate at every model size",
        "Small models diverge at the learning rate that is optimal for large models"
      ],
      correct: [0],
      explanation: "Each curve is one model size; the minima line up at roughly the same learning rate (~0.01). That is muP delivering on its promise: after the reparametrization, the LR shift across scales becomes minimal, so hyperparameters tuned on small models transfer. Without muP, the optimal LR would drift systematically with width."
    },
    {
      type: "single",
      question: "Why is fitting a Chinchilla-style scaling law expensive when using a cosine learning-rate schedule?",
      options: [
        "Each data budget needs its own full run from scratch, making total cost roughly $n^2$",
        "Cosine schedules require double-precision optimizer states, roughly doubling memory cost",
        "Cosine schedules only produce valid scaling fits for models above 1B parameters",
        "The cosine period must equal the number of accelerators, limiting data parallelism"
      ],
      correct: [0],
      explanation: "A model trained with cosine decay for 40N tokens is NOT equivalent at 20N tokens to a model whose cosine was scheduled for 20N — intermediate checkpoints of a long run are systematically worse than dedicated short runs. So Chinchilla-style fitting needs a separate complete run per (model size, data size) pair, turning $n$ data points into $O(n^2)$ training cost. This is the problem WSD solves."
    },
    {
      type: "single",
      question: "This figure compares learning-rate schedules. What characterizes the WSD (green/dark) schedules versus cosine (orange)?",
      image: "data/figures/lec11-wsd-schedule.png",
      imageCredit: "Figure from MiniCPM (Hu et al., 2024), via CS336 Lecture 11",
      options: [
        "Warmup, then a long flat hold at peak LR, then a sharp final decay",
        "A continuously decreasing LR from the very first step to the last",
        "Multiple warm restarts, with the LR repeatedly returning to its peak mid-run",
        "A constant LR for the whole run, with no warmup and no decay phase"
      ],
      correct: [0],
      explanation: "WSD = Warmup-Stable-Decay: quick warmup, hold at max LR for most of training, then decay rapidly at the end. Crucially, WSD(40N,4N) and WSD(80N,8N) are IDENTICAL during the stable phase — the runs only diverge when each branches into its decay. Cosine, by contrast, commits to the total length from the start, which is exactly why it can't share computation across data budgets."
    },
    {
      type: "single",
      question: "How does the WSD schedule make Chinchilla-style scaling analysis dramatically cheaper?",
      options: [
        "All data budgets branch short decay runs off one shared stable-phase run",
        "It removes the need to evaluate loss on a held-out validation set",
        "It lets several model sizes share the same weights during the stable phase",
        "It shrinks the vocabulary during the stable phase so each step is faster"
      ],
      correct: [0],
      explanation: "Because checkpoints in the stable phase are all 'the same run', you can take the checkpoint at 10N, 20N, ..., 60N tokens and decay each briefly to get the loss a dedicated run would have achieved. One long run plus cheap decay branches makes the sweep $O(n)$ in the number of data budgets instead of $O(n^2)$. This single trick is what made MiniCPM's (and later DeepSeek-style) dense scaling sweeps affordable."
    },
    {
      type: "single",
      question: "The Chinchilla paper's three fitting approaches are referenced throughout this lecture. What are methods 1, 2, and 3?",
      options: [
        "1: lower envelope of training curves; 2: IsoFLOP parabolas; 3: parametric joint fit of $L(N,D)$",
        "1: grid search over sizes; 2: random search over data; 3: Bayesian optimization of both",
        "1: train one huge model; 2: distill it into smaller students; 3: measure the students' losses",
        "1: fit the learning rate; 2: fit the batch size; 3: fit weight decay as laws in compute"
      ],
      correct: [0],
      explanation: "Method 1 takes the lower envelope of loss-vs-compute across training curves; method 2 trains models of several sizes at fixed FLOP budgets and finds each parabola's minimum (IsoFLOP); method 3 fits the parametric form $L(N,D) = C_N N^{-\\alpha} + C_D D^{-\\beta} + L_0$ jointly. MiniCPM uses 1 and 3; DeepSeek uses 2; Llama 3 and Hunyuan use IsoFLOP-style analyses."
    },
    {
      type: "single",
      question: "This contour plot shows MiniCPM's Chinchilla method-3 joint fit. The fitted optimum implies a data-to-model ratio of ~192. Why is this striking?",
      image: "data/figures/lec11-minicpm-method3.png",
      imageCredit: "Figure from MiniCPM (Hu et al., 2024), via CS336 Lecture 11",
      options: [
        "It is roughly $10\\times$ Chinchilla's famous ~20 tokens per parameter",
        "It exactly reproduces Chinchilla's 20 tokens-per-parameter ratio on new data",
        "It implies compute-optimal models should have more parameters than training tokens",
        "It shows the optimal ratio does not shift with the compute budget at all"
      ],
      correct: [0],
      explanation: "Chinchilla's famous answer was ~20 tokens/parameter; MiniCPM's joint fit on their data/setup lands near 192 — roughly $10\\times$ higher, favoring much smaller models trained on far more data. The lecture notes recent models like Llama 3 also use significantly higher data-to-model ratios, suggesting careful optimization can go far beyond the 20×model_size rule of thumb. This is a running theme: the exact ratio is recipe-dependent."
    },
    {
      type: "single",
      question: "These DeepSeek plots illustrate Chinchilla method 2. How does the IsoFLOP procedure work?",
      image: "data/figures/lec11-deepseek-isoflop.png",
      imageCredit: "Figure from DeepSeek LLM (2024), via CS336 Lecture 11",
      options: [
        "Train several sizes per FLOP budget, take each parabola's minimum, regress optima vs compute",
        "Train one model per compute budget and linearly interpolate between the endpoints",
        "Fix the model size across all budgets and vary only the amount of training data",
        "Fit one parametric loss surface to all runs simultaneously in a single regression"
      ],
      correct: [0],
      explanation: "Each colored parabola is one compute budget (1e17 → 3e20): vary the model-size/data split at fixed FLOPs, and loss traces a U-shape whose bottom is the optimal allocation. Regressing the minima gives clean power laws for optimal model scale and data scale — extrapolated here to predict DeepSeek 67B's configuration. The parabola bottoms are notably flat: modest sizing errors cost little loss."
    },
    {
      type: "single",
      question: "In the relation $C \\approx 6ND$ used for compute-optimal analysis (e.g., $N_{\\text{opt}}/D_{\\text{opt}} = K^2(C/6)^{\\eta}$), what does $6ND$ represent?",
      options: [
        "Approximate training FLOPs: about 6 per parameter per token, forward plus backward",
        "The bytes of activation memory stored per parameter per token during training",
        "The number of optimizer state entries maintained for N parameters over D tokens",
        "The wall-clock time for one pass over the dataset on six accelerators"
      ],
      correct: [0],
      explanation: "Training compute $C \\approx 6ND$: about 2 FLOPs/param/token for the forward pass and ~4 for backward. This approximation ties the $(N, D)$ allocation problem to a single budget $C$, which is what all the compute-optimal machinery (Chinchilla, IsoFLOP, joint fits) optimizes over. DeepSeek's M-based variant refines exactly this approximation."
    },
    {
      type: "single",
      question: "Llama 3 predicts downstream benchmark accuracy from scaling laws in two steps, shown here. What are the steps?",
      image: "data/figures/lec11-llama3-isoflop.png",
      imageCredit: "Figures from The Llama 3 Herd of Models (2024), via CS336 Lecture 11",
      options: [
        "Compute → NLL via a power law, then NLL → accuracy via a sigmoid fitted on smaller models",
        "First predict accuracy directly from compute, then correct it using the validation loss",
        "First fit accuracy on Llama 2 models, then copy it unchanged to Llama 3",
        "First measure accuracy at 405B, then interpolate down to smaller scales"
      ],
      correct: [0],
      explanation: "Chaining two fits stabilizes an otherwise noisy extrapolation: compute → normalized NLL per char follows a clean power law, and NLL → task accuracy follows a sigmoid (fitted with help from Llama 2 generation models). Composing them predicted the 405B model's benchmark accuracy remarkably well. Their IsoFLOP analysis (left) also gave a ~39:1 token-to-parameter ratio — again above Chinchilla's 20."
    },
    {
      type: "single",
      question: "What is DeepSeek's strategy for setting batch size and learning rate at scale?",
      options: [
        "Grid-search at small scale, fit power laws in compute, and extrapolate",
        "Adopt muP so batch and LR transfer without any fitting",
        "Reuse the hyperparameters published for Llama 2 without changes",
        "Adjust batch and LR on the fly during the final large training run"
      ],
      correct: [0],
      explanation: "DeepSeek explicitly avoids muP. They grid-search batch/LR at small compute (e.g., 1e17 FLOPs), keep configurations within 0.25% of the best loss ('near-optimal'), and fit $\\eta_{\\text{opt}} = 0.3118\\,C^{-0.125}$, $B_{\\text{opt}} = 0.2920\\,C^{0.3271}$. The final 7B/67B settings come from evaluating these formulas at the target compute — pure empirical extrapolation."
    },
    {
      type: "single",
      question: "These are DeepSeek's fitted scaling curves for optimal batch size and learning rate. As compute grows, how do the optima move?",
      image: "data/figures/lec11-deepseek-batchlr.png",
      imageCredit: "Figure from DeepSeek LLM (2024), via CS336 Lecture 11",
      options: [
        "Optimal batch size increases while optimal learning rate decreases",
        "Both optimal batch size and optimal learning rate increase",
        "Optimal batch size decreases while optimal learning rate increases",
        "Neither changes systematically with compute"
      ],
      correct: [0],
      explanation: "The left panel: $B_{\\text{opt}}$ rises with training FLOPs ($\\propto C^{0.33}$, reaching ~9M and ~19.7M tokens for the 7B/2T and 67B/2T runs). The right panel: $\\eta_{\\text{opt}}$ falls slowly ($\\propto C^{-0.125}$, down to 4.2e-4 and 3.2e-4). The lecture notes the LR fit 'looks a bit questionable' — the scatter is wide — but the directions are robust and match broader practice: bigger runs, bigger batches, gentler learning rates."
    },
    {
      type: "single",
      question: "CerebrasGPT trained 0.1B–13B models with the Chinchilla recipe. What was its core finding?",
      options: [
        "Using the muP parametrization makes scaling more stable and predictable",
        "Cosine learning-rate schedules outperform WSD schedules at every model scale",
        "Chinchilla's 20 tokens-per-parameter rule breaks down below 1B parameters",
        "Hyperparameters must be re-tuned from scratch at every model size"
      ],
      correct: [0],
      explanation: "CerebrasGPT is the first of the lecture's three detailed public scaling recipes (with MiniCPM and DeepSeek). Its authors found more predictable scaling from the muP parametrization: combined with aggressive small-scale hyperparameter optimization (scale_emb=10, lr=6e-3, init_std=0.08), hyperparameters stayed generally stable across the 0.1B–13B sweep. MiniCPM later adopted the same muP-based strategy with its own constants."
    },
    {
      type: "multi",
      question: "Different labs model optimal batch size as a function of different variables. Which pairings are correct? (Select all that apply)",
      options: [
        "OpenAI / Kaplan: batch as a function of the loss (critical batch size)",
        "MiniCPM: follows the Kaplan analysis, plotting optimal batch against final loss",
        "DeepSeek: batch as a power law in the compute budget C",
        "CerebrasGPT: batch fitted as a power law of the network's depth and aspect ratio"
      ],
      correct: [0, 1, 2],
      explanation: "MiniCPM identifies optimal batch sizes from training-curve sweeps at three model sizes, then follows the Kaplan 2020 analysis: plot optimal batch versus final loss, giving a clean trend of polynomially larger batches as loss decreases. DeepSeek instead fits $B_{\\text{opt}}$ directly as a power law in compute ($\\propto C^{0.327}$). CerebrasGPT does no batch-size scaling fit — it uses muP plus the Chinchilla formula directly."
    },
    {
      type: "single",
      question: "Hunyuan-1 extends IsoFLOP-style Chinchilla analysis to MoE models. What optimal ratio did it find?",
      options: [
        "About 96 training tokens per active parameter",
        "About 20 tokens per active parameter, exactly matching Chinchilla",
        "About 6 tokens per active parameter, favoring very large models",
        "About 1000 tokens per active parameter, favoring tiny models"
      ],
      correct: [0],
      explanation: "For a mixture-of-experts model the natural unit is ACTIVE (per-token) parameters, not total parameters, and Hunyuan-1's isoflop sweep puts the optimum near 96 tokens per active parameter. Alongside Llama 3's 39:1 and MiniCPM's ~192, it is another data point in the lecture's theme: modern recipes consistently land well above Chinchilla's ~20 tokens per parameter."
    },
    {
      type: "single",
      question: "The lecture derives muP from two conditions on how the network behaves as width grows. What are they?",
      options: [
        "Activations stay $\\Theta(1)$ at initialization, and change by $\\Theta(1)$ after one gradient step",
        "Gradients vanish at initialization, and the loss decreases monotonically each step",
        "Weight norms stay $\\Theta(1)$ at initialization, and the loss change is exactly zero per step",
        "Activations grow as $\\sqrt{\\text{width}}$ at initialization, and shrink after each update"
      ],
      correct: [0],
      explanation: "muP rests on A1: individual activations at initialization remain $\\Theta(1)$ as a function of width $n_l$ (so the norm is $\\Theta(\\sqrt{n_l})$), and A2: after one gradient step the CHANGE in activations is also $\\Theta(1)$. Working out which initialization variances and per-layer learning rates enforce both conditions in a deep linear network yields the muP scaling rules for init std and LR."
    },
    {
      type: "single",
      question: "In the lecture's muP replication study, which architectural component actually breaks muP's learning-rate transfer?",
      options: [
        "Learnable gains in RMSNorm",
        "SwiGLU activations in the MLP",
        "Squared-ReLU activations in the MLP",
        "Zero-initialized query projections"
      ],
      correct: [0],
      explanation: "The study stress-tests muP against modern deviations from its theory: nonlinearities (SwiGLU, squared ReLU), batch-size changes, and initialization tricks like zero-query and SP unembedding all preserve the stable optimal LR. Learnable RMSNorm gains break transfer — but can be removed with little performance loss. Sign-based optimizers (e.g., Lion) and strong (0.1) weight decay also break it, the latter being perhaps the most significant failure."
    },
    {
      type: "single",
      question: "Under muP, how should the Adam learning rate for hidden matrix-like parameters scale with width?",
      options: [
        "Proportional to $1/\\text{width}$ — smaller LRs on those tensors as models widen",
        "Proportional to $\\sqrt{\\text{width}}$ — wider models need larger hidden LRs",
        "Independent of width, exactly as in standard parametrization practice",
        "Proportional to $1/\\text{width}^2$ for all parameters including embeddings"
      ],
      correct: [0],
      explanation: "The A2 derivation gives per-layer LRs: for SGD $\\eta \\propto n_l/n_{l-1}$ (≈ constant for square layers), but for ADAM — whose update magnitudes are normalized — the rule becomes $\\eta \\propto 1/n_{l-1}$, i.e., $1/\\text{width}$ for hidden matrices, while 'other' params (embeddings, gains) keep $\\Theta(1)$ LR. This is the practical content of Yang's Table 2 ($l \\to l/r$ under a width multiplier $r$) and MiniCPM's $1/(d_m/d_{\\text{base}})$ LR scaling."
    },
    {
      type: "single",
      question: "This figure contrasts standard parametrization (left) with muP (right) as width grows from 128 to 8192. What does it show?",
      image: "data/figures/lec11-mup-shift.png",
      imageCredit: "Figure from Tensor Programs V (Yang et al., 2021), via CS336 Lecture 11",
      options: [
        "Under SP the optimal LR drifts with width; under muP it stays put",
        "Both parametrizations keep the optimal LR fixed; muP just reaches lower loss faster",
        "muP shifts the optimal LR to progressively larger values as width grows",
        "SP keeps a stable optimum while muP's optimal LR drifts with width"
      ],
      correct: [0],
      explanation: "Left panel: each width's loss-vs-LR curve bottoms out at a different LR, marching leftward as width grows — and curves turn up sharply (instability) where wider models can no longer tolerate LRs that narrow ones could. Right panel: minima aligned at one LR for every width. This is THE picture of what muP buys you: tune once at small width, reuse at large width."
    }
  ]
};

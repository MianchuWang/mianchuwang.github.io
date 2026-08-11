// CS336 Lecture 11 — Scaling: case study and details (Basics)
// 50 questions covering the core content: MiniCPM & DeepSeek scaling recipes,
// WSD learning rates, Chinchilla methods in practice, Step Law, optimizer
// scaling, and muP theory. 10 questions include figures from the slides.
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture11-basics"] = {
  title: "Lecture 11 — Scaling Laws II: Basics",
  questions: [
    // ---------- Big picture ----------
    {
      type: "multi",
      question: "Which questions frame this lecture on scaling in practice? (Select all that apply)",
      options: [
        "Does Chinchilla's approach to scaling actually work?",
        "Can we save compute when training and fitting scaling laws?",
        "Should we pick architectures / parametrizations that scale nicely?",
        "How do we serve models cheaply once they are trained?"
      ],
      correct: [0, 1, 2],
      explanation: "The lecture is a case study of scaling in the wild, organized around three questions: whether Chinchilla-style analysis holds up in real training runs, how to make fitting scaling laws affordable (e.g., WSD learning rates), and whether parametrizations like muP can make hyperparameters scale-invariant. Serving/inference was the previous lecture's topic."
    },
    {
      type: "single",
      question: "Why do modern LM reports (DeepSeek, MiniCPM, Llama 3, Qwen, Kimi...) all include scaling-law analyses?",
      options: [
        "Final runs cannot be tuned directly, so hyperparameters and sizing must be extrapolated from small models",
        "Regulators increasingly require published scaling analyses before approving large training runs",
        "Scaling laws are needed to compute the model's parameter count exactly",
        "It is a convention inherited from vision models, with little practical effect"
      ],
      correct: [0],
      explanation: "A frontier-scale run is a one-shot, multi-million-dollar experiment: you cannot grid-search the learning rate on the final model. Every serious effort therefore fits scaling trends (for LR, batch size, model-vs-data sizing) on a ladder of cheap small models and extrapolates to the big one. The lecture walks through exactly how MiniCPM, DeepSeek, and others do this."
    },
    {
      type: "single",
      question: "At a high level, how do the MiniCPM and DeepSeek scaling recipes differ?",
      options: [
        "MiniCPM uses muP for scale-invariant hyperparameters; DeepSeek directly fits power laws for batch and LR",
        "MiniCPM trains only MoE models, while DeepSeek's analysis covers only dense models",
        "MiniCPM avoids scaling analysis entirely; DeepSeek relies on muP transfer",
        "MiniCPM tunes on the final model directly; DeepSeek never tunes hyperparameters at all"
      ],
      correct: [0],
      explanation: "These are the lecture's two detailed case studies. MiniCPM: apply muP so LR (and friends) stay stable across widths, then use a WSD schedule for cheap Chinchilla fitting. DeepSeek: no muP — instead run small-scale sweeps and fit η_opt and B_opt as power laws of compute, plus IsoFLOP analysis for model sizing. Both arrive at strong models, showing there is more than one workable recipe."
    },

    // ---------- MiniCPM & muP in practice ----------
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
      type: "multi",
      question: "Which operations are part of MiniCPM's muP-style parametrization? (Select all that apply)",
      options: [
        "Multiplying the embedding output by a constant scale_emb",
        "Scaling residual-branch increments by scale_depth / √(num_layers)",
        "Scaling LRs and output logits by 1/(d_m/d_base) for matrix-like tensors",
        "Replacing softmax attention with linear attention at large widths"
      ],
      correct: [0, 1, 2],
      explanation: "MiniCPM's Table 7 lists the tensor-program operations: embedding output scaling (×12), residual connection scaling (scale_depth/√L), initialization std set via init_std/√(d_m/d_base), and 1/(d_m/d_base) scaling for both learning rates and LM-head logits of matrix-like parameters. Changing the attention mechanism is not part of muP — it is a parametrization, not an architecture change."
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
      question: "What is MiniCPM's overall scaling strategy for model architecture?",
      options: [
        "Fix the aspect ratio and scale overall size through a model ladder (9M → 0.5B)",
        "Grow depth only, keeping width constant across all model sizes",
        "Search a different architecture shape independently at every scale",
        "Grow width only, keeping the number of layers fixed at 8"
      ],
      correct: [0],
      explanation: "MiniCPM keeps proportions fixed (e.g., head dim 64 everywhere, d_ff/d_m constant) and scales overall size: 9M, 30M, 70M, 0.1B, 0.17B, 0.2B, 0.5B. With muP handling hyperparameter transfer, optimal batch, LR, and token-to-size ratios are then fitted directly via scaling analysis. Note the released model is still ~5× larger than the largest scaling-ladder model — extrapolation is always involved."
    },
    {
      type: "single",
      question: "This figure plots MiniCPM's fitted optimal batch size against loss (following Kaplan-style analysis). What trend does it show?",
      image: "data/figures/lec11-batch-loss.png",
      imageCredit: "Figure from MiniCPM (Hu et al., 2024), via CS336 Lecture 11",
      options: [
        "Optimal batch size grows polynomially as the achievable loss decreases",
        "Optimal batch size is constant once training stabilizes",
        "Optimal batch size shrinks steadily as models train to lower loss",
        "Batch size and loss are unrelated in these experiments"
      ],
      correct: [0],
      explanation: "The fit log(BS) = −6.24·log(L) + 20.91 says: the lower the loss you are training toward, the larger the batch you can productively use. Intuition: near the end of training (or with stronger models) gradients are less noisy relative to the signal, so larger batches waste less compute. This 'batch as a function of loss' view originates from OpenAI's critical batch size analysis (Kaplan 2020 era)."
    },
    {
      type: "single",
      question: "Kimi K2 fits a 'sparsity scaling law' for its MoE architecture (sparsity = total experts ÷ activated experts). What does it find?",
      options: [
        "Increasing sparsity steadily improves loss at fixed activated compute; K2 picks sparsity 48",
        "Sparsity beyond 8 total experts per active expert starts to hurt the loss",
        "Sparsity affects only inference speed and has no effect on training loss",
        "The best models keep total and activated expert counts exactly equal"
      ],
      correct: [0],
      explanation: "Holding activated parameters fixed, raising the total number of experts (higher sparsity) consistently lowers both training and validation loss: at the same target loss, sparsity 48 needs 1.69×/1.39×/1.15× fewer FLOPs than sparsity 8/16/32. The gain comes with infrastructure complexity, so K2 balances at sparsity 48 — activating 8 of 384 experts per forward pass. A nice example of scaling laws guiding an architecture choice, not just sizing."
    },
    {
      type: "single",
      question: "How did CerebrasGPT use muP in its 0.1B–13B model family?",
      options: [
        "Tune a small 40M muP model, then transfer the hyperparameters up the family",
        "Use muP only for the largest model while tuning the smaller ones by hand",
        "Apply muP at inference time to stabilize generation",
        "Use muP to compress the training data before fitting scaling laws"
      ],
      correct: [0],
      explanation: "CerebrasGPT trained with the Chinchilla recipe and followed muTransfer: tune a 40M-parameter proxy, then carry those hyperparameters along the muP scaling law up to 2.7B (with small changes like element-wise activation scaling and layer-wise LR scaling). Core finding: muP parametrization made scaling more stable — their muP models tracked the scaling-law trend more tightly than standard-parametrization ones."
    },

    // ---------- Cosine problem & WSD ----------
    {
      type: "single",
      question: "Why is fitting a Chinchilla-style scaling law expensive when using a cosine learning-rate schedule?",
      options: [
        "Each data budget needs its own full run from scratch, making total cost roughly n²",
        "Cosine schedules require double-precision optimizer states, doubling memory cost",
        "Cosine schedules only work for models above 1B parameters",
        "The cosine period must equal the number of GPUs, limiting parallelism"
      ],
      correct: [0],
      explanation: "A model trained with cosine decay for 40N tokens is NOT equivalent at 20N tokens to a model whose cosine was scheduled for 20N — intermediate checkpoints of a long run are systematically worse than dedicated short runs. So Chinchilla-style fitting needs a separate complete run per (model size, data size) pair, turning n data points into ~n² training cost. This is the problem WSD solves."
    },
    {
      type: "single",
      question: "This figure compares learning-rate schedules. What characterizes the WSD (green/dark) schedules versus cosine (orange)?",
      image: "data/figures/lec11-wsd-schedule.png",
      imageCredit: "Figure from MiniCPM (Hu et al., 2024), via CS336 Lecture 11",
      options: [
        "Warmup, a long flat phase at peak LR, then a sharp decay — runs share the stable phase",
        "A continuously decreasing LR from the first step to the last",
        "Multiple warm restarts, with the LR repeatedly returning to its peak value mid-run",
        "A constant LR for the whole run with no decay at all"
      ],
      correct: [0],
      explanation: "WSD = Warmup-Stable-Decay: quick warmup, hold at max LR for most of training, then decay rapidly at the end. Crucially, WSD(40N,4N) and WSD(80N,8N) are IDENTICAL during the stable phase — the runs only diverge when each branches into its decay. Cosine, by contrast, commits to the total length from the start, which is exactly why it can't share computation across data budgets."
    },
    {
      type: "single",
      question: "How does the WSD schedule make Chinchilla-style scaling analysis dramatically cheaper?",
      options: [
        "All budgets share one stable run; each branches a short decay, making cost linear",
        "It removes the need to evaluate loss on a validation set",
        "It lets several model sizes share the same weights during the stable phase",
        "It shrinks the vocabulary so each training step is faster"
      ],
      correct: [0],
      explanation: "Because checkpoints in the stable phase are all 'the same run', you can take the checkpoint at 10N, 20N, ..., 60N tokens and decay each briefly to get the loss a dedicated run would have achieved. One long run + m cheap decay branches ≈ O(mC) instead of m full retrainings. This single trick is what made MiniCPM's (and later DeepSeek-style) dense scaling sweeps affordable."
    },
    {
      type: "single",
      question: "This figure shows loss curves for WSD variants versus cosine on C4. How does WSD behave across its phases?",
      image: "data/figures/lec11-wsd-loss.png",
      imageCredit: "Figure from MiniCPM (Hu et al., 2024), via CS336 Lecture 11",
      options: [
        "Slower than cosine in the stable phase, then a sharp drop during decay catches up",
        "Loss tracks cosine exactly throughout the entire run",
        "Loss is lower than cosine early on but gradually falls behind later in training",
        "Loss oscillates during the stable phase and never recovers"
      ],
      correct: [0],
      explanation: "During the flat high-LR stable phase, WSD's loss sits visibly above the cosine run; when the decay kicks in, loss plunges and lands at (or below) the cosine endpoint. The final-loss equivalence is what licenses using WSD branches as stand-ins for dedicated cosine runs in scaling fits — you get cheap sweeps without sacrificing the quality of each measurement."
    },
    {
      type: "single",
      question: "Roughly what fraction of a run does MiniCPM allocate to the WSD decay phase?",
      options: [
        "About 10% of the tokens",
        "About 50% of the tokens",
        "About 90% of the tokens",
        "A fixed 100 steps regardless of run length"
      ],
      correct: [0],
      explanation: "The decay is short — around a tenth of the run (e.g., WSD(80N, 8N): 80N total with an 8N decay). That is what makes branching cheap: each additional data-budget measurement costs only ~10% of a full run. DeepSeek's variant uses two discrete drops (to 31.6% at 80% of tokens, to 10% at 90%) with the same practical effect."
    },
    {
      type: "single",
      question: "DeepSeek uses a multi-step (WSD-like) schedule instead of cosine: warmup, then LR drops at 80% and 90% of training. How does it compare to cosine in final loss?",
      options: [
        "It generally matches cosine's final performance",
        "It consistently beats cosine by a wide margin at every scale",
        "It underperforms cosine but is kept for implementation simplicity",
        "It only works when combined with muP initialization"
      ],
      correct: [0],
      explanation: "DeepSeek's 2000-step warmup + drops to 31.6%/10% of peak at 80%/90% of tokens produces training curves that track cosine closely and end at essentially the same loss. That equivalence matters: you keep cosine-level quality while gaining the reusable-checkpoint property that makes scaling analysis (and continued training) far cheaper."
    },

    // ---------- Chinchilla methods & sizing results ----------
    {
      type: "single",
      question: "The Chinchilla paper's three fitting approaches are referenced throughout this lecture. What are methods 1, 2, and 3?",
      options: [
        "1: lower envelope of training curves; 2: IsoFLOP parabolas; 3: parametric joint fit of L(N, D)",
        "1: grid search; 2: random search; 3: Bayesian optimization",
        "1: train one huge model; 2: distill it into students; 3: measure the students' losses",
        "1: fit LR; 2: fit batch size; 3: fit weight decay"
      ],
      correct: [0],
      explanation: "Method 1 takes the lower envelope of loss-vs-compute across training curves; method 2 trains models of several sizes at fixed FLOP budgets and finds each parabola's minimum (IsoFLOP); method 3 fits the parametric form L(N,D) = C_N·N^−α + C_D·D^−β + L0 jointly. MiniCPM uses 1 and 3; DeepSeek uses 2; Llama 3 and Hunyuan use IsoFLOP-style analyses."
    },
    {
      type: "single",
      question: "This contour plot shows MiniCPM's Chinchilla method-3 joint fit. The fitted optimum implies D_opt/N_opt ≈ 96 at C=10²¹. Why is this striking?",
      image: "data/figures/lec11-minicpm-method3.png",
      imageCredit: "Figure from MiniCPM (Hu et al., 2024), via CS336 Lecture 11",
      options: [
        "It is ~5× Chinchilla's ~20 tokens per parameter, favoring smaller models on more data",
        "It exactly reproduces Chinchilla's 20:1 ratio on a new dataset",
        "It implies compute-optimal models should have more parameters than training tokens",
        "It shows compute has no influence on the optimal ratio"
      ],
      correct: [0],
      explanation: "Chinchilla's famous answer was ~20 tokens/parameter; MiniCPM's joint fit on their data/setup lands near 96 (and η≈0 — the ratio barely moves with compute). Modern recipes (with better data and schedules) systematically find higher optimal ratios, and teams often over-train even beyond compute-optimal because smaller models are cheaper to SERVE. This is a running theme: the exact ratio is recipe-dependent."
    },
    {
      type: "single",
      question: "These DeepSeek plots illustrate Chinchilla method 2. How does the IsoFLOP procedure work?",
      image: "data/figures/lec11-deepseek-isoflop.png",
      imageCredit: "Figure from DeepSeek LLM (2024), via CS336 Lecture 11",
      options: [
        "Train several sizes per FLOP budget, take each parabola's minimum, regress optima vs compute",
        "Train one model per budget and interpolate between the endpoints",
        "Fix the model size and vary only the data across budgets",
        "Fit one parametric loss surface to all runs simultaneously in a single regression"
      ],
      correct: [0],
      explanation: "Each colored parabola is one compute budget (1e17 → 3e20): vary the model-size/data split at fixed FLOPs, and loss traces a U-shape whose bottom is the optimal allocation. Regressing the minima gives clean power laws for optimal model scale and data scale — extrapolated here to predict DeepSeek 67B's configuration. The parabola bottoms are notably flat: modest sizing errors cost little loss."
    },
    {
      type: "single",
      question: "Instead of parameter count N, DeepSeek expresses model scale as M — non-embedding FLOPs per token. Why?",
      options: [
        "Embeddings distort 6N at small scale; M counts compute actually spent",
        "M is easier to measure in practice than the exact parameter count",
        "Using N would leak information about the proprietary architecture",
        "M makes the numbers larger and easier to plot on log axes"
      ],
      correct: [0],
      explanation: "The standard C≈6ND treats every parameter as contributing equally, but embedding parameters do almost no compute — and at small model sizes they are a big fraction of N, biasing the fit. Counting non-embedding FLOPs/token (M, so C=MD) fixes the small-scale end of the ladder, which matters because that is exactly where scaling experiments live. A good example of the lecture's theme: details matter in scaling fits."
    },
    {
      type: "single",
      question: "In the relation C ≈ 6ND used for compute-optimal analysis (e.g., N_opt/D_opt = K²(C/6)^η), what does 6ND represent?",
      options: [
        "Approximate training FLOPs: ~6 per parameter per token (forward + backward)",
        "The memory in bytes needed to store activations during training",
        "The number of optimizer states for N parameters over D tokens",
        "The wall-clock seconds for one epoch on 6 GPUs"
      ],
      correct: [0],
      explanation: "Training compute ≈ 6·N·D: about 2 FLOPs/param/token for the forward pass and ~4 for backward. This approximation ties the (N, D) allocation problem to a single budget C, which is what all the compute-optimal machinery (Chinchilla, IsoFLOP, joint fits) optimizes over. DeepSeek's M-based variant refines exactly this approximation."
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
      question: "Across the recipes surveyed (Chinchilla ~20:1, Llama 3 ~39:1, Hunyuan ~96:1 per active parameter, MiniCPM ~96+:1), what is the overall trend in fitted data-to-model ratios?",
      options: [
        "Recent recipes find substantially higher ratios than Chinchilla's 20",
        "All modern fits converge tightly to the original 20:1 value",
        "Recent fits favor far fewer tokens per parameter than Chinchilla",
        "The ratios are random, with no pattern across papers"
      ],
      correct: [0],
      explanation: "Every modern fit lands well above 20:1 — better data, tokenizers, and schedules shift the optimum toward more tokens per parameter, and MoE models count only ACTIVE parameters (Hunyuan's 96:1). On top of that, labs deliberately over-train past compute-optimal because a smaller model with equal quality is cheaper to serve for its lifetime. Treat '20 tokens/param' as a historical anchor, not a law."
    },

    // ---------- DeepSeek batch/LR & Step Law ----------
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
      explanation: "DeepSeek explicitly avoids muP. They grid-search batch/LR at small compute (e.g., 1e17 FLOPs), keep configurations within 0.25% of the best loss ('near-optimal'), and fit η_opt = 0.3118·C^−0.125, B_opt = 0.2920·C^0.3271. The final 7B/67B settings come from evaluating these formulas at the target compute — pure empirical extrapolation."
    },
    {
      type: "single",
      question: "DeepSeek's grid searches revealed something reassuring about the batch/LR landscape. What?",
      options: [
        "Loss stays near-optimal across a wide region of batch and LR choices",
        "Only one exact (batch, LR) pair avoids divergence at each scale",
        "Batch size matters enormously but learning rate barely matters",
        "The landscape is so noisy that no trend is discernible"
      ],
      correct: [0],
      explanation: "Their heatmaps at 1e17 and 1e20 FLOPs show a broad basin: generalization error is stable across a wide range of batch sizes and LRs, with near-optimal performance available throughout a large region. This flatness is what makes power-law extrapolation viable at all — modest prediction errors in the fitted hyperparameters cost very little loss. StepFun's later study confirms the same convexity/flatness."
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
      explanation: "The left panel: B_opt rises with training FLOPs (∝C^0.33, reaching ~9M and ~19.7M tokens for the 7B/2T and 67B/2T runs). The right panel: η_opt falls slowly (∝C^−0.125, down to 4.2e-4 and 3.2e-4). The lecture notes the LR fit 'looks a bit questionable' — the scatter is wide — but the directions are robust and match broader practice: bigger runs, bigger batches, gentler learning rates."
    },
    {
      type: "single",
      question: "StepFun's 'Step Law' fits optimal LR and batch over a large grid. Which variable turns out to primarily drive optimal batch size?",
      options: [
        "Dataset size D (B_opt ∝ D^0.571), largely independent of model size",
        "Model width alone, with dataset size playing no measurable role",
        "The number of layers in the network",
        "The weight decay used during training"
      ],
      correct: [0],
      explanation: "Step Law: η_opt = 1.79·N^−0.713·D^0.307 and B_opt = 0.58·D^0.571 — batch depends on how much DATA you process, not on the model. Their fits achieve 0.94‰ relative error versus 9–10‰ for OpenAI/DeepSeek-style laws on the same grid. They also find higher optimal LR with larger D at fixed model size, though the lecture cautions this may be fragile under WSD-style schedules (see the InternLM scaling paper)."
    },
    {
      type: "single",
      question: "StepFun's first empirical observation makes hyperparameter fitting tractable. What is it?",
      options: [
        "Loss is bowl-shaped in batch and LR, so minimizers are cleanly identified",
        "Loss decreases monotonically in both batch size and learning rate",
        "The loss surface has many equally good but disconnected minima",
        "Loss depends only on the product of batch size and LR"
      ],
      correct: [0],
      explanation: "Slicing the loss surface along batch (at fixed LR) or along LR (at fixed batch) yields smooth convex curves with clear minima — the 3D surface is a well-behaved bowl. Convexity means a coarse grid suffices to locate optima reliably, which is what makes the 'purely empirical, grid-search the space' approach practical at hundreds of configurations."
    },
    {
      type: "multi",
      question: "Different labs model optimal batch size as a function of different variables. Which pairings are correct? (Select all that apply)",
      options: [
        "OpenAI / Kaplan: batch as a function of the loss (critical batch size)",
        "DeepSeek: batch as a power law in the compute budget C",
        "StepFun Step Law: batch as a power law in the dataset size D",
        "MiniCPM: batch as a function of the tokenizer's vocabulary size"
      ],
      correct: [0, 1, 2],
      explanation: "The functional-form disagreement is a real open question the lecture highlights: critical batch B(L) (OpenAI; also adopted by MiniCPM), B(C) (DeepSeek), and B(D) (Step Law). All fit their own data well — loss, compute, and data are correlated, so the laws partially coincide — but they extrapolate differently. MiniCPM ties batch to loss, not vocabulary."
    },
    {
      type: "single",
      question: "How robust is the fitted Step Law beyond its original setting?",
      options: [
        "It stays near-optimal across MoE sparsity levels and data recipes",
        "It holds for dense models only and fails for any MoE",
        "It works only on English-dominated training corpora",
        "It requires re-fitting from scratch for every architecture tweak"
      ],
      correct: [0],
      explanation: "StepFun's ablations show the law's predicted optima land inside the low-loss basin for MoE models at multiple sparsity ratios (N_a/N from 0.27 to 0.58) and across bilingual, code-heavy, and code-dominant data mixes. That robustness — plus explicitly modeling data recipe and sparsity — is their pitch over earlier laws. (Still an empirical fit, though: outside tested regimes, caution applies.)"
    },

    // ---------- Optimizer scaling ----------
    {
      type: "single",
      question: "This figure evaluates optimizers against AdamW across model sizes at 8× Chinchilla data. What is the headline finding?",
      image: "data/figures/lec11-opt-speedup.png",
      imageCredit: "Figures from 'Fantastic Pretraining Optimizers and Where to Find Them' (Wen et al., 2025), via CS336 Lecture 11",
      options: [
        "Speedups over AdamW shrink as model size grows — from ~1.4× at 130M to ~1.1× at 1.2B",
        "Speedups over AdamW grow with model size, favoring exotic optimizers at scale",
        "All optimizers perform identically at every model size",
        "AdamW is fastest at small sizes but the slowest at 1.2B"
      ],
      correct: [0],
      explanation: "The left panel shows Muon/Soap's advantage decaying with scale; the right shows matrix-based optimizers still consistently beating scalar-based ones, but by narrowing margins. The sober lesson: a 1.4× speedup measured on a 130M speedrun does NOT promise 1.4× at 10B. Any optimizer claim must be tested across scale before being trusted for a frontier run."
    },
    {
      type: "single",
      question: "In the same benchmark, which family of optimizers consistently outperforms which?",
      options: [
        "Matrix-based (Muon, Soap, Kron) consistently beat scalar-based (AdamW, NAdamW, Mars)",
        "Scalar-based optimizers consistently beat the matrix-based family",
        "Sign-based optimizers dominate both of the other families at all scales",
        "No family ordering survives across model sizes"
      ],
      correct: [0],
      explanation: "Across model sizes and Chinchilla ratios, the solid curves (matrix-based: they precondition using matrix structure of the weights) sit below the dashed scalar-based ones in loss. The gap narrows with scale but the ORDERING is stable. This motivates serious attention to optimizers like Muon — which Kimi K2 then validated at trillion-parameter MoE scale."
    },
    {
      type: "single",
      question: "Why does the lecture insist that optimizer comparisons be run across multiple compute budgets AND Chinchilla ratios?",
      options: [
        "Both are major confounders: an optimizer's advantage can shrink as either changes",
        "Because optimizers only differ when the batch size is exactly critical",
        "To satisfy statistical-significance requirements for publication",
        "Because LR schedules only behave well at specific Chinchilla ratios"
      ],
      correct: [0],
      explanation: "The speedup-vs-scale decay is one axis; the Chinchilla ratio is another (speedups also compress from 1.4× toward 1.1× as tokens/Chinchilla grows from 1× to 8×). A comparison at one (size, ratio) point can flatter almost any method. 'Always check scaling with respect to compute and chinchilla ratios' is stated as a general algorithm-development principle."
    },
    {
      type: "single",
      question: "The 'Fantastic Pretraining Optimizers' study found that many published optimizer speedups were inflated. What was one major reason?",
      options: [
        "Baselines were under-tuned — a properly tuned AdamW LR alone gave ~2× speedups",
        "The new optimizers were run with double the compute of the baselines",
        "Speedups were measured only in wall-clock time and never in optimization steps",
        "Datasets were switched between the baseline and the new method"
      ],
      correct: [0],
      explanation: "Their 130M example: AdamW with lr 8e-3 versus a mistuned 6e-4 differs by ~2× in steps-to-loss — comparable to entire claimed 'optimizer breakthroughs'. If the baseline is weak, anything looks fast. The paper's discipline: tune EVERY optimizer's hyperparameters properly at every scale before comparing. Under that protocol, real but modest gains remain (Muon-family ~1.1–1.4×)."
    },
    {
      type: "single",
      question: "The same study warns that hyperparameters do not transfer between optimizers. What is their concrete example?",
      options: [
        "Lion's optimal weight decay is ≈0.6 — far from AdamW-style defaults",
        "Muon requires a batch size below 32 to converge at all",
        "SGD needs a higher LR than Adam by exactly 100×",
        "NAdamW only converges when paired with a cosine schedule"
      ],
      correct: [0],
      explanation: "Sweeping weight decay for Lion shows a clean optimum near 0.6, while AdamW's sits much lower — an evaluation that fixes wd at AdamW's value would unfairly cripple Lion (and vice versa). Different optimizers need different hyperparameters AND likely different hyperparameter scaling rules, compounding the difficulty of fair comparisons across scale."
    },
    {
      type: "single",
      question: "What does the Muon optimizer do, mechanically?",
      options: [
        "Newton-Schulz iterations approximately orthogonalize the momentum buffer (USVᵀ → UVᵀ)",
        "It quantizes gradients to int8 before applying an otherwise Adam-style update",
        "It replaces gradients with their sign, like Lion, but adds weight decay",
        "It averages weights across several independent training runs each step"
      ],
      correct: [0],
      explanation: "Muon (for matrix-valued parameters): B_t = μB_{t−1} + G_t, then O_t = NewtonSchulz5(B_t), then θ_t = θ_{t−1} − ηO_t. The Newton-Schulz iteration approximately maps the momentum matrix's SVD USVᵀ to UVᵀ — an orthogonal update that equalizes the step across singular directions. That matrix-aware preconditioning is why it belongs to the 'matrix-based' family that consistently beats scalar methods."
    },
    {
      type: "single",
      question: "What is the strongest evidence so far that Muon 'works at scale'?",
      options: [
        "Kimi K2 — a trillion-parameter-class MoE pretrained with Muon",
        "It won the NanoGPT speedrun, which is a demanding large-scale benchmark",
        "A theoretical proof that its updates are optimal for transformers",
        "Its adoption as the default optimizer in PyTorch"
      ],
      correct: [0],
      explanation: "The lecture is careful about evidence tiers: NanoGPT speedruns are very small, and controlled scaling studies show shrinking (though real) gains — but Kimi K2 actually pretrained a frontier MoE with Muon (plus its own sparsity scaling laws), demonstrating production viability. 'Scaling gains are tricky to measure, but clearly muon works at scale' is the lecture's summary."
    },
    {
      type: "single",
      question: "The lecture shows a cautionary tale (from the Delphi/Marin runs): a scaling law that looked clean at small scale blew up when extrapolated. What is the lesson?",
      options: [
        "Good-looking small-scale fits can still diverge — validate at held-out larger scales",
        "Scaling laws should only ever be fitted with three data points",
        "Divergence at scale is always caused by data contamination",
        "Once IsoFLOP parabolas look smooth, extrapolation is guaranteed to be safe"
      ],
      correct: [0],
      explanation: "The IsoFLOP parabolas per compute bucket looked textbook-clean, yet the extrapolated large run diverged (0.8% worse → 2.5% worse → blow-up). The fix involved more careful parametrization and optimizer changes (cautious AdamC, √batch LR scaling). Moral: a scaling fit is a hypothesis; validate it out-of-sample before betting a frontier-scale budget on it."
    },

    // ---------- muP theory ----------
    {
      type: "single",
      question: "The 'spectral' formulation of muP is built on two conditions, A1 and A2. What are they?",
      options: [
        "A1: activations stay Θ(1) at init; A2: the change in activations after one step is Θ(1)",
        "A1: gradients vanish at initialization; A2: weights stay bounded forever",
        "A1: the loss starts below 1; A2: the loss decreases monotonically",
        "A1: all layers share one LR; A2: all layers share one initialization"
      ],
      correct: [0],
      explanation: "As width n_l grows, muP demands that (A1) each activation coordinate is Θ(1) at init — no blow-up, no vanishing — and (A2) one optimization step changes activations by Θ(1) — updates neither stall nor explode. Everything else (init variances, per-layer LRs) is DERIVED by enforcing these two conditions layer by layer. This is the 'muP for babies' framing from Yang, Simon & Bernstein's spectral-condition paper."
    },
    {
      type: "single",
      question: "If each coordinate of an n-dimensional activation vector is Θ(1), what is the vector's Euclidean norm?",
      options: [
        "Θ(√n)",
        "Θ(1)",
        "Θ(n)",
        "Θ(log n)"
      ],
      correct: [0],
      explanation: "Summing n squared coordinates of size Θ(1) gives ‖h‖² = Θ(n), so ‖h‖ = Θ(√n). This bookkeeping identity is used constantly in the muP derivation: conditions on 'coordinates of size 1' translate into norm conditions like ‖h_l‖ = Θ(√n_l), which then constrain the spectral norms of weights and updates."
    },
    {
      type: "single",
      question: "How does muP's initialization rule differ from standard practice?",
      options: [
        "std gains a min(1, √(fan-out/fan-in)) factor, differing from standard when fan-out < fan-in",
        "muP initializes every weight to exactly zero",
        "muP uses uniform instead of Gaussian distributions",
        "muP scales the initialization standard deviation by the square root of the batch size"
      ],
      correct: [0],
      explanation: "Standard init: σ = 1/√n_{l−1} (fan-in only). muP's derivation from condition A1 adds the min(1, √(n_l/n_{l−1})) factor, which only bites when the output dimension is SMALLER than the input — e.g., projection/unembedding layers (the Lingle table's 1/M² for the unembedding init variance is this correction). For square hidden layers the two coincide."
    },
    {
      type: "single",
      question: "Under muP, how should the Adam learning rate for hidden matrix-like parameters scale with width?",
      options: [
        "Proportional to 1/width — smaller LRs on those tensors as models widen",
        "Proportional to √width — wider models need larger LRs",
        "Independent of width, exactly as in standard parametrization practice",
        "Proportional to 1/width² for all parameters including embeddings"
      ],
      correct: [0],
      explanation: "The A2 derivation gives per-layer LRs: for SGD η ∝ n_l/n_{l−1} (≈ constant for square layers), but for ADAM — whose update magnitudes are normalized — the rule becomes η ∝ 1/n_{l−1}, i.e., 1/width for hidden matrices, while 'other' params (embeddings, gains) keep Θ(1) LR. This is the practical content of Yang's Table 2 (l → l/r under a width multiplier r) and MiniCPM's 1/(d_m/d_base) LR scaling."
    },
    {
      type: "single",
      question: "Besides init and LR rules, muP prescribes a change to attention itself. Which?",
      options: [
        "Scale attention logits by 1/d_head instead of the usual 1/√d_head",
        "Replace softmax with a linear kernel at large widths",
        "Double the number of attention heads at each width increase",
        "Remove positional embeddings, which violate Θ(1) activations"
      ],
      correct: [0],
      explanation: "Because q·k sums d coordinates of correlated (post-training) Θ(1) entries, its natural scale is d, not √d — so muP uses τ⁻¹ = Θ(1/d). The 1/√d convention is right at random init (where q,k are independent) but wrong once training correlates them. Lingle's μ-transfer study finds this attention scale has a MAJOR impact on transfer quality — one of the less obvious but most consequential muP details."
    },
    {
      type: "single",
      question: "This figure contrasts standard parametrization (left) with muP (right) as width grows from 128 to 8192. What does it show?",
      image: "data/figures/lec11-mup-shift.png",
      imageCredit: "Figure from Tensor Programs V (Yang et al., 2021), via CS336 Lecture 11",
      options: [
        "Under SP the optimal LR drifts and wide models blow up; under muP it stays put",
        "Both parametrizations keep the optimal LR fixed; muP just reaches lower loss faster",
        "muP shifts the optimal LR to larger values as width grows",
        "SP is stable while muP's optimum drifts with width"
      ],
      correct: [0],
      explanation: "Left panel: each width's loss-vs-LR curve bottoms out at a different LR, marching leftward as width grows — and curves turn up sharply (instability) where wider models can no longer tolerate LRs that narrow ones could. Right panel: minima aligned at one LR for every width. This is THE picture of what muP buys you: tune once at small width, reuse at large width."
    },
    {
      type: "single",
      question: "When Lingle's large-scale study replicated muP on transformers, what happened to the optimal base LR as width scaled 128 → 2048?",
      options: [
        "It stayed at 2⁻⁶ — the small model's optimum predicted the larger models'",
        "It shifted by two powers of two at every width doubling",
        "It transferred cleanly to width 512 but broke down entirely at 2048",
        "It transferred only when RMSNorm gains were added"
      ],
      correct: [0],
      explanation: "Baseline μP: optimal base LR 2⁻⁶ at widths 128, 512, AND 2048 (each model 4× wider / 16× larger than the last) — clean transfer, replicated again in their 10B-scale experiment (2M → 10B params, base LR 2⁻⁶ throughout). The study then becomes interesting precisely where transfer FAILS — gains, Lion, strong weight decay — mapping muP's practical boundaries."
    },
    {
      type: "multi",
      question: "Per the μ-transfer study, which modifications BREAK muP's learning-rate transfer? (Select all that apply)",
      options: [
        "Trainable RMSNorm gains (vector or scalar)",
        "The Lion optimizer (sign-based updates)",
        "Strong decoupled weight decay (0.1)",
        "SwiGLU activations and zero-init attention projections"
      ],
      correct: [0, 1, 2],
      explanation: "Transfer fails (optimal LR shifts across widths) for trainable norm gains, for Lion — whose sign updates fall outside the SGD/Adam analysis muP is derived for — and for strong decoupled weight decay, which the lecture calls 'maybe the only significant muP failure' in practice. SwiGLU, squared-ReLU, batch-size changes, and zero attention init all transfer fine — muP is robust to those."
    },
    {
      type: "single",
      question: "What did the study conclude about RMSNorm's trainable gain parameters in muP transformers?",
      options: [
        "They break LR transfer, hurt the largest models, and can be removed at little cost",
        "They are essential for training stability and must always be kept",
        "They only matter when combined with the Lion optimizer",
        "They improve transfer as long as they are trained with a lower learning rate"
      ],
      correct: [0],
      explanation: "With Θ(1) LR scaling on the gains, optimal LRs no longer transferred, and trainable gains actively harmed the largest μP models at optimal LR. Dropping the gains entirely cost almost nothing. A tidy practical recipe emerges: μP + gainless RMSNorm + standard AdamW (mild weight decay) preserves the transfer property that makes small-scale tuning trustworthy."
    },
    {
      type: "single",
      question: "How does the lecture summarize the practical value of muP versus standard parametrization (SP)?",
      options: [
        "muP seems genuinely useful; SP is noticeably less stable at large width",
        "muP and SP behave identically in every measured setting",
        "muP is strictly required — no large model has ever trained without it",
        "muP only helps models below roughly 100M parameters"
      ],
      correct: [0],
      explanation: "In SP sweeps, width 2048 at LR 2⁻⁶ jumps to loss 7.2 (blow-up) while μP stays smooth, and SP's optimum wanders across widths. Verdict: 'muP parametrization / initialization may be easier to tune' — useful, not magic, and DeepSeek proves you can succeed without it by fitting hyperparameter scaling empirically. Two viable philosophies; know both."
    },

    // ---------- Recap ----------
    {
      type: "single",
      question: "The lecture closes with three main challenges of scaling 'in the wild'. Which list matches?",
      options: [
        "Architecture hypers, optimizer hypers, and the compute cost of Chinchilla sweeps",
        "Buying GPUs, hiring researchers, and cleaning data",
        "Choosing a tokenizer, a positional encoding, and an activation function",
        "Avoiding overfitting, underfitting, and double descent"
      ],
      correct: [0],
      explanation: "Challenges: (1) model-shape hypers like width/depth, (2) optimizer hypers like LR and batch, (3) the sheer compute of fitting model-vs-data tradeoffs. The paired solutions: assume stability or use muP for (1)-(2); search small-scale and fix or extrapolate (DeepSeek/Step Law) for (2); and WSD-like schedules to make (3) linear instead of quadratic."
    },
    {
      type: "multi",
      question: "Which techniques from this lecture directly reduce the COST of running a scaling-law study? (Select all that apply)",
      options: [
        "WSD/multi-step schedules — branch short decays off one shared stable run",
        "muP — tune hyperparameters once at small width instead of at every scale",
        "Exploiting the flatness of loss near hyperparameter optima — coarse grids suffice",
        "Training every candidate configuration to full convergence with cosine decay"
      ],
      correct: [0, 1, 2],
      explanation: "The lecture's cost-saving toolkit: WSD turns the n² retraining problem into ~linear cost; muP eliminates per-scale hyperparameter sweeps; and the broad/convex optima (DeepSeek's wide basins, StepFun's convexity) mean few grid points are needed. Full cosine retraining for every point is precisely the expensive baseline all of these techniques replace."
    }
  ]
};

// CS336 Lecture 9 — Scaling Laws I (curated 20 essentials)
// Selected from lecture9-basics.js: power laws & why they appear, data scaling
// theory, model-engineering scaling, critical batch size, Kaplan vs Chinchilla,
// IsoFLOP methodology, and practical applications.
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture9"] = {
  title: "Lecture 9 — Scaling Laws I",
  questions: [
    {
      type: "single",
      question: "This is the scaling-curve shape hypothesized by Hestness et al. In which region do practically interesting LLM training runs live, and why does that matter?",
      image: "data/figures/lec9-hestness-regions.png",
      imageCredit: "Figure from Hestness et al. (2017), via CS336 Lecture 9",
      options: [
        "The power-law region — where loss falls predictably with data",
        "The small-data region — where models are only as good as guessing",
        "The irreducible-error region — where more data cannot help at all",
        "None of them: real training curves have no consistent shape"
      ],
      correct: [0],
      explanation: "The three regions: small-data (best-guess error — the model has too little signal), power-law (log-linear descent), and irreducible error (the floor set by data entropy). All the useful predictive machinery — fitting on small runs and extrapolating — assumes you are operating inside the power-law region, which practical LLM runs are. Reaching the floor would require unrealistic data; staying above the cliff requires enough data to escape guessing."
    },
    {
      type: "single",
      question: "These are the famous Kaplan et al. (2020) plots. What do they establish?",
      image: "data/figures/lec9-kaplan-3axes.png",
      imageCredit: "Figures from Kaplan et al. (2020), via CS336 Lecture 9",
      options: [
        "Loss follows power laws in all three resources — compute, data, and parameters",
        "Loss follows a power law in compute only; data and parameters behave erratically",
        "Loss reaches exactly zero once any single resource is scaled far enough",
        "The three resources are freely interchangeable one-for-one at every scale"
      ],
      correct: [0],
      explanation: "$L(C) = (C_{\\text{min}}/2.3 \\cdot 10^8)^{-0.050}$, $L(D) = (D/5.4 \\cdot 10^{13})^{-0.095}$, $L(N) = (N/8.8 \\cdot 10^{13})^{-0.076}$: clean straight lines on log-log axes across ~7 orders of magnitude. Note how SMALL the exponents are — each constant decrement of loss costs a multiplicative increase in resources. That is why frontier progress is so expensive, and why the laws even extend to nonstandard settings (they hold even when train and test distributions differ)."
    },
    {
      type: "single",
      question: "Classical statistics predicts error slopes near $-1$ (rates like $1/n$). What do neural scaling laws actually show, and what is the puzzle?",
      options: [
        "Much shallower slopes, around $-0.05$ to $-0.13$ — far less data-efficient than $1/n$",
        "Much steeper slopes, around $-2$ to $-3$ — nets extract more per sample than theory allows",
        "Slopes of exactly $-1$, matching the classical mean-estimation analysis directly",
        "Slightly positive slopes — past a threshold, extra data marginally hurts test loss"
      ],
      correct: [0],
      explanation: "MT fits show $\\epsilon(m) \\approx 3.87\\,m^{-0.13}$, language modeling $L = (D/5.4 \\cdot 10^{13})^{-0.095}$, speech similar — an order of magnitude shallower than $1/n$. If simple mean estimation gets slope $-1$, why does a giant transformer get $\\approx -0.1$? The lecture presents this as an intriguing mystery, with the nonparametric/intrinsic-dimension story as the leading (but not airtight) explanation."
    },
    {
      type: "single",
      question: "The nonparametric detour: estimating a smooth function on $d$-dimensional inputs by partitioning space into boxes gives what error rate?",
      options: [
        "$\\text{Error} \\approx n^{-1/d}$ — a slope that flattens as the dimension $d$ grows",
        "$\\text{Error} \\approx n^{-d}$ — decay that steepens rapidly in higher dimension",
        "$\\text{Error} \\approx 1/n$ regardless of the input dimension $d$",
        "$\\text{Error} \\approx$ constant — nonparametric methods cannot learn at all"
      ],
      correct: [0],
      explanation: "Cutting $d$-dimensional space into bins and averaging within each bin spreads $n$ samples over exponentially many cells: error decays as $n^{-1/d}$ (slope $-1/d$). Takeaway: flexible 'nonparametric' learners — and neural nets acting like them — have DIMENSION-DEPENDENT scaling laws. A slope of $-0.1$ then reads naturally as 'the data behaves ~10-dimensional'."
    },
    {
      type: "single",
      question: "Muennighoff et al.'s data-constrained scaling: how valuable is REPEATED data compared to fresh data?",
      options: [
        "Up to ~4 epochs repeats are nearly as good as new data; by ~40 epochs they are worthless",
        "Any amount of repetition at all immediately and irreversibly damages the model",
        "Repeats are exactly as valuable as fresh data indefinitely, so data never runs out",
        "Value decays only after 1,000 epochs, far beyond practical budgets"
      ],
      correct: [0],
      explanation: "Their effective-data formula $D' = U_D + U_D R_D^*(1 - e^{-R_D/R_D^*})$ captures it: early repetitions count almost fully, then exponentially saturate. Practical upshots: modest multi-epoch training is fine when unique data is scarce, and the compute-optimal response to data constraints is a somewhat larger model trained on more epochs — but no amount of repetition substitutes for genuinely new tokens at ~40+ epochs."
    },
    {
      type: "single",
      question: "This Kaplan et al. figure compares LSTMs and transformers. What does the scaling-law lens reveal?",
      image: "data/figures/lec9-lstm-transformer.png",
      imageCredit: "Figure from Kaplan et al. (2020), via CS336 Lecture 9",
      options: [
        "Transformers scale better — lower loss at every size, with LSTM curves plateauing",
        "LSTMs and transformers scale identically once parameter counts match",
        "LSTMs win at small scale and overtake transformers again at the largest sizes",
        "Transformers only win because they were trained on more data in this comparison"
      ],
      correct: [0],
      explanation: "Instead of spending tens of millions training an 'LSTM GPT-3' to settle the question, plot both architectures' scaling curves: the transformer line sits below AND keeps descending, while LSTM curves (1/2/4 layers) bend and plateau — a worse trend, not just a worse point. This is the template for cheap, scaling-aware architecture comparisons."
    },
    {
      type: "single",
      question: "Tay et al. (2022) ran cross-architecture scaling comparisons over many transformer alternatives. What was the sobering result?",
      options: [
        "The vanilla transformer scaled best — most claimed improvements scaled worse",
        "Every alternative scaled better than the vanilla transformer baseline",
        "All architectures scaled identically, so architecture choice is irrelevant",
        "Only the convolution-based alternatives kept improving as compute grew"
      ],
      correct: [0],
      explanation: "Across ALBERT, Performer, Switch, Universal Transformer, MLP-Mixer, Funnel and more, few alternatives matched the plain transformer's scaling trend — many wins evaporated (or reversed) as compute grew. The lesson echoes through the course: evaluate architecture claims BY THEIR SCALING CURVES, not by a single-scale comparison, because slope beats intercept at scale."
    },
    {
      type: "single",
      question: "Hestness's Adam-vs-SGD comparison (on pre-transformer RHNs) found what pattern — one the lecture treats as typical for optimizer choices?",
      options: [
        "Nearly identical slopes with different offsets — a shifted curve, not a new rate",
        "Completely different slopes — Adam fundamentally changes how error decays with data",
        "SGD failed to converge entirely at every data size tested in the comparison",
        "The comparison was impossible because optimizer effects cannot be plotted against scale"
      ],
      correct: [0],
      explanation: "$\\epsilon(m) = 5.37\\,m^{-0.094}$ for SGD vs $\\epsilon(m) = 5.25\\,m^{-0.095}$ for Adam: same exponent to two decimals, slightly different constant. Optimizer choice (like many engineering choices) tends to move the OFFSET, not the slope — real but bounded gains. Contrast with architecture (LSTM vs transformer), where the trend itself differed. Diagnosing 'slope change vs offset change' is a core scaling-law skill."
    },
    {
      type: "single",
      question: "This Kaplan figure varies transformer depth. What is the takeaway about the number of layers?",
      image: "data/figures/lec9-depth.png",
      imageCredit: "Figure from Kaplan et al. (2020), via CS336 Lecture 9",
      options: [
        "1 → 2 layers matters enormously; extra depth beyond that shows diminishing returns",
        "Every additional layer improves loss by the same large margin, indefinitely",
        "Depth is irrelevant: a 1-layer model matches deep models at equal parameters",
        "Only models deeper than 50 layers reach the power-law region"
      ],
      correct: [0],
      explanation: "The 1-layer curve is clearly worse; 2 layers closes most of the gap; 3/6/>6 layers bunch together (diminishing returns below $10^7$ parameters). Combined with the aspect-ratio result, the message is liberating: once past minimal depth, transformer performance depends mainly on the parameter BUDGET, not on finely tuned shape."
    },
    {
      type: "single",
      question: "McCandlish et al.'s 'critical batch size' separates two training regimes. What are they?",
      options: [
        "Below $B_{\\text{crit}}$, larger batches give near-linear speedups; above it, returns diminish",
        "Below $B_{\\text{crit}}$ training diverges, while above it training is always numerically stable",
        "Below $B_{\\text{crit}}$ memory bandwidth is the bottleneck; above it raw compute throughput is",
        "Batch size only trades off memory use; it never changes the number of steps to a target loss"
      ],
      correct: [0],
      explanation: "Small batches give noisy gradient estimates, so doubling the batch nearly halves the steps needed ('perfect scaling'). Once the batch is large enough that gradients are already accurate — around the critical batch size — extra samples per step stop buying progress ('ineffective scaling'). $B_{\\text{crit}}$ is the knee of that curve, and it's claimed to track the gradient noise scale: the ratio of gradient covariance to squared gradient norm."
    },
    {
      type: "single",
      question: "This figure plots empirical critical batch size against training loss. What is the key empirical finding?",
      image: "data/figures/lec9-critical-batch.png",
      imageCredit: "Figure from Kaplan et al. (2020), via CS336 Lecture 9",
      options: [
        "$B_{\\text{crit}}$ follows a power law in the LOSS, largely independent of model size",
        "$B_{\\text{crit}}$ is a fixed constant (around 32k tokens) throughout the whole of training",
        "$B_{\\text{crit}}$ shrinks as loss falls, so batches should decay like the learning rate",
        "$B_{\\text{crit}}$ depends only on model size, with loss playing no role at any point"
      ],
      correct: [0],
      explanation: "The 3M and 85M curves lie on top of each other: what predicts the useful batch is not the model but HOW GOOD it currently is — $B_{\\text{crit}} = 2.1 \\times 10^8\\,\\text{tokens} \\cdot L^{-4.8}$ grows steeply as loss drops. Practical readings: ramp the batch during training, and expect stronger models/harder targets to support much larger batches. This 'batch as a function of loss' view is the OpenAI school that MiniCPM later adopts (Lecture 11)."
    },
    {
      type: "single",
      question: "If you scale a transformer's width naively (standard parametrization), what happens to the optimal learning rate — and what is the lecture's prescription?",
      options: [
        "It drifts with scale, motivating scaling-aware initialization and LR rules (muP)",
        "It stays exactly fixed across widths, so one tuned LR works at every scale",
        "It grows in proportion to width, so larger models need proportionally larger LRs",
        "It becomes irrelevant because large models are insensitive to LR"
      ],
      correct: [0],
      explanation: "The Yang et al. figure shows standard practice's loss-vs-LR minima marching across the axis as width grows 128 to 8192 ('optimum shifts'), versus a muP-parametrized model whose optimum stays put. The Lecture 9 message is just the phenomenon and the need for scale-aware LR/init choices; Lecture 11 does the muP derivation and its practical limits in depth."
    },
    {
      type: "single",
      question: "What did Kaplan et al. (2020) claim about compute-optimal allocation, and what behavior did it encourage?",
      options: [
        "$N_{\\text{opt}} \\propto C^{0.73}$ — mostly into parameters, encouraging huge under-trained models",
        "$D_{\\text{opt}} \\propto C^{0.73}$ — mostly into data, encouraging small models fed enormous token counts",
        "An exact 50/50 split of $N$ and $D$, encouraging balanced $\\approx 20$ tokens-per-parameter training",
        "All compute into longer training of a fixed-size model, encouraging ever-longer schedules"
      ],
      correct: [0],
      explanation: "Kaplan's exponents ($N_{\\text{opt}} \\propto C^{0.73}$, $D_{\\text{opt}} \\propto C^{0.27}$) imply tokens-per-parameter DECREASES as budgets grow — bigger budget, proportionally less data per parameter. This shaped the GPT-3 era: 175B parameters on ~300B tokens is ~2 tokens/param. Chinchilla later argued these fits were systematically off, and that the field had been training models far larger (and less data-fed) than optimal."
    },
    {
      type: "single",
      question: "The Chinchilla paper re-estimated compute-optimal scaling with three methods. What did they converge on?",
      options: [
        "$a \\approx b \\approx 0.50$ — grow parameters and data in equal proportion",
        "$a \\approx 0.73$, $b \\approx 0.27$ — confirming Kaplan's parameter-heavy allocation",
        "$a \\approx 0.10$, $b \\approx 0.90$ — models barely grow while the data budget explodes",
        "The three methods disagreed so wildly that no allocation rule could be drawn"
      ],
      correct: [0],
      explanation: "Method 1 (envelope): 0.50/0.50; method 2 (IsoFLOP): 0.49/0.51; method 3 (parametric): 0.46/0.54 — versus Kaplan's 0.73/0.27. Equal exponents mean $N$ and $D$ scale together: compute-optimal training keeps a roughly constant token-to-parameter ratio ($\\approx 20$ tokens/param in their setup). Besiroglu et al. (2024) later re-fit method 3 from recovered data and brought it in line with methods 1-2. The practical consequence was immediate: Chinchilla-70B, trained on 4x the data of Gopher-280B at the same compute, beat it."
    },
    {
      type: "single",
      question: "This is Chinchilla's Figure 3. Describe method 2 (IsoFLOP profiles) and what the panels show.",
      image: "data/figures/lec9-isoflop.png",
      imageCredit: "Figure from Hoffmann et al. (2022), via CS336 Lecture 9",
      options: [
        "Each budget gives a loss valley over model size; the minima trace the optimal $N$, $D$ power laws",
        "Each curve shows one model retrained repeatedly on the same tokens until its loss converges",
        "The flat valley bottoms show loss is essentially independent of model size at fixed compute",
        "The right panels extrapolate each budget's final loss out to effectively infinite compute"
      ],
      correct: [0],
      explanation: "Left: each color is one FLOP budget ($6 \\times 10^{18}$ to $3 \\times 10^{21}$); loss versus model size forms a clear parabola — too small underfits capacity, too big starves each parameter of tokens. Center/right: regressing the valley minima gives $N_{\\text{opt}}(C)$ and $D_{\\text{opt}}(C)$ as clean power laws (green lines mark the Gopher-budget projections). Simple, robust, and now the industry default (Llama 3, DeepSeek, Hunyuan all run IsoFLOPs)."
    },
    {
      type: "single",
      question: "This plot overlays the three Chinchilla fits, Kaplan's law, and famous models. What story does it tell?",
      image: "data/figures/lec9-kaplan-vs-chin.png",
      imageCredit: "Figure from Hoffmann et al. (2022), via CS336 Lecture 9",
      options: [
        "The three fits agree; Kaplan's line is steeper; flagship models sit far above the optimal line",
        "All four fits coincide exactly, and the famous models sit right on the shared optimal line",
        "The flagship models sit far below every fitted line, meaning they were built much too small",
        "Approaches 1-3 disagree with each other so strongly that no conclusion can be drawn"
      ],
      correct: [0],
      explanation: "Approaches 1-3 (solid lines) nearly coincide; Kaplan's dashed line diverges upward — and the era's flagship models (GPT-3 175B, Gopher 280B, MT-NLG 530B) cluster along KAPLAN's line, not the optimal one. Read as a report card: following the 2020 law, the field built models several times larger than compute-optimal. Chinchilla-70B, placed ON the corrected line, validated the correction by outperforming Gopher."
    },
    {
      type: "single",
      question: "Kaplan's and Chinchilla's compute-optimal exponents disagreed sharply. What does the lecture flag as the main methodological difference behind the gap?",
      options: [
        "Accounting for learning-rate schedules — cosine cycles longer than the run inflate loss",
        "Chinchilla trained on a strictly larger and cleaner dataset than Kaplan's WebText2",
        "Kaplan fitted downstream benchmark accuracy while Chinchilla fitted held-out test loss",
        "Chinchilla ran SGD with momentum while all of Kaplan's runs used the Adam optimizer"
      ],
      correct: [0],
      explanation: "Kaplan reused intermediate checkpoints from long runs, so short 'runs' were really trained with a cosine cycle far longer than their actual duration. Chinchilla's ablation shows the cost: stretch the cosine cycle to 1.1-5x the number of steps and the learning rate stays too high, leaving both training and C4 loss visibly worse than a schedule matched to the run length. That bias made short/small runs look worse than they are, tilting Kaplan's fit toward oversized, under-trained models."
    },
    {
      type: "single",
      question: "Why is Chinchilla's 'compute-optimal' recipe often NOT what a deployed model should follow?",
      options: [
        "It optimizes training compute only, but lifetime cost is dominated by inference",
        "Because compute-optimal training makes large models overfit their training data",
        "Because Chinchilla's analysis applies only to models under 1B parameters",
        "Because over-training reduces final accuracy even though it is cheaper per epoch"
      ],
      correct: [0],
      explanation: "Chinchilla answers 'best loss for a fixed TRAINING budget'. But a popular model spends most of its lifetime compute on inference, where a smaller model of equal quality is cheaper forever. Hence the industry's steady march past $\\approx 20$ tokens/param: LLaMA-65B 22, Llama 2 70B 29, Mistral 7B 110, Llama 3 70B 215. The more usage you expect, the more it pays to over-train — a bridge to the inference economics of Lecture 10."
    },
    {
      type: "single",
      question: "The lecture distills a three-step 'scaling-law based design procedure'. What is it?",
      options: [
        "Train a few smaller models, fit a scaling law, and choose settings from its prediction",
        "Train the largest affordable model first, distill it, then fit laws to the students",
        "Fit laws to results published by other groups only, never running your own experiments",
        "Grid-search every hyperparameter at full scale, relying on early stopping to cap cost"
      ],
      correct: [0],
      explanation: "(1) Run a ladder of small models varying the choice under study; (2) fit the scaling trend (e.g., an Adam-vs-SGD pair of curves); (3) pick the option/value the trend favors at your target scale. It is the procedure behind every case study in this lecture and the recipes of Lecture 11 — the disciplined alternative to cargo-culting hyperparameters from other people's models."
    },
    {
      type: "multi",
      question: "Which of these can be predicted from small-scale runs BEFORE training the big model, per this lecture? (Select all that apply)",
      options: [
        "The effect of optimizer choice on final loss",
        "The effect of model depth and architecture choices",
        "The compute-optimal split between model size and data",
        "Exact downstream benchmark rankings at every scale"
      ],
      correct: [0, 1, 2],
      explanation: "The 'surprising takeaway': optimizer, depth, and architecture effects on big LMs are predictable before training — as are $(N, D)$ allocations via joint laws and IsoFLOPs. The exception is downstream behavior: Tay et al. showed task rankings can reorder across scales, so downstream claims need their own validation rather than free-riding on loss curves."
    }
  ]
};

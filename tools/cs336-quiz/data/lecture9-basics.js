// CS336 Lecture 9 — Scaling laws: basics
// 50 questions covering: history of scaling laws, data scaling and its theory,
// model-engineering scaling (architecture/depth/optimizer), critical batch size,
// Kaplan vs Chinchilla compute-optimal analysis, and applications. 8 figures.
window.QUIZ_DATA = window.QUIZ_DATA || {};
window.QUIZ_DATA["lecture9-basics"] = {
  title: "Lecture 9 — Scaling Laws I: Basics",
  questions: [
    // ---------- Motivation & history ----------
    {
      type: "single",
      question: "The lecture opens with a scenario: you have 10,000 B200s for one month to build a good open LM. Which decision are scaling laws meant to answer?",
      options: [
        "Which model to train — its size, shape, and hyperparameters",
        "How to configure the distributed training framework efficiently",
        "How to collect and clean the pretraining dataset",
        "How to serve the finished model to users cheaply"
      ],
      correct: [0],
      explanation: "Infrastructure (Assignment 2) and data (Assignment 4) are covered elsewhere; the open question is 'run a big model — but WHICH one??'. You get essentially one shot at the big run, so architecture, size, and hyperparameters must be decided beforehand. Scaling laws — simple predictive rules fitted on small models — are the course's answer to making those decisions without burning the budget on trial and error."
    },
    {
      type: "single",
      question: "How does the scaling-law paradigm differ from the older approach to building large models?",
      options: [
        "Tune on small models and extrapolate, instead of tuning hyperparameters directly on big models",
        "Train a single enormous model once, instead of training many small ones repeatedly",
        "Replace empirical experiments entirely with theoretical generalization bounds",
        "Search architectures with reinforcement learning instead of by hand"
      ],
      correct: [0],
      explanation: "The lecture frames it as 'old and unpleasant: tune hyperparameters on big models' versus 'new (over?) optimism: tune on small models, extrapolate to large ones'. The parenthetical '(over?)' is deliberate — the whole lecture examines when this extrapolation works (remarkably often) and when it quietly fails (downstream tasks, blindly applied recipes)."
    },
    {
      type: "single",
      question: "The earliest data scaling-law paper (Cortes et al., 1993) modeled learning curves as power laws. What practical problem was it solving?",
      options: [
        "Predicting a classifier's final accuracy without training on the full dataset",
        "Choosing the learning rate schedule for neural networks",
        "Compressing datasets to fit into the memory of early workstations",
        "Proving that neural networks cannot overfit large datasets"
      ],
      correct: [0],
      explanation: "Cortes, Jackel, Solla, Vapnik & Denker fit ε_test = a + b/l^α on small training subsets to predict a classifier's asymptotic error — so that compute could go to the most promising candidates instead of fully training every model. Thirty years later this is exactly the modern playbook: fit on cheap runs, predict the expensive one. Scaling laws are older than they look."
    },
    {
      type: "single",
      question: "Banko & Brill (2001) trained several learners on up to a billion words for confusion-set disambiguation. What was their provocative conclusion?",
      options: [
        "None of the learners was close to plateauing, so corpus growth might beat algorithm development",
        "All learners plateaued quickly, so algorithmic innovation is the only path forward",
        "Simple learners beat complex ones at every data size, so complexity never pays",
        "Accuracy is independent of training data size beyond one million words"
      ],
      correct: [0],
      explanation: "Their accuracy curves kept climbing log-linearly with corpus size, with no asymptote in sight — suggesting the field should 'reconsider the trade-off between spending time and money on algorithm development versus spending it on corpus development'. An early data-centric manifesto: two decades before LLMs, data scaling was already outrunning clever algorithms."
    },
    {
      type: "single",
      question: "Why is Hestness et al. (2017) considered a landmark in the history of scaling laws?",
      options: [
        "It was the earliest large-scale NEURAL scaling study, spanning MT, LM, and speech",
        "It derived the scaling laws mathematically from first principles",
        "It was the first paper ever to train a model with more than one billion parameters",
        "It introduced the transformer architecture that later scaling work relied on"
      ],
      correct: [0],
      explanation: "Hestness et al. systematically measured learning curves across machine translation, language modeling, and speech, found ε(m) ≈ αm^β power laws everywhere, and hypothesized the now-standard three-region curve shape. It predates Kaplan by three years and the transformer scaling era entirely — the empirical regularity came before the models that made it famous."
    },
    {
      type: "multi",
      question: "The lecture calls Hestness et al. (2017) 'very ahead of its time'. Which later ideas did it anticipate? (Select all that apply)",
      options: [
        "Emergence-like accuracy cliffs, where models are no better than guessing until data suffices",
        "Using learning curves to project the compute needed for a target accuracy",
        "Trading accuracy for speed (e.g., quantization) and recovering it by scaling up",
        "The Chinchilla rule of training with about 20 tokens for every model parameter"
      ],
      correct: [0, 1, 2],
      explanation: "Hestness discussed accuracy cliffs in the small-data region (an early framing of 'emergence'), projecting compute requirements from predictable curves ('scaling by compute'), and the performance-accuracy trade-off where efficiency techniques give up accuracy that larger models can win back. The 20:1 token ratio came from Chinchilla in 2022 — that one it did not anticipate."
    },
    {
      type: "single",
      question: "Classical learning theory offers bounds like ε(ĥ) ≤ min ε(h) + 2√((1/m)·log(2k/δ)). Why don't these serve as scaling laws for practice?",
      options: [
        "They are upper bounds on error, not predictions of realized loss values",
        "They only apply to neural networks, not to classical models",
        "They require infinitely many samples to evaluate even approximately",
        "They predict error should increase with more data"
      ],
      correct: [0],
      explanation: "Theory gives worst-case guarantees: the true error is AT MOST this much. Practice needs the actual number — 'what loss will my 10B model hit on 500B tokens?'. Scaling laws fill that gap: empirical fits that predict realized loss, often startlingly well. The lecture's stance: theory motivates the FORM (polynomial decay), experiments supply the constants."
    },

    // ---------- Data scaling & theory ----------
    {
      type: "single",
      question: "What is a 'data scaling law', and what functional form does it take for language models?",
      options: [
        "A formula mapping dataset size to loss — empirically a power law, i.e., linear on a log-log plot",
        "A formula mapping dataset size to loss — empirically exponential decay toward zero error",
        "A rule for how to shuffle and deduplicate training data at scale",
        "A schedule for growing the batch size as the dataset grows"
      ],
      correct: [0],
      explanation: "Kaplan's LM fit is the canonical example: L(D) = (D/5.4·10¹³)^−0.095 — a straight line on log-log axes ('scale-free' behavior, no special dataset size where the rules change). Exponential decay would be far faster and would hit the irreducible-error floor almost immediately; the observed decay is polynomial and slow."
    },
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
        "Loss reaches zero once any single resource is scaled far enough",
        "The three resources are interchangeable one-for-one at any scale"
      ],
      correct: [0],
      explanation: "L(C) = (C_min/2.3·10⁸)^−0.050, L(D) = (D/5.4·10¹³)^−0.095, L(N) = (N/8.8·10¹³)^−0.076: clean straight lines on log-log axes across ~7 orders of magnitude. Note how SMALL the exponents are — each constant decrement of loss costs a multiplicative increase in resources. That is why frontier progress is so expensive, and why the laws even extend to nonstandard settings (emergent-task accuracy fits sigmoids in log-FLOPs)."
    },
    {
      type: "single",
      question: "The lecture's toy example: estimate the mean of n Gaussian samples. What scaling law does this produce, and why is it instructive?",
      options: [
        "E[(μ̂−μ)²] = σ²/n — a log-log slope of −1, showing polynomial error decay is natural",
        "Error decays as e^−n, showing why neural scaling should be exponential",
        "Error is independent of n, showing scaling laws need neural networks",
        "Error decays as 1/log(n), matching observed LM slopes"
      ],
      correct: [0],
      explanation: "log(Error) = −log n + 2log σ: a perfect straight line with slope −1. The broader point: essentially every classical estimation problem has polynomially decaying error (any 1/n^α rate IS a scaling law), so log-log linearity in neural training is less mysterious than it first appears. The mystery lies elsewhere — in the VALUE of the exponent."
    },
    {
      type: "single",
      question: "Classical statistics predicts error slopes near −1 (rates like 1/n). What do neural scaling laws actually show, and what is the puzzle?",
      options: [
        "Much shallower slopes (−0.05 to −0.13), far less data-efficient than the classical rate",
        "Much steeper slopes (−2 to −3) — neural nets extract more per sample than theory allows",
        "Exactly −1, confirming the classical analysis directly",
        "Positive slopes — more data slightly hurts loss"
      ],
      correct: [0],
      explanation: "MT fits show ε(m) ≈ 3.87m^−0.13, LM 0.095, speech similar — an order of magnitude shallower than 1/n. If simple mean estimation gets slope −1, why does a giant transformer get −0.1? The lecture presents this as an intriguing mystery, with the nonparametric/intrinsic-dimension story as the leading (but not airtight) explanation."
    },
    {
      type: "single",
      question: "The nonparametric detour: estimating a smooth function on d-dimensional inputs by partitioning space into boxes gives what error rate?",
      options: [
        "Error ≈ n^(−1/d) — a slope that flattens as dimension grows",
        "Error ≈ n^(−d) — steeper decay in higher dimension",
        "Error ≈ 1/n regardless of the dimension d",
        "Error ≈ constant: nonparametric methods cannot learn at all"
      ],
      correct: [0],
      explanation: "Cutting d-dimensional space into bins and averaging within each bin spreads n samples over exponentially many cells: error decays as n^(−1/d) (slope −1/d). Takeaway: flexible 'nonparametric' learners — and neural nets acting like them — have DIMENSION-DEPENDENT scaling laws. A slope of −0.1 then reads naturally as 'the data behaves ~10-dimensional'."
    },
    {
      type: "single",
      question: "The intrinsic-dimensionality theory (Bahri et al., 2021) connects scaling exponents to data geometry. What is the claim, and its status?",
      options: [
        "The slope α relates to the data manifold's intrinsic dimension, though estimates are shaky",
        "The slope is fully determined by parameter count, with data geometry irrelevant — this is proven",
        "The slope equals the number of attention heads divided by depth — confirmed on CIFAR",
        "Scaling exponents are pure measurement noise with no structure — widely accepted"
      ],
      correct: [0],
      explanation: "Combining 'error decays polynomially' with the nonparametric rate suggests α is tied to the INTRINSIC dimension of the data manifold; plots of 4/α against measured dimension (teacher-student, CIFAR, MNIST...) line up encouragingly. But the lecture is careful: estimators of intrinsic dimension are sketchy, 'and this is not airtight' — the best available story, not a law."
    },
    {
      type: "single",
      question: "Hashimoto (2021) studied scaling under different data COMPOSITIONS (mixtures of sources). What is the headline finding?",
      options: [
        "Composition shifts the curve's offset while the slope stays essentially unchanged",
        "Composition changes the slope drastically while offsets stay fixed",
        "Composition only matters for datasets under a million examples",
        "Mixtures always interpolate linearly between their sources' curves"
      ],
      correct: [0],
      explanation: "Excess-error curves for different mixture proportions run parallel on log-log axes: same slope, different intercepts. That is useful structure — the mixture choice becomes a one-number 'offset' story (with a U-shaped optimum over proportions), and these distribution-shift scaling laws quantify the value of collecting diverse data."
    },
    {
      type: "single",
      question: "For choosing pretraining data mixtures in practice, what does the DataDecide-style empirical evidence suggest?",
      options: [
        "Picking the best dataset at small scale transfers well (~80% correct at 1B from 150M runs)",
        "Small-scale comparisons are useless: dataset rankings reverse completely at scale",
        "Elaborate fitted mixing laws are strictly required to beat random mixtures",
        "The best mixture is always uniform sampling over all available sources"
      ],
      correct: [0],
      explanation: "There are sophisticated proposals (Data Mixing Laws fit performance as a function of mixture proportions), but the sobering empirical result is that the naive baseline — evaluate candidate datasets on a small proxy model and take the winner — predicts the 1B-scale winner ~80% of the time. Fancy mixture optimization has to beat THAT cheap baseline, which the lecture notes is genuinely hard."
    },

    // ---------- Data: repetition, filtering, limits ----------
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
      explanation: "Their effective-data formula D' = U_D + U_D·R*_D(1−e^(−R_D/R*_D)) captures it: early repetitions count almost fully, then exponentially saturate. Practical upshots: modest multi-epoch training is fine when unique data is scarce, and the compute-optimal response to data constraints is a somewhat larger model trained on more epochs — but no amount of repetition substitutes for genuinely new tokens at ~40+ epochs."
    },
    {
      type: "single",
      question: "The 'pre-training under infinite compute' study (Kim et al.) sounds two cautions about scaling laws. Which pair?",
      options: [
        "Blindly applied laws can break (loss turns U-shaped in epochs), and laws are lower bounds",
        "Scaling laws are exact and immutable, so extrapolating them is always safe at any scale",
        "Only compute matters, and data curves are irrelevant once models are large",
        "Scaling laws apply to vision but reliably fail for language"
      ],
      correct: [0],
      explanation: "Two lessons: (1) naively cranking epochs at fixed data eventually INCREASES loss — the fitted law breaks outside its regime; (2) a scaling law describes one recipe's frontier, not a law of nature — with proper regularization or ensembling the asymptote improves (better fits like 1.03/D^0.23+1.96 vs the standard recipe). 'You can always potentially do better' than the curve you fitted."
    },
    {
      type: "single",
      question: "'Data Curation cannot be Compute Agnostic' (Goyal et al.): how should data FILTERING strategy change with the compute budget?",
      options: [
        "Small budgets favor aggressive filtering; large budgets favor keeping more, lower-quality data",
        "Filtering strategy is independent of compute: always keep only the very best data",
        "Large budgets should filter more aggressively since they can afford to discard data",
        "Filtering only matters below one million training samples"
      ],
      correct: [0],
      explanation: "Because repeated data loses value, the quality-quantity trade-off (QQT) shifts with scale: at small compute you train ~1 epoch on the highest-quality pool (aggressive filtering wins); at large compute that pool would be repeated many times, so admitting broader, lower-quality pools (E → E+D → E+D+C) wins. The best data pool CHANGES with total compute — curation must be scale-aware."
    },

    // ---------- Model engineering scaling ----------
    {
      type: "single",
      question: "Beyond data, the lecture applies scaling laws to 'model engineering'. What kinds of questions does this cover?",
      options: [
        "Architecture and optimizer choices, plus resource allocation between model and data",
        "Only the choice of the tokenizer's vocabulary size for a fixed model architecture",
        "Only inference-time serving settings such as temperature and beam width",
        "Legal and licensing questions around training data usage"
      ],
      correct: [0],
      explanation: "The motivating questions: how do we efficiently design huge LMs (which architecture? which optimizer?) and how do we allocate limited resources (train longer vs train bigger; collect more data vs buy more GPUs)? The claim — surprising in 2020, standard now — is that fitting scaling curves on small models answers all of these BEFORE committing to the big run."
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
        "Every alternative scaled better than the vanilla transformer",
        "All architectures scaled identically, so architecture choice is irrelevant",
        "Only convolution-based models kept improving with compute"
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
        "SGD failed to converge at every data size tested",
        "The comparison was impossible because optimizers cannot be scaled"
      ],
      correct: [0],
      explanation: "ε(m) = 5.37m^−0.094 for SGD vs 5.25m^−0.095 for Adam: same exponent to two decimals, slightly different constant. Optimizer choice (like many engineering choices) tends to move the OFFSET, not the slope — real but bounded gains. Contrast with architecture (LSTM vs transformer), where the trend itself differed. Diagnosing 'slope change vs offset change' is a core scaling-law skill."
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
      explanation: "The 1-layer curve is clearly worse; 2 layers closes most of the gap; 3/6/>6 layers bunch together (diminishing returns below ~10⁷ parameters). Combined with the aspect-ratio result, the message is liberating: once past minimal depth, transformer performance depends mainly on the parameter BUDGET, not on finely tuned shape."
    },
    {
      type: "single",
      question: "How sensitive is transformer loss to model SHAPE (aspect ratio, feed-forward ratio, head dimension) at a fixed parameter count, per Kaplan?",
      options: [
        "Very mildly — aspect ratio can vary ~40× with only a few percent change in loss",
        "Extremely sensitive: a 2× change in aspect ratio can roughly double the loss",
        "Shape matters more than total parameter count in every experiment",
        "Only the number of attention heads matters; other shape choices are free"
      ],
      correct: [0],
      explanation: "Kaplan's Figure 5: with N fixed, wide ranges of d_ff/d_model, d_model/n_layer, and head dimension move loss by only a few percent — an (n_layer, d_model) of (6, 4288) lands within 3% of the (48, 1600) GPT-2 shape. Practical consequence: spend your effort on SIZE and data, not on exquisitely tuning shape; small mistakes there are cheap to buy back with compute."
    },
    {
      type: "single",
      question: "These two plots show depth curves against parameters WITH embeddings (left) and WITHOUT (right). What do they demonstrate?",
      image: "data/figures/lec9-embedding.png",
      imageCredit: "Figure from Kaplan et al. (2020), via CS336 Lecture 9",
      options: [
        "Counting non-embedding parameters collapses the depth curves onto one trend",
        "Embedding parameters are the only ones that matter for scaling",
        "The two counting conventions give identical curves at every size",
        "Embeddings only matter for models above a billion parameters"
      ],
      correct: [0],
      explanation: "With embeddings included (left), each depth traces a different curve — the 'law' looks messy. Excluding embedding parameters (right), the curves collapse onto a single clean power law. Lesson: not all parameters are equal, and WHAT YOU COUNT changes the law you fit. This seemingly bookkeeping-level choice returns with force in the Kaplan-vs-Chinchilla dispute and in MoE scaling."
    },
    {
      type: "single",
      question: "For Mixture-of-Experts models, Abnar et al. (2025) fit scaling laws over sparsity. At a fixed compute budget, what happens as sparsity increases?",
      options: [
        "Optimal TOTAL parameters rise while optimal ACTIVE parameters fall",
        "Both total and active parameters must shrink to stay compute-optimal",
        "Sparsity has no effect on optimal parameter allocation at fixed FLOPs",
        "Only dense models (zero sparsity) can be compute-optimal"
      ],
      correct: [0],
      explanation: "IsoFLOP surfaces over (parameters, sparsity) show the optimum drifting to higher total parameter counts and lower active parameter counts as sparsity S grows — 'value per parameter' differs between total capacity and per-token compute. This is the modern continuation of the embedding lesson: parameter counts are not interchangeable, and scaling laws must respect which kind you mean. (Kimi K2's sparsity law, in Lecture 11, builds exactly here.)"
    },

    // ---------- Critical batch size & LR ----------
    {
      type: "single",
      question: "McCandlish et al.'s 'critical batch size' separates two training regimes. What are they?",
      options: [
        "Below B_crit, bigger batches give near-perfect speedups; above it, returns diminish sharply",
        "Below B_crit training diverges, while above it training is always numerically stable",
        "Below B_crit memory is the bottleneck; above it compute is",
        "Batch size has no effect on training speed in either regime"
      ],
      correct: [0],
      explanation: "Small batches give noisy gradient estimates, so doubling the batch nearly halves the steps needed ('perfect scaling'). Once the batch is large enough that gradients are already accurate — around the critical batch size — extra samples per step stop buying progress ('ineffective scaling'). B_crit is the knee of that curve, and it's claimed to track the gradient noise scale: the ratio of gradient covariance to squared gradient norm."
    },
    {
      type: "single",
      question: "How is the critical batch size operationally defined from training curves?",
      options: [
        "Sweep batches at a target loss, fit the S-E trade-off curve, and set B_crit = E_min/S_min",
        "It is simply the largest batch size that fits in GPU memory at full precision",
        "It is the batch size at which the loss first stops decreasing entirely",
        "It is fixed at 0.1% of the dataset size by convention"
      ],
      correct: [0],
      explanation: "For a target loss, each batch size needs some number of steps S and examples E; the trade-off curve fits S/S_min − 1 = (E/E_min − 1)^(−1), with asymptotes S_min (steps at infinite batch) and E_min (examples at tiny batch). B_crit = E_min/S_min balances the two — training there costs about 2× the minimum steps AND 2× the minimum examples, a sensible compromise between time and compute."
    },
    {
      type: "single",
      question: "This figure plots empirical critical batch size against training loss. What is the key empirical finding?",
      image: "data/figures/lec9-critical-batch.png",
      imageCredit: "Figure from McCandlish et al. (2018), via CS336 Lecture 9",
      options: [
        "B_crit follows a power law in the LOSS, largely independent of model size",
        "B_crit is a fixed constant (around 32k tokens) throughout the whole of training",
        "B_crit shrinks as loss falls, so batches should decay like the learning rate",
        "B_crit depends only on model size, with loss playing no role"
      ],
      correct: [0],
      explanation: "The 3M and 85M curves lie on top of each other: what predicts the useful batch is not the model but HOW GOOD it currently is — B_crit ∝ L^−4.8 grows steeply as loss drops. Practical readings: ramp the batch during training, and expect stronger models/harder targets to support much larger batches. This 'batch as a function of loss' view is the OpenAI school that MiniCPM later adopts (Lecture 11)."
    },
    {
      type: "single",
      question: "Why do larger batches stop helping beyond the critical point? The gradient-noise intuition:",
      options: [
        "Once gradient noise is small relative to the true gradient, extra averaging adds little",
        "GPUs cannot compute matrix products above certain batch dimensions",
        "Larger batches change the effective objective, raising the loss floor permanently",
        "The learning rate must be zero for large batches, freezing training"
      ],
      correct: [0],
      explanation: "A batch gradient is an average of per-example gradients: noise variance falls as 1/B. While noise dominates the signal, bigger B ≈ proportionally fewer steps. Once B exceeds the noise scale, the estimate is already close to the true gradient — further averaging polishes something that no longer limits progress. Hence 'perfect scaling' below the noise scale and a hard knee above it."
    },
    {
      type: "single",
      question: "If you scale a transformer's width naively (standard parametrization), what happens to the optimal learning rate — and what is the lecture's prescription?",
      options: [
        "It drifts with scale, motivating scaling-aware initialization and LR rules (muP)",
        "It stays exactly fixed, so one tuned LR works forever",
        "It grows in proportion to width, so larger models need proportionally larger LRs",
        "It becomes irrelevant because large models are insensitive to LR"
      ],
      correct: [0],
      explanation: "The Yang et al. figure shows standard practice's loss-vs-LR minima marching across the axis as width grows 128→8192 ('optimum shifts'), versus a muP-parametrized model whose optimum stays put. The Lecture 9 message is just the phenomenon and the need for scale-aware LR/init choices; Lecture 11 does the muP derivation and its practical limits in depth."
    },
    {
      type: "single",
      question: "Tay et al. (2023) sound a caution about extrapolating from pretraining loss. What did they find?",
      options: [
        "Upstream perplexity scales predictably, but downstream rankings can reorder across scales",
        "Downstream accuracy is always a smooth, deterministic function of the perplexity",
        "Perplexity stops improving with scale even as downstream tasks improve",
        "Downstream tasks are perfectly predictable while perplexity is noisy"
      ],
      correct: [0],
      explanation: "Comparing models by negative log-perplexity versus by SuperGLUE accuracy produces visibly different orderings — architectures that look best upstream are not always best downstream. Scaling laws are cleanest for the pretraining objective itself; treat 'lower loss ⇒ better at everything' as a hypothesis to verify, not a theorem. (Llama 3's two-step compute→NLL→accuracy fits, in Lecture 11, are one response.)"
    },

    // ---------- Joint scaling & Chinchilla ----------
    {
      type: "single",
      question: "'Do we need more data or bigger models?' What tool answers this, and what are the two classic functional forms?",
      options: [
        "Joint laws: Rosenfeld's n^−α + m^−β + C and Kaplan's [m^−α + n^−1]^β",
        "Separate single-variable laws for data and model, which cannot be combined",
        "A lookup table of best (N, D) pairs published by each hardware vendor",
        "The elbow method applied to a single training curve"
      ],
      correct: [0],
      explanation: "Joint laws model loss as a function of BOTH model size m and data size n. Rosenfeld's additive form (independent error sources plus an irreducible constant) and Kaplan's nested form both fit empirical (m, n) loss surfaces surprisingly well — and once you have a fitted surface plus costs, 'more data or bigger model?' becomes a simple optimization."
    },
    {
      type: "single",
      question: "How did Rosenfeld et al. validate that joint data-model laws genuinely EXTRAPOLATE?",
      options: [
        "Fit exponents on small models/data, then accurately predict larger held-out settings",
        "Fit on the largest models and interpolate downward to the small configurations",
        "Show the law fits the training data it was estimated on",
        "Prove convergence theoretically without any held-out configurations"
      ],
      correct: [0],
      explanation: "The green points (small fractions of model/data grid) were used to fit; the red points (larger configurations) were held out and predicted — with errors around a few percent on both vision and language. This is the crucial epistemic test for any scaling law: not goodness-of-fit, but out-of-sample extrapolation toward the regime you actually care about."
    },
    {
      type: "single",
      question: "What did Kaplan et al. (2020) claim about compute-optimal allocation, and what behavior did it encourage?",
      options: [
        "N_opt ∝ C^0.73 — compute mostly into parameters, encouraging huge under-trained models",
        "N_opt ∝ C^0.27, D_opt ∝ C^0.73 — extra compute should go mostly into data",
        "N and D should always grow in exact proportion, 50/50",
        "Compute should go entirely into longer training of a fixed-size model"
      ],
      correct: [0],
      explanation: "Kaplan's exponents imply tokens-per-parameter DECREASES as budgets grow — bigger budget, proportionally less data per parameter. This shaped the GPT-3 era: 175B parameters on ~300B tokens is ~2 tokens/param. Chinchilla later argued these fits were systematically off, and that the field had been training models far larger (and less data-fed) than optimal."
    },
    {
      type: "single",
      question: "The Chinchilla paper re-estimated compute-optimal scaling with three methods. What did they converge on?",
      options: [
        "a ≈ b ≈ 0.50 — grow parameters and data in equal proportion (~20 tokens per parameter)",
        "a ≈ 0.73, essentially confirming Kaplan's parameter-heavy allocation was right",
        "a ≈ 0.10 — models should barely grow while data explodes",
        "The three methods disagreed wildly, settling nothing"
      ],
      correct: [0],
      explanation: "Method 1 (envelope): 0.50/0.50; method 2 (IsoFLOP): 0.49/0.51; method 3 (parametric): 0.46/0.54 — versus Kaplan's 0.73/0.27. Equal exponents mean N and D scale together: compute-optimal training keeps a roughly constant token-to-parameter ratio (~20:1 in their setup). The practical consequence was immediate: Chinchilla-70B, trained on 4× the data of Gopher-280B at the same compute, beat it."
    },
    {
      type: "single",
      question: "Chinchilla method 1 ('minimum over training curves') works how?",
      options: [
        "Take the lower envelope of loss-vs-FLOPs over many runs and fit power laws to it",
        "Train one model and read optima directly off its single curve",
        "Average together all of the training curves and fit the mean trajectory instead",
        "Pick the largest affordable model and measure its final loss only"
      ],
      correct: [0],
      explanation: "Each run (70M–10B, four cosine lengths) traces loss against FLOPs; the pointwise minimum over all curves is the empirical efficiency frontier — at each budget, the best (N, D) seen. Fitting N_opt(C) and D_opt(C) on envelope points gives the 0.50/0.50 exponents, projected forward to the Gopher budget (5.76·10²³ FLOPs). Lecture 11's WSD trick exists precisely to make gathering these curves cheap."
    },
    {
      type: "single",
      question: "This is Chinchilla's Figure 3. Describe method 2 (IsoFLOP profiles) and what the panels show.",
      image: "data/figures/lec9-isoflop.png",
      imageCredit: "Figure from Hoffmann et al. (2022), via CS336 Lecture 9",
      options: [
        "Each budget traces a loss valley over model size; the minima give power laws for N_opt, D_opt",
        "Each curve shows one model retrained repeatedly on the same data until convergence",
        "The valleys show that loss is independent of model size at fixed compute",
        "The right panels extrapolate loss to infinite compute budgets"
      ],
      correct: [0],
      explanation: "Left: each color is one FLOP budget (6e18→3e21); loss versus model size forms a clear parabola — too small underfits capacity, too big starves each parameter of tokens. Center/right: regressing the valley minima gives N_opt(C) and D_opt(C) as clean power laws (green lines mark the Gopher-budget projections). Simple, robust, and now the industry default (Llama 3, DeepSeek, Hunyuan all run IsoFLOPs)."
    },
    {
      type: "single",
      question: "Chinchilla method 3 fits a parametric loss surface L̂(N, D). What extra structure does this method give beyond methods 1-2?",
      options: [
        "A full loss model with IsoLoss contours and an efficient frontier over budgets",
        "It removes the need for any training runs at all",
        "It predicts downstream benchmark accuracy directly rather than the pretraining loss",
        "It applies only to vision models, unlike the other two methods"
      ],
      correct: [0],
      explanation: "Fitting L̂(N,D) = E + A/N^α + B/D^β to the whole run grid produces the loss LANDSCAPE, not just frontier points: IsoLoss contours, IsoFLOP slices, and the efficient frontier line in log-log space. It projected ~40B as optimal at Gopher's 5.76e23 budget — versus Gopher's actual 280B, a stark statement of how oversized that generation was. (Method 3's original fit also had subtle errors — see the Besiroglu question.)"
    },
    {
      type: "single",
      question: "This plot overlays the three Chinchilla fits, Kaplan's law, and famous models. What story does it tell?",
      image: "data/figures/lec9-kaplan-vs-chin.png",
      imageCredit: "Figure from Hoffmann et al. (2022), via CS336 Lecture 9",
      options: [
        "The three methods agree; Kaplan's is steeper; the flagship models sit far above the optimal line",
        "All fits coincide exactly, and the famous models sit right on the optimal line",
        "The models sit far below the line, meaning they were too small",
        "Approaches 1-3 disagree with each other so strongly that no conclusion can be drawn"
      ],
      correct: [0],
      explanation: "Approaches 1-3 (solid lines) nearly coincide; Kaplan's dashed line diverges upward — and the era's flagship models (GPT-3 175B, Gopher 280B, MT-NLG 530B) cluster along KAPLAN's line, not the optimal one. Read as a report card: following the 2020 law, the field built models several times larger than compute-optimal. Chinchilla-70B, placed ON the corrected line, validated the correction by outperforming Gopher."
    },
    {
      type: "single",
      question: "Wortsman et al. ('Resolving Discrepancies...') reproduced Kaplan's setup to find WHY the exponents disagreed. Which factors did they identify?",
      options: [
        "Not counting last-layer FLOPs, plus too-long warmup at small compute budgets",
        "A units bug that mislabeled FLOPs as MACs throughout the paper",
        "Kaplan's datasets were too small for any law to be fitted",
        "Different GPU hardware between the two papers changed the math"
      ],
      correct: [0],
      explanation: "Step by step: reproduce Kaplan (a≈0.84) → count last-layer/head FLOPs properly (0.71) → correct the disproportionately long warmup at small compute (0.60) → tune optimizer/batch per scale (≈0.50, matching Chinchilla; cosine decay itself mattered less once tuned). Moral of the story: scaling exponents are sensitive to accounting and hyperparameter details — 'details matter' is the lecture's refrain."
    },
    {
      type: "single",
      question: "Pearce & Song ('Reconciling Kaplan and Chinchilla') offer a complementary explanation. What is it?",
      options: [
        "Total parameters yield ≈0.51 (Chinchilla); non-embedding yield ≈0.78 (Kaplan)",
        "The two papers used incompatible definitions of the token count when computing D",
        "Chinchilla secretly used a different architecture family than Kaplan",
        "Random seeds alone account for the difference in exponents"
      ],
      correct: [0],
      explanation: "Starting from Chinchilla's own fitted loss model, they regenerate Kaplan-style small-scale curves: computing the compute-optimal frontier in TOTAL parameters yields 0.51 (Chinchilla-like), while the same data analyzed in NON-embedding parameters (Kaplan's convention) yields a local slope of 0.78 (Kaplan-like). At small scales embeddings are a large parameter fraction, so the bookkeeping choice — plus small nonlinearities — reproduces the historical disagreement."
    },
    {
      type: "single",
      question: "Besiroglu et al. did 'data forensics' on the Chinchilla paper itself. What did they find about method 3?",
      options: [
        "The original fit was flawed; refitting recovered data agrees with methods 1 and 2",
        "Method 3 was correct and methods 1-2 were both wrong",
        "The paper's raw data could never be recovered, leaving it unresolved",
        "All three methods were invalidated, reinstating Kaplan's exponents"
      ],
      correct: [0],
      explanation: "They reconstructed the underlying data (from the paper's figures), re-ran the parametric fit, and found the published method-3 estimate suffered from fitting errors — the corrected fit agrees with methods 1-2 (residuals centered on zero, and the implied optimal token ratio aligns with the ~20/param rule of thumb). Wholesome meta-lesson: even the paper that corrected Kaplan needed correcting; scaling analyses deserve the same scrutiny as models."
    },
    {
      type: "single",
      question: "Why is Chinchilla's 'compute-optimal' recipe often NOT what a deployed model should follow?",
      options: [
        "It optimizes training compute only, but lifetime cost is dominated by inference",
        "Because compute-optimal training makes large models overfit their training data",
        "Because Chinchilla's analysis applies only to models under 1B parameters",
        "Because over-training reduces accuracy but is cheaper per epoch"
      ],
      correct: [0],
      explanation: "Chinchilla answers 'best loss for a fixed TRAINING budget'. But a popular model spends most of its lifetime compute on inference, where a smaller model of equal quality is cheaper forever. Hence the industry's steady march past 20 tokens/param: LLaMA-65B 22, Llama 2 70B 29, Mistral 7B 110, Llama 3 70B 215. The more usage you expect, the more it pays to over-train — a bridge to the inference economics of Lecture 10."
    },
    {
      type: "single",
      question: "In practice today, how central has the IsoFLOP method become — and where has it been applied beyond dense LMs?",
      options: [
        "It is the workhorse — used from diffusion-vs-AR comparisons to MoE sparsity surfaces",
        "It has been largely abandoned in favor of purely theoretical derivations of optima",
        "It applies only to text models and fails for other modalities",
        "It requires proprietary data, limiting it to a few large labs"
      ],
      correct: [0],
      explanation: "'IsoFLOPs everywhere': the same fix-compute-vary-allocation template compares generative paradigms (Gulrajani's diffusion vs autoregressive LMs), maps MoE sparsity surfaces (Abnar), and drives the sizing analyses in Llama 3, DeepSeek, and Hunyuan (Lecture 11). Its virtues are operational — each point is an independent short run, the valleys are visually checkable, and no parametric form must be assumed."
    },

    // ---------- Applications & recap ----------
    {
      type: "single",
      question: "The lecture distills a three-step 'scaling-law based design procedure'. What is it?",
      options: [
        "Train a few smaller models, fit a scaling law, and choose settings from its prediction",
        "Train the largest affordable model first, distill it, then fit laws to the students",
        "Fit laws from published papers only; never run your own experiments",
        "Grid-search everything at full scale with early stopping"
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
      explanation: "The 'surprising takeaway': optimizer, depth, and architecture effects on big LMs are predictable before training — as are (N, D) allocations via joint laws and IsoFLOPs. The exception is downstream behavior: Tay et al. showed task rankings can reorder across scales, so downstream claims need their own validation rather than free-riding on loss curves."
    },
    {
      type: "single",
      question: "A rival lab shows their new architecture beating the transformer in a paper — at a single model size. Based on this lecture, what is the right response?",
      options: [
        "Ask for the scaling curve — single-scale wins often vanish or reverse as compute grows",
        "Adopt the architecture immediately, since one fair comparison suffices",
        "Reject it outright, since the transformer can never be beaten",
        "Average the two architectures' weights and use the blend"
      ],
      correct: [0],
      explanation: "This is the lecture's thinking-tool applied: Tay's cross-architecture study showed most claimed improvements scale WORSE than vanilla transformers (slope beats intercept), while the LSTM comparison shows what a genuine trend difference looks like. The scaling curve — a few sizes, same data and tuning care — is the minimum bar of evidence for architecture claims."
    },
    {
      type: "single",
      question: "The closing recap gives three reasons scaling laws matter. Which triple?",
      options: [
        "Data theory, cheaper training decisions, and predicting what can be 'brute forced'",
        "Faster GPUs, better tokenizers, and cheaper storage",
        "Proving generalization bounds, removing hyperparameters, and avoiding data collection",
        "Replacing evaluation, replacing pretraining, and replacing fine-tuning"
      ],
      correct: [0],
      explanation: "Data scaling connects to estimation theory (the mean-estimation story) and guides collection/curation; model scaling turns costly design decisions (architecture, optimizer, sizing) into cheap small-scale experiments; and scaling-as-prediction tells you which capabilities more compute will buy — and which problems won't yield to brute force. Lectures 10-11 then apply this toolkit to inference and to modern training recipes."
    }
  ]
};

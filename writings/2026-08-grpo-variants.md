---
title: "GRPO, Dr. GRPO, DAPO Comparisons"
date: 2026-08-30
tags: [DAPO, GRPO, RL, verl]
summary: Three critic-free RL objectives, one pipeline — what each changes in the token-weighting math, and what actually differs in training.
---

## The three objectives

All critic-free: sample $G$ responses per question, compare within the group. The difference is **how a group-relative reward becomes a per-token gradient weight**. $\rho_{i,t} = \pi_\theta(o_{i,t} \mid \cdot)\,/\,\pi_{\mathrm{old}}(o_{i,t} \mid \cdot)$.

### GRPO

$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}
$$

$$
\mathcal{J} = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
\min\!\big(\rho_{i,t}\hat{A}_i,\ \mathrm{clip}(\rho_{i,t},\,1\pm\varepsilon)\,\hat{A}_i\big)
\;-\; \beta\, D_{\mathrm{KL}}\!\big(\pi_\theta \,\|\, \pi_{\mathrm{ref}}\big)
$$

### Dr. GRPO — delete two biases

$$
\hat{A}_i = r_i - \mathrm{mean}(r_1,\dots,r_G),
\qquad
\mathcal{J} = \frac{1}{G}\sum_{i=1}^{G} \sum_{t=1}^{|o_i|}
\min\!\big(\rho_{i,t}\hat{A}_i,\ \mathrm{clip}(\rho_{i,t},\,1\pm\varepsilon)\,\hat{A}_i\big)
$$

- **$1/|o_i|$ gone** — it gave every rollout equal total weight, but per *token* that suppresses long failures (and reinforces long successes) more weakly than short ones — effectively encouraging long wrong rollouts and short correct ones.
- **$/\,\mathrm{std}$ gone** — near-solved / near-impossible questions have tiny std and got outsized advantages.

> [!info] **Objective hacking.** The reward is never gamed — nothing earns more by being long. The gap sits between the intended objective and the implemented surrogate: under $1/|o_i|$ the optimizer lowers the surrogate by parking failure mass in its least-punished (long) variants, correctness unchanged.

### DAPO — four changes, no KL

$$
\mathcal{J} = \frac{1}{\sum_i |o_i|}\sum_{i=1}^{G} \sum_{t=1}^{|o_i|}
\min\!\big(\rho_{i,t}\hat{A}_i,\ \mathrm{clip}(\rho_{i,t},\,1-\varepsilon_{\mathrm{low}},\,1+\varepsilon_{\mathrm{high}})\,\hat{A}_i\big)
$$

- **Token-level $1/\sum_i|o_i|$** — same length-bias fix as Dr. GRPO ($\hat{A}_i$ stays GRPO's); the different denominator only stabilizes step size across batches.
- **Clip-higher** ($\varepsilon_{\mathrm{high}}=0.28 > \varepsilon_{\mathrm{low}}=0.2$) — a symmetric ceiling chokes rare tokens' growth → entropy collapse; raise only the ceiling.
- **Dynamic sampling** — all-correct / all-wrong groups have $\hat{A}=0$, zero gradient; filter and resample until the batch is live.
- **Overlong shaping** — truncated responses get a soft penalty, not a spurious reward.
- **$\beta = 0$** — no KL, no ref model, one fewer forward pass per step (8.4 s on our A40, per [W260828](post.html?p=2026-08-grpo-a40-scaling)).

## Ablations

**Setup.** Qwen2.5-1.5B-Instruct on MATH (levels 3–5), max response 2,048 tokens — hard enough that accuracy stays mid-range, so groups keep mixing correct and wrong rollouts: the raw material both biases need. Three runs, each one switch apart:

**R1** GRPO  →  **R2** R1 minus $1/\vert o_i\vert$ (token-level aggregation)  →  **R3** R2 minus $/\,\mathrm{std}$ (= Dr. GRPO)

| Exp | Compares | Isolates |
|---|---|---|
| E1 | R1 vs R2 | the $1/\vert o_i\vert$ length bias |
| E2 | R2 vs R3 | the $/\,\mathrm{std}$ difficulty bias |

A higher accuracy curve is not a verdict — it is noisy and confounded. Each experiment must produce direct evidence that its mechanism operated.

### E1 — what does $1/|o_i|$ actually do?

Claim: per-token suppression of a wrong rollout scales as $1/|o_i|$, so long failures are punished gently and survive. Evidence to collect:

1. **Length by correctness over training** — mean response length split into wrong / correct. R1: wrong lengthens, correct shortens; R2: both flat.
2. **Per-token weight within one batch** — among wrong rollouts of the same step, per-token loss magnitude vs length: $\propto 1/|o|$ under R1, flat under R2 (checks the mechanism is present in the loss, not just implied by the formula).
3. **Truncation share among wrong answers** — fraction of wrong rollouts hitting the 2,048 cap: rises under R1 only.

*(to fill: plots + numbers)*

### E2 — what does dividing by std actually do?

Claim: near-unanimous groups have tiny std, so the lone deviant's advantage is inflated — 1 correct out of 5 gives that rollout $\hat{A} = 2.0$, vs $1.0$ in a balanced group. Evidence to collect:

1. **$|\hat{A}|$ vs group outcome** — mean $|\hat{A}|$ binned by correct-count in the group (0…G): with std the extremes dominate (U-shape); without, balanced groups do (∩-shape).
2. **Gradient mass by difficulty** — share of total advantage mass contributed by 1-of-G and (G−1)-of-G groups: higher under R2 than R3.
3. **Gradient-norm spikes** — R2's grad norm spikes on lone-success batches; R3 stays smoother.

*(to fill: plots + numbers)*

*(Next: mapping each switch onto verl config, then the runs.)*

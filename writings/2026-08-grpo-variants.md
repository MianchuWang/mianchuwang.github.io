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

- **$1/|o_i|$ gone** — it made each token of a *long* wrong answer cheaper, so wrong answers inflate. Prediction: wrong-answer length grows under GRPO, not here.
- **$/\,\mathrm{std}$ gone** — near-solved / near-impossible questions have tiny std and got outsized advantages.

### DAPO — four changes, no KL

$$
\mathcal{J} = \frac{1}{\sum_i |o_i|}\sum_{i=1}^{G} \sum_{t=1}^{|o_i|}
\min\!\big(\rho_{i,t}\hat{A}_i,\ \mathrm{clip}(\rho_{i,t},\,1-\varepsilon_{\mathrm{low}},\,1+\varepsilon_{\mathrm{high}})\,\hat{A}_i\big)
$$

- **Token-level $1/\sum_i|o_i|$** — same length-bias fix as Dr. GRPO ($\hat{A}_i$ stays GRPO's). Within a batch the two weight tokens identically (the denominator is one constant per batch); it only stabilizes the step size across batches, where Dr. GRPO's $1/G$ lets verbose batches take bigger steps.
- **Clip-higher** ($\varepsilon_{\mathrm{high}}=0.28 > \varepsilon_{\mathrm{low}}=0.2$) — a symmetric ceiling chokes rare tokens' growth → entropy collapse; raise only the ceiling.
- **Dynamic sampling** — all-correct / all-wrong groups have $\hat{A}=0$, zero gradient; filter and resample until the batch is live.
- **Overlong shaping** — truncated responses get a soft penalty, not a spurious reward.
- **$\beta = 0$** — no KL, no ref model, one fewer forward pass per step (8.4 s on our A40, per [W260828](post.html?p=2026-08-grpo-a40-scaling)).

## The ablation matrix

Every run: Qwen2.5-1.5B-Instruct on GSM8K, 1×A40, 2 epochs (232 steps), config identical to [W260828](post.html?p=2026-08-grpo-a40-scaling)'s 1-GPU run except the listed switch. E0's 2-GPU twin doubles as the noise band (±0.01 acc): a difference smaller than that is not a finding.

| ID | Change vs GRPO baseline | Isolates | Registered prediction |
|---|---|---|---|
| E0 | — (done: W260828) | baseline + noise band | — |
| E1 | per-sequence mean → token-level aggregation | the $1/\|o_i\|$ length bias | mean length of *wrong* answers rises under E0, flat under E1 |
| E2 | drop $/\,\mathrm{std}$ from $\hat{A}$ | difficulty bias | less gradient mass on near-solved prompts; late-training curves differ most |
| E3 | E1 + E2 (= Dr. GRPO) | the combined fix | ≈ E1 (length bias should dominate on GSM8K) |
| E4 | clip $1\pm0.2$ → $(1-0.2,\,1+0.28)$ | entropy collapse | entropy decays slower; upper clip fraction drops |
| E5 | + dynamic sampling | dead-batch compute | live-gradient fraction stays high late in training, when most groups go all-correct |
| E6 | $\beta = 0$, drop ref | the KL leash | step time −8.4 s; KL-to-ref drifts freely; acc unchanged on this task |
| E7 | E1 + E4 + E5 + E6 (≈ DAPO) | the full recipe | — |

The mechanism behind E1's prediction, in one line: $1/|o_i|$ makes each token of a 100-token failure ten times cheaper than a token of a 10-token failure, so short failure modes are suppressed faster and the *surviving* wrong answers skew long — a selection effect, not a reward for continuing.

*(Next: mapping each switch onto verl config, then the runs.)*

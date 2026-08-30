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

> [!note] **Do the normalizers prefer long responses?** No. Near $\rho = 1$,
> $$J_{\mathrm{Dr.GRPO}} \approx \frac{1}{G}\sum_i |o_i|\,\hat{A}_i, \qquad \sum_i \hat{A}_i = 0$$
> so lengthening every response just scales a mean-centered sum: a two-response group with $\hat{A} = \pm 1$ and length $L$ gives $\frac{1}{2}(L - L) = 0$ for any $L$. The normalizers — $G$ for Dr. GRPO, $G\,\overline{|o|}$ for DAPO — therefore change only the **magnitude and variance** of the gradient (verbose batches take bigger, noisier steps under Dr. GRPO, stable ones under DAPO), never its **direction**. The one *directional* length bias in this family is GRPO's $1/|o_i|$, which shrinks long sequences' weight asymmetrically.

*(Next: mapping each change onto verl switches, then the runs.)*

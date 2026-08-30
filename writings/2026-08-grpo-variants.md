---
title: "GRPO, Dr. GRPO, DAPO Comparisons"
date: 2026-08-30
tags: [DAPO, GRPO, RL, verl]
draft: true
summary: Three critic-free RL objectives, one pipeline — what each changes in the token-weighting math, and what actually differs in training.
---

## The three objectives

All three are critic-free: the baseline comes from sampling $G$ responses per question and comparing within the group. They differ in **how a group-relative reward becomes a per-token gradient weight**. Notation: response $o_i$ with reward $r_i$, token ratio $\rho_{i,t} = \pi_\theta(o_{i,t} \mid \cdot)\,/\,\pi_{\mathrm{old}}(o_{i,t} \mid \cdot)$.

### GRPO

$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}
$$

$$
\mathcal{J} = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
\min\!\big(\rho_{i,t}\hat{A}_i,\ \mathrm{clip}(\rho_{i,t},\,1\pm\varepsilon)\,\hat{A}_i\big)
\;-\; \beta\, D_{\mathrm{KL}}\!\big(\pi_\theta \,\|\, \pi_{\mathrm{ref}}\big)
$$

The baseline. Group-normalized advantage (mean *and* std), per-sequence length normalization ($1/|o_i|$), symmetric PPO clip, KL leash to the frozen reference.

### Dr. GRPO — remove two biases

$$
\hat{A}_i = r_i - \mathrm{mean}(r_1,\dots,r_G),
\qquad
\mathcal{J} = \frac{1}{G}\sum_{i=1}^{G} \sum_{t=1}^{|o_i|}
\min\!\big(\rho_{i,t}\hat{A}_i,\ \mathrm{clip}(\rho_{i,t},\,1\pm\varepsilon)\,\hat{A}_i\big)
$$

Two deletions, each fixing a bias GRPO builds in:

- **Drop $1/|o_i|$ (length bias).** Dividing a sequence's loss by its own length makes each token of a *long* wrong answer cheaper than a token of a short one — the optimizer learns that verbose failures hurt less, and incorrect responses inflate over training. Falsifiable prediction: under GRPO, mean length of *wrong* answers grows; under Dr. GRPO it shouldn't.
- **Drop $/\,\mathrm{std}$ (difficulty bias).** Near-solved and near-impossible questions have tiny group std, so dividing by it hands them outsized advantages. Removing it weights questions by their raw reward spread instead of amplifying the extremes.

### DAPO — four engineering changes

$$
\mathcal{J} = \frac{1}{\sum_i |o_i|}\sum_{i=1}^{G} \sum_{t=1}^{|o_i|}
\min\!\big(\rho_{i,t}\hat{A}_i,\ \mathrm{clip}(\rho_{i,t},\,1-\varepsilon_{\mathrm{low}},\,1+\varepsilon_{\mathrm{high}})\,\hat{A}_i\big),
\qquad \varepsilon_{\mathrm{high}} > \varepsilon_{\mathrm{low}}
$$

keeping GRPO's normalized $\hat{A}_i$, plus rules about *which data reaches the loss at all*:

- **Token-level aggregation** ($1/\sum_i|o_i|$): the same length-bias medicine as Dr. GRPO, with a different normalizer.
- **Clip-higher** ($\varepsilon_{\mathrm{high}} = 0.28$ vs $\varepsilon_{\mathrm{low}} = 0.2$): the symmetric upper clip caps how fast a *rare* token's probability may grow — exploration suffocates and entropy collapses. Raising only the ceiling frees low-probability tokens while keeping the floor that guards against collapse of existing behavior.
- **Dynamic sampling**: a group that is all-correct or all-wrong has $\hat{A}_i = 0$ for every member — zero gradient, dead compute — and the dead fraction *grows* as accuracy improves. Filter those groups and resample until the batch is full of live ones.
- **Overlong shaping**: responses cut by the length limit get a soft penalty instead of a spurious full reward or silent truncation.

DAPO also sets $\beta = 0$: no KL term, no reference model — reasoning training drifts far from the initial policy anyway, and the leash costs a full forward pass per step (8.4 s/step on our A40, per W260830).

*(Next: mapping each change onto verl switches, then the runs.)*

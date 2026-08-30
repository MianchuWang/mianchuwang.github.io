---
title: "GRPO Variants: Empirical Results"
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

**Setup.** Qwen2.5-1.5B-Instruct on MATH (levels 3–5), 64 prompts × $G = 12$ rollouts per step, max response 2,048 tokens — hard enough that accuracy stays mid-range, so groups keep mixing correct and wrong rollouts: the raw material both biases need. A large $G$ sharpens everything at once: fewer all-wrong zero-gradient groups, more lone-success events, finer $k$ resolution. All three runs share identical config and seed; only the listed switches differ.

**Variants.** Three training runs (R = run), each one switch apart:

- **R1** — GRPO, unmodified.
- **R2** — R1 minus $1/\vert o_i\vert$: token-level aggregation, std kept.
- **R3** — R2 minus $/\,\mathrm{std}$: both fixes, i.e. Dr. GRPO.

| Exp | Compares | Isolates |
|---|---|---|
| E1 | R1 vs R2 | the $1/\vert o_i\vert$ length bias |
| E2 | R2 vs R3 | the $/\,\mathrm{std}$ difficulty bias |

A higher accuracy curve is not a verdict — it is noisy and confounded. Each experiment must produce direct evidence that its mechanism operated.

### E1 — what does $1/|o_i|$ actually do?

Claim: per-token suppression of a wrong rollout scales as $1/|o_i|$, so long failures are punished gently and survive. Evidence to collect:

1. **Length by correctness over training** — mean length of training rollouts at each step, split by reward into wrong / correct.
2. **Per-token weight within one batch** — from periodically dumped rollout batches: among wrong rollouts of one step, per-token loss weight vs response length (checks the mechanism is actually present in the loss, not just implied by the formula).
3. **Truncation share among wrong answers** — fraction of wrong training rollouts hitting the 2,048 cap, per step.

*(to fill: plots + numbers)*

### E2 — what does removing the std division buy?

Claim: dividing by group std amplifies near-unanimous groups — precisely the least learnable prompts — and turns a lone lucky success into an outsized update. Removing it should buy stability and better-allocated learning, not just an unbiased estimator. Writing $k$ for the number of correct rollouts in a group (0…$G$), evidence to collect:

1. **Gradient stability** — the P95/P50 ratio of grad norm (tail heaviness; the ratio cancels the two runs' different advantage scales), plus spike rate (grad norm > 3× the rolling median of the last 50 updates), and whether spikes coincide with batches containing $k = 1$ or $G-1$ groups.
2. **Where the learning goes** — probe once at step 0: sample 8 answers per val prompt with the base model and freeze each prompt's solve rate as its difficulty label — hard (0–20%), mid (20–80%), easy (80–100%). Then plot val accuracy per bucket, alongside the share of total $|\hat{A}|$ mass contributed by near-unanimous groups ($k \le 1$ or $k \ge G-1$).
3. **Churn** — fraction of val prompts whose greedy answer flips from correct to wrong between consecutive validations. A prompt propped up by one amplified lucky trajectory has no redundant support — later, conflicting gradients from other prompts erode it; a prompt solved by consistent signal stays solved.

If none of these separate, the honest verdict is "unbiased and simpler, at no measurable cost" — also a result.

*(to fill: plots + numbers)*

*(Next: mapping each switch onto verl config, then the runs.)*

---
title: "GRPO Variants: Empirical Results"
date: 2026-08-30
tags: [DAPO, "Dr. GRPO", GRPO]
summary: Three critic-free RL objectives in one pipeline — what each one changes in the per-token weighting, and what actually differs in training.
---

## The three objectives

All three are critic-free: sample $G$ responses for each question, then compare them within the group. The difference between them is **how a group-relative reward becomes a per-token gradient weight**. Throughout, $\rho_{i,t} = \pi_\theta(o_{i,t} \mid \cdot)\,/\,\pi_{\mathrm{old}}(o_{i,t} \mid \cdot)$.

### GRPO

$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}
$$

$$
\mathcal{J} = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
\min\!\big(\rho_{i,t}\hat{A}_i,\ \mathrm{clip}(\rho_{i,t},\,1\pm\varepsilon)\,\hat{A}_i\big)
\;-\; \beta\, D_{\mathrm{KL}}\!\big(\pi_\theta \,\|\, \pi_{\mathrm{ref}}\big)
$$

### Dr. GRPO — remove two biases

$$
\hat{A}_i = r_i - \mathrm{mean}(r_1,\dots,r_G),
\qquad
\mathcal{J} = \frac{1}{G}\sum_{i=1}^{G} \sum_{t=1}^{|o_i|}
\min\!\big(\rho_{i,t}\hat{A}_i,\ \mathrm{clip}(\rho_{i,t},\,1\pm\varepsilon)\,\hat{A}_i\big)
$$

- **$1/|o_i|$ removed** — it gave every rollout the same total weight. But per *token*, this means a long wrong rollout is punished more weakly than a short one, and a long correct rollout is rewarded more weakly than a short one. In effect it encourages long wrong rollouts and short correct ones.
- **$/\,\mathrm{std}$ removed** — questions that are almost always solved, or almost never solved, have a very small std, so their advantages were scaled up too much.

> [!info] **Objective hacking.** The reward itself is never gamed — a long answer does not earn more reward. The gap is between the objective we intend and the surrogate we actually optimize: under $1/|o_i|$, the optimizer lowers the surrogate loss by moving failures into their least-punished (long) form, without changing correctness.

### DAPO — four changes, no KL

$$
\mathcal{J} = \frac{1}{\sum_i |o_i|}\sum_{i=1}^{G} \sum_{t=1}^{|o_i|}
\min\!\big(\rho_{i,t}\hat{A}_i,\ \mathrm{clip}(\rho_{i,t},\,1-\varepsilon_{\mathrm{low}},\,1+\varepsilon_{\mathrm{high}})\,\hat{A}_i\big)
$$

- **Token-level $1/\sum_i|o_i|$** — the same fix for the length bias as Dr. GRPO ($\hat{A}_i$ stays as in GRPO). The different denominator only keeps the step size stable across batches.
- **Clip-higher** ($\varepsilon_{\mathrm{high}}=0.28 > \varepsilon_{\mathrm{low}}=0.2$) — DAPO's reasoning: a symmetric ceiling holds back the growth of rare tokens and pushes the policy toward entropy collapse; so raise only the ceiling. E3 tests this.
- **Dynamic sampling** — groups where every rollout is correct, or every rollout is wrong, have $\hat{A}=0$ and give no gradient. Filter them out and sample again until the batch has signal.
- **Overlong shaping** — responses cut off by the length limit get a soft penalty instead of a misleading reward.
- **$\beta = 0$** — no KL term, no reference model, one forward pass fewer per step (8.4 s on our A40, see [W260828](post.html?p=2026-08-grpo-a40-scaling)).

## Ablations

**Setup.** Qwen2.5-1.5B-Instruct on MATH (levels 3–5), 64 prompts × $G = 12$ rollouts per step, maximum response 2,048 tokens. The task is hard enough that accuracy stays in the middle range, so groups keep a mix of correct and wrong rollouts — which is what both biases need to show up. A large $G$ keeps the groups mixed and gives fine resolution in $k$. Within a round, all runs share the same config and seed; only the listed switches differ.

**Variants.** Training runs (R = run), each one switch apart. First round:

- **R1** — GRPO, unchanged.
- **R2** — R1 without $1/\vert o_i\vert$: token-level aggregation, std kept.
- **R3** — R2 without $/\,\mathrm{std}$: both fixes, i.e. Dr. GRPO.

Second round (same data, model and budget):

- **R4** — Dr. GRPO with mini-batch 8 (8 gradient updates per batch instead of 2 — see E3 for why).
- **R5** — R4 + Clip-Higher: $\varepsilon_{\text{high}} = 0.28$, $\varepsilon_{\text{low}} = 0.2$.
- **R6** — R4 with $\beta = 0$; the reference model stays loaded so that KL can still be measured.

| Exp | Compares | Isolates |
|---|---|---|
| E1 | R1 vs R2 | the $1/\vert o_i\vert$ length bias |
| E2 | R2 vs R3 | the $/\,\mathrm{std}$ difficulty bias |
| E3 | R4 vs R5 (R3 as near-on-policy reference) | the $\varepsilon_{\text{high}}$ gradient cut-off |
| E4 | R4 vs R6 | the KL anchor |

Accuracy curves are noisy and mix many effects. Each experiment must show its mechanism directly.

### E1 — what does $1/|o_i|$ actually do?

Claim: the per-token punishment of a wrong rollout scales as $1/|o_i|$, so long failures are punished gently and survive. Evidence to collect:

1. **Length by correctness over training** — the mean length of training rollouts at each step, split by reward into wrong and correct.
2. **Per-token weight within one batch** — from periodically saved rollout batches: among the wrong rollouts of one step, the per-token loss weight against response length.
3. **Truncation share among wrong answers** — the fraction of wrong training rollouts that hit the 2,048 cap, per step.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e11_len">E1.1 — mean training-rollout length by correctness, R1 vs R2 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

| Series | steps 1–20 | steps 153–172 | Δ |
|---|---|---|---|
| R1 wrong | 693 | 841 | +148 |
| R2 wrong | 689 | 787 | +98 |
| R1 correct | 474 | 500 | +26 |
| R2 correct | 471 | 503 | +32 |

*(mean tokens, averaged over the first / last 20 steps)*

Findings:

1. **Everything gets longer.** Both runs and both classes grow — RL on MATH favors longer reasoning. So the evidence is the gap *between* the runs, not the slope.
2. **Wrong answers separate.** From the same starting point, R1's wrong answers end up longer than R2's. This is the selection effect: under $1/|o_i|$, long failures are punished weakly per token, so they survive.
3. **Correct answers do not.** A confound that affects the whole run would move both classes. A gap that appears only in wrong answers points to the mechanism.

**E1.2** — no data: the per-rollout dumps were lost when the pods were terminated.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e13_clip">E1.3 — share of wrong rollouts truncated at 2,048 tokens, R1 vs R2 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

| Series | steps 1–20 | steps 153–172 |
|---|---|---|
| R1 wrong | 1.0% | 3.9% |
| R2 wrong | 1.0% | 2.3% |

Findings:

1. **Truncation happens to wrong answers.** Correct rollouts almost never reach the cap (not plotted).
2. **The extreme tail grows faster under $1/|o_i|$.** The length gap in E1.1 is not a uniform shift. It comes from long failures that survive and grow until they hit the cap.

### E2 — what does removing the std division buy?

Claim: dividing by the group std scales up the advantages of near-unanimous groups — exactly the prompts that are hardest to learn from — and turns a single lucky success into an outsized update. Removing it should give ~~more stability and~~ better-allocated learning, not only an unbiased estimator. Writing $k$ for the number of correct rollouts in a group (0…$G$), evidence to collect:

1. **Gradient stability** — the P95/P50 ratio of the grad norm (how heavy the tail is; the ratio cancels the different advantage scales of the two runs), plus the spike rate (grad norm above 3× the rolling median of the last 50 updates).
2. **Where the learning goes** — validation accuracy per difficulty bucket (base-model solve rate at step 0: hard / mid / easy), together with the share of $|\hat{A}|$ mass that comes from near-unanimous groups:
   $$\text{mass share} = \frac{\sum_{i \,\in\, \text{near-unanimous}} |\hat{A}_i|}{\sum_{i \,\in\, \text{batch}} |\hat{A}_i|}, \qquad \text{near-unanimous:}\; k \le 1 \text{ or } k \ge G-1,$$
   with one $\hat{A}_i$ per rollout. All-wrong and all-correct groups have $\hat{A} \equiv 0$, so in practice the share measures groups with a single success or a single failure — where $/\,\mathrm{std}$ scales up the most (at $k=1$, std $\approx 0.28$, a $3.5\times$ boost).
3. **Churn** — the fraction of validation prompts whose greedy answer flips from correct to wrong between two consecutive validations. A prompt that is held up by one scaled-up lucky trajectory has no other support and slips back; a prompt learned from consistent signal stays solved.

If none of these separate, the honest verdict is "unbiased and simpler, at no measurable cost" — which is also a result.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e21_grad">E2.1 — actor grad norm per step, R2 vs R3 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

| Run | median | P95 | P95/P50 | spikes (>3× rolling median) |
|---|---|---|---|---|
| R2 with /std | 0.177 | 0.213 | 1.20 | 0 / 171 |
| R3 Dr. GRPO | 0.074 | 0.089 | 1.20 | 0 / 171 |

*(spikes counted from step 2, once the rolling window exists)*

Findings:

1. **The scaling-up is real in magnitude.** For 0/1 rewards, the group std is at most 0.5, so dividing by it at least doubles every advantage. That is why R2's grad norm is larger.
2. **The instability is not.** The tail heaviness and the spike count are the same. Each update averages over 32 groups, so one scaled-up lone-lucky group is diluted before it reaches the weights.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e22_mass">E2.2 — share of |A| mass from near-unanimous groups, R2 vs R3 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

Findings:

1. **The mass concentration is real but absorbed.** The scaling-up lands on the near-unanimous groups, as predicted; E2.1 shows that the batch average absorbs it.
2. **The difficulty-bucket half has no data** — it needed the per-prompt validation records that died with the pods.

**E2.3** — no data: same loss.

### E3 — what does raising $\varepsilon_{high}$ actually change?

Claim: the clip is a per-token gradient cut-off, not a wall. Once $\rho > 1+\varepsilon_{high}$ on a token with positive advantage, that token stops contributing to the update, and nothing pulls it back. Raising the ceiling (DAPO's Clip-Higher) moves the cut-off, which can only matter where tokens actually reach it. In R1–R3 tokens rarely did: with 2 updates per batch, `pg_clipfrac` stayed around 0.05%, and the lower boundary effectively never fired — so E3 runs both arms at mini-batch 8. ~~Even then, we expect the effect to be small — the ceiling touches about 1% of tokens, while sharpening acts on all of them.~~ What E3 really measures is how the clip's share changes with off-policyness. Evidence to collect:

1. **Clip incidence** — the fraction of tokens cut off at the upper boundary (`pg_clipfrac`), R5 vs R4. The lower boundary (the same $\varepsilon_{low}$ in both) is the control: ~~it should stay similar~~.
2. **Entropy** — the policy's mean next-token entropy: high means the probability is spread over many continuations, low means near-deterministic sampling. This is the downstream effect of the cut-off: gradients stay alive → rare tokens rise → entropy falls more slowly. R3's entropy collapsed *with the clip inactive*, so part of the fall is plain sharpening; the gap between R4 and R5 measures the part the clip is responsible for.
3. **Where the gains land** — validation accuracy split by MATH level. ~~If the freed exploration matters, Level 5 should improve first.~~ Noisy; secondary to 1–2.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e31_upper">E3.1 — share of tokens cut off at the upper clip boundary, R3 vs R4 vs R5 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e31_lower">E3.1 control — lower-boundary clip share, same three runs — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

Findings:

1. **Off-policyness sets the base rate.** More updates per batch means more clipping, and it grows over training: as the policy sharpens, the ratios drift further in each step and crowd the ceiling.
2. **Ceiling height beats off-policy pressure.** Raising $\varepsilon_{high}$ brings the clip rate below even the near-on-policy R3 and flattens the growth.
3. **What Clip-Higher buys is signal, not stability.** A clipped token is silenced, not destabilizing — the clip *is* the stabilizer. What R5 recovers are exactly the fastest-rising tokens, the ones that carry exploration.
4. **R4 also drifts most at the lower boundary, which fires about a thousand times less often.** As updates stack within a batch, the ratios drift away from 1 in both directions. R5 drifting less downward than R4 may be due to slower sharpening (see E3.2).

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e32_entropy">E3.2 — actor entropy over training, R3 vs R4 vs R5 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

Findings:

1. **A higher ceiling preserves exploration.** Raising $\varepsilon_{high}$ keeps R5's entropy far above R4's. Entropy is a precondition for exploration, not a result of it: once it collapses, sampling can no longer reach rare tokens, so there is nothing left to explore. Clip-Higher is therefore defensive — it stops sharpening from strangling exploration; it does not add any. But a preserved capacity is a necessary condition for better answers, not a sufficient one — and E3.3 does not find better answers here.
2. **Why clipping only 0.1% of tokens can move entropy this much.** The clipped tokens are exactly the low-probability tokens that are rising — the only force that spreads the distribution out. Silencing them at every step sends growth to tokens that are already likely, and over 172 steps × 8 updates this effect compounds until the two policies diverge.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e33_levels">E3.3 — validation accuracy by MATH level, R4 (solid) vs R5 (dashed) — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

Findings:

1. **End to end, the two are close.** L4 and L5 stay tangled throughout — at this scale, Clip-Higher's clear mechanistic effect (E3.1, E3.2) does not turn into accuracy.
2. **The one separation is on L3.** R5 rises to a plateau around step 150 and holds it, while R4 drops at the last validation; the plateau is the more reliable half, since R4's drop is a single point.

The experiments do not show a clear difference between R4 and R5; a larger-scale run may differ.

### E4 — what does $\beta = 0$ actually do?

Claim: the KL term anchors the policy to the initial model, mainly to stop reward hacking. But the reward here is a rule-based verifier with little to hack, and in R3 the term's contribution to the loss was $0.023 \times 0.001 \approx 2\times10^{-5}$. ~~The anchor probably does nothing;~~ R6 removes it while keeping the reference model loaded, so the divergence is still measured — measurement without regularization. Evidence to collect:

1. **KL to the reference** — R3's curve rose steadily with no explosion; does removing the anchor change the slope?
2. **Learning-curve parity** — accuracy and response lengths, R6 vs R4. If the curves cannot be told apart, the anchor has no effect on performance and the reference model is pure cost.
3. **Drift proxies** — the share of non-English characters and n-gram repetition in validation outputs. Expected: no drift attributable to $\beta$ at this scale — a bound, not a discovery.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e41_kl">E4.1 — KL to the reference model, R4 (β = 0.001) vs R6 (β = 0) — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

Finding: even at $\beta = 0.001$ the anchor measurably limits drift — without it, R6 moves away from the reference a little faster and further.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e42_acc">E4.2 — overall validation accuracy, R4 vs R6 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e42_len">Mean response length, same comparison — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

Finding: the accuracy and length curves cannot be told apart — R4 runs slightly higher through most of training, and the two meet at the last validation. With the anchor's contribution to the loss around $3\times10^{-5}$, that gap is a small perturbation, not a mechanism. Removing the anchor has no measurable cost in performance, and gives back the cost of the reference model.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e43_rep">E4.3 — 4-gram repetition in validation outputs, R4 vs R6 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

Finding: no drift that can be attributed to $\beta$. Non-English characters stay at 0% throughout. The 4-gram repetition rises steeply in *both* runs together — it is degenerate looping on hard problems, not an effect of the anchor. The worst R6 outputs simply repeat until they hit the length cap:

> `\cos(2x - (x + (y + z) - x)) = \cos(2x - (x + (y + z) - x)) = \cos(2x - (x + (y + z) - x)) = …`

> `So, 11 is divisible by 11. We can divide by 11 again: [1 ÷ 11 = 0] So, 11 is divisible by 11. We can divide by 11 again: …`

R4 loops in the same way; removing the anchor adds no degeneration.

## Summary

1. **Every switch shows up in the mechanism metrics; none shows up in accuracy** where we measured it (E3, E4). Whether a fix pays off depends on the regime.
2. **$1/|o_i|$**: long failures survive because their per-token punishment is diluted (E1).
3. **$/\,\mathrm{std}$**: the scaling-up is real, and batch averaging absorbs it (E2).
4. **Clip-Higher**: only matters under off-policy pressure; there it keeps entropy alive — a precondition for exploration, not a guarantee of better answers (E3).
5. **$\beta = 0.001$**: still limits drift, still changes nothing else; drop it and get the reference model's cost back (E4).

---
title: "GRPO Variants: Empirical Results"
date: 2026-08-30
tags: [DAPO, "Dr. GRPO", GRPO]
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

**Setup.** Qwen2.5-1.5B-Instruct on MATH (levels 3–5), 64 prompts × $G = 12$ rollouts per step, max response 2,048 tokens — hard enough that accuracy stays mid-range, so groups keep mixing correct and wrong rollouts: the raw material both biases need. A large $G$ keeps groups mixed and resolves $k$ finely. Within a round, runs share config and seed; only the listed switches differ.

**Variants.** Training runs (R = run), each one switch apart. First round:

- **R1** — GRPO, unmodified.
- **R2** — R1 minus $1/\vert o_i\vert$: token-level aggregation, std kept.
- **R3** — R2 minus $/\,\mathrm{std}$: both fixes, i.e. Dr. GRPO.

Second round (same data, model and budget):

- **R4** — Dr. GRPO, mini-batch 8 (8 gradient updates per batch instead of 2 — see E3 for why).
- **R5** — R4 + Clip-Higher: $\varepsilon_{\text{high}} = 0.28$, $\varepsilon_{\text{low}} = 0.2$.
- **R6** — R4 with $\beta = 0$; the reference model stays loaded so KL can still be measured.

| Exp | Compares | Isolates |
|---|---|---|
| E1 | R1 vs R2 | the $1/\vert o_i\vert$ length bias |
| E2 | R2 vs R3 | the $/\,\mathrm{std}$ difficulty bias |
| E3 | R4 vs R5 | the $\varepsilon_{\text{high}}$ gradient cut-off |
| E4 | R4 vs R6 | the KL anchor |

Accuracy curves are noisy and confounded; each experiment must show its mechanism directly.

### E1 — what does $1/|o_i|$ actually do?

Claim: per-token suppression of a wrong rollout scales as $1/|o_i|$, so long failures are punished gently and survive. Evidence to collect:

1. **Length by correctness over training** — mean length of training rollouts at each step, split by reward into wrong / correct.
2. **Per-token weight within one batch** — from periodically dumped rollout batches: among wrong rollouts of one step, per-token loss weight vs response length.
3. **Truncation share among wrong answers** — fraction of wrong training rollouts hitting the 2,048 cap, per step.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e11_len">E1.1 — mean training-rollout length by correctness, R1 vs R2 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

| Series | steps 1–20 | steps 153–172 | Δ |
|---|---|---|---|
| R1 wrong | 693 | 841 | +148 |
| R2 wrong | 689 | 787 | +98 |
| R1 correct | 474 | 500 | +26 |
| R2 correct | 471 | 503 | +32 |

*(mean tokens, averaged over the first / last 20 steps)*

Findings:

1. **Everything drifts longer.** Both runs, both classes rise — RL on MATH favors longer chains. The evidence is the gap *between* runs, not the slope.
2. **Wrong answers separate.** From identical starts, R1's wrong answers end longer — the selection effect: under $1/|o_i|$, long failures are suppressed weakly per token and survive.
3. **Correct answers don't.** A global confound would move both classes; a gap confined to wrong answers points at the mechanism.

**E1.2** — no data: the per-rollout dumps were lost to pod termination.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e13_clip">E1.3 — share of wrong rollouts truncated at 2,048 tokens, R1 vs R2 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

| Series | steps 1–20 | steps 153–172 |
|---|---|---|
| R1 wrong | 1.0% | 3.9% |
| R2 wrong | 1.0% | 2.3% |

Findings:

1. **Truncation is a wrong-answer phenomenon.** Correct rollouts almost never hit the cap (not plotted).
2. **The extreme tail grows faster under $1/|o_i|$.** The mean-length gap of E1.1 is not a uniform shift — it is long failures surviving and stretching into the cap.

### E2 — what does removing the std division buy?

Claim: dividing by group std amplifies near-unanimous groups — precisely the least learnable prompts — and turns a lone lucky success into an outsized update. ~~Removing it should buy stability and better-allocated learning, not just an unbiased estimator.~~ *Wrong on stability — E2.1 finds none to buy; allocation went untested.* Writing $k$ for the number of correct rollouts in a group (0…$G$), evidence to collect:

1. **Gradient stability** — the P95/P50 ratio of grad norm (tail heaviness; the ratio cancels the two runs' different advantage scales), plus spike rate (grad norm > 3× the rolling median of the last 50 updates).
2. **Where the learning goes** — val accuracy per difficulty bucket (base-model solve rate at step 0: hard / mid / easy), alongside the share of $|\hat{A}|$ mass from near-unanimous groups:
   $$\text{mass share} = \frac{\sum_{i \,\in\, \text{near-unanimous}} |\hat{A}_i|}{\sum_{i \,\in\, \text{batch}} |\hat{A}_i|}, \qquad \text{near-unanimous:}\; k \le 1 \text{ or } k \ge G-1,$$
   one $\hat{A}_i$ per rollout. All-wrong and all-correct groups have $\hat{A} \equiv 0$, so the share effectively measures lone-success and lone-failure groups — where $/\,\mathrm{std}$ amplifies most (at $k=1$, std $\approx 0.28$, a $3.5\times$ boost).
3. **Churn** — fraction of val prompts whose greedy answer flips from correct to wrong between consecutive validations. A prompt propped up by one amplified lucky trajectory has no redundant support and erodes; consistent signal stays solved.

If none of these separate, the honest verdict is "unbiased and simpler, at no measurable cost" — also a result.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e21_grad">E2.1 — actor grad norm per step, R2 vs R3 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

| Run | median | P95 | P95/P50 | spikes (>3× rolling median) |
|---|---|---|---|---|
| R2 with /std | 0.177 | 0.213 | 1.20 | 0 / 171 |
| R3 Dr. GRPO | 0.074 | 0.089 | 1.20 | 0 / 171 |

Findings:

1. **The amplification is real at the scale level.** For 0/1 rewards the group std is at most 0.5, so dividing by it at least doubles every advantage, hence R2's larger grad norm.
2. **The instability is not.** Tail heaviness and spike count are identical. Each update averages 32 groups, so one amplified lone-lucky group is diluted before it touches the weights.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e22_mass">E2.2 — share of |A| mass from near-unanimous groups, R2 vs R3 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

Findings:

1. **The mass concentration is real but absorbed.** The amplification lands on the near-unanimous groups as predicted; E2.1 shows the batch average absorbs it.
2. **The difficulty-bucket half has no data** — it needed the per-prompt validation records that died with the pods.

**E2.3** — no data: same loss.

### E3 — what does raising $\varepsilon_{high}$ actually change?

Claim: the clip is a per-token gradient cut-off, not a wall — once $\rho > 1+\varepsilon_{high}$ on a positive-advantage token, that token stops contributing to the update, and nothing pulls it back. Raising the ceiling (DAPO's Clip-Higher) moves the cut-off, which can only matter where tokens actually hit it. R1–R3 barely did: with 2 updates per batch, `pg_clipfrac` stayed at ~0.05% and the lower boundary never fired once — so E3 runs both arms at mini-batch 8. ~~Even then the expected effect is small — the ceiling touches ~1% of tokens while sharpening pressure acts on all of them.~~ *Wrong — 0.1% of tokens moved entropy threefold (E3.2).* What E3 really measures is how the clip's share scales with off-policyness. Evidence to collect:

1. **Clip incidence** — fraction of tokens cut off at the upper boundary (`pg_clipfrac`), R5 vs R4. The lower boundary (same $\varepsilon_{low}$ in both) is the control: ~~it should stay similar~~ *— it did not; see finding 4*.
2. **Entropy** — mean next-token entropy of the policy: high means mass spread across many continuations, low means near-deterministic sampling. The cut-off's downstream observable: gradients stay alive → rare tokens rise → entropy falls slower. R3's collapsed *with clip inactive*, so part of the fall is plain sharpening; the R4–R5 gap measures the clip's part.
3. **Where gains land** — validation accuracy split by MATH level. ~~If freed exploration matters, Level 5 should move first.~~ *Wrong — the only separation appeared on Level 3.* Noisy; auxiliary to 1–2.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e31_upper">E3.1 — share of tokens cut off at the upper clip boundary, R3 vs R4 vs R5 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e31_lower">E3.1 control — lower-boundary clip share, same three runs — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

Findings:

1. **Off-policyness sets the base rate.** More updates per batch means more clipping, with a climb over training: sharpening drifts ratios further each step, crowding the ceiling.
2. **Ceiling height beats off-policy pressure.** Raising $\varepsilon_{high}$ cuts incidence below even the near-on-policy R3 and flattens the climb.
3. **The purchase is signal, not stability.** A clipped token is silenced, not destabilizing — the clip *is* the stabilizer. What R5 recovers are precisely the fastest-rising tokens, the ones carrying exploration.
4. **The lower boundary shows the same ordering, two orders of magnitude down.** As updates stack within a batch, ratios drift off 1 in both directions. R5 drifting less downward than R4 may be due to slower sharpening (see E3.2).

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e32_entropy">E3.2 — actor entropy over training, R3 vs R4 vs R5 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

Findings:

1. **A higher ceiling preserves exploration.** Raising $\varepsilon_{high}$ holds R5's entropy far above R4's. Entropy is the precondition for exploration, not its by-product: once it collapses, sampling can no longer reach rare tokens, so there is nothing left to explore. Clip-Higher is thus defensive — it keeps sharpening from strangling exploration rather than adding any. But a preserved capacity is a necessary, not sufficient, condition for better answers — which E3.3 does not find here.
2. **Why 0.1% of tokens clipped can move entropy this much.** The clipped tokens are exactly the low-probability, rising ones — the distribution's only spreading force. Silencing them each step redirects growth toward already-likely tokens, and over 172 steps × 8 updates the compounding diverges the two policies.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e33_levels">E3.3 — validation accuracy by MATH level, R4 (solid) vs R5 (dashed) — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

Findings:

1. **End to end, the two are close.** L4 and L5 stay tangled throughout — at this scale Clip-Higher's clear mechanistic effect (E3.1, E3.2) does not cash out as accuracy.
2. **The one separation is on L3.** After ~150 steps R5 keeps climbing while R4 turns down; the rising trend is the sturdier half.

The experiments do not show a clear difference between R4 and R5; a larger-scale run may differ.

### E4 — what does $\beta = 0$ actually do?

Claim: the KL term anchors the policy to the init model, mainly to stop reward hacking — but the reward here is a rule verifier with little to hack, and in R3 the term's loss contribution measured $0.023 \times 0.001 \approx 2\times10^{-5}$. ~~The anchor is likely decorative;~~ *Half wrong — it measurably curbs drift (E4.1), with no performance consequence (E4.2).* R6 removes it while keeping the reference model loaded, so the divergence is still measured — measurement without regularization. Evidence to collect:

1. **KL to the reference** — R3's curve rose steadily with no explosion; does removing the anchor change the slope?
2. **Learning-curve parity** — accuracy and response lengths, R6 vs R4. Indistinguishable curves would mean the anchor was cosmetic and the reference model pure cost.
3. **Drift proxies** — non-English-character share and n-gram repetition in validation outputs. Expected: no visible drift at this scale — a bound, not a discovery.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e41_kl">E4.1 — KL to the reference model, R4 (β = 0.001) vs R6 (β = 0) — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

Finding: even at $\beta = 0.001$ the anchor measurably curbs drift — R6 without it diverges from the reference a little faster and further.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e42_acc">E4.2 — overall validation accuracy, R4 vs R6 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e42_len">Mean response length, same comparison — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

Finding: accuracy and length curves are indistinguishable — R4 runs a touch higher early on and the two meet by the last step; with the anchor's loss contribution at $\sim 10^{-5}$, that gap is perturbation, not mechanism. Removing the anchor costs nothing measurable and returns the reference model's cost.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e43_rep">E4.3 — 4-gram repetition in validation outputs, R4 vs R6 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

Finding: no drift attributable to $\beta$. Non-English characters stay at 0% throughout. The 4-gram repetition climbs steeply in *both* runs together — it is degenerate looping on hard problems, not an anchor effect. The worst R6 outputs simply repeat until the length cap:

> `\cos(2x - (x + (y + z) - x)) = \cos(2x - (x + (y + z) - x)) = \cos(2x - (x + (y + z) - x)) = …`

> `So, 11 is divisible by 11. We can divide by 11 again: [1 ÷ 11 = 0] So, 11 is divisible by 11. We can divide by 11 again: …`

R4 loops the same way; removing the anchor adds no degeneration.

## Summary

1. **Every switch shows up in the mechanism metrics; none shows up in accuracy** at this scale. Whether a fix pays depends on the regime.
2. **$1/|o_i|$**: long failures survive because their per-token punishment is diluted (E1).
3. **$/\,\mathrm{std}$**: the amplification is real, and batch averaging absorbs it (E2).
4. **Clip-Higher**: only matters under off-policy pressure; there it keeps entropy alive — a precondition for exploration, not a guarantee of better answers (E3).
5. **$\beta = 0.001$**: still curbs drift, still changes nothing else; drop it and get the reference model's cost back (E4).

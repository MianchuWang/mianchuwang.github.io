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

**Setup.** Qwen2.5-1.5B-Instruct on MATH (levels 3–5), 64 prompts × $G = 12$ rollouts per step, max response 2,048 tokens — hard enough that accuracy stays mid-range, so groups keep mixing correct and wrong rollouts: the raw material both biases need. A large $G$ sharpens everything at once: fewer all-wrong zero-gradient groups, more lone-success events, finer $k$ resolution. All three runs share identical config and seed; only the listed switches differ.

**Variants.** Three training runs (R = run), each one switch apart:

- **R1** — GRPO, unmodified.
- **R2** — R1 minus $1/\vert o_i\vert$: token-level aggregation, std kept.
- **R3** — R2 minus $/\,\mathrm{std}$: both fixes, i.e. Dr. GRPO.

Second round (planned; same data, model and budget):

- **R4** — Dr. GRPO, mini-batch 16 (4 gradient updates per batch instead of 2 — see E3 for why).
- **R5** — R4 + Clip-Higher: $\varepsilon_{\text{high}} = 0.28$, $\varepsilon_{\text{low}} = 0.2$.
- **R6** — R3 with $\beta = 0$; the reference model stays loaded so KL can still be measured.

| Exp | Compares | Isolates |
|---|---|---|
| E1 | R1 vs R2 | the $1/\vert o_i\vert$ length bias |
| E2 | R2 vs R3 | the $/\,\mathrm{std}$ difficulty bias |
| E3 | R4 vs R5 | the $\varepsilon_{\text{high}}$ gradient cut-off |
| E4 | R3 vs R6 | the KL anchor |

A higher accuracy curve is not a verdict — it is noisy and confounded. Each experiment must produce direct evidence that its mechanism operated.

### E1 — what does $1/|o_i|$ actually do?

Claim: per-token suppression of a wrong rollout scales as $1/|o_i|$, so long failures are punished gently and survive. Evidence to collect:

1. **Length by correctness over training** — mean length of training rollouts at each step, split by reward into wrong / correct.
2. **Per-token weight within one batch** — from periodically dumped rollout batches: among wrong rollouts of one step, per-token loss weight vs response length (checks the mechanism is actually present in the loss, not just implied by the formula).
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

1. **Everything drifts longer.** RL on MATH favors longer reasoning chains regardless of objective — both runs, both correctness classes rise. The evidence is therefore the gap *between* runs, not the absolute slope.
2. **Wrong answers separate.** From near-identical starts (693 vs 689 — a parity check passed), R1's wrong answers end 53 tokens (~7%) longer than R2's. Direction matches the selection effect: under $1/|o_i|$, long failures are suppressed weakly per token and survive.
3. **Correct answers don't.** +26 vs +32 is within noise. A global confound (R1 simply generating longer) would move both classes; a gap confined to wrong answers points at the mechanism, which operates exactly there.

**E1.2** — no data: the per-rollout dumps were lost to pod termination.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e13_clip">E1.3 — share of wrong rollouts truncated at 2,048 tokens, R1 vs R2 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

| Series | steps 1–20 | steps 153–172 |
|---|---|---|
| R1 wrong | 1.0% | 3.9% |
| R2 wrong | 1.0% | 2.3% |

Findings:

1. **Truncation is a wrong-answer phenomenon.** Correct rollouts almost never hit the cap in either run (peak 0.3%, not plotted) — the cap prunes exactly the population E1 is about.
2. **The extreme tail grows faster under $1/|o_i|$.** From the same 1% start, R1 ends with ~70% more truncated failures than R2 (3.9% vs 2.3%). The mean-length gap of E1.1 is not a uniform shift — it is driven by long failures surviving and stretching into the cap.

### E2 — what does removing the std division buy?

Claim: dividing by group std amplifies near-unanimous groups — precisely the least learnable prompts — and turns a lone lucky success into an outsized update. Removing it should buy stability and better-allocated learning, not just an unbiased estimator. Writing $k$ for the number of correct rollouts in a group (0…$G$), evidence to collect:

1. **Gradient stability** — the P95/P50 ratio of grad norm (tail heaviness; the ratio cancels the two runs' different advantage scales), plus spike rate (grad norm > 3× the rolling median of the last 50 updates), and whether spikes coincide with batches containing $k = 1$ or $G-1$ groups.
2. **Where the learning goes** — probe once at step 0: sample 8 answers per val prompt with the base model and freeze each prompt's solve rate as its difficulty label — hard (0–20%), mid (20–80%), easy (80–100%). Then plot val accuracy per bucket, alongside the share of total $|\hat{A}|$ mass contributed by near-unanimous groups:
   $$\text{mass share} = \frac{\sum_{i \,\in\, \text{near-unanimous}} |\hat{A}_i|}{\sum_{i \,\in\, \text{batch}} |\hat{A}_i|}, \qquad \text{near-unanimous:}\; k \le 1 \text{ or } k \ge G-1,$$
   one $\hat{A}_i$ per rollout (a batch is $64 \times G = 768$). All-wrong and all-correct groups have $\hat{A} \equiv 0$, so the share effectively measures lone-success and lone-failure groups — where $/\,\mathrm{std}$ amplifies most (at $k=1$, std $\approx 0.28$, a $3.5\times$ boost).
3. **Churn** — fraction of val prompts whose greedy answer flips from correct to wrong between consecutive validations. A prompt propped up by one amplified lucky trajectory has no redundant support — later, conflicting gradients from other prompts erode it; a prompt solved by consistent signal stays solved.

If none of these separate, the honest verdict is "unbiased and simpler, at no measurable cost" — also a result.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e21_grad">E2.1 — actor grad norm per step, R2 vs R3 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

| Run | median | P95 | P95/P50 | spikes (>3× rolling median) |
|---|---|---|---|---|
| R2 with /std | 0.177 | 0.213 | 1.20 | 0 / 171 |
| R3 Dr. GRPO | 0.074 | 0.089 | 1.20 | 0 / 171 |

Findings:

1. **The amplification is real at the scale level.** R2's grad norm runs 2.4× larger throughout — for 0/1 rewards the group std is at most 0.5, so dividing by it at least doubles every advantage. The switch demonstrably reaches the gradients.
2. **The instability is not.** Tail heaviness is identical (P95/P50 = 1.20 in both), zero spikes in 171 updates each, and the maximum (0.24) never approaches the 1.0 grad clip. The likely reason: each update averages 32 prompts' groups, so one amplified lone-lucky group is diluted ~32-fold before it touches the weights.

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e22_mass">E2.2 — share of |A| mass from near-unanimous groups, R2 vs R3 — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

Findings:

1. **The mass concentration is real but absorbed.** Near-unanimous groups carry 0.20 of the $|\hat{A}|$ mass under R2 vs 0.13 under R3 (run means) — the amplification lands where predicted, and E2.1 shows the batch average absorbs it.
2. **The difficulty-bucket half has no data** — it needed the per-prompt validation records that died with the pods.

**E2.3** — no data: same loss.

### E3 — what does raising $\varepsilon_{high}$ actually change?

Claim: the clip is a per-token gradient cut-off, not a wall — once $\rho > 1+\varepsilon_{high}$ on a positive-advantage token, that token stops contributing to the update, and nothing pulls it back. Raising the ceiling (DAPO's Clip-Higher) moves the cut-off, which can only matter where tokens actually hit it. R1–R3 barely did: with 2 updates per batch, `pg_clipfrac` stayed at ~0.05% and the lower boundary never fired once — so E3 runs both arms at mini-batch 16, giving the boundary four updates of surface area per batch. Evidence to collect:

1. **Clip incidence** — fraction of tokens cut off at the upper boundary (`pg_clipfrac`), R5 vs R4. The lower boundary (same $\varepsilon_{low}$ in both) is the control: it should stay similar.
2. **Entropy** — actor entropy over training. R3's entropy collapsed 0.27 → 0.085 *with clip inactive*, so objective sharpening alone kills entropy here; Clip-Higher can only rescue the part the clip is responsible for. Coinciding curves would mean this medicine does not treat this configuration's disease — also a result.
3. **Where gains land** — validation accuracy split by MATH level (labels ride in `extra_info`). If freed exploration matters, Level 5 should move first. ~300 prompts per level, so noisy; auxiliary to 1–2.

*(to fill: plots + numbers)*

### E4 — what does $\beta = 0$ actually do?

Claim: the KL term anchors the policy to the init model, mainly to stop reward hacking — but the reward here is a rule verifier with little to hack, and in R3 the term's loss contribution measured $0.023 \times 0.001 \approx 2\times10^{-5}$. The anchor is likely decorative; R6 removes it while keeping the reference model loaded, so the divergence is still measured — measurement without regularization. Evidence to collect:

1. **KL to the reference** — R3's curve rose 0.0035 → 0.023 nats/token with no explosion. Does removing the anchor change the slope, or was it decorative all along?
2. **Learning-curve parity** — accuracy and response lengths, R6 vs R3. Indistinguishable curves are the honest verdict: the anchor was already cosmetic, and dropping the reference model buys pure engineering — no ref forward pass, one less model copy in memory.
3. **Drift proxies** — non-English-character share and n-gram repetition in rollouts, from dumps (streamed to wandb this time). Expectation: no visible drift at 1.5B × 172 steps — this evidence is a bound, not a discovery.

*(to fill: plots + numbers)*

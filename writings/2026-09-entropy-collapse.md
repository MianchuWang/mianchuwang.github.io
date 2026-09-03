---
title: "The Collapse of Entropy"
date: 2026-09-03
tags: [Entropy, GRPO, "In Progress"]
summary: One disease behind several GRPO fixes — what entropy collapse is, why it caps exploration, and what our own runs already show.
---

## The problem

### What collapses

Entropy here is the policy's mean next-token entropy over its own rollouts: high means probability mass spread across many plausible continuations, low means near-deterministic sampling. Under RL with verifiable rewards it falls fast — in our runs ([W260830](post.html?p=2026-08-grpo-variants), E3.2) from 0.27 to 0.085 in 172 steps, and faster still with more updates per batch:

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e32_entropy">Actor entropy over training for three Dr. GRPO runs — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

Some of that fall is the point. Learning *is* sharpening: a policy that has found the right steps should sample them more often. The problem is not that entropy falls but that it falls past the floor where exploration dies, and faster than learning justifies. So the question is quantitative: **how much of each step's entropy loss is learning, and how much is collateral?**

### Why it matters

RL does not invent, it selects: every update reweights tokens the policy already samples. Entropy is therefore the precondition for exploration, not its by-product — once it collapses, rare tokens stop being sampled, so nothing new can ever be reinforced. The group signal starves at the same time: unanimous groups have $\hat{A} \equiv 0$, and the share of unanimous groups rises as the policy sharpens. A collapsed policy is not wrong, it is finished.

### What we have already seen

Three shapes from W260830, all on Qwen2.5-1.5B / MATH L3–5:

- **Off-policy pressure accelerates collapse** — eight updates per batch (R4) collapses harder than two (R3).
- **A gradient-side fix holds it** — raising the clip ceiling (R5) keeps entropy at three times R4's endpoint, by keeping the rare rising tokens' gradients alive.
- **Preserved entropy did not pay** — R5's accuracy matched R4's. Necessary, not sufficient: this scale never got to spend the capacity it kept.

That last point sets the bar for what follows. Preserving entropy is cheap; the real question is which interventions preserve the *right* entropy — spread that a larger regime can turn into answers — and how to tell them apart at a scale where accuracy cannot.

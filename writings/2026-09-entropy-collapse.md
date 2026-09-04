---
title: "The Collapse of Entropy"
date: 2026-09-03
tags: [Entropy, GRPO, "In Progress"]
summary: One problem behind several GRPO fixes — what entropy collapse is, why it limits exploration, and what our own runs already show.
---

## Problem

At each position $t$ of a rollout, the policy's next-token entropy is

$$
H_t = -\sum_{v \in \mathcal{V}} \pi_\theta(v \mid x, o_{<t}) \,\log \pi_\theta(v \mid x, o_{<t}),
$$

where $x$ is the prompt, $o_{<t}$ the tokens generated so far, $v$ ranges over every candidate in the vocabulary $\mathcal{V}$, and $\pi_\theta$ is the current policy. The entropy we track is the average of $H_t$ over all tokens the policy generates in one training step. High entropy means the probability is spread over many possible next tokens. Low entropy means the policy almost always picks the same token.

Under RL with verifiable rewards, entropy drops quickly. In our runs ([W260830](post.html?p=2026-08-grpo-variants), E3.2) it fell from 0.27 to 0.085 over 172 steps (R3; means of the first and last 20 steps):

<div class="chart" data-src="writings/figures/w260830.json" data-metric="e32_entropy">Actor entropy over training for three Dr. GRPO runs — interactive figure; raw data: <a href="writings/figures/w260830.json">w260830.json</a></div>

Part of this drop is expected. Learning means the policy becomes more certain: once it finds the right steps, it should choose them more often. The problem is not that entropy drops. The problem is that it drops below the level where exploration is still possible, and faster than learning requires. So the question is a quantitative one: **in each step, how much of the entropy loss is learning, and how much is a side effect?**

Three results from those runs (Qwen2.5-1.5B, MATH levels 3–5):

- **More off-policy updates make the collapse worse** — eight updates per batch (R4) collapses further than two (R3).
- **A gradient-side fix slows it down** — raising the clip ceiling (R5) keeps entropy at three times R4's final value, because the rare tokens that are rising keep their gradients.
- **Keeping entropy did not improve accuracy** — R5 and R4 reached the same accuracy. Entropy is necessary for exploration, but not sufficient for better answers.

The last point sets the standard for the rest of this article. Keeping entropy high is easy. The real question is which methods keep the *useful* kind of entropy — the kind a larger model or a longer run can turn into better answers — and how to tell them apart when accuracy alone cannot.

## Identification

*(to fill)*

## Accounting

*(to fill)*

## Treatments

*(to fill)*

## Comparison

*(to fill)*

## Summary

*(to fill)*

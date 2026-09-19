---
title: "The System Design of verl: GRPO as an Example"
date: 2026-09-18
tags: ["system design", verl, "In Progress"]
summary: Exploring how verl coordinates data, computation, and GPU resources across generation and training.
---

> [!info] **Version.** This article studies [verl v0.8.0](https://github.com/verl-project/verl/tree/v0.8.0), released on June 1, 2026.

We examine verl's system design through a single training step of Group Relative Policy Optimization (GRPO) on math problems. The discussion has three parts:

1. **Training loop:** the order of tasks in one step, and the data passed between them.
2. **Workers:** how one call from the loop runs on many GPUs, in the generation and training engines.
3. **GPU placement:** how generation and training are placed on GPUs, on the same ones or on separate ones, and what each choice costs in memory, weight transfer, and communication.

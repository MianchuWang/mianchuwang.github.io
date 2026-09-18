---
title: "The System Design of verl: GRPO as an Example"
date: 2026-09-18
tags: ["system design", verl, "In Progress"]
summary: Exploring how verl coordinates data, computation, and GPU resources across generation and training.
---

We examine verl's system design through a single training step of Group Relative Policy Optimization (GRPO) on math problems. The discussion proceeds in three stages:

1. **Coordination:** how the training loop orders tasks and passes data between them.
2. **Execution:** how workers execute these tasks using generation and training engines.
3. **Resource management:** how GPU memory requirements, weight transfers, and communication costs affect the design.

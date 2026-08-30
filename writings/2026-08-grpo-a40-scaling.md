---
title: "Fine-tuning Qwen2.5-1.5B-Instruct on GSM8K with Single/Double A40 GPUs"
date: 2026-08-28
tags: [GPU, GRPO, verl]
summary: A 1-vs-2 GPU GRPO comparison — where the speedup went, phase by phase.
---

## Setup

**Task.** GRPO on GSM8K with Qwen2.5-1.5B-Instruct, via verl 0.8.0 (vLLM 0.11 rollouts, FSDP training, colocated on the same GPUs). Reward is rule-based: exact match on the `#### <answer>` line.

**Data & schedule.** 7,473 train / 1,319 test problems. Global batch 64 prompts × 5 rollouts, max response 1,024 tokens → 116 steps per epoch; both runs train 2 epochs (232 steps).

**One step, as pseudocode** — which model runs, how many forward passes, and what each is for:

```python
responses = vllm.generate(prompts)              # 64 prompts x 5 samples, actor weights synced in
old_logp  = actor.forward(responses)            # fwd #1, no grad: rollout-time snapshot
ref_logp  = ref.forward(responses)              # fwd #2, frozen initial model: feeds the KL only
adv       = group_normalize(reward(responses))  # GRPO: no critic
for mb in split(batch, 2):                      # 2 gradient updates per step
    logp  = actor.forward(mb)                   # training fwd (this is also the backward graph)
    ratio = exp(logp - old_logp)                # = 1 on update 1; corrects staleness on update 2
    loss  = -clip(ratio * adv) + 0.001 * KL(logp, ref_logp)
    actor.update(loss)
```

**Hardware.** RunPod community-cloud A40s (48 GB, $0.44/GPU·h): one pod with 1× A40, one with 2× A40. The pair has no NVLink — `nvidia-smi topo` reports `SYS`, i.e. inter-GPU traffic crosses the CPU-socket interconnect.

**Parity.** The two runs differ in exactly one flag, `trainer.n_gpus_per_node`. The global batch stays 64, so both see identical optimization trajectories and the curves compare step-for-step. No FSDP offload in either run.

## Learning curves

Both runs consume the same global batches in the same order, so correct data parallelism should trace *the same curve*. It does: final val accuracy **0.777 (1 GPU) vs 0.789 (2 GPUs)**, inside the ±0.01 step-to-step noise. The overlap is the correctness certificate for the speed comparison that follows.

<div class="chart" data-src="writings/figures/w260828.json" data-metric="acc">GSM8K validation accuracy, 1 vs 2 GPUs — interactive figure; raw data: <a href="writings/figures/w260828.json">w260828.json</a></div>

Shape: accuracy jumps 0.09 → 0.72 in 10 steps — that is the model learning the `#### <answer>` format, not math. The remaining 220 steps grind 0.75 → 0.79; the second epoch is worth ~3–4 points.

> [!note] (i) Rollouts are sampled at temperature 1.0 — GRPO's advantage is the within-group variance of the 5 samples, which greedy decoding would zero out; validation decodes greedily (hence `mean@1`). (ii) The tiny gap between the curves is implementation, not math: splitting rollouts across two engines changes RNG streams and kernel scheduling, and DP's cross-GPU gradient summation has a different floating-point order than single-GPU accumulation. RL's feedback loop amplifies these, so the curves agree in expectation rather than bit-for-bit.

## Where the speedup went

Two GPUs bought **1.30×** (54.8 → 42.3 s/step; 3h 46m → 2h 52m wall clock).

| Phase | 1×A40 | 2×A40 | Speedup | Source (mean over 232 steps) |
|---|---|---|---|---|
| rollout generation (vLLM) | 11.3 s | 8.7 s | 1.29× | `timing_s/gen` |
| old_log_prob (fwd, no grad) | 5.9 s | 4.2 s | 1.41× | `timing_s/old_log_prob` |
| ref log_prob (fwd, no grad) | 8.4 s | 4.5 s | **1.88×** | `timing_s/ref` |
| actor update (2 × fwd+bwd+optim) | 23.8 s | 21.2 s | **1.12×** | `timing_s/update_actor` |
| weight sync to vLLM | 2.1 s | 2.2 s | 0.97× | `timing_s/update_weights` |
| **full step** | **54.8 s** | **42.3 s** | **1.30×** | `timing_s/step` |

The pattern: the less communication a phase needs, the better it scales. The cause is the interconnect — `SYS` means every FSDP collective crosses the CPU-socket link at a fraction of NVLink bandwidth. Generation is the exception: its 1.29× is bandwidth economics, not communication — each engine re-reads the full weights on every decode step regardless of batch size, and the step count is set by the longest response, which splitting the batch barely shortens.

MFU falls **31% → 18%** while `nvidia-smi` reads ~100% on both cards — NCCL kernels busy-wait, so "busy" includes "waiting". Utilization ≠ useful math; that is what MFU is for.

<div class="chart" data-src="writings/figures/w260828.json" data-metric="mfu">MFU during the actor update, 1 vs 2 GPUs — interactive figure; raw data: <a href="writings/figures/w260828.json">w260828.json</a></div>

### The mechanics behind 1.12×

Full-shard FSDP costs three collectives per micro-batch: all-gather params for the forward, again for the backward, reduce-scatter the gradients. (No fourth "sync params" step — each GPU updates only the shard it owns; the next forward's all-gather reassembles the result.) FSDP hides this traffic by prefetching the next layer's parameters during compute, but a 1.5B model computes each layer faster than the SYS link delivers the next one, so the overlap fails and both GPUs stall at every layer boundary. Raising `ppo_micro_batch_size_per_gpu` from 8 to `mini`=32 would cut the collective rounds 4× — the cheapest lever on a bad interconnect.

### Open question

`old_log_prob` and `ref` are nominally the same no-grad forward over the same sequences — why is ref 42% slower on 1 GPU (8.4 vs 5.9 s)? (Its better scaling, 1.88× vs 1.41×, is just the overlap law again: slower per-layer compute hides more communication.)

## Economics & takeaways

The run cost **\$1.66 on 1×A40** (3h 46m × \$0.44) vs **\$2.52 on 2×A40** (2h 52m × \$0.88): the second GPU delivered 1.30× the speed for 1.52× the money. On this interconnect, doubling GPUs is a way to buy time, not efficiency — worth it only when wall clock matters more than budget.

1. **Buy interconnect, not GPU count.** `nvidia-smi topo -m` is the first command on any multi-GPU pod; on `SYS`, redeploy and draw a different host.
2. **Read utilization, power, and MFU together.** NCCL busy-waits at 100% utilization; only MFU says whether the math is useful.
3. **RL steps are expensive by construction.** Four forward passes per step — generation, old_log_prob, ref, training — and only the last carries gradients.
4. **Small models scale worst.** 1.5B computes each layer too fast to hide communication behind it; at this scale one bigger GPU beats two small ones.
5. **The cheapest lever was free.** `micro_batch = mini_batch` (memory permitting) cuts FSDP collective rounds 4×; the default of 8 was an unexamined copy from the quickstart.

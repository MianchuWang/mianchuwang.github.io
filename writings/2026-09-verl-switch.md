---
title: "verl Internals: The Training–Rollout Switch"
date: 2026-09-01
tags: [verl]
summary: How colocated verl swaps one GPU between FSDP training and vLLM inference — sleep levels, weight sync, and the memory choreography of a single step.
---

In colocated (hybrid-engine) verl, one GPU runs two engines that can never be
resident at full size together: the FSDP training engine (params + grads +
optimizer states) and the vLLM inference engine (weights + KV cache). Every
training step the GPU changes hands twice. This note reads the switch's
source, pinned to the commit our runs used
([`7aed6b2`](https://github.com/volcengine/verl/tree/7aed6b230776f963fa09509c10d9c3a767d1102c)).

## The cast

Three files carry the whole mechanism:

| File | Role |
|---|---|
| `verl/workers/engine_workers.py` — `ActorRolloutRefWorker` | Owns both engines; its `update_weights()` *is* the switch |
| `verl/workers/rollout/vllm_rollout/vllm_rollout.py` — `ServerAdapter` | The vLLM side: `sleep()`, `resume(tags)`, `update_weights()` |
| `verl/third_party/vllm/__init__.py` | Picks `VLLM_SLEEP_LEVEL` (what "sleep" releases) |

## The choreography of one hand-off

`ActorRolloutRefWorker.update_weights()` (training → rollout, run right before
generation) does five moves in strict order:

1. **`set_expandable_segments(False)`** — vLLM's sleep mode is built on
   `CuMemAllocator`, which cannot coexist with PyTorch's
   `expandable_segments` allocator feature; verl toggles it off around the
   switch. (We learned this empirically before reading it: exporting
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` broke vLLM engine
   init in our smoke test.)
2. **`rollout.resume(tags=["weights"])`** — re-materialize *only* vLLM's
   weight buffers. The KV cache stays released: the next move needs the
   training engine's params on the GPU too, and both would not fit with a
   KV cache in place.
3. **`rollout.update_weights(per_tensor_param)`** — the actual sync.
   `get_per_tensor_param()` walks the FSDP engine's parameters and hands
   them tensor-by-tensor to vLLM, which copies them into its own layout.
   No disk, no network — a GPU-to-GPU copy inside one process group.
4. **Offload + `aggressive_empty_cache()`** — if param offload is enabled,
   the training copy moves to CPU; either way the allocator cache is
   flushed so the freed VRAM is actually reusable.
5. **`rollout.resume(tags=["kv_cache"])`** — only now, with the training
   engine out of the way, does vLLM claim its KV cache
   (`gpu_memory_utilization` of the card) and stand ready to generate.

The reverse hand-off (rollout → training) is the mirror: after generation,
`rollout.sleep()` releases KV cache and — depending on the sleep level —
weights, then the training engine loads back.

*(next: sleep levels 1 vs 2, and what `get_per_tensor_param` does under
FSDP sharding)*

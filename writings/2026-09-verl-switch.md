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

## 1. Training loop

The training loop is `RayPPOTrainer.fit`, at line 1362 of [`verl/trainer/ppo/ray_trainer.py`](https://github.com/verl-project/verl/blob/v0.8.0/verl/trainer/ppo/ray_trainer.py#L1362).

**What it does.** A task is one unit of work in a training step, such as generating responses, computing advantages, or updating the actor. `fit` decides the order of the tasks and passes data between them. It does not run the model. Each heavy task is a remote call to GPU workers. Only the advantage computation, which is light, runs inside `fit`.

**When it is called.** Once per run. `python3 -m verl.trainer.main_ppo` reaches it in three steps:

1. `main` in [`verl/trainer/main_ppo.py`](https://github.com/verl-project/verl/blob/v0.8.0/verl/trainer/main_ppo.py#L38) reads the configuration and calls `run_ppo`.
2. `run_ppo` starts Ray and creates `TaskRunner`, a Ray actor with one CPU and no GPU.
3. `TaskRunner.run` builds the trainer, creates the GPU workers with `init_workers()`, and calls `fit()`.

The process that runs `fit` is the driver. `fit` returns when training reaches the last step.

### 1.1 Before the loop

Lines 1369–1420 run once, before the first step. Restore state loads a checkpoint if one exists, sends the weights to the generation side, and computes the starting epoch. Validate before training evaluates the model once and logs the metrics. If `val_only` is also set, `fit` returns there.

**Restore state** (lines 1383–1389).

```python
self.global_steps = 0

# load checkpoint and update weights before doing anything
self._load_checkpoint()
self.checkpoint_manager.update_weights(self.global_steps)

current_epoch = self.global_steps // len(self.train_dataloader)
```

`_load_checkpoint()` looks for the latest checkpoint on the local disk. If one exists, it restores the step counter, the actor, and the position of the dataloader. Its behavior is set by `self.config.trainer.resume_mode`: `disable` skips the loading and trains from scratch; `auto`, the default, loads the latest checkpoint if one exists; `resume_path` loads the checkpoint given in `resume_from_path`. It loads the actor through `actor_rollout_wg`. Here `wg` means worker group: the GPU workers that hold the actor. GRPO has no critic, so `critic_wg` does not exist and no critic is loaded.

`update_weights` copies the actor's weights from the training side to the generation side. There are three engines in verl: the model engine, the rollout engine, and the checkpoint engine. The model engine, such as FSDP, trains the actor. The rollout engine, such as vLLM, generates responses; it runs as one or more rollout replicas, and each replica is one inference server with its own copy of the model. When the two share GPUs (`backend="naive"`), the model engine hands the weights to the rollout engine directly. If not, the checkpoint engine moves the weights. In both cases `CheckpointEngineManager` coordinates the transfer.

**Validate before training** (lines 1391–1400).

```python
# perform validation before training
# currently, we only support validation using the reward_function.
if self.config.trainer.get("val_before_train", True):
    val_metrics = self._validate()
    assert val_metrics, f"{val_metrics=}"
    pprint(f"Initial validation metrics: {val_metrics}")
    logger.log(data=val_metrics, step=self.global_steps)
    if self.config.trainer.get("val_only", False):
        self._shutdown_dump_executor()
        return
```

The `assert` stops the run if validation returns no metrics. The dump executor is a background thread that writes generated samples to disk, so that the loop does not wait for the disk. It writes only if `trainer.rollout_data_dir` or `trainer.validation_data_dir` is set. With `val_only`, `fit` waits for it to finish, and then returns without training.

### 1.2 In the loop

### 1.3 After the loop

## 2. Workers

## 3. GPU placement

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

Lines 1422–1768 are the loop. Each pass of the inner loop is one training step. The parts below are the ones a GRPO step runs, in order. Branches that GRPO does not use, such as the critic and REMAX, are left out.

```python
for epoch in range(current_epoch, self.config.trainer.total_epochs):
    for batch_dict in self.train_dataloader:
```

**1. Build the batch** (lines 1435–1448).

```python
batch: DataProto = DataProto.from_single_dict(batch_dict)
batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

# add uid to batch
batch.non_tensor_batch["uid"] = np.array(
    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
)

gen_batch = self._get_gen_batch(batch)

# pass global_steps to trace
gen_batch.meta_info["global_steps"] = self.global_steps
rollout_n = self.config.actor_rollout_ref.rollout.n
gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)
```

**2. Generate** (lines 1467–1471).

```python
with marked_timer("gen", timing_raw, color="red"):
    ...
    combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
    self.checkpoint_manager.sleep_replicas()
```

**3. Merge and balance** (lines 1496–1507).

```python
# repeat to align with repeated responses in rollout
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
batch = batch.union(gen_batch_output)

if "response_mask" not in batch.batch.keys():
    batch.batch["response_mask"] = compute_response_mask(batch)
# Balance the number of valid tokens across DP ranks.
# NOTE: This usually changes the order of data in the `batch`,
# which won't affect the advantage calculation (since it's based on uid),
# but might affect the loss calculation (due to the change of mini-batching).
if self.config.trainer.balance_batch:
    self._balance_batch(batch, metrics=metrics)
```

**4. Reward** (lines 1518–1525).

```python
with marked_timer("reward", timing_raw, color="yellow"):
    # compute reward model score
    if self.use_rm and "rm_scores" not in batch.batch.keys():
        batch_reward = self._compute_reward_colocate(batch)
        batch = batch.union(batch_reward)

    # extract reward_tensor and reward_extra_infos_dict for training
    reward_tensor, reward_extra_infos_dict = extract_reward(batch)
```

**5. Old log-probabilities** (lines 1542–1567).

```python
with marked_timer("old_log_prob", timing_raw, color="blue"):
    old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
    ...
    batch = batch.union(old_log_prob)
```

**6. Reference log-probabilities** (lines 1576–1580).

```python
if self.use_reference_policy:
    # compute reference log_prob
    with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
        ref_log_prob = self._compute_ref_log_prob(batch)
        batch = batch.union(ref_log_prob)
```

**7. Advantage** (lines 1588–1633).

```python
with marked_timer("adv", timing_raw, color="brown"):
    ...
    batch.batch["token_level_scores"] = reward_tensor
    ...
    # compute rewards. apply_kl_penalty if available
    if self.config.algorithm.use_kl_in_reward:
        batch, kl_metrics = apply_kl_penalty(
            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
        )
        metrics.update(kl_metrics)
    else:
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
    ...
    # compute advantages, executed on the driver process
    norm_adv_by_std_in_grpo = self.config.algorithm.get(
        "norm_adv_by_std_in_grpo", True
    )  # GRPO adv normalization factor

    batch = compute_advantage(
        batch,
        adv_estimator=self.config.algorithm.adv_estimator,
        gamma=self.config.algorithm.gamma,
        lam=self.config.algorithm.lam,
        num_repeat=self.config.actor_rollout_ref.rollout.n,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=self.config.algorithm,
    )
```

**8. Update the actor** (lines 1647–1649).

```python
# update actor
with marked_timer("update_actor", timing_raw, color="red"):
    actor_output = self._update_actor(batch)
```

**9. Save a checkpoint** (lines 1663–1671).

```python
if self.config.trainer.save_freq > 0 and (
    is_last_step
    or self.global_steps % self.config.trainer.save_freq == 0
    or esi_close_to_expiration
):
    if esi_close_to_expiration:
        print("Force saving checkpoint: ESI instance expiration approaching.")
    with marked_timer("save_checkpoint", timing_raw, color="green"):
        self._save_checkpoint()
```

**10. Sync the weights** (lines 1673–1675).

```python
# update weights from trainer to rollout
with marked_timer("update_weights", timing_raw, color="red"):
    self.checkpoint_manager.update_weights(self.global_steps)
```

**11. Validate** (lines 1685–1693).

```python
# validate
if self.config.trainer.test_freq > 0 and (
    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
):
    with marked_timer("testing", timing_raw, color="green"):
        val_metrics: dict = self._validate()
        if is_last_step:
            last_val_metrics = val_metrics
    metrics.update(val_metrics)
```

**12. Log and advance** (lines 1750–1753).

```python
logger.log(data=metrics, step=self.global_steps)

progress_bar.update(1)
self.global_steps += 1
```

### 1.3 After the loop

## 2. Workers

## 3. GPU placement

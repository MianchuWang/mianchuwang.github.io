---
title: "The System Design of verl: GRPO as an Example"
date: 2026-09-18
tags: ["system design", verl, "In Progress"]
summary: Exploring how verl coordinates data, computation, and GPU resources across generation and training.
---

> [!info] **Version.** This article studies [verl v0.8.0](https://github.com/verl-project/verl/tree/v0.8.0), released on June 1, 2026.

We examine verl's system design through a single training step of Group Relative Policy Optimization (GRPO) on math problems. The discussion has three parts:

1. **Training loop:** the order of tasks in one step, and the data passed between them.
2. **Workers:** how one call from the loop runs on many GPUs, in the model engine and the rollout engine.
3. **GPU placement:** how the model engine and the rollout engine are placed on GPUs, on the same ones or on separate ones, and what each choice costs in memory, weight transfer, and communication.

## 1. Training loop

The training loop is `RayPPOTrainer.fit`, at line 1362 of [`verl/trainer/ppo/ray_trainer.py`](https://github.com/verl-project/verl/blob/v0.8.0/verl/trainer/ppo/ray_trainer.py#L1362).

**What it does.** A task is one unit of work in a training step, such as generating responses, computing advantages, or updating the actor. `fit` decides the order of the tasks and passes data between them. It does not run the model. Each heavy task is a remote call to GPU workers. Only the advantage computation, which is light, runs inside `fit`.

**When it is called.** Once per run. `python3 -m verl.trainer.main_ppo` reaches it in three steps:

1. `main` in [`verl/trainer/main_ppo.py`](https://github.com/verl-project/verl/blob/v0.8.0/verl/trainer/main_ppo.py#L38) reads the configuration and calls `run_ppo`.
2. `run_ppo` starts Ray and creates `TaskRunner`, a Ray actor with one CPU and no GPU.
3. `TaskRunner.run` builds the trainer, creates the GPU workers with `init_workers()`, and calls `fit()`.

The process that runs `fit` is the driver. `fit` returns when training reaches the last step.

### 1.1 Before the loop

Lines 1369–1420 run once, before the first step. Restore state loads a checkpoint if one exists, sends the weights to the rollout engine, and computes the starting epoch. Validate before training evaluates the model once and logs the metrics. If `val_only` is also set, `fit` returns there.

#### 1.1.1 Restore state (lines 1383–1389)

```python
self.global_steps = 0

# load checkpoint and update weights before doing anything
self._load_checkpoint()
self.checkpoint_manager.update_weights(self.global_steps)

current_epoch = self.global_steps // len(self.train_dataloader)
```

`_load_checkpoint()` looks for the latest checkpoint on the local disk. If one exists, it restores the step counter, the actor, and the position of the dataloader. Its behavior is set by `self.config.trainer.resume_mode`: `disable` skips the loading and trains from scratch; `auto`, the default, loads the latest checkpoint if one exists; `resume_path` loads the checkpoint given in `resume_from_path`. It loads the actor through `actor_rollout_wg`. Here `wg` means worker group: the GPU workers that hold the actor. GRPO has no critic, so `critic_wg` does not exist and no critic is loaded.

`update_weights` copies the actor's weights from the model engine to the rollout engine. There are three engines in verl: the model engine, the rollout engine, and the checkpoint engine. The model engine, such as FSDP, trains the actor. The rollout engine, such as vLLM, generates responses; it runs as one or more rollout replicas, and each replica is one inference server with its own copy of the model. When the two share GPUs (`backend="naive"`), the model engine hands the weights to the rollout engine directly. If not, the checkpoint engine moves the weights. In both cases `CheckpointEngineManager` coordinates the transfer.

#### 1.1.2 Validate before training (lines 1391–1400)

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

The table shows where each part runs and what it adds to `batch`.

| Part | Runs on | Adds to `batch` |
|---|---|---|
| 1.2.1 Build the batch | driver | `uid`, `temperature` |
| 1.2.2 Generate | rollout engine, reward workers | nothing yet; returns `gen_batch_output` |
| 1.2.3 Merge and balance | driver | `prompts`, `responses`, `response_mask`, `input_ids`, `attention_mask`, `position_ids`, `rm_scores`, `acc` |
| 1.2.4 Reward | driver | nothing; reads `rm_scores` |
| 1.2.5 Old log-probabilities | model engine (actor) | `old_log_probs` |
| 1.2.6 Reference log-probabilities | model engine (reference policy) | `ref_log_prob` |
| 1.2.7 Advantage | driver | `token_level_scores`, `token_level_rewards`, `advantages`, `returns` |
| 1.2.8 Update the actor | model engine (actor) | nothing; returns metrics |
| 1.2.9 Save a checkpoint | model engine, driver | nothing |
| 1.2.10 Sync the weights | model engine, rollout engine | nothing |
| 1.2.11 Validate | rollout engine, reward workers | nothing |

#### 1.2.1 Build the batch (lines 1435–1448)

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

A `DataProto` has three fields:

| Field | Type | Holds |
|---|---|---|
| `batch` | `TensorDict` | Tensors with one row per sample, such as `response_mask` and `old_log_probs`. |
| `non_tensor_batch` | dict of NumPy arrays | Other per-sample data, such as `uid`. |
| `meta_info` | dict | Values for the whole batch, such as `temperature`. |

The first lines wrap the dataloader's output in a `DataProto`, set the rollout temperature, and give each prompt a unique `uid`. At this point a prompt is still a list of chat messages (`raw_prompt`), not tokens; it is tokenized during generation. `_get_gen_batch` moves the fields that generation needs out of `batch` into a new `DataProto`, `gen_batch`. `batch` keeps only the reward fields and `uid`. `repeat` copies each prompt `rollout.n` times, with the copies next to each other (`interleave=True`). The copies share one `uid`, which later groups the responses to the same prompt.

#### 1.2.2 Generate (lines 1467–1471)

```python
with marked_timer("gen", timing_raw, color="red"):
    ...
    combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
    self.checkpoint_manager.sleep_replicas()
```

`marked_timer` records how long the block takes, under the name `gen`. `generate_sequences` sends the repeated prompts to the rollout replicas and returns the responses as a `DataProto`. It also computes the reward: each finished response is scored, and the scores come back in the same `DataProto` (see 1.2.4). When the two engines share GPUs, `sleep_replicas` then frees the rollout engine's GPU memory, both the weights and the KV cache, so that the model engine can use it for training. The weights are discarded, not moved to the CPU (vLLM sleep level 2, verl's default): the model engine holds the real copy, and `update_weights` writes the updated weights into the rollout engine at the end of the step.

#### 1.2.3 Merge and balance (lines 1496–1507)

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

`repeat` makes `batch` line up with the responses, row by row. `union` then adds the fields of `gen_batch_output` to `batch`: tensors go into `batch.batch`, arrays into `non_tensor_batch`, and values into `meta_info`. It adds fields, not rows. The two must have the same number of rows, and a field that exists in both must be equal.

`response_mask` needs two facts about the layout:

- Prompts are padded on the left and responses on the right, so the boundary between prompt and response is at the same column for every sample.
- `attention_mask` separates real tokens from padding. `response_mask` separates the real response tokens from everything else.

```
                prompt (left-padded)|  response (right-padded)
sample 1        pad  pad  p1   p2   |  r1   r2   r3   pad  pad
sample 2        p1   p2   p3   p4   |  r1   r2   r3   r4   r5

masks of sample 1
attention_mask  0    0    1    1    |  1    1    1    0    0
response_mask                       |  1    1    1    0    0
```

The generation output usually has `response_mask` already; `compute_response_mask` is the fallback. It takes the last columns of `attention_mask`, as many as the response length.

`_balance_batch` reorders the rows so that each data-parallel (DP) rank gets a similar amount of work. It estimates the work of a sample from its number of real tokens, and splits the samples into groups with the same number of samples and similar total work. A rank that gets a long sample also gets short ones. DP ranks run in step with each other, so without this a rank with long sequences would keep the others waiting. Section 2 discusses it in detail.

#### 1.2.4 Reward (lines 1518–1525)

```python
with marked_timer("reward", timing_raw, color="yellow"):
    # compute reward model score
    if self.use_rm and "rm_scores" not in batch.batch.keys():
        batch_reward = self._compute_reward_colocate(batch)
        batch = batch.union(batch_reward)

    # extract reward_tensor and reward_extra_infos_dict for training
    reward_tensor, reward_extra_infos_dict = extract_reward(batch)
```

The score is already in `batch`: it was computed during generation. When a response finishes, a reward worker decodes it and calls the scoring function with the response text and the ground truth. For GSM8K, the function extracts the final answer and compares it with the ground truth. `extract_reward` only reads the result:

- `reward_tensor` is `batch.batch["rm_scores"]`. Its shape is (batch size, response length), the same as `response_mask`. It is zero everywhere except at the last real response token, which holds the score.
- `reward_extra_infos_dict` holds the other values that the scoring function returns, one per response. If the function returns a single number, the dict has one key, `acc`.

The `if` branch runs only with a learned reward model (`reward.reward_model.enable`). A rule-based reward, as for GSM8K, skips it.

#### 1.2.5 Old log-probabilities (lines 1542–1567)

```python
with marked_timer("old_log_prob", timing_raw, color="blue"):
    old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
    ...
    batch = batch.union(old_log_prob)
```

`_compute_old_log_prob` runs one forward pass of the actor over all responses, without gradients, and returns the log-probability of every response token. `union` adds it to `batch` as `old_log_probs`. It is the red term in the GRPO objective:

$$
\mathcal{J} = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
\min\!\big(\rho_{i,t}\hat{A}_i,\ \mathrm{clip}(\rho_{i,t},\,1\pm\varepsilon)\,\hat{A}_i\big)
\;-\; \beta\, D_{\mathrm{KL}}\!\big(\pi_\theta \,\|\, \pi_{\mathrm{ref}}\big)
$$

$$
\rho_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q,\, o_{i,<t})}{\textcolor{#e0433a}{\pi_{\mathrm{old}}(o_{i,t} \mid q,\, o_{i,<t})}}
$$

When `actor.ppo_mini_batch_size` is smaller than `data.train_batch_size`, one step makes several gradient updates on the same responses. After the first update, the actor is no longer the policy that generated them, so the later updates are off-policy. $\rho_{i,t}$ corrects for this. verl recomputes $\pi_{\mathrm{old}}$ with the model engine instead of reusing the rollout engine's values, because the two engines give slightly different numbers for the same weights.

#### 1.2.6 Reference log-probabilities (lines 1576–1580)

```python
if self.use_reference_policy:
    # compute reference log_prob
    with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
        ref_log_prob = self._compute_ref_log_prob(batch)
        batch = batch.union(ref_log_prob)
```

`_compute_ref_log_prob` runs one forward pass of the reference policy, the frozen starting model, and `union` adds the result to `batch`. It is the red term in the KL penalty of the objective:

$$
\beta\, D_{\mathrm{KL}}\!\big(\pi_\theta \,\|\, \textcolor{#e0433a}{\pi_{\mathrm{ref}}}\big)
$$

The step runs only when the KL penalty is on (`actor.use_kl_loss` or `algorithm.use_kl_in_reward`). The numerator of $\rho_{i,t}$, $\pi_\theta$, is not computed here. It changes at every gradient update, so the actor computes it during the update (1.2.8).

#### 1.2.7 Advantage (lines 1588–1633)

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

`token_level_scores` is the raw score from 1.2.4, and `token_level_rewards` is what the advantage uses. They differ only with `algorithm.use_kl_in_reward`, which subtracts a KL penalty from the score. GRPO normally puts the KL term in the loss instead (`actor.use_kl_loss`), so the two are equal.

`compute_advantage` runs on the driver, not on a GPU worker. For GRPO it sums the reward of each response, groups the responses by `uid`, and computes the red term of the objective:

$$
\textcolor{#e0433a}{\hat{A}_i} = \frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}
$$

The group mean is always subtracted. The division by the standard deviation is on by default; `norm_adv_by_std_in_grpo=False` turns it off, as in Dr. GRPO. Every token of a response gets the same advantage. `gamma` and `lam` belong to GAE, the estimator for PPO with a critic. GRPO does not use them.

#### 1.2.8 Update the actor (lines 1647–1649)

```python
# update actor
with marked_timer("update_actor", timing_raw, color="red"):
    actor_output = self._update_actor(batch)
```

`_update_actor` sends the whole `batch` to `actor_rollout_wg` in one remote call. The workers split it into mini-batches. For each mini-batch they compute $\pi_\theta$ with gradients, build the objective from the fields prepared above (`old_log_probs`, the reference log-probabilities, `advantages`, `response_mask`), and take one optimizer step. Only metrics come back, such as the loss and the gradient norm. Section 2 discusses what happens inside the workers.

#### 1.2.9 Save a checkpoint (lines 1663–1671)

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

ESI stands for Elastic Server Instance: a cloud instance that is rented for a fixed time and shuts down when the time is over. `should_save_ckpt_esi` reads the expiration time from an environment variable:

- Volcano Engine Machine Learning Platform (veMLP): `MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP`
- AWS SageMaker: `SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP`

It returns true when the remaining time is no more than the longest step so far, plus the time to save a checkpoint (60 seconds by default), plus `trainer.esi_redundant_time`. Then `fit` saves a checkpoint at once, before the instance shuts down. On RunPod neither variable is set, and the function returns false.

#### 1.2.10 Sync the weights (lines 1673–1675)

```python
# update weights from trainer to rollout
with marked_timer("update_weights", timing_raw, color="red"):
    self.checkpoint_manager.update_weights(self.global_steps)
```

This is the same call as in 1.1.1. It copies the updated weights from the model engine to the rollout engine. It also wakes the rollout engine, which has been asleep since 1.2.2: the GPU memory for its weights and its KV cache is allocated again. After this call the rollout engine is ready to generate for the next step.

#### 1.2.11 Validate (lines 1685–1693)

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

The same `_validate()` as in 1.1.2. It runs every `trainer.test_freq` steps, and on the last step.

### 1.3 After the loop

## 2. Workers

## 3. GPU placement

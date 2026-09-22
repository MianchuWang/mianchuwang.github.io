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

**What it does.** A task is one unit of work in a training step, such as generating responses, computing advantages, or updating the actor. `fit` decides the order of the tasks and passes data between them. It does not run the model. Each heavy task is a remote call to GPU workers. Light work, such as the advantage computation, runs inside `fit`.

**When it is called.** Once per run. `python3 -m verl.trainer.main_ppo` reaches it in three steps:

1. `main` in [`verl/trainer/main_ppo.py`](https://github.com/verl-project/verl/blob/v0.8.0/verl/trainer/main_ppo.py#L38) reads the configuration and calls `run_ppo`.
2. `run_ppo` starts Ray and creates `TaskRunner`, a Ray actor with one CPU and no GPU.
3. `TaskRunner.run` builds the trainer, creates the GPU workers with `init_workers()`, and calls `fit()`.

The process that runs `fit` is the driver. `fit` returns when training reaches the last step.

### 1.1 Before the loop

Lines 1369–1420 run once, before the first step. Restore state loads a checkpoint if one exists, sends the weights to the rollout engine, and computes the starting epoch. Validate before training evaluates the model once and logs the metrics. If `val_only` is also set, `fit` returns there.

#### 1.1.1 Restore state (lines 1383–1389)

```python lines=1383-1389
self.global_steps = 0

# load checkpoint and update weights before doing anything
self._load_checkpoint()
self.checkpoint_manager.update_weights(self.global_steps)

current_epoch = self.global_steps // len(self.train_dataloader)
```

`_load_checkpoint()` looks for the latest checkpoint on the local disk. If one exists, it restores the step counter, the actor, and the position of the dataloader. Its behavior is set by `self.config.trainer.resume_mode`: `disable` skips the loading and trains from scratch; `auto`, the default, loads the latest checkpoint if one exists; `resume_path` loads the checkpoint given in `resume_from_path`. It loads the actor through `actor_rollout_wg`. Here `wg` means worker group: the GPU workers that hold the actor. GRPO has no critic, so `critic_wg` does not exist and no critic is loaded.

`update_weights` copies the actor's weights from the model engine to the rollout engine. There are three engines in verl: the model engine, the rollout engine, and the checkpoint engine. The model engine, such as FSDP, trains the actor. The rollout engine, such as vLLM, generates responses. When the two share GPUs (`backend="naive"`), the model engine hands the weights to the rollout engine directly. If not, the checkpoint engine moves the weights. In both cases `CheckpointEngineManager` coordinates the transfer.

#### 1.1.2 Validate before training (lines 1391–1400)

```python lines=1391-1400
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

Lines 1422–1770 are the loop. Each pass of the inner loop is one training step. The parts below are the ones a GRPO step runs, in order. Branches that GRPO does not use, such as the critic and REMAX, are left out.

```python lines=1422-1423,1426-1427
for epoch in range(current_epoch, self.config.trainer.total_epochs):
    for batch_dict in self.train_dataloader:
        metrics = {}
        timing_raw = {}
```

`metrics` collects the numbers to log at the end of the step, and `timing_raw` the time of each part.

`batch_dict` comes from the dataloader. `RLHFDataset` reads one row of the parquet file and adds a few fields; `collate_fn` stacks a batch of rows. For GSM8K a row has:

| Field | Holds |
|---|---|
| `raw_prompt` | The prompt as a list of chat messages: `[{"role": "user", "content": question}]`. It is not tokenized yet; the agent loop applies the chat template. |
| `data_source` | `"openai/gsm8k"`. The scoring function is chosen by it. |
| `reward_model` | `{"style": "rule", "ground_truth": "72"}`: the answer to score against. |
| `prompt`, `ability`, `extra_info`, `index`, `tools_kwargs`, `interaction_kwargs`, `dummy_tensor` | Not used by GRPO on GSM8K. `dummy_tensor` is a one-byte placeholder that keeps `DataProto.batch` from being empty. |

#### 1.2.1 Build the batch (lines 1435–1448)

```python lines=1435-1448
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
| <span class="code-blue">`batch`</span> | `TensorDict` | Tensors with one row per sample, such as `response_mask` and `old_log_probs`. |
| <span class="code-green">`non_tensor_batch`</span> | dict of NumPy arrays | Other per-sample data, such as `uid`. |
| <span class="code-amber">`meta_info`</span> | dict | Values for the whole batch, such as `temperature`. |

The first lines wrap the dataloader's output in a `DataProto`, set the rollout temperature, and give each prompt a unique `uid`. At this point a prompt is still a list of chat messages (`raw_prompt`), not tokens. `_get_gen_batch` splits `batch` in two. `gen_batch` gets every non-tensor field and goes to generation. `batch` keeps `dummy_tensor` and the four fields that the reward and the advantage need: `data_source`, `reward_model`, `extra_info` and `uid`. These four are in both, because the reward is computed during generation (1.2.4) and needs the ground truth. `gen_batch` is consumed by generation and does not come back, so `batch` is the driver's own copy; 1.2.3 joins the generated responses to it. `repeat` copies each prompt `rollout.n` times, with the copies next to each other (`interleave=True`). The copies share one `uid`, which later groups the responses to the same prompt.

#### 1.2.2 Generate (lines 1467–1471)

```python lines=1467,1470-1471
with marked_timer("gen", timing_raw, color="red"):
    ...
    combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
    self.checkpoint_manager.sleep_replicas()
```

`marked_timer` records how long the block takes, under the name `gen`. For GRPO, `combined_gen_batch` is just `gen_batch_output`. `generate_sequences` sends the repeated prompts to the rollout engine and returns the responses as a `DataProto`. Each finished response is scored, and the scores come back in the same `DataProto`, as the tensor `rm_scores` in its `batch` field (see 1.2.4).

When the two engines share GPUs, `sleep_replicas` then frees the rollout engine's GPU memory, both the weights and the KV cache, so that the model engine can use it for training.

#### 1.2.3 Merge and balance (lines 1496–1510)

```python lines=1496-1510
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

# compute global_valid tokens
batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
```

`repeat` makes `batch` line up with the responses, row by row. `union` then adds the fields of `gen_batch_output` to `batch`: the tensors `prompts`, `responses`, `response_mask`, `input_ids`, `attention_mask`, `position_ids` and `rm_scores`, and the arrays `acc` and `__num_turns__`.

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

The generation output usually has `response_mask` already; `compute_response_mask` is the fallback.

`_balance_batch` reorders the rows so that each data-parallel (DP) rank gets a similar amount of work. It estimates the work of a sample from its number of real tokens, and splits the samples into groups with the same number of samples and similar total work. Without this a rank with long sequences would keep the others waiting. Section 2 discusses it in detail.

`global_token_num` is the number of real tokens in each sample, prompt and response together, as a list with one entry per row. It is computed after balancing, so its order matches the rows. The workers use it to estimate the FLOPs of a pass over the batch, which gives the MFU metric.

#### 1.2.4 Reward (lines 1518–1525)

```python lines=1518-1525
with marked_timer("reward", timing_raw, color="yellow"):
    # compute reward model score
    if self.use_rm and "rm_scores" not in batch.batch.keys():
        batch_reward = self._compute_reward_colocate(batch)
        batch = batch.union(batch_reward)

    # extract reward_tensor and reward_extra_infos_dict for training
    reward_tensor, reward_extra_infos_dict = extract_reward(batch)
```

The `if` branch runs only with a learned reward model that shares GPUs with training. A rule-based reward, as for GSM8K, skips it: the score is already in `batch`, computed during generation. When a response finishes, a reward worker decodes it and calls the scoring function with the response text and the ground truth. For GSM8K, the function extracts the final answer and compares it with the ground truth. `extract_reward` only reads the result:

- `reward_tensor` is `batch.batch["rm_scores"]`. Its shape is (batch size, response length), the same as `response_mask`. It is zero everywhere except at the last real response token, which holds the score.
- `reward_extra_infos_dict` holds the other values that the scoring function returns, one per response. If the function returns a single number, the dict has one key, `acc`.

#### 1.2.5 Old log-probabilities (lines 1542–1567)

```python lines=1542-1543,1567
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

```python lines=1576-1580
if self.use_reference_policy:
    # compute reference log_prob
    with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
        ref_log_prob = self._compute_ref_log_prob(batch)
        batch = batch.union(ref_log_prob)
```

`_compute_ref_log_prob` runs one forward pass of the reference policy, the frozen starting model, and `union` adds the result to `batch` as `ref_log_prob`. It is the red term in the KL penalty of the objective:

$$
\beta\, D_{\mathrm{KL}}\!\big(\pi_\theta \,\|\, \textcolor{#e0433a}{\pi_{\mathrm{ref}}}\big)
$$

The step runs only when the KL penalty is on (`actor.use_kl_loss` or `algorithm.use_kl_in_reward`). The numerator of $\rho_{i,t}$, $\pi_\theta$, is not computed here. It changes at every gradient update, so the actor computes it during the update (1.2.8).

#### 1.2.7 Advantage (lines 1588–1633)

```python lines=1588,1591,1596-1603,1620-1633
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

```python lines=1647-1649
# update actor
with marked_timer("update_actor", timing_raw, color="red"):
    actor_output = self._update_actor(batch)
```

`_update_actor` sends the whole `batch` to `actor_rollout_wg` in one remote call. The workers split it into mini-batches. For each mini-batch they compute $\pi_\theta$ with gradients, build the objective from the fields prepared above (`old_log_probs`, the reference log-probabilities, `advantages`, `response_mask`), and take one optimizer step. Only metrics come back, such as the loss and the gradient norm. Section 2 discusses what happens inside the workers.

#### 1.2.9 Save a checkpoint (lines 1663–1671)

```python lines=1663-1671
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

ESI stands for Elastic Server Instance: a cloud instance that is rented for a fixed time and shuts down when the time is over. `should_save_ckpt_esi` reads the expiration time from an environment variable. It returns true when the remaining time is no more than the longest step so far, plus the time to save a checkpoint (60 seconds by default), plus `trainer.esi_redundant_time`. Then `fit` saves a checkpoint at once, before the instance shuts down. On RunPod it is not set, and the function returns false.

#### 1.2.10 Sync the weights (lines 1673–1675)

```python lines=1673-1675
# update weights from trainer to rollout
with marked_timer("update_weights", timing_raw, color="red"):
    self.checkpoint_manager.update_weights(self.global_steps)
```

This is the same call as in 1.1.1. It copies the updated weights from the model engine to the rollout engine. It also wakes the rollout engine, which has been asleep since 1.2.2: the GPU memory for its weights and its KV cache is allocated again. After this call the rollout engine is ready to generate for the next step.

#### 1.2.11 Validate (lines 1685–1693)

```python lines=1685-1693
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

#### 1.2.12 Log the metrics and advance the step (lines 1695–1770)

```python lines=1709-1720,1731,1734,1737,1750-1755,1758-1761
steps_duration = timing_raw["step"]
self.max_steps_duration = max(self.max_steps_duration, steps_duration)

# training metrics
metrics.update(
    {
        "training/global_step": self.global_steps,
        "training/epoch": epoch,
    }
)
# collect metrics
metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
...
metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
...
metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
...
metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
...
logger.log(data=metrics, step=self.global_steps)

progress_bar.update(1)
self.global_steps += 1

if is_last_step:
    ...
    self._shutdown_dump_executor()
    pprint(f"Final validation metrics: {last_val_metrics}")
    progress_bar.close()
    return
```

`timing_raw["step"]` is the time of the whole step, from generation to validation. `max_steps_duration` is the longest step so far; the ESI check in 1.2.9 uses it.

The four helpers add the numbers that a run reports at every step:

- `compute_data_metrics`: statistics of the batch. Mean, max and min of the scores, rewards, advantages and returns, and of the prompt and response lengths, plus the share of responses that hit the length limit (`response_length/clip_ratio`).
- `compute_timing_metrics`: the time of each part in seconds (`timing_s/gen`, `timing_s/update_actor`, and so on), and the same per token.
- `compute_throughout_metrics`: the total number of tokens in the step, the time per step, and tokens per second per GPU (`perf/throughput`).
- `compute_variance_proxy_metrics`: an estimate of the variance of the policy gradient, from `old_log_probs` and the advantages.

`logger.log` sends the metrics to the backends listed in `trainer.logger`, such as the console and Weights & Biases, under the current step number.

To sum up 1.2, the table shows where each part runs and what it adds to `batch`.

| Part | Runs on | Adds to `batch` |
|---|---|---|
| 1.2.1 Build the batch | driver | <span class="code-green">`uid`</span>, <span class="code-amber">`temperature`</span> |
| 1.2.2 Generate | rollout engine, reward workers | nothing yet; returns `gen_batch_output` |
| 1.2.3 Merge and balance | driver | <span class="code-blue">`prompts`, `responses`, `response_mask`, `input_ids`, `attention_mask`, `position_ids`, `rm_scores`</span>, <span class="code-green">`acc`, `__num_turns__`</span>, <span class="code-amber">`global_token_num`</span> |
| 1.2.4 Reward | driver | nothing; reads <span class="code-blue">`rm_scores`</span> |
| 1.2.5 Old log-probabilities | model engine (actor) | <span class="code-blue">`old_log_probs`</span> |
| 1.2.6 Reference log-probabilities | model engine (reference policy) | <span class="code-blue">`ref_log_prob`</span> |
| 1.2.7 Advantage | driver | <span class="code-blue">`token_level_scores`, `token_level_rewards`, `advantages`, `returns`</span> |
| 1.2.8 Update the actor | model engine (actor) | nothing; returns metrics |
| 1.2.9 Save a checkpoint | model engine, driver | nothing |
| 1.2.10 Sync the weights | model engine, rollout engine | nothing |
| 1.2.11 Validate | rollout engine, reward workers | nothing |
| 1.2.12 Log the metrics and advance the step | driver | nothing |

The color of a name is the field of the `DataProto` that holds it: <span class="code-blue">`batch`</span> for tensors, <span class="code-green">`non_tensor_batch`</span> for NumPy arrays, and <span class="code-amber">`meta_info`</span> for values of the whole batch.

## 2. Workers

### 2.1 One worker per GPU

GRPO runs on one resource pool, `global_pool`, with one worker process per GPU. Each worker is an `ActorRolloutRefWorker`: the actor, the rollout engine and the reference policy live in one process. Three steps set this up. `main_ppo.py` decides the roles and the pool (2.1.1). `init_workers` in `ray_trainer.py` turns the pool into a worker group, `actor_rollout_wg`, and calls `init_model` on every worker (2.1.2). `init_model` builds the three engines inside each process (2.1.3).

#### 2.1.1 Roles and the resource pool (`add_actor_rollout_worker` and `init_resource_pool_mgr`, main_ppo.py, lines 132–190)

```python lines=132-134,140-145
actor_rollout_cls = ActorRolloutRefWorker
ray_worker_group_cls = RayWorkerGroup

...
if need_reference_policy(config) and not ref_in_actor:
    role = Role.ActorRolloutRef
else:
    role = Role.ActorRollout
self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
self.mapping[role] = "global_pool"
```

`actor_rollout_cls` is `ActorRolloutRefWorker`, the class that every GRPO worker runs. A role is a member of the `Role` enum: what a worker plays, such as `Actor`, `Critic` or `RefPolicy`, or a combination such as `ActorRolloutRef`. Without LoRA, `ref_in_actor` is false, so the role is `Role.ActorRolloutRef` whenever the KL penalty needs a reference policy. Lines 144 and 145 fill two tables: the worker class of the role, wrapped by `ray.remote` so that it can be created in another process, and the resource pool of the role, by name. Nothing is created yet.

```python lines=161-164,190
global_pool_id = "global_pool"
resource_pool_spec = {
    global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
}
...
resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
```

`resource_pool_spec` says how many GPUs each pool has, one entry per node. GRPO uses one pool, `global_pool`, with every GPU. `ResourcePoolManager` joins the two tables. For a role, it reads the pool name from `mapping` and takes the GPUs of that pool from `resource_pool_spec`. 2.1.2 creates the pools from it.

#### 2.1.2 Create the worker group (`init_workers`, ray_trainer.py, lines 782–898)

```python lines=782,787-792,794-796,861,864-871,894-898
self.resource_pool_manager.create_resource_pool()
...
actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
if self.hybrid_engine:
    actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
    actor_rollout_cls = RayClassWithInitArgs(
        cls=self.role_worker_mapping[actor_role],
        config=self.config.actor_rollout_ref,
        ...
        role=str(actor_role),
    )
    self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
...
for resource_pool, class_dict in self.resource_pool_to_cls.items():
    ...
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    wg_dict = self.ray_worker_group_cls(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_dict_cls,
        **wg_kwargs,
    )
    spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
    all_wg.update(spawn_wg)
...
self.actor_rollout_wg = all_wg[str(actor_role)]
self.actor_rollout_wg.init_model()

if self.ref_in_actor:
    self.ref_policy_wg = self.actor_rollout_wg
```

#### 2.1.3 Inside a worker (`ActorRolloutRefWorker.__init__` and `init_model`, engine_workers.py, lines 441–629)

```python lines=441-445,448-455
def __init__(
    self, config: DictConfig, role: str, distillation_config: Optional[DistillationConfig] = None, **kwargs
):
    Worker.__init__(self)
    self.config = config
    ...
    self.role = role
    self.actor: TrainingWorker = None
    self.ref: TrainingWorker = None
    self.rollout: BaseRollout = None
    assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
    self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
    self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
    self._is_ref = self.role in ["ref", "actor_rollout_ref"]
```

```python lines=500-504,537-542,585-592,608-611,618-619,627-629
def init_model(self):
    model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)

    # 1. build reference model
    if "ref" in self.role:
        ...
        self.ref = TrainingWorker(config=ref_training_config)
        self.ref.reset()
        self.set_dispatch_collect(mesh_name="ref", **self.ref.get_dispatch_collect())

    # 2. build actor model
    if "actor" in self.role:
        ...
        self.actor = TrainingWorker(config=actor_training_config)
        self.actor.reset()
        self.actor.set_loss_fn(self.loss_fn)
        self.set_dispatch_collect(mesh_name="actor", **self.actor.get_dispatch_collect())

    # 3. build rollout engine
    if "rollout" in self.role:
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        ...
        rollout_cls: type[BaseRollout] = get_rollout_class(rollout_config.name, rollout_config.mode)
        self.rollout = rollout_cls(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        ...
    # 4. build checkpoint engine
    if "actor" in self.role:
        ...
        self.checkpoint_engine = CheckpointEngineRegistry.new(
            backend, is_master=(torch.distributed.get_rank() == 0), bucket_size=bucket_size, **engine_kwargs
        )
```

### 2.2 One remote call

### 2.3 Inside the model engine

### 2.4 Inside the rollout engine

## 3. GPU placement

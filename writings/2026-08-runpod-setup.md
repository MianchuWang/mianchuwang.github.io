---
title: "Setting Up RunPod for verl"
date: 2026-08-27
tags: ["Agent Runbook", RunPod, verl]
summary: An agent-executable runbook for setting up verl on a fresh RunPod pod.
---

This is a runbook for an **AI agent with SSH access to the pod** (`Host runpod` in the user's `~/.ssh/config`). Sections 1–3 are console/human steps; sections 4–5 are shell steps the agent executes over SSH, all idempotent. The contract: **the Docker image owns the heavy stack** (torch, vLLM, flash-attn — version-matched, precompiled); **`/workspace` owns everything personal**. This page builds the verl *environment* only — project code, data pipelines, and training configs belong to each project's own repo.

> [!warning] **Agent scope**: set up and verify the environment (through §4), then stop and report back. Do **not** launch training or any long-running job on the user's behalf — §5 documents the launch pattern for the user to invoke, or for the agent only when the user explicitly asks.

## 1. Template (console, once)

- **Container image**: `verlai/verl:vllm011.latest`
- **Container start command** — must override *both* entrypoint and cmd (the image inherits vLLM's ENTRYPOINT; a plain `bash -c` start command becomes arguments to the vLLM server and sshd never starts):

```json
{"entrypoint": ["bash", "-c"], "cmd": ["apt-get update && apt-get install -y openssh-server && mkdir -p ~/.ssh && echo \"$PUBLIC_KEY\" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && service ssh start && sleep infinity"]}
```

- Container disk 50 GB; persistent storage mounted at `/workspace`; **expose TCP port 22** (no port 22 → no "SSH over exposed TCP" → no remote dev).

## 2. Pod (console, per deployment)

1. Attach persistent storage first if a network volume exists (EU-SE-1 no longer offers new network volumes; the template's 100 GB volume disk is the fallback — it **dies with the pod**, so git/wandb are the real persistence).
2. Filter for **Public IP**; proxy-only machines cannot carry VS Code / direct SSH.
3. GPU: A40 (~$0.44/h/GPU, Ampere — bf16 yes, FP8 no). GPU count is fixed at deploy; scale by redeploying.
4. After boot, trust env vars over the pod card: `RUNPOD_MEM_GB` is the real CPU RAM cap (**50 GB per GPU**; the card's larger number is the host's).

Pods can also be deployed entirely via the GraphQL API — **`PUBLIC_KEY` must then be passed explicitly as an env var** (nothing injects it for you; a pod without it accepts no SSH key and greets you with a password prompt):

```bash
curl -s -X POST https://api.runpod.io/graphql -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -d '{"query":"mutation { podFindAndDeployOnDemand(input: { cloudType: ALL, gpuTypeId: \"NVIDIA A40\", gpuCount: 1, name: \"my-pod\", templateId: \"<template-id>\", containerDiskInGb: 50, volumeInGb: 100, supportPublicIp: true, env: [{ key: \"PUBLIC_KEY\", value: \"<contents of id_ed25519.pub>\" }] }) { id } }"}'
```

Poll `pod { runtime { ports } }` for the SSH ip/port once it boots.

## 3. SSH access

`~/.ssh/config` on the user's machine (IP/port change per pod — read them from the pod's Connect panel, or `RUNPOD_PUBLIC_IP` / `RUNPOD_TCP_PORT_22` in `/proc/1/environ`):

```
Host runpod
    HostName <RUNPOD_PUBLIC_IP>
    Port <RUNPOD_TCP_PORT_22>
    User root
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ForwardAgent yes
    StrictHostKeyChecking accept-new
    ServerAliveInterval 30
    ServerAliveCountMax 6
```

Verify: `ssh -o BatchMode=yes runpod 'echo ok'`. If **Permission denied**, the pod is missing `PUBLIC_KEY`. Fix it via the API — add the env with `podEditJob`, then `podStop` + `podResume` (the port changes) — or, as a last resort, paste the key into `~/.ssh/authorized_keys` from the Web Terminal.

With `ForwardAgent yes` and the key loaded locally (`ssh-add`), the pod can also push to GitHub (use SSH remotes, not HTTPS) and SSH onward to sibling pods — no credentials ever land on the pod.

## 4. Environment (agent, over SSH, idempotent)

**Secrets**: `/workspace/setup.sh` needs `WANDB_API_KEY` and an account-level `RUNPOD_API_KEY` (the auto-injected pod-scoped key can `stop pod` but fails read queries). Ask the user for both if no existing `setup.sh` has them. Never commit them to git.

**4.1 — `/workspace/setup.sh`** (single entrypoint for every shell; `~/.bashrc` dies with the container):

```bash
export PATH=/workspace/bin:$PATH
export UV_CACHE_DIR=/workspace/.cache/uv
export HF_HOME=/workspace/.cache/huggingface
export XDG_CACHE_HOME=/workspace/.cache          # vLLM, torch.compile, flashinfer
export TRITON_CACHE_DIR=/workspace/.cache/triton # triton ignores XDG
export WANDB_API_KEY=<ASK_USER>
export RUNPOD_API_KEY=<ASK_USER>                 # account-level; must come AFTER any line exporting the injected key
export $(cat /proc/1/environ | tr "\0" "\n" | grep -E "^RUNPOD_POD_ID=" | xargs)
[ -f /workspace/.venv/bin/activate ] && source /workspace/.venv/bin/activate  # venv arrives in §4.3
```

The `/proc/1/environ` extraction exists because SSH sessions do not inherit container env vars. Before appending anything to `setup.sh`, `grep` for an existing export of the same name — a later duplicate silently wins.

**4.2 — per-pod bootstrap** (container disk, redo after every pod recreation):

```bash
echo 'source /workspace/setup.sh' >> ~/.bashrc
apt-get update && apt-get install -y tmux
mkdir -p /workspace/bin && wget -qO /workspace/bin/runpodctl https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 && chmod +x /workspace/bin/runpodctl
```

**4.3 — thin venv** at `/workspace/.venv` (sees the image's stack via `--system-site-packages`; holds only additions; independent of any project checkout):

```bash
python3 -m venv --system-site-packages /workspace/.venv
source /workspace/setup.sh
uv pip install --no-deps "verl==0.8.0"
```

Two hard rules: **never let torch / vllm / flash-attn (or `nvidia-*`) into the venv** — install torch-adjacent packages with `--no-deps` and watch the resolver output (recovery: `uv pip uninstall torch`); **verl's version must match the image's vLLM** — PyPI `verl==0.9.0` fails to import against this image's vLLM 0.11, `0.8.0` works.

**4.4 — verification** (all must pass):

```bash
source /workspace/setup.sh
python -c "import verl, torch, vllm; print(verl.__version__, torch.__version__, vllm.__version__)"
python -c "import torch; print(torch.__file__)"   # must be /usr/local/..., NOT /workspace/.venv/...
echo $RUNPOD_POD_ID && runpodctl get pod $RUNPOD_POD_ID
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

**4.5 — smoke test** (proves verl actually *runs*, using its official GSM8K quickstart data):

```bash
curl -fsSL https://raw.githubusercontent.com/volcengine/verl/v0.8.0/examples/data_preprocess/gsm8k.py -o /tmp/gsm8k.py
python /tmp/gsm8k.py --local_dir /workspace/data/gsm8k
python -m verl.trainer.main_ppo --cfg job > /dev/null && echo VERL_CONFIG_OK
```

Environment is done. Clone the project repo (check **all branches** — work may live on a feature branch; cloning uses the user's forwarded agent, so run it from an SSH session, not the Web Terminal) and follow its README from here.

**Cloning a whole pod** (fan-out for parallel runs): a sibling pod on the same account can be filled from a finished one in minutes — env, venv, model cache and all — instead of rebuilding:

```bash
ssh -A pod1 'cd /workspace && tar czf - setup.sh bin verl verl-playground data .venv .cache/huggingface \
  | ssh -o StrictHostKeyChecking=accept-new -p <port2> root@<ip2> "cd /workspace && tar xzf -"'
```

(`tar` ownership warnings on the network volume are harmless.) Then per-pod bootstrap (§4.2) on the new pod and it is ready.

## 5. Running

Long runs live in tmux, never a bare SSH terminal:

```bash
tmux new -s train      # or: tmux new-session -d -s train "<command>"
tmux attach -t train   # reconnect after any disconnect
```

Launch pattern — log to the volume, auto-stop the pod when the run ends *whether it succeeds or crashes* (`;`, not `&&`):

```bash
<run-command> 2>&1 | tee /workspace/train.log; runpodctl stop pod $RUNPOD_POD_ID
```

`runpodctl get pod $RUNPOD_POD_ID` is the harmless auth probe; `runpodctl stop pod` takes effect immediately with no confirmation. Verify launch health within the first minutes: every GPU shows load in `nvidia-smi`, and step timing appears on wandb.

---
title: "Setting Up RunPod: A Memo"
date: 2026-08-28
tags: [Infrastructure, RunPod]
summary: A practical guide to configuring RunPod GPU pods for research workflows.
---

This memo records my RunPod setup, kept short for future reference: deploy a pod with a persistent network volume, connect VS Code and Claude Code over SSH, and keep the whole uv-based Python environment on the volume — so a brand-new pod is ready to work in about a minute, with nothing to reinstall.

本备忘录记录了我的 RunPod 配置流程：部署挂载持久化 network volume 的 pod，通过 SSH 连接 VS Code 与 Claude Code，并把基于 uv 的 Python 环境完整放在 volume 上——新开 pod 约一分钟即可恢复工作，无需重新安装任何东西。

## Storage

RunPod offers three storage options when deploying a pod:

- **Network volume** — persistent storage independent of any pod. Survives pod termination and can be attached to a new pod in the same datacenter. Mounted at `/workspace`. Billed per GB per month even when no pod is running.
- **Volume disk** — persistent storage tied to one pod. Survives stops and restarts, but is deleted when the pod is terminated. Also mounted at `/workspace`.
- **None (container disk only)** — everything lives on the container disk and is wiped on every stop or restart. Cheapest; fine for throwaway experiments.

Rule of thumb: use a network volume for anything you want to keep (code, datasets, checkpoints); the container disk is scratch space.

When deploying later pods, pick the **existing** volume in the storage dropdown instead of "Automatically create" — a fresh volume means an empty `/workspace`. A volume is pinned to its datacenter, so later pods must be deployed there too.

> [!info] An example deployment, for reference:
>
> - **Template**: RunPod PyTorch 2.8.0 (CUDA 13.0)
> - **GPU**: 1× RTX PRO 4500 — 32 GB VRAM, 62 GB RAM, 12 vCPU
> - **Disk**: 30 GB container disk + 90 GB network volume
> - **Cost**: `$0.73/hour` (almost all GPU; storage is about a cent per hour), billed per millisecond. The network volume alone bills `~$6/month` while no pod is running

## Connection

Add your SSH public key once, under RunPod → Settings → SSH Public Keys; every pod deployed afterwards accepts it.

The pod's Connect panel shows two SSH options — use **SSH over exposed TCP** (the proxy one doesn't support VS Code). It gives a command like:

```
ssh root@194.26.196.6 -p 22117 -i ~/.ssh/id_ed25519
```

Map it into `~/.ssh/config`: the user is the part before `@` (always `root`), the hostname is the IP after `@`, and the port is the number after `-p`. Both change with every pod:

```
Host runpod
    HostName 194.26.196.6
    Port 22117
    User root
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ForwardAgent yes
    StrictHostKeyChecking accept-new
    ServerAliveInterval 30
    ServerAliveCountMax 6
```

The extra lines are quality-of-life: `accept-new` auto-accepts the host key of a brand-new pod (otherwise VS Code stalls on a confirmation prompt every time), the keep-alives stop idle sessions from dropping, and `ForwardAgent` lets git on the pod authenticate with the local SSH key — private repos clone without storing any credentials on the pod.

Then in VS Code: **Remote-SSH: Connect to Host** → `runpod`, and open `/workspace`.

VS Code installs its server component into `~/.vscode-server`, which is erased with the container disk — it re-downloads on every fresh pod. To keep it (and its extensions) on the volume instead, add to the local `settings.json`, keyed by the `Host` name:

```json
"remote.SSH.serverInstallPath": {
    "runpod": "/workspace/.vscode-server"
}
```

## Environment

Anything installed outside `/workspace` (system packages, the default conda, `~/.bashrc`) lives on the container disk and disappears with the pod. The trick is to install the whole environment onto the network volume once, then reattach it to every new pod.

The RunPod PyTorch template already ships uv, and the image restores it on every pod — no install needed. What must persist are its download dirs, so write a small script pointing them (and Claude Code, below) at the volume:

```bash
cat > /workspace/setup.sh << 'EOF'
export PATH=/workspace/bin:$PATH
export UV_PYTHON_INSTALL_DIR=/workspace/.uv/python
export UV_CACHE_DIR=/workspace/.cache/uv
export HF_HOME=/workspace/.cache/huggingface
export CLAUDE_CONFIG_DIR=/workspace/.claude
EOF
```

(If a template ever lacks uv: `curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/workspace/bin sh` puts it on the volume.)

With `setup.sh` in place, install Claude Code — also once. Copy the binary to `/workspace/bin` (already on the `PATH`), and sign in; `CLAUDE_CONFIG_DIR` keeps the credentials on the volume:

```bash
source /workspace/setup.sh
curl -fsSL https://claude.ai/install.sh | bash
mkdir -p /workspace/bin && cp ~/.local/bin/claude /workspace/bin/
claude   # sign in once
```

Running on the pod, it edits files and launches jobs directly there. On later pods `claude` is already installed and signed in. (The copied binary can't auto-update itself — re-run the installer occasionally.)

The project itself declares its dependencies (`pyproject.toml` + `uv.lock`), so there is no environment to hand-build — clone and sync. Use the SSH URL (`git@github.com:...`) for private repos; agent forwarding from the Connection section handles the authentication:

```bash
source /workspace/setup.sh
cd /workspace
git clone ...
cd myproject
uv sync
source .venv/bin/activate
```

`uv sync` creates `.venv` inside the project from the lockfile. Because the cache lives on the volume, the heavy wheels (CUDA, torch) download only the first time — **on later pods `uv sync` just re-links them**:

```bash
# Only on a new/restarted pod — if you just ran the block above, you're done
source /workspace/setup.sh
cd /workspace/myproject && uv sync
source .venv/bin/activate
```

(`~/.bashrc` is wiped with the pod, which is why the activation lives in `/workspace/setup.sh` instead.)


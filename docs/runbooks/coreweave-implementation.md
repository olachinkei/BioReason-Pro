# CoreWeave SSH/Slurm Implementation Notes

Use this runbook for normal CoreWeave GPU work on the SUNK/Slurm login node. This is the default path when someone asks to run Senpai on CoreWeave GPUs. Use the CKS/SUNK Kubernetes runbook only when the request specifically calls for CKS or Kubernetes Jobs.

Use the global Codex skill `coreweave-gpu-implementation` for live cluster work. Do not add that skill to this repository.

## Connect

```bash
ssh -o IdentitiesOnly=yes kkamata+cwb607@sunk.cwb607-training.coreweave.app
```

The login node is only for lightweight orchestration: inspect queues, check disk, sync code, and submit/attach to Slurm jobs. Do not edit code, install packages, import GPU frameworks, compile extensions, or run training on the login node.

## Storage Preflight

Before allocating GPUs:

```bash
hostname
df -h /mnt/home /mnt/data 2>/dev/null || df -h
du -h -d 2 /mnt/home/$USER /mnt/data/$USER 2>/dev/null | sort -h | tail -40
find /mnt/home/$USER /mnt/data/$USER -type f -size +5G -mtime +7 -print 2>/dev/null
```

Move reusable caches and scratch outputs to `/mnt/data/$USER`. Do not delete datasets, model checkpoints, W&B artifacts, registry manifests, or active run outputs unless the exact path has been confirmed.

## Runtime Paths

Set these before installs, imports, training, or eval:

```bash
export BIOREASON_RUNTIME_ROOT=/mnt/data/$USER/BioReason-Pro
export BIOREASON_ARTIFACTS_ROOT=$BIOREASON_RUNTIME_ROOT/data/artifacts
export BIOREASON_CACHE_ROOT=$BIOREASON_RUNTIME_ROOT/cache
export WANDB_DIR=$BIOREASON_RUNTIME_ROOT/wandb
export WEAVE_SERVER_CACHE_DIR=$WANDB_DIR/weave_server_cache
export HF_HOME=$BIOREASON_CACHE_ROOT/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export XDG_CACHE_HOME=$BIOREASON_CACHE_ROOT/xdg
export TRITON_CACHE_DIR=$BIOREASON_CACHE_ROOT/triton
export TORCHINDUCTOR_CACHE_DIR=$BIOREASON_CACHE_ROOT/torch_inductor
export TORCHINDUCTOR_COMPILE_THREADS=4
export TMPDIR=$BIOREASON_RUNTIME_ROOT/tmp
export VLLM_USE_V1=0
mkdir -p "$BIOREASON_ARTIFACTS_ROOT" "$BIOREASON_CACHE_ROOT" "$WANDB_DIR" \
  "$WEAVE_SERVER_CACHE_DIR" "$HF_HOME" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"
```

## W&B Online Preflight

W&B online tracking is mandatory for Senpai smoke, gate, and full runs. Do not
set `WANDB_MODE=offline`, `WANDB_MODE=dryrun`, `WANDB_MODE=disabled`, or pass
`--wandb_mode offline` to work around auth or SDK failures.

After entering a GPU allocation, but before launching a long training run,
verify W&B from the exact Python environment that will run Senpai:

```bash
python -m pip install --upgrade wandb weave
python - <<'PY'
import wandb
import weave

print("wandb", getattr(wandb, "__version__", "unknown"))
print("weave", getattr(weave, "__version__", "unknown"))
api = wandb.Api(timeout=30)
viewer = getattr(api, "viewer", None)
viewer = viewer() if callable(viewer) else viewer
print("wandb viewer", getattr(viewer, "username", viewer))
PY
```

If this preflight returns 401, fails to import, or cannot resolve the viewer,
stop. Fix the W&B key or upgrade the `wandb`/`weave` libraries in the target
runtime, then rerun the preflight. Do not launch Senpai until this passes.

## GPU Allocation

Use one H100 node with 8 GPUs for the real Senpai path:

```bash
srun --partition=h100 --nodes=1 --ntasks=1 --gpus=8 \
  --cpus-per-task=64 --mem=0 --time=12:00:00 --pty bash -l
```

Inside the allocation:

```bash
hostname
nvidia-smi
scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n '1,80p'
```

If `nvidia-smi` is unavailable or no Slurm allocation is active, stop and allocate a GPU node.

## Run Smoke

```bash
python train.py \
  --gate_steps 1 \
  --continue_steps 0 \
  --max_val_samples 2 \
  --wandb_name smoke/senpai
```

Verify that outputs, W&B data, Weave cache, Hugging Face cache, eval scratch, and checkpoints are under `/mnt/data/$USER/BioReason-Pro`.

## Run Full Gate

```bash
python train.py \
  --wandb_name "$STUDENT_NAME/<hypothesis-slug>" \
  --wandb_group "<pr-or-hypothesis>"
```

The command stops after the 5-step gate if validation `overall_mean_fmax` does not strictly beat baseline. It continues to 20 total steps only on improvement.

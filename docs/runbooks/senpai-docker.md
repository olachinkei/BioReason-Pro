# Senpai Docker Runtime

This image packages the BioReason-Pro Senpai training dependencies for GPU
nodes. It does not include W&B, Hugging Face, or registry secrets.

## Required External APIs

| API | Required for | Runtime variable |
| --- | --- | --- |
| Anthropic Claude Agent SDK | Run Senpai advisor/student coding agents that inspect code, edit files, launch jobs, and summarize results. Keep this in the orchestration environment, not the GPU training image. | `ANTHROPIC_API_KEY` |
| W&B Artifacts and Runs | Download frozen data/model artifacts and log Senpai runs. | `WANDB_API_KEY` |
| Weave | Store rollout traces when `--trace_rollouts_to_weave true`. Uses the same W&B identity. | `WANDB_API_KEY` |
| Hugging Face Hub | Download public or gated model/tokenizer dependencies when a source is not already materialized. | `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` |
| Container registry | Push/pull the built image when the GPU node does not build it locally. | Registry login token, for example `docker login` |

No hosted LLM inference API is required inside the default training path.
Experiment generation runs locally through vLLM on the allocated GPUs. Claude
is for the Senpai control plane that proposes code changes, opens PRs, submits
jobs, and writes result summaries.

Use the current Claude Agent SDK package names in the orchestration project:

```bash
uv add claude-agent-sdk
npm install @anthropic-ai/claude-agent-sdk
```

The W&B key must have read access to the configured artifact refs and write
access to the target run project, normally `wandb-healthcare/bioreasoning-pro`.
The Hugging Face token only needs read access.

Do not bake `ANTHROPIC_API_KEY`, `WANDB_API_KEY`, or `HF_TOKEN` into the Docker
image. Pass them through the job environment or your cluster secret manager.

## Important Runtime Variables

Use cluster-mounted storage for all mutable outputs:

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
mkdir -p "$BIOREASON_ARTIFACTS_ROOT" "$BIOREASON_CACHE_ROOT" "$WANDB_DIR" \
  "$WEAVE_SERVER_CACHE_DIR" "$HF_HOME" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"
```

Optional registry overrides are documented in
`configs/disease_benchmark/wandb_registry_paths.env.example`.

## Sync Local Secrets To CoreWeave

Keep the local file outside the repository:

```bash
mkdir -p ~/.secrets
chmod 700 ~/.secrets
$EDITOR ~/.secrets/bioreason-senpai.env
chmod 600 ~/.secrets/bioreason-senpai.env
```

The file should contain these keys:

```bash
ANTHROPIC_API_KEY=...
WANDB_API_KEY=...
HF_TOKEN=...
```

Sync it to CoreWeave before submitting Senpai jobs:

```bash
./scripts/coreweave_sync_secrets.sh
```

If the local file lives elsewhere:

```bash
LOCAL_SECRET_ENV=/path/to/bioreason-senpai.env ./scripts/coreweave_sync_secrets.sh
```

On CoreWeave, source the file before launching the orchestrator or pass it to
Docker with `--env-file ~/.secrets/bioreason-senpai.env`. For training-only
containers, pass only `WANDB_API_KEY` and `HF_TOKEN`; keep `ANTHROPIC_API_KEY`
on the Senpai orchestration side.

## Build

Do not build this image on a local macOS workstation unless you are explicitly
using a Linux/amd64 remote builder. The practical default is to build on
CoreWeave or another Linux CUDA-capable builder, then optionally push it to a
container registry:

```bash
docker build -t bioreason-pro-senpai:cuda12.6 .
```

FlashAttention is disabled by default because it is compiled during image
build. Enable it only on a builder with a working CUDA toolchain:

```bash
docker build \
  --build-arg INSTALL_FLASH_ATTN=true \
  -t bioreason-pro-senpai:cuda12.6-flash-attn .
```

## Smoke Run

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e WANDB_API_KEY \
  -e HF_TOKEN \
  -e BIOREASON_RUNTIME_ROOT=/mnt/data/$USER/BioReason-Pro \
  -v /mnt/data/$USER/BioReason-Pro:/mnt/data/$USER/BioReason-Pro \
  bioreason-pro-senpai:cuda12.6 \
  python train.py \
    --gate_steps 1 \
    --continue_steps 0 \
    --max_val_samples 2 \
    --wandb_name smoke/senpai
```

## Full Senpai Run

Run on the target 1 node x 8 GPU allocation:

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e WANDB_API_KEY \
  -e HF_TOKEN \
  -e WANDB_PROJECT=bioreasoning-pro \
  -e BIOREASON_RUNTIME_ROOT=/mnt/data/$USER/BioReason-Pro \
  -v /mnt/data/$USER/BioReason-Pro:/mnt/data/$USER/BioReason-Pro \
  bioreason-pro-senpai:cuda12.6 \
  python train.py \
    --wandb_name "$STUDENT_NAME/<hypothesis-slug>" \
    --wandb_group "<hypothesis-or-pr>"
```

Keep benchmark registries, GO ontology, IA weights, and hidden labels unchanged.

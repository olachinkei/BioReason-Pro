#!/usr/bin/env bash
set -euo pipefail

cd ~/BioReason-Pro

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

if [ -f configs/disease_benchmark/wandb_registry_paths.env ]; then
  set -a
  source configs/disease_benchmark/wandb_registry_paths.env
  set +a
fi

source .venv-gpu/bin/activate
mkdir -p runtime_logs

TS=${TS:-$(date +%Y%m%d-%H%M%S)}

export WANDB_ENTITY=wandb-healthcare
export BASE_WANDB_PROJECT=bioreasoning-pro
export WANDB_PROJECT="$BASE_WANDB_PROJECT"
export EXPECTED_WANDB_ENTITY=wandb-healthcare
export EXPECTED_WANDB_PROJECT=bioreasoning-pro
export WEAVE_PROJECT="${WEAVE_PROJECT:-${WANDB_ENTITY}/${BASE_WANDB_PROJECT}}"

export DATA_BUNDLE=main_production
export TRAIN_PARTITION=h100
export TRAIN_NUM_GPUS=8
export TRAIN_CPUS_PER_TASK=96
export TRAIN_MEM=512G
export TRAIN_TIME_LIMIT=01:30:00
export TRAIN_JOB_NAME=bioreason-rl-ddp-smoke
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$TRAIN_CPUS_PER_TASK}"

export WANDB_RUN_NAME="${WANDB_RUN_NAME:-rl-ddp-smoke-${TS}}"
export CHECKPOINT_ARTIFACT_NAME="${CHECKPOINT_ARTIFACT_NAME:-train-rl-output-ddp-smoke-${TS}}"

export BASE_CHECKPOINT_LOCAL_DIR="${BASE_CHECKPOINT_LOCAL_DIR:-$HOME/BioReason-Pro/data/artifacts/models/train_sft_output}"
export IA_FILE_PATH="${IA_FILE_PATH:-data/artifacts/benchmarks/213_221_225_228/temporal_split/IA.txt}"
export REQUIRE_IA_FILE=True

export MAX_STEPS="${MAX_STEPS:-3}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-16}"
export EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-3}"
export SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-0}"
export MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"
export MAX_LOSS_COMPLETION_TOKENS="${MAX_LOSS_COMPLETION_TOKENS:-768}"
export ROLLOUT_LOGPROB_MICROBATCH_SIZE="${ROLLOUT_LOGPROB_MICROBATCH_SIZE:-1}"
export DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS:-7200}"

export CONTINUATION_MODE=paper_native
export REASONING_PROMPT_STYLE=auto
export COMPACT_INTERPRO_LIMIT=8
export COMPACT_PPI_LIMIT=10
export COMPACT_GO_SPECULATION_LIMIT=4
export SAMPLING_CONTRACT=auto
export REWARD_PREDICTION_SOURCE=auto
export REWARD_FINAL_ANSWER_ONLY=False

bash scripts/sh_train_protein_grpo.sh

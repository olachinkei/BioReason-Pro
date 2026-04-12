#!/bin/bash

# ===================================================================================================
# SLURM Configuration
# Adjust these to match your cluster. Example values shown.
# ===================================================================================================
# #SBATCH --job-name=eval_bioreason
# #SBATCH --partition=your_gpu_partition
# #SBATCH --account=your_account
# #SBATCH --time=24:00:00
# #SBATCH --nodes=1
# #SBATCH --gpus=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=128gb
# #SBATCH --output=eval_bioreason_%j.out
# #SBATCH --error=eval_bioreason_%j.err

set -eo pipefail

# Run from project root
cd "$(dirname "$0")/.."

# ===================================================================================================
# Environment Setup
# Set these to your conda environment and project root.
# ===================================================================================================
# export PATH=/path/to/your/conda/envs/bin:$PATH     # e.g., /home/user/miniconda/envs/bio/bin
# source /path/to/conda/etc/profile.d/conda.sh       # e.g., /home/user/miniconda/etc/profile.d/conda.sh
# conda activate your_env_name
# ROOT_DIR=/path/to/BioReason-Pro                     # e.g., /home/user/BioReason-Pro
# cd $ROOT_DIR

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

PYTHON_BIN=${PYTHON_BIN:-}
if [ -z "$PYTHON_BIN" ]; then
    if [ -x "./.venv-gpu/bin/python" ]; then
        PYTHON_BIN="./.venv-gpu/bin/python"
    elif [ -x "./.venv/bin/python" ]; then
        PYTHON_BIN="./.venv/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

EVALS_DIR=${EVALS_DIR:-"./evals"}
mkdir -p "$EVALS_DIR"

export TRANSFORMERS_CACHE="$EVALS_DIR/transformers"
mkdir -p "$TRANSFORMERS_CACHE"

# ===================================================================================================
# Model Checkpoint
# This is the low-level wrapper. Prefer scripts/run_registered_eval.py for registry-driven execution.
# ===================================================================================================
MODEL_PATH=${MODEL_PATH:-"/path/to/bioreason-pro-rl"}
DATA_BUNDLE_RESOLVER=${DATA_BUNDLE_RESOLVER:-"scripts/materialize_data_bundle.py"}
DATA_MANIFEST_PATH=${DATA_MANIFEST_PATH:-"configs/disease_benchmark/data_registry.json"}
DATA_BUNDLE=${DATA_BUNDLE:-"main_production"}

# ===================================================================================================
# Paths: scratch + caches used during evaluation
# ===================================================================================================
PROTEIN_MODEL_NAME=${PROTEIN_MODEL_NAME:-"esm3_sm_open_v1"}
GO_OBO_PATH=${GO_OBO_PATH:-"bioreason2/dataset/go-basic.obo"}  # repo-local default
IA_FILE_PATH=${IA_FILE_PATH:-""}                   # e.g., /path/to/IA.txt
GO_EMBEDDINGS_PATH=${GO_EMBEDDINGS_PATH:-""}       # e.g., /data/bioreason/go_embeddings
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-"data/artifacts/hf_cache"}
STRUCTURE_DIR=${STRUCTURE_DIR:-"data/structures"}
KEEP_LOCAL_EVAL_OUTPUTS=${KEEP_LOCAL_EVAL_OUTPUTS:-0}
WANDB_DIR=${WANDB_DIR:-"./wandb"}
WEAVE_SERVER_CACHE_DIR=${WEAVE_SERVER_CACHE_DIR:-"$WANDB_DIR/weave_server_cache"}

EVAL_SCRIPT="eval.py"
EVALS_PATH=${EVALS_PATH:-"$EVALS_DIR/results"}

# Chunking parameters (can be overridden via command line)
NUM_CHUNKS=${NUM_CHUNKS:-1}
CHUNK_ID=${CHUNK_ID:-0}

# ===================================================================================================
# Evaluation Parameters
# ===================================================================================================
EVAL_SPLIT=${EVAL_SPLIT:-validation}
MAX_SAMPLES=${MAX_SAMPLES:-}
if [ -z "${MAX_SAMPLES// }" ]; then
  if [ "$EVAL_SPLIT" = "validation" ]; then
    MAX_SAMPLES=100
  else
    MAX_SAMPLES=-1
  fi
fi
SAMPLE_STRATEGY=${SAMPLE_STRATEGY:-stratified_aspect_profile}
MAX_LENGTH_PROTEIN=2000
MAX_NEW_TOKENS=5000
TEMPERATURE=0
TOP_P=0.95
REPETITION_PENALTY=1.0
INFERENCE_BATCH_SIZE=${INFERENCE_BATCH_SIZE:-1}
PASS_AT_K=1
PROTEIN_EMBEDDING_LAYER=37
UNIFIED_GO_ENCODER=True
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.4}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-256}

# GO Model Architecture (must match training config)
GO_HIDDEN_DIM=512
GO_NUM_GAT_LAYERS=3
GO_NUM_HEADS=8
GO_NUM_REDUCED_EMBEDDINGS=200
GO_EMBEDDING_DIM=2560

# Dataset configuration
CAFA5_DATASET=${CAFA5_DATASET:-"wanglab/cafa5"}
DATASET_NAME=${DATASET_NAME:-"interlabel_test_dataset_with_gogpt_memorized_copy"}
REASONING_DATASET_NAME=${REASONING_DATASET_NAME:-"interlabel_test_dataset_with_gogpt_memorized_copy"}
SPLIT_GO_ASPECTS=False
INTERPRO_IN_PROMPT=True
PREDICT_INTERPRO=False
PPI_IN_PROMPT=True
GO_GPT_PREDICTIONS_COLUMN="go_pred"
ADD_UNIPROT_SUMMARY=False

# GO Filtering (must match training config)
MIN_GO_MF_FREQ=1
MIN_GO_BP_FREQ=1
MIN_GO_CC_FREQ=1
APPLY_GO_FILTERING_TO_VAL_TEST=False
BENCHMARK_VERSION=${BENCHMARK_VERSION:-}
MODEL_NAME=${MODEL_NAME:-}
TEMPORAL_SPLIT_ARTIFACT=${TEMPORAL_SPLIT_ARTIFACT:-}
DATASET_ARTIFACT=${DATASET_ARTIFACT:-}
MODEL_ARTIFACT=${MODEL_ARTIFACT:-}
SHORTLIST_QUERY=${SHORTLIST_QUERY:-}
SHORTLIST_MODE=${SHORTLIST_MODE:-}
TRAIN_START_RELEASE=${TRAIN_START_RELEASE:-}
TRAIN_END_RELEASE=${TRAIN_END_RELEASE:-}
DEV_END_RELEASE=${DEV_END_RELEASE:-}
TEST_END_RELEASE=${TEST_END_RELEASE:-}
METRICS_SUMMARY_PATH=${METRICS_SUMMARY_PATH:-}
METRIC_THREADS=${METRIC_THREADS:-0}
METRIC_THRESHOLD_STEP=${METRIC_THRESHOLD_STEP:-0.99}
WANDB_PROJECT=${WANDB_PROJECT:-"bioreason-pro"}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
WANDB_ARTIFACT_NAME=${WANDB_ARTIFACT_NAME:-}
WANDB_MODE=${WANDB_MODE:-}
EXPECTED_WANDB_ENTITY=${EXPECTED_WANDB_ENTITY:-"wandb-healthcare"}
EXPECTED_WANDB_PROJECT=${EXPECTED_WANDB_PROJECT:-"bioreason-pro"}
ALLOW_WANDB_PROJECT_MISMATCH=${ALLOW_WANDB_PROJECT_MISMATCH:-0}
WEAVE_PROJECT=${WEAVE_PROJECT:-}
WEAVE_EVAL_NAME=${WEAVE_EVAL_NAME:-}

if [ -z "${WANDB_MODE// }" ]; then
    unset WANDB_MODE
fi

if [ "$ALLOW_WANDB_PROJECT_MISMATCH" != "1" ]; then
    if [ -n "$WANDB_ENTITY" ] && [ "$WANDB_ENTITY" != "$EXPECTED_WANDB_ENTITY" ]; then
        echo "Error: WANDB_ENTITY=$WANDB_ENTITY does not match expected $EXPECTED_WANDB_ENTITY"
        exit 1
    fi
    if [ -n "$WANDB_PROJECT" ] && [ "$WANDB_PROJECT" != "$EXPECTED_WANDB_PROJECT" ]; then
        echo "Error: WANDB_PROJECT=$WANDB_PROJECT does not match expected $EXPECTED_WANDB_PROJECT"
        exit 1
    fi
fi

# ===================================================================================================
# Execute Evaluation
# ===================================================================================================
echo "Starting reasoning evaluation..."
echo "Model checkpoint: $MODEL_PATH"
echo "Protein model: $PROTEIN_MODEL_NAME"
echo "Evaluation split: $EVAL_SPLIT"
echo "Scratch directory: $EVALS_PATH"
echo "Inference batch size: $INFERENCE_BATCH_SIZE"
echo "vLLM gpu_memory_utilization: $VLLM_GPU_MEMORY_UTILIZATION"
echo "vLLM max_num_seqs: $VLLM_MAX_NUM_SEQS"

DEFAULT_DATASET_SOURCE="wanglab/cafa5"
if [ -n "$DATASET_ARTIFACT" ] && { [ -z "$CAFA5_DATASET" ] || [ "$CAFA5_DATASET" = "$DEFAULT_DATASET_SOURCE" ]; }; then
    echo "--- Resolving reasoning dataset source for eval from W&B Artifact"
    RESOLVED_REASONING_DATASET_DIR=$("$PYTHON_BIN" "$DATA_BUNDLE_RESOLVER" \
        --data-manifest-path "$DATA_MANIFEST_PATH" \
        --data-bundle "$DATA_BUNDLE" \
        --asset-key reasoning_dataset \
        --print-field local_dir)
    if [ -z "$RESOLVED_REASONING_DATASET_DIR" ]; then
        echo "Error: failed to resolve reasoning dataset source for eval"
        exit 1
    fi
    CAFA5_DATASET="$RESOLVED_REASONING_DATASET_DIR"
    if [ -z "$DATASET_NAME" ] || [ "$DATASET_NAME" = "interlabel_test_dataset_with_gogpt_memorized_copy" ]; then
        RESOLVED_REASONING_DATASET_NAME=$("$PYTHON_BIN" "$DATA_BUNDLE_RESOLVER" \
            --data-manifest-path "$DATA_MANIFEST_PATH" \
            --data-bundle "$DATA_BUNDLE" \
            --asset-key reasoning_dataset \
            --print-field dataset_name)
        if [ -n "$RESOLVED_REASONING_DATASET_NAME" ]; then
            DATASET_NAME="$RESOLVED_REASONING_DATASET_NAME"
            REASONING_DATASET_NAME="$RESOLVED_REASONING_DATASET_NAME"
        fi
    fi
    echo "--- Reasoning dataset materialized at $CAFA5_DATASET"
fi

mkdir -p "$EVALS_PATH"
mkdir -p "$WANDB_DIR"
mkdir -p "$WEAVE_SERVER_CACHE_DIR"
export WEAVE_SERVER_CACHE_DIR

CHUNK_ARGS=""
if [ "$NUM_CHUNKS" -gt 1 ]; then
    CHUNK_ARGS="--num_chunks $NUM_CHUNKS --chunk_id $CHUNK_ID"
    echo "Processing chunk $((CHUNK_ID + 1))/$NUM_CHUNKS"
fi

TRACKING_ARGS=()
MODEL_OPTIONAL_ARGS=()
if [ -n "$BENCHMARK_VERSION" ]; then
    TRACKING_ARGS+=(--benchmark_version "$BENCHMARK_VERSION")
fi
if [ -n "$MODEL_NAME" ]; then
    TRACKING_ARGS+=(--model_name "$MODEL_NAME")
fi
if [ -n "$TEMPORAL_SPLIT_ARTIFACT" ]; then
    TRACKING_ARGS+=(--temporal_split_artifact "$TEMPORAL_SPLIT_ARTIFACT")
fi
if [ -n "$DATASET_ARTIFACT" ]; then
    TRACKING_ARGS+=(--dataset_artifact "$DATASET_ARTIFACT")
fi
if [ -n "$MODEL_ARTIFACT" ]; then
    TRACKING_ARGS+=(--model_artifact "$MODEL_ARTIFACT")
fi
if [ -n "$SHORTLIST_QUERY" ]; then
    TRACKING_ARGS+=(--shortlist_query "$SHORTLIST_QUERY")
fi
if [ -n "$SHORTLIST_MODE" ]; then
    TRACKING_ARGS+=(--shortlist_mode "$SHORTLIST_MODE")
fi
if [ -n "$TRAIN_START_RELEASE" ]; then
    TRACKING_ARGS+=(--train_start_release "$TRAIN_START_RELEASE")
fi
if [ -n "$TRAIN_END_RELEASE" ]; then
    TRACKING_ARGS+=(--train_end_release "$TRAIN_END_RELEASE")
fi
if [ -n "$DEV_END_RELEASE" ]; then
    TRACKING_ARGS+=(--dev_end_release "$DEV_END_RELEASE")
fi
if [ -n "$TEST_END_RELEASE" ]; then
    TRACKING_ARGS+=(--test_end_release "$TEST_END_RELEASE")
fi
if [ -n "$METRICS_SUMMARY_PATH" ]; then
    TRACKING_ARGS+=(--metrics_summary_path "$METRICS_SUMMARY_PATH")
fi
if [ -n "$IA_FILE_PATH" ]; then
    TRACKING_ARGS+=(--ia_file_path "$IA_FILE_PATH")
fi
TRACKING_ARGS+=(--metric_threshold_step "$METRIC_THRESHOLD_STEP")
if [ -n "$WANDB_PROJECT" ]; then
    TRACKING_ARGS+=(--wandb_project "$WANDB_PROJECT")
fi
if [ -n "$WANDB_ENTITY" ]; then
    TRACKING_ARGS+=(--wandb_entity "$WANDB_ENTITY")
fi
if [ -n "$WANDB_RUN_NAME" ]; then
    TRACKING_ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")
fi
if [ -n "$WANDB_ARTIFACT_NAME" ]; then
    TRACKING_ARGS+=(--wandb_artifact_name "$WANDB_ARTIFACT_NAME")
fi
if [ -n "$WANDB_DIR" ]; then
    TRACKING_ARGS+=(--wandb_dir "$WANDB_DIR")
fi
if [ -n "$WANDB_MODE" ]; then
    TRACKING_ARGS+=(--wandb_mode "$WANDB_MODE")
fi
if [ -n "$WEAVE_PROJECT" ]; then
    TRACKING_ARGS+=(--weave_project "$WEAVE_PROJECT")
fi
if [ -n "$WEAVE_EVAL_NAME" ]; then
    TRACKING_ARGS+=(--weave_eval_name "$WEAVE_EVAL_NAME")
fi
if [ -n "$GO_EMBEDDINGS_PATH" ]; then
    MODEL_OPTIONAL_ARGS+=(--precomputed_embeddings_path "$GO_EMBEDDINGS_PATH")
fi
case "${KEEP_LOCAL_EVAL_OUTPUTS,,}" in
    1|true|yes)
        TRACKING_ARGS+=(--keep_local_eval_outputs)
        ;;
esac

"$PYTHON_BIN" "$EVAL_SCRIPT" \
    --ckpt_dir "$MODEL_PATH" \
    --protein_model_name "$PROTEIN_MODEL_NAME" \
    --protein_embedding_layer "$PROTEIN_EMBEDDING_LAYER" \
    --go_obo_path "$GO_OBO_PATH" \
    --ia_file_path "$IA_FILE_PATH" \
    --unified_go_encoder "$UNIFIED_GO_ENCODER" \
    --go_hidden_dim $GO_HIDDEN_DIM \
    --go_num_gat_layers $GO_NUM_GAT_LAYERS \
    --go_num_heads $GO_NUM_HEADS \
    --go_num_reduced_embeddings $GO_NUM_REDUCED_EMBEDDINGS \
    --go_embedding_dim $GO_EMBEDDING_DIM \
    --cafa5_dataset "$CAFA5_DATASET" \
    --cafa5_dataset_name "$DATASET_NAME" \
    --reasoning_dataset_name "$REASONING_DATASET_NAME" \
    --go_gpt_predictions_column "$GO_GPT_PREDICTIONS_COLUMN" \
    --dataset_cache_dir "$DATASET_CACHE_DIR" \
    --structure_dir "$STRUCTURE_DIR" \
    --split_go_aspects "$SPLIT_GO_ASPECTS" \
    --interpro_in_prompt "$INTERPRO_IN_PROMPT" \
    --predict_interpro "$PREDICT_INTERPRO" \
    --ppi_in_prompt "$PPI_IN_PROMPT" \
    --add_uniprot_summary "$ADD_UNIPROT_SUMMARY" \
    --min_go_mf_freq $MIN_GO_MF_FREQ \
    --min_go_bp_freq $MIN_GO_BP_FREQ \
    --min_go_cc_freq $MIN_GO_CC_FREQ \
    --apply_go_filtering_to_val_test $APPLY_GO_FILTERING_TO_VAL_TEST \
    --seed 23 \
    --debug False \
    --metric_threads $METRIC_THREADS \
    --eval_split "$EVAL_SPLIT" \
    --max_samples $MAX_SAMPLES \
    --sample_strategy "$SAMPLE_STRATEGY" \
    --max_length_protein $MAX_LENGTH_PROTEIN \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --inference_batch_size $INFERENCE_BATCH_SIZE \
    --pass_at_k $PASS_AT_K \
    --repetition_penalty $REPETITION_PENALTY \
    --vllm_gpu_memory_utilization $VLLM_GPU_MEMORY_UTILIZATION \
    --vllm_max_num_seqs $VLLM_MAX_NUM_SEQS \
    --evals_path "$EVALS_PATH" \
    "${MODEL_OPTIONAL_ARGS[@]}" \
    "${TRACKING_ARGS[@]}" \
    $CHUNK_ARGS

if [[ "${KEEP_LOCAL_EVAL_OUTPUTS,,}" =~ ^(1|true|yes)$ ]]; then
    echo "Evaluation finished. Local scratch was retained at $EVALS_PATH"
else
    echo "Evaluation finished. W&B is the source of truth; local scratch may have been cleaned up."
fi

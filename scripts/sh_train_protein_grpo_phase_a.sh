#!/usr/bin/env bash
#
# Phase A ablation launcher for train_protein_grpo.py.
#
# Thin wrapper over scripts/sh_train_protein_grpo.sh that translates an
# ablation name (A0/A1/A2/A3) from the Phase A grid into the right
# --reward_mode / --disease_loss_weight / --ablation_tag / --wandb_tags /
# --run_name arguments, then exec's the production launcher so every
# ablation inherits the exact distributed + rollout contract.
#
# Phase A grid (see domain/learning-log/2026-04-16-rl-tuning-proposal-disease-pilot.md §3):
#
#   A0 (baseline): reward=ia_f1,             loss_weight=1.0   (current behavior)
#   A1           : reward=per_aspect_ia_f1,  loss_weight=1.0
#   A2           : reward=per_aspect_lin,    loss_weight=1.0
#   A3           : reward=per_aspect_lin,    loss_weight=1.5
#
# Usage:
#
#   ABLATION=A1 WANDB_ENTITY=... WANDB_PROJECT=... \
#     scripts/sh_train_protein_grpo_phase_a.sh [--any extra train_protein_grpo.py flags]
#
# Filter in W&B by the tag "phase-a-A1" (or any A0..A3) to see that run's
# curves in isolation; by "phase-a-disease-pilot" to see all four together.

set -euo pipefail

cd "$(dirname "$0")/.."

ABLATION=${ABLATION:-""}
if [ -z "$ABLATION" ]; then
  echo "Error: set ABLATION=A0|A1|A2|A3 (see Phase A grid in the tuning proposal)." >&2
  exit 1
fi

ABLATION_UPPER=$(printf '%s' "$ABLATION" | tr '[:lower:]' '[:upper:]')

case "$ABLATION_UPPER" in
  A0)
    ABLATION_REWARD_MODE="ia_f1"
    ABLATION_DISEASE_LOSS_WEIGHT="1.0"
    ABLATION_DESCRIPTION="baseline-ia-f1"
    ;;
  A1)
    ABLATION_REWARD_MODE="per_aspect_ia_f1"
    ABLATION_DISEASE_LOSS_WEIGHT="1.0"
    ABLATION_DESCRIPTION="per-aspect-ia-f1"
    ;;
  A2)
    ABLATION_REWARD_MODE="per_aspect_lin"
    ABLATION_DISEASE_LOSS_WEIGHT="1.0"
    ABLATION_DESCRIPTION="per-aspect-lin"
    ;;
  A3)
    ABLATION_REWARD_MODE="per_aspect_lin"
    ABLATION_DISEASE_LOSS_WEIGHT="1.5"
    ABLATION_DESCRIPTION="per-aspect-lin-disease-weighted"
    ;;
  *)
    echo "Error: unknown ABLATION='$ABLATION' (expected A0|A1|A2|A3)." >&2
    exit 1
    ;;
esac

ABLATION_TAG=${ABLATION_TAG:-"phase-a-${ABLATION_UPPER}"}
PHASE_A_GROUP_TAG=${PHASE_A_GROUP_TAG:-"phase-a-disease-pilot"}
ABLATION_TIMESTAMP=${ABLATION_TIMESTAMP:-"$(date -u +%Y%m%dT%H%M%SZ)"}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"rl-phase-a-${ABLATION_UPPER}-${ABLATION_TIMESTAMP}"}

ABLATION_LOWER=$(printf '%s' "$ABLATION_UPPER" | tr '[:upper:]' '[:lower:]')

# Each ablation writes to its own checkpoint directory / artifact alias so
# concurrent or back-to-back runs do not clobber each other.
OUTPUT_DIR=${OUTPUT_DIR:-"data/artifacts/models/train_rl_output_phase_a_${ABLATION_LOWER}"}
CHECKPOINT_ARTIFACT_NAME=${CHECKPOINT_ARTIFACT_NAME:-"train-rl-output-phase-a-${ABLATION_LOWER}"}
CHECKPOINT_ARTIFACT_ALIASES=${CHECKPOINT_ARTIFACT_ALIASES:-"latest,phase-a-${ABLATION_LOWER}"}
export OUTPUT_DIR CHECKPOINT_ARTIFACT_NAME CHECKPOINT_ARTIFACT_ALIASES

ABLATION_TRAIN_ARGS=(
  --reward_mode "$ABLATION_REWARD_MODE"
  --disease_loss_weight "$ABLATION_DISEASE_LOSS_WEIGHT"
  --ablation_tag "$ABLATION_TAG"
  --wandb_tags "$PHASE_A_GROUP_TAG" "$ABLATION_DESCRIPTION"
)

echo "[phase-a] ablation=$ABLATION_UPPER reward_mode=$ABLATION_REWARD_MODE disease_loss_weight=$ABLATION_DISEASE_LOSS_WEIGHT tag=$ABLATION_TAG run_name=$WANDB_RUN_NAME" >&2

exec "$(dirname "$0")/sh_train_protein_grpo.sh" "${ABLATION_TRAIN_ARGS[@]}" "$@"

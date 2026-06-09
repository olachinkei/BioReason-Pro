#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/launch_wandb_senpai_control_plane.sh <research-tag> [upstream launch.py args...]

Launch the upstream wandb/senpai advisor + student control plane against this
BioReason-Pro target repository. This is the real Senpai GitHub PR workflow:
advisor pod creates/reviews PRs, student pods implement assigned PRs and run
train.py, and GitHub labels route work.

Environment defaults:
  LOCAL_SECRET_ENV       ~/.secrets/bioreason-senpai.env
  SENPAI_RUNNER_DIR      ~/.cache/bioreason-senpai/wandb-senpai
  SENPAI_RUNNER_REPO     https://github.com/wandb/senpai.git
  SENPAI_RUNNER_BRANCH   main
  TARGET_REPO_URL        https://github.com/olachinkei/BioReason-Pro.git
  TARGET_REPO_BRANCH     main
  ADVISOR_BRANCH         senpai-bioreason
  WANDB_ENTITY           wandb-healthcare
  WANDB_PROJECT          bioreasoning-pro-senpai
  PVC_CLAIM_NAME         bioreason-runtime-pvc
  PVC_MOUNT_PATH         /mnt/data
  N_STUDENTS             1
  STUDENT_NAMES          frieren

Required secrets in LOCAL_SECRET_ENV or environment:
  GITHUB_TOKEN, ANTHROPIC_API_KEY, EXA_API_KEY, WANDB_API_KEY
Optional but recommended:
  HF_TOKEN or HUGGING_FACE_HUB_TOKEN
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TAG="$1"
shift

LOCAL_SECRET_ENV="${LOCAL_SECRET_ENV:-$HOME/.secrets/bioreason-senpai.env}"
SENPAI_RUNNER_DIR="${SENPAI_RUNNER_DIR:-$HOME/.cache/bioreason-senpai/wandb-senpai}"
SENPAI_RUNNER_REPO="${SENPAI_RUNNER_REPO:-https://github.com/wandb/senpai.git}"
SENPAI_RUNNER_BRANCH="${SENPAI_RUNNER_BRANCH:-main}"
TARGET_REPO_URL="${TARGET_REPO_URL:-https://github.com/olachinkei/BioReason-Pro.git}"
TARGET_REPO_BRANCH="${TARGET_REPO_BRANCH:-main}"
ADVISOR_BRANCH="${ADVISOR_BRANCH:-senpai-bioreason}"
WANDB_ENTITY="${WANDB_ENTITY:-wandb-healthcare}"
WANDB_PROJECT="${WANDB_PROJECT:-bioreasoning-pro-senpai}"
PVC_CLAIM_NAME="${PVC_CLAIM_NAME:-bioreason-runtime-pvc}"
PVC_MOUNT_PATH="${PVC_MOUNT_PATH:-/mnt/data}"
N_STUDENTS="${N_STUDENTS:-1}"
STUDENT_NAMES="${STUDENT_NAMES:-frieren}"
GPUS_PER_STUDENT="${GPUS_PER_STUDENT:-8}"
CPU_PER_GPU="${CPU_PER_GPU:-15}"
MEMORY_GI_PER_GPU="${MEMORY_GI_PER_GPU:-120}"
GH_HISTORY_SCOPE="${GH_HISTORY_SCOPE:-branch}"
TIMEOUT_MINUTES="${TIMEOUT_MINUTES:-720}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-300}"
POLL_JITTER_S="${POLL_JITTER_S:-60}"
SENPAI_CREATE_RBAC="${SENPAI_CREATE_RBAC:-1}"
SENPAI_LAUNCH_PYTHON="${SENPAI_LAUNCH_PYTHON:-}"

if [[ -z "$SENPAI_LAUNCH_PYTHON" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
  SENPAI_LAUNCH_PYTHON="$REPO_ROOT/.venv/bin/python"
fi
if [[ -z "$SENPAI_LAUNCH_PYTHON" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      SENPAI_LAUNCH_PYTHON="$candidate"
      break
    fi
  done
fi

if [[ -f "$LOCAL_SECRET_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$LOCAL_SECRET_ENV"
  set +a
fi

dry_run=0
preflight_only=0
for arg in "$@"; do
  case "$arg" in
    --dry_run|--dry-run) dry_run=1 ;;
    --preflight_only|--preflight-only) preflight_only=1 ;;
  esac
done

if [[ "$dry_run" -eq 0 || "$preflight_only" -eq 1 ]]; then
  required_keys=(GITHUB_TOKEN ANTHROPIC_API_KEY EXA_API_KEY WANDB_API_KEY)
  for key in "${required_keys[@]}"; do
    if [[ -z "${!key:-}" ]]; then
      echo "Missing required secret: $key. Set it in $LOCAL_SECRET_ENV or the environment." >&2
      exit 1
    fi
  done
fi

HF_TOKEN_VALUE="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

if [[ ! -d "$SENPAI_RUNNER_DIR/.git" ]]; then
  mkdir -p "$(dirname "$SENPAI_RUNNER_DIR")"
  git clone "$SENPAI_RUNNER_REPO" "$SENPAI_RUNNER_DIR" >&2
fi

git -C "$SENPAI_RUNNER_DIR" fetch origin "$SENPAI_RUNNER_BRANCH" >&2
git -C "$SENPAI_RUNNER_DIR" checkout "$SENPAI_RUNNER_BRANCH" >&2
git -C "$SENPAI_RUNNER_DIR" pull --ff-only origin "$SENPAI_RUNNER_BRANCH" >&2

cd "$SENPAI_RUNNER_DIR"

"$SENPAI_LAUNCH_PYTHON" - <<'PY'
from pathlib import Path

insert = """        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: senpai-secrets
              key: hf-token
              optional: true
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: senpai-secrets
              key: hf-token
              optional: true
"""
marker = """        - name: EXA_API_KEY
          valueFrom:
            secretKeyRef:
              name: {{LAUNCH_SECRET_NAME}}
              key: exa-api-key
"""
for rel in ("k8s/advisor-deployment.yaml", "k8s/student-deployment.yaml"):
    path = Path(rel)
    text = path.read_text(encoding="utf-8")
    if "name: HF_TOKEN" not in text:
        if marker not in text:
            raise SystemExit(f"Could not patch {rel}: EXA_API_KEY marker missing")
        text = text.replace(marker, marker + insert)
        path.write_text(text, encoding="utf-8")
PY

if [[ "$dry_run" -eq 0 && "$preflight_only" -eq 0 ]]; then
  kubectl create secret generic senpai-secrets \
    --from-literal=wandb-api-key="$WANDB_API_KEY" \
    --from-literal=hf-token="$HF_TOKEN_VALUE" \
    --dry-run=client -o yaml | kubectl apply -f -

  if [[ "$SENPAI_CREATE_RBAC" == "1" ]]; then
    kubectl apply -f - <<'EOF'
apiVersion: v1
kind: ServiceAccount
metadata:
  name: senpai-orchestrator
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: senpai-orchestrator
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: senpai-orchestrator
subjects:
- kind: ServiceAccount
  name: senpai-orchestrator
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: senpai-orchestrator
EOF
  fi
fi

launch_args=(
  --tag "$TAG"
  --advisor
  --target_repo_url "$TARGET_REPO_URL"
  --target_repo_branch "$TARGET_REPO_BRANCH"
  --advisor_branch "$ADVISOR_BRANCH"
  --gh_history_scope "$GH_HISTORY_SCOPE"
  --wandb_entity "$WANDB_ENTITY"
  --wandb_project "$WANDB_PROJECT"
  --pvc_claim_name "$PVC_CLAIM_NAME"
  --pvc_mount_path "$PVC_MOUNT_PATH"
  --n_students "$N_STUDENTS"
  --names "$STUDENT_NAMES"
  --gpus_per_student "$GPUS_PER_STUDENT"
  --cpu_per_gpu "$CPU_PER_GPU"
  --memory_gi_per_gpu "$MEMORY_GI_PER_GPU"
  --timeout_minutes "$TIMEOUT_MINUTES"
  --max_epochs "$MAX_EPOCHS"
  --poll_interval_s "$POLL_INTERVAL_S"
  --poll_jitter_s "$POLL_JITTER_S"
)

if command -v uv >/dev/null 2>&1; then
  uv run --with simple-parsing --with PyYAML python k8s/launch.py "${launch_args[@]}" "$@"
else
  launch_venv="$SENPAI_RUNNER_DIR/.launch-venv"
  if [[ -x "$launch_venv/bin/python" ]] && ! "$launch_venv/bin/python" - >/dev/null 2>&1 <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
  then
    rm -rf "$launch_venv"
  fi
  if [[ ! -x "$launch_venv/bin/python" ]]; then
    "$SENPAI_LAUNCH_PYTHON" -m venv "$launch_venv"
    "$launch_venv/bin/python" -m pip install --upgrade pip simple-parsing PyYAML >&2
  fi
  if ! "$launch_venv/bin/python" - >/dev/null 2>&1 <<'PY'
import simple_parsing  # noqa: F401
import yaml  # noqa: F401
PY
  then
    "$launch_venv/bin/python" -m pip install --upgrade simple-parsing PyYAML >&2
  fi
  "$launch_venv/bin/python" k8s/launch.py "${launch_args[@]}" "$@"
fi

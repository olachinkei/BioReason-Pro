#!/usr/bin/env bash
set -euo pipefail

LOCAL_SECRET_ENV="${LOCAL_SECRET_ENV:-$HOME/.secrets/bioreason-senpai.env}"
REMOTE="${COREWEAVE_REMOTE:-kkamata+cwb607@sunk.cwb607-training.coreweave.app}"
REMOTE_SECRET_ENV="${REMOTE_SECRET_ENV:-~/.secrets/bioreason-senpai.env}"

if [[ ! -f "$LOCAL_SECRET_ENV" ]]; then
  echo "Secret env file not found: $LOCAL_SECRET_ENV" >&2
  echo "Set LOCAL_SECRET_ENV=/path/to/bioreason-senpai.env if you keep it elsewhere." >&2
  exit 1
fi

if [[ "$(stat -f %Lp "$LOCAL_SECRET_ENV" 2>/dev/null || stat -c %a "$LOCAL_SECRET_ENV")" != "600" ]]; then
  echo "Refusing to copy $LOCAL_SECRET_ENV because permissions are not 600." >&2
  echo "Run: chmod 600 \"$LOCAL_SECRET_ENV\"" >&2
  exit 1
fi

required_keys=(ANTHROPIC_API_KEY WANDB_API_KEY)
for key in "${required_keys[@]}"; do
  if ! grep -qE "^${key}=" "$LOCAL_SECRET_ENV"; then
    echo "Missing required key in $LOCAL_SECRET_ENV: $key" >&2
    exit 1
  fi
done

if ! grep -qE "^(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)=" "$LOCAL_SECRET_ENV"; then
  echo "Missing Hugging Face token in $LOCAL_SECRET_ENV: HF_TOKEN or HUGGING_FACE_HUB_TOKEN" >&2
  exit 1
fi

ssh -o IdentitiesOnly=yes "$REMOTE" 'install -d -m 700 ~/.secrets'
scp -o IdentitiesOnly=yes "$LOCAL_SECRET_ENV" "${REMOTE}:${REMOTE_SECRET_ENV}"
ssh -o IdentitiesOnly=yes "$REMOTE" "chmod 600 ${REMOTE_SECRET_ENV}"

echo "Synced secret env to ${REMOTE}:${REMOTE_SECRET_ENV}"

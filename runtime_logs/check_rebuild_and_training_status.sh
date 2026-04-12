#!/usr/bin/env bash
set -euo pipefail

cd ~/BioReason-Pro

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

source .venv-gpu/bin/activate

echo "== $(date) =="
echo "-- squeue --"
squeue -j 2832,2833 -o "%i %t %M %j %R" || true

echo "-- rebuild log tail --"
tail -n 60 ~/BioReason-Pro/runtime_logs/coreweave/rebuild_reasoning_paper_context_2832.log || true

echo "-- training log tail --"
tail -n 60 ~/BioReason-Pro/runtime_logs/coreweave/rl_postrebuild_smoke_2833.log || true

echo "-- artifact production --"
python - <<'PY'
import json
import wandb

api = wandb.Api(timeout=30)
art = api.artifact("wandb-healthcare/bioreason-pro/disease-temporal-reasoning:production", type="dataset")
md = art.metadata or {}
print(json.dumps({
    "version": art.version,
    "aliases": art.aliases,
    "dataset_name": md.get("dataset_name"),
    "local_dir": md.get("local_dir"),
}, sort_keys=True))
PY

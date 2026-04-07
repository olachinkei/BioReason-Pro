#!/usr/bin/env python3
"""Launch stage2-only SFT search runs on CoreWeave."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_REMOTE_HOST = "kkamata+cwb607@sunk.cwb607-training.coreweave.app"
DEFAULT_REMOTE_REPO = "~/BioReason-Pro"
DEFAULT_SEARCH_CONFIG = "configs/disease_benchmark/sft_stage2_search_v1.json"


def shell_path(path: str) -> str:
    if path.startswith("~/"):
        return path
    return shlex.quote(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-config", type=str, default=DEFAULT_SEARCH_CONFIG)
    parser.add_argument("--remote-host", type=str, default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-repo", type=str, default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--trial-ids", type=str, default="")
    parser.add_argument("--max-launch", type=int, default=3)
    parser.add_argument("--wandb-entity", type=str, default="wandb-healthcare")
    parser.add_argument("--wandb-project", type=str, default="bioreason-pro-custom")
    parser.add_argument("--train-time-limit", type=str, default="12:00:00")
    parser.add_argument("--train-partition", type=str, default="h100")
    parser.add_argument("--train-num-gpus", type=int, default=1)
    parser.add_argument("--train-num-nodes", type=int, default=1)
    parser.add_argument("--train-cpus-per-task", type=int, default=16)
    parser.add_argument("--train-mem", type=str, default="128G")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_trials(config_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(config_path.read_text())
    runs = payload.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"No runs found in {config_path}")
    return runs


def build_remote_command(args: argparse.Namespace, trial: dict[str, Any]) -> str:
    trial_id = str(trial["trial_id"])
    run_name = f"sft-s2-{trial_id}"
    checkpoint_root = f"data/artifacts/models/sft_search/{trial_id}"
    log_path = f"logs/coreweave/{run_name}.log"

    env_vars = {
        "WANDB_ENTITY": args.wandb_entity,
        "BASE_WANDB_PROJECT": args.wandb_project,
        "TRAIN_JOB_NAME": run_name,
        "TRAIN_TIME_LIMIT": args.train_time_limit,
        "TRAIN_PARTITION": args.train_partition,
        "TRAIN_NUM_GPUS": str(args.train_num_gpus),
        "TRAIN_NUM_NODES": str(args.train_num_nodes),
        "TRAIN_CPUS_PER_TASK": str(args.train_cpus_per_task),
        "TRAIN_MEM": args.train_mem,
        "TRAIN_USE_SRUN": "True",
        "TRAIN_EXCLUSIVE": "True",
        "RUN_STAGE1": "False",
        "VALIDATION_SUBSET_SIZE": "100",
        "VALIDATION_SUBSET_STRATEGY": "stratified_aspect_profile",
        "STAGE2_RUN_LABEL": trial_id,
        "WANDB_RUN_NAME_S2": run_name,
        "BASE_CHECKPOINT_DIR": checkpoint_root,
        "STAGE2_LEARNING_RATE": str(trial["learning_rate"]),
        "STAGE2_WARMUP_RATIO": str(trial["warmup_ratio"]),
        "STAGE2_BATCH_SIZE": str(trial["batch_size"]),
        "STAGE2_GRADIENT_ACCUMULATION_STEPS": str(trial["gradient_accumulation_steps"]),
        "STAGE2_VAL_CHECK_INTERVAL": str(trial["val_check_interval"]),
        "STAGE2_MAX_EPOCHS": str(trial["max_epochs"]),
        "STAGE2_EARLY_STOPPING_PATIENCE": str(trial.get("early_stopping_patience", 2)),
        "STAGE2_EARLY_STOPPING_MIN_DELTA": str(trial.get("early_stopping_min_delta", 0.0)),
    }
    env_blob = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_vars.items())
    inner_shell = (
        f"cd {shell_path(args.remote_repo)} && "
        f"source .venv/bin/activate && "
        f"mkdir -p logs/coreweave && "
        f"env {env_blob} bash scripts/sh_train_protein_qwen_staged.sh "
        f"> {shlex.quote(log_path)} 2>&1"
    )
    remote_shell = (
        f"nohup bash -lc {shlex.quote(inner_shell)} "
        f"</dev/null >/dev/null 2>&1 & echo $!"
    )
    return remote_shell


def main() -> None:
    args = parse_args()
    config_path = Path(args.search_config)
    trials = load_trials(config_path)

    selected_ids = [item.strip() for item in args.trial_ids.split(",") if item.strip()]
    if selected_ids:
        selected = [trial for trial in trials if str(trial["trial_id"]) in selected_ids]
    else:
        selected = trials[: args.max_launch]

    if not selected:
        raise SystemExit("No trials selected.")

    for trial in selected:
        remote_cmd = build_remote_command(args, trial)
        printable = f"ssh -o IdentitiesOnly=yes {args.remote_host} {shlex.quote(remote_cmd)}"
        if args.dry_run:
            print(printable)
            continue

        result = subprocess.run(
            ["ssh", "-o", "IdentitiesOnly=yes", args.remote_host, remote_cmd],
            check=True,
            capture_output=True,
            text=True,
        )
        pid = result.stdout.strip()
        print(f"{trial['trial_id']}: pid={pid} lr={trial['learning_rate']} warmup={trial['warmup_ratio']}")


if __name__ == "__main__":
    main()

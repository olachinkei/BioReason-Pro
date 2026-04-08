#!/usr/bin/env python3
"""Wait for SFT trials, pick the best one, launch RL, then run final test eval."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import wandb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_ENV_FILE = ROOT / "configs" / "disease_benchmark" / "wandb_registry_paths.env"
TERMINAL_STATES = {"finished", "failed", "crashed", "killed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-entity", type=str, default="wandb-healthcare")
    parser.add_argument("--wandb-project", type=str, default="bioreasoning-pro")
    parser.add_argument("--sft-run-names", type=str, required=True)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--registry-env-file", type=Path, default=DEFAULT_REGISTRY_ENV_FILE)
    parser.add_argument("--rl-job-name", type=str, default="bioreason-rl-best-sft")
    parser.add_argument("--rl-run-name-prefix", type=str, default="rl-sft-best")
    parser.add_argument("--rl-checkpoint-artifact-name", type=str, default="train-rl-output")
    parser.add_argument("--rl-checkpoint-artifact-aliases", type=str, default="latest,213.221.225.228")
    parser.add_argument("--eval-target", type=str, default="train-rl-output")
    parser.add_argument("--eval-split", type=str, default="test", choices=["validation", "test"])
    return parser.parse_args()


def run_ref(entity: str, project: str) -> str:
    return f"{entity}/{project}"


def parse_created_at(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def latest_named_run(api: wandb.Api, project_ref: str, run_name: str, not_before: Optional[datetime] = None):
    runs = [run for run in api.runs(project_ref) if run.name == run_name]
    if not_before is not None:
        filtered = []
        for run in runs:
            created_at = parse_created_at(getattr(run, "created_at", None))
            if created_at is not None and created_at >= not_before:
                filtered.append(run)
        runs = filtered
    if not runs:
        return None
    return sorted(runs, key=lambda run: run.created_at)[-1]


def wait_for_runs(
    api: wandb.Api,
    project_ref: str,
    run_names: Iterable[str],
    poll_seconds: int,
    not_before: Optional[datetime],
):
    remaining = set(run_names)
    latest = {}
    while remaining:
        for run_name in list(remaining):
            run = latest_named_run(api, project_ref, run_name, not_before=not_before)
            if run is None:
                print(f"[wait] {run_name}: not visible on W&B yet", flush=True)
                continue
            latest[run_name] = run
            print(f"[wait] {run_name}: state={run.state}", flush=True)
            if run.state in TERMINAL_STATES:
                remaining.remove(run_name)
        if remaining:
            time.sleep(poll_seconds)
    return latest


def best_finished_run(runs_by_name: dict[str, object]):
    candidates = []
    for run_name, run in runs_by_name.items():
        summary = dict(run.summary)
        val_loss_epoch = summary.get("val_loss_epoch")
        if run.state == "finished" and isinstance(val_loss_epoch, (int, float)):
            candidates.append((float(val_loss_epoch), run_name, run))
    if not candidates:
        raise RuntimeError("No finished SFT run with numeric val_loss_epoch was found.")
    candidates.sort(key=lambda item: item[0])
    return candidates[0][2]


def resolve_model_artifact_ref(run, entity: str, project: str) -> str:
    for artifact in run.logged_artifacts():
        if getattr(artifact, "type", None) == "model":
            return f"{entity}/{project}/{artifact.name}"
    configured_name = run.config.get("model_artifact")
    if configured_name:
        return f"{entity}/{project}/{configured_name}:latest"
    raise RuntimeError(f"Run {run.name} did not log a model artifact.")


def update_env_export(env_file: Path, key: str, value: str) -> None:
    env_file.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if env_file.exists():
        lines = env_file.read_text().splitlines()
    updated = False
    prefix = f"export {key}="
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = f'export {key}="{value}"'
            updated = True
            break
    if not updated:
        lines.append(f'export {key}="{value}"')
    env_file.write_text("\n".join(lines) + "\n")


def run_local_command(command: list[str], env_updates: dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(env_updates)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    project_ref = run_ref(args.wandb_entity, args.wandb_project)
    sft_run_names = [name.strip() for name in args.sft_run_names.split(",") if name.strip()]
    if not sft_run_names:
        raise SystemExit("No SFT run names were provided.")

    api = wandb.Api()
    run_not_before = datetime.now(timezone.utc) - timedelta(minutes=10)
    print(f"[orchestrator] waiting for SFT runs: {', '.join(sft_run_names)}", flush=True)
    runs_by_name = wait_for_runs(api, project_ref, sft_run_names, args.poll_seconds, not_before=run_not_before)
    best_run = best_finished_run(runs_by_name)
    best_val_loss = float(dict(best_run.summary).get("val_loss_epoch"))
    best_artifact_ref = resolve_model_artifact_ref(best_run, args.wandb_entity, args.wandb_project)
    print(
        f"[orchestrator] best_sft_run={best_run.name} run_id={best_run.id} "
        f"val_loss_epoch={best_val_loss:.6f} artifact={best_artifact_ref}",
        flush=True,
    )

    update_env_export(args.registry_env_file, "BIOREASON_TRAIN_SFT_MODEL_REGISTRY_PATH", best_artifact_ref)

    rl_run_name = f"{args.rl_run_name_prefix}-{best_run.name}"
    rl_env = {
        "WANDB_ENTITY": args.wandb_entity,
        "BASE_WANDB_PROJECT": args.wandb_project,
        "BIOREASON_TRAIN_SFT_MODEL_REGISTRY_PATH": best_artifact_ref,
        "ALLOW_PAPER_RL_ABLATION": "False",
        "TRAIN_JOB_NAME": args.rl_job_name,
        "WANDB_RUN_NAME": rl_run_name,
        "CHECKPOINT_ARTIFACT_NAME": args.rl_checkpoint_artifact_name,
        "CHECKPOINT_ARTIFACT_ALIASES": args.rl_checkpoint_artifact_aliases,
    }
    print(f"[orchestrator] launching RL from {best_artifact_ref}", flush=True)
    run_local_command(["bash", "scripts/sh_train_protein_grpo.sh"], rl_env)

    rl_run = latest_named_run(api, project_ref, rl_run_name, not_before=run_not_before)
    if rl_run is None:
        raise RuntimeError(f"RL run {rl_run_name} was not found on W&B.")
    if rl_run.state != "finished":
        raise RuntimeError(f"RL run {rl_run_name} ended in state={rl_run.state}.")
    rl_artifact_ref = resolve_model_artifact_ref(rl_run, args.wandb_entity, args.wandb_project)
    print(f"[orchestrator] rl_run={rl_run.name} run_id={rl_run.id} artifact={rl_artifact_ref}", flush=True)

    update_env_export(args.registry_env_file, "BIOREASON_TRAIN_RL_MODEL_REGISTRY_PATH", rl_artifact_ref)

    eval_env = {
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_PROJECT": args.wandb_project,
        "WEAVE_PROJECT": project_ref,
        "BIOREASON_TRAIN_RL_MODEL_REGISTRY_PATH": rl_artifact_ref,
    }
    print(f"[orchestrator] launching final {args.eval_split} eval for {rl_artifact_ref}", flush=True)
    run_local_command(
        [
            "python3",
            "scripts/run_registered_eval.py",
            "--target",
            args.eval_target,
            "--split",
            args.eval_split,
            "--wandb-entity",
            args.wandb_entity,
            "--wandb-project",
            args.wandb_project,
            "--registry-env-file",
            str(args.registry_env_file),
        ],
        eval_env,
    )
    print("[orchestrator] final eval completed", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

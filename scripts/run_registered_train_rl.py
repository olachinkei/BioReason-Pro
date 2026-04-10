#!/usr/bin/env python3
"""Launch spec-first registered RL training from the canonical paper checkpoint."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioreason2.utils.research_registry import load_exported_env_file, normalize_text


DEFAULT_REGISTRY_ENV_FILE = ROOT / "configs" / "disease_benchmark" / "wandb_registry_paths.env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-env-file", type=Path, default=DEFAULT_REGISTRY_ENV_FILE)
    parser.add_argument("--wandb-entity", type=str, default=os.environ.get("WANDB_ENTITY", "wandb-healthcare"))
    parser.add_argument("--wandb-project", type=str, default=os.environ.get("WANDB_PROJECT", "bioreasoning-pro"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint-artifact-name", type=str, default="train-rl-output")
    parser.add_argument("--checkpoint-artifact-aliases", type=str, default="latest,213.221.225.228")
    parser.add_argument("--weave-project", type=str, default=None)
    parser.add_argument("--nnodes", type=int, default=int(os.environ.get("NNODES", "2")))
    parser.add_argument("--gpus-per-node", type=int, default=int(os.environ.get("GPUS_PER_NODE", "4")))
    parser.add_argument("--hostfile", type=Path, default=Path(os.environ["HOSTFILE"]) if os.environ.get("HOSTFILE") else None)
    parser.add_argument("--master-addr", type=str, default=os.environ.get("MASTER_ADDR"))
    parser.add_argument("--master-port", type=str, default=os.environ.get("MASTER_PORT"))
    parser.add_argument("--node-rank", type=str, default=os.environ.get("NODE_RANK"))
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


def latest_named_run(api, project_ref: str, run_name: str, not_before: Optional[datetime] = None):
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


def resolve_model_artifact_ref(run, entity: str, project: str) -> str:
    for artifact in run.logged_artifacts():
        if getattr(artifact, "type", None) == "model":
            return f"{entity}/{project}/{artifact.name}"
    configured_name = normalize_text(run.config.get("model_artifact")).strip()
    if configured_name:
        return f"{entity}/{project}/{configured_name}:latest"
    raise RuntimeError(f"Run {run.name} did not log a model artifact.")


def update_env_export(env_file: Path, key: str, value: str) -> None:
    env_file.parent.mkdir(parents=True, exist_ok=True)
    lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
    updated = False
    prefix = f"export {key}="
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = f'export {key}="{value}"'
            updated = True
            break
    if not updated:
        lines.append(f'export {key}="{value}"')
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_local_command(command: list[str], env_updates: dict[str, str]) -> None:
    env = os.environ.copy()
    env.update({key: value for key, value in env_updates.items() if normalize_text(value).strip()})
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def resolve_registry_env_file(path_value: Path) -> Path:
    if path_value.is_absolute():
        return path_value
    return (ROOT / path_value).resolve()


def build_launch_env(args: argparse.Namespace) -> dict[str, str]:
    run_name = normalize_text(args.run_name).strip() or f"train-rl-paper-{int(time.time())}"
    weave_project = normalize_text(args.weave_project).strip()
    if not weave_project and normalize_text(args.wandb_entity).strip() and normalize_text(args.wandb_project).strip():
        weave_project = f"{args.wandb_entity}/{args.wandb_project}"

    if args.nnodes != 2:
        raise ValueError(f"Spec-first production launch requires --nnodes 2, got {args.nnodes}.")
    if args.gpus_per_node != 4:
        raise ValueError(f"Spec-first production launch requires --gpus-per-node 4, got {args.gpus_per_node}.")

    env = {
        "REGISTRY_ENV_FILE": str(args.registry_env_file),
        "WANDB_ENTITY": normalize_text(args.wandb_entity).strip(),
        "WANDB_PROJECT": normalize_text(args.wandb_project).strip(),
        "BASE_WANDB_PROJECT": normalize_text(args.wandb_project).strip(),
        "WANDB_RUN_NAME": run_name,
        "CHECKPOINT_ARTIFACT_NAME": normalize_text(args.checkpoint_artifact_name).strip(),
        "CHECKPOINT_ARTIFACT_ALIASES": normalize_text(args.checkpoint_artifact_aliases).strip(),
        "NNODES": str(args.nnodes),
        "GPUS_PER_NODE": str(args.gpus_per_node),
    }
    if weave_project:
        env["WEAVE_PROJECT"] = weave_project

    if args.hostfile is not None:
        env["HOSTFILE"] = str(args.hostfile)
        return env

    if not normalize_text(args.master_addr).strip():
        raise ValueError("Spec-first production launch requires --master-addr when --hostfile is not used.")
    if not normalize_text(args.master_port).strip():
        raise ValueError("Spec-first production launch requires --master-port when --hostfile is not used.")
    if not normalize_text(args.node_rank).strip():
        raise ValueError("Spec-first production launch requires --node-rank when --hostfile is not used.")

    env["MASTER_ADDR"] = normalize_text(args.master_addr).strip()
    env["MASTER_PORT"] = normalize_text(args.master_port).strip()
    env["NODE_RANK"] = normalize_text(args.node_rank).strip()
    return env


def main() -> None:
    args = parse_args()
    args.registry_env_file = resolve_registry_env_file(args.registry_env_file)
    load_exported_env_file(str(args.registry_env_file))

    launch_env = build_launch_env(args)
    args.run_name = launch_env["WANDB_RUN_NAME"]
    project_ref = run_ref(args.wandb_entity, args.wandb_project)
    run_not_before = datetime.now(timezone.utc) - timedelta(minutes=10)

    import wandb

    api = wandb.Api()
    print(f"[train_rl] launching registered RL run {args.run_name}", flush=True)
    run_local_command(["bash", "scripts/sh_train_protein_grpo.sh"], launch_env)

    rl_run = latest_named_run(api, project_ref, args.run_name, not_before=run_not_before)
    if rl_run is None:
        raise RuntimeError(f"RL run {args.run_name} was not found on W&B.")
    if rl_run.state != "finished":
        raise RuntimeError(f"RL run {args.run_name} ended in state={rl_run.state}.")

    rl_artifact_ref = resolve_model_artifact_ref(rl_run, args.wandb_entity, args.wandb_project)
    update_env_export(args.registry_env_file, "BIOREASON_TRAIN_RL_MODEL_REGISTRY_PATH", rl_artifact_ref)
    print(f"[train_rl] updated BIOREASON_TRAIN_RL_MODEL_REGISTRY_PATH={rl_artifact_ref}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

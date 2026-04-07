#!/usr/bin/env python3
"""Resolve the exact W&B artifact ref logged by a specific run."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--run-name", required=True, help="W&B display name to match.")
    parser.add_argument("--artifact-name", required=True, help="Artifact collection name, e.g. train-sft-output.")
    parser.add_argument("--attempts", type=int, default=24, help="How many times to poll W&B before failing.")
    parser.add_argument("--sleep-seconds", type=float, default=5.0, help="Seconds to wait between polling attempts.")
    parser.add_argument("--api-timeout", type=int, default=120, help="Timeout passed to wandb.Api().")
    return parser.parse_args()


def iter_matching_runs(api, entity: str, project: str, run_name: str) -> Iterable:
    path = f"{entity}/{project}"
    runs = api.runs(path, filters={"display_name": run_name}, per_page=100)
    return sorted(runs, key=lambda run: getattr(run, "created_at", "") or "", reverse=True)


def main() -> int:
    args = parse_args()
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - runtime environment dependent
        print(f"wandb import failed: {exc}", file=sys.stderr)
        return 1

    api = wandb.Api(timeout=args.api_timeout)
    artifact_prefix = f"{args.artifact_name}:"
    entity_project_prefix = f"{args.wandb_entity}/{args.wandb_project}/{artifact_prefix}"

    for attempt in range(1, max(args.attempts, 1) + 1):
        for run in iter_matching_runs(api, args.wandb_entity, args.wandb_project, args.run_name):
            artifacts = list(run.logged_artifacts())
            for artifact in artifacts:
                ref = f"{args.wandb_entity}/{args.wandb_project}/{artifact.name}"
                if ref.startswith(entity_project_prefix):
                    print(ref)
                    return 0

        if attempt < args.attempts:
            time.sleep(max(args.sleep_seconds, 0.0))

    print(
        f"Could not find logged artifact '{args.artifact_name}' for run '{args.run_name}' "
        f"in {args.wandb_entity}/{args.wandb_project} after {args.attempts} attempts.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

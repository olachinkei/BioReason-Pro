#!/usr/bin/env python
"""Verify the Phase A W&B logging path end-to-end, without touching a GPU.

For each Phase A ablation (A0/A1/A2/A3) this script:
  1. Parses the exact CLI flags the launcher
     (``scripts/sh_train_protein_grpo_phase_a.sh``) will pass.
  2. Resolves the W&B tag list via ``train_protein_grpo.resolve_wandb_tags``.
  3. Calls ``wandb.init(...)`` with the same tags, run name, and entity the
     real training run would use — into the provided ``--project`` (default
     ``bioreason-smoke``) and ``job_type=train_rl_smoke`` so the verification
     runs never pollute the production job type.
  4. Logs one fake metric and finishes the run.

At the end, prints a table of ``<ablation> <run_name> <tags> <run_url>`` so
you can click through and confirm the 4 runs showed up with the expected
tags before spending GPU time on SUNK.

Example::

    WANDB_PROJECT=bioreason-smoke \\
        python scripts/smoke_phase_a_wandb_logging.py

    # Or: override the project name and/or only exercise one ablation:
    python scripts/smoke_phase_a_wandb_logging.py --project bioreason-smoke --only A2
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[1]
GRPO_PATH = ROOT / "train_protein_grpo.py"


def _load_grpo_module():
    spec = importlib.util.spec_from_file_location(
        "train_protein_grpo_wandb_smoke", GRPO_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass
class AblationSpec:
    name: str
    reward_mode: str
    disease_loss_weight: float
    description: str


PHASE_A_GRID: List[AblationSpec] = [
    AblationSpec("A0", "ia_f1", 1.0, "baseline-ia-f1"),
    AblationSpec("A1", "per_aspect_ia_f1", 1.0, "per-aspect-ia-f1"),
    AblationSpec("A2", "per_aspect_lin", 1.0, "per-aspect-lin"),
    AblationSpec("A3", "per_aspect_lin", 1.5, "per-aspect-lin-disease-weighted"),
]


def build_cli_args(spec: AblationSpec) -> List[str]:
    return [
        "--text_model_name",
        "stub-for-wandb-smoke",
        "--reward_mode",
        spec.reward_mode,
        "--disease_loss_weight",
        f"{spec.disease_loss_weight}",
        "--ablation_tag",
        f"phase-a-{spec.name}",
        "--wandb_tags",
        "phase-a-disease-pilot",
        spec.description,
    ]


def run_one_ablation(
    spec: AblationSpec,
    GRPO,
    wandb,
    *,
    project: str,
    entity: str | None,
) -> dict:
    parsed = GRPO.parse_args(build_cli_args(spec))
    tags = GRPO.resolve_wandb_tags(parsed)
    run_name = f"smoke-phase-a-{spec.name}-{int(time.time())}"

    init_kwargs = {
        "project": project,
        "name": run_name,
        "tags": tags,
        "job_type": "train_rl_smoke",
        "config": {
            "ablation": spec.name,
            "reward_mode": parsed.reward_mode,
            "disease_loss_weight": parsed.disease_loss_weight,
            "ablation_tag": parsed.ablation_tag,
            "description": spec.description,
        },
    }
    if entity:
        init_kwargs["entity"] = entity

    run = wandb.init(**init_kwargs)
    run.log({"smoke/alive": 1.0, "smoke/disease_loss_weight": spec.disease_loss_weight}, step=0)
    run_url = getattr(run, "url", "")
    run_path = getattr(run, "path", "")
    run.finish()

    return {
        "ablation": spec.name,
        "run_name": run_name,
        "tags": tags,
        "run_url": run_url,
        "run_path": run_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default="bioreason-smoke",
        help="W&B project to log the 4 smoke runs into (default: bioreason-smoke).",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="Optional W&B entity. Falls back to your default entity if unset.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Optional ablation name (A0/A1/A2/A3) to restrict to a single run.",
    )
    args = parser.parse_args(argv)

    try:
        import wandb  # type: ignore  # noqa: WPS433
    except ImportError:
        print(
            "error: wandb is not installed. Run `uv pip install wandb` (or activate the project venv).",
            file=sys.stderr,
        )
        return 2

    GRPO = _load_grpo_module()

    selected = PHASE_A_GRID
    if args.only:
        filter_name = args.only.upper()
        selected = [spec for spec in PHASE_A_GRID if spec.name == filter_name]
        if not selected:
            print(
                f"error: --only={args.only!r} did not match any Phase A ablation.",
                file=sys.stderr,
            )
            return 2

    print(f"Logging {len(selected)} Phase A smoke run(s) into project={args.project!r}.")
    summaries = []
    for spec in selected:
        summary = run_one_ablation(
            spec,
            GRPO,
            wandb,
            project=args.project,
            entity=args.entity,
        )
        summaries.append(summary)
        print(
            f"  [{summary['ablation']}] run_name={summary['run_name']!r} "
            f"tags={summary['tags']} url={summary['run_url']!r}"
        )

    print(
        f"ok: logged {len(summaries)} run(s) to project={args.project!r}. "
        "Open W&B and confirm each run has the expected tags (phase-a-A0..A3, phase-a-disease-pilot, description)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

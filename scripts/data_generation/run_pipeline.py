#!/usr/bin/env python3
"""Run the public BioReason-Pro data-generation pipeline.

By default this builds:

1. disease temporal split artifact,
2. reasoning DatasetDict,
3. IA.txt from propagated train annotations.

Use --dry-run to print the exact commands without running network-heavy steps.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REQUIRED_TEMPORAL_SPLIT_FILES = [
    "summary.json",
    "report.md",
    "train_assigned_labels.tsv",
    "dev_assigned_labels.tsv",
    "test_assigned_labels.tsv",
    "train_assigned_propagated.tsv",
    "dev_assigned_propagated.tsv",
    "test_assigned_propagated.tsv",
]


@dataclass(frozen=True)
class VariantConfig:
    name: str
    benchmark_tag: str
    benchmark_dir_name: str
    train_start_release: int
    train_end_release: int
    dev_end_release: int
    test_end_release: int
    artifact_aliases: Tuple[str, ...]

    @property
    def temporal_split_output_dir(self) -> str:
        return f"data/artifacts/benchmarks/{self.benchmark_dir_name}/temporal_split"

    @property
    def reasoning_output_dir(self) -> str:
        return f"data/artifacts/datasets/disease_temporal_hc_reasoning_v2/{self.benchmark_dir_name}"


VARIANTS = {
    "main": VariantConfig(
        name="main",
        benchmark_tag="213.221.225.228",
        benchmark_dir_name="213_221_225_228",
        train_start_release=213,
        train_end_release=221,
        dev_end_release=225,
        test_end_release=228,
        artifact_aliases=("213.221.225.228", "production"),
    ),
    "comparison": VariantConfig(
        name="comparison",
        benchmark_tag="214.221.225.228",
        benchmark_dir_name="214_221_225_228",
        train_start_release=214,
        train_end_release=221,
        dev_end_release=225,
        test_end_release=228,
        artifact_aliases=("214.221.225.228",),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["main", "comparison", "all"], default="main")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-temporal-split", action="store_true")
    parser.add_argument("--skip-reasoning-dataset", action="store_true")
    parser.add_argument("--skip-ia", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-refresh-metadata", action="store_true")
    parser.add_argument("--shortlist-mode", choices=["main", "high-confidence"], default="high-confidence")
    parser.add_argument("--shortlist-tsv", default="")
    parser.add_argument("--gaf-dir", default="")
    parser.add_argument("--obo", default="")
    parser.add_argument("--validation-proteins", type=int, default=200)
    parser.add_argument("--holdout-proteins", type=int, default=400)
    parser.add_argument("--partition-seed", type=int, default=23)
    parser.add_argument("--source-metadata-dataset", default=os.environ.get("BIOREASON_SOURCE_METADATA_DATASET", ""))
    parser.add_argument("--source-metadata-name", default=os.environ.get("BIOREASON_SOURCE_METADATA_NAME", ""))
    parser.add_argument("--source-metadata-local-dir", default=os.environ.get("BIOREASON_SOURCE_METADATA_LOCAL_DIR", ""))
    parser.add_argument("--source-metadata-cache-dir", default=os.environ.get("BIOREASON_SOURCE_METADATA_CACHE_DIR", ""))
    parser.add_argument("--interpro-metadata-dataset", default=os.environ.get("BIOREASON_INTERPRO_METADATA_DATASET", ""))
    parser.add_argument("--interpro-metadata-name", default=os.environ.get("BIOREASON_INTERPRO_METADATA_NAME", "interpro_metadata"))
    parser.add_argument("--interpro-metadata-local-dir", default=os.environ.get("BIOREASON_INTERPRO_METADATA_LOCAL_DIR", ""))
    parser.add_argument("--allow-missing-paper-context", action="store_true")
    parser.add_argument("--upload-to-wandb", action="store_true")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", ""))
    parser.add_argument("--temporal-artifact-family", default="disease-temporal-split")
    parser.add_argument("--reasoning-artifact-family", default="disease-temporal-reasoning")
    parser.add_argument("--ia-artifact-family", default="disease-temporal-ia")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_variants(selection: str) -> List[VariantConfig]:
    if selection == "all":
        return [VARIANTS["main"], VARIANTS["comparison"]]
    return [VARIANTS[selection]]


def add_optional(command: List[str], flag: str, value: Optional[str]) -> None:
    if value:
        command.extend([flag, value])


def build_temporal_split_command(args: argparse.Namespace, variant: VariantConfig) -> List[str]:
    command = [
        args.python_bin,
        "scripts/data_generation/build_temporal_split_artifact.py",
        "--output-dir",
        variant.temporal_split_output_dir,
        "--train-start-release",
        str(variant.train_start_release),
        "--train-end-release",
        str(variant.train_end_release),
        "--dev-end-release",
        str(variant.dev_end_release),
        "--test-end-release",
        str(variant.test_end_release),
        "--validation-proteins",
        str(args.validation_proteins),
        "--holdout-proteins",
        str(args.holdout_proteins),
        "--partition-seed",
        str(args.partition_seed),
        "--shortlist-mode",
        args.shortlist_mode,
    ]
    add_optional(command, "--shortlist-tsv", args.shortlist_tsv)
    add_optional(command, "--gaf-dir", args.gaf_dir)
    add_optional(command, "--obo", args.obo)
    if args.force_download:
        command.append("--force-download")
    return command


def build_reasoning_dataset_command(args: argparse.Namespace, variant: VariantConfig) -> List[str]:
    command = [
        args.python_bin,
        "scripts/data_generation/build_reasoning_dataset.py",
        "--temporal-split-dir",
        variant.temporal_split_output_dir,
        "--reasoning-output-dir",
        variant.reasoning_output_dir,
    ]
    add_optional(command, "--source-metadata-dataset", args.source_metadata_dataset)
    add_optional(command, "--source-metadata-name", args.source_metadata_name)
    add_optional(command, "--source-metadata-local-dir", args.source_metadata_local_dir)
    add_optional(command, "--source-metadata-cache-dir", args.source_metadata_cache_dir)
    add_optional(command, "--interpro-metadata-dataset", args.interpro_metadata_dataset)
    add_optional(command, "--interpro-metadata-name", args.interpro_metadata_name)
    add_optional(command, "--interpro-metadata-local-dir", args.interpro_metadata_local_dir)
    if args.allow_missing_paper_context:
        command.append("--allow-missing-paper-context")
    if args.force_refresh_metadata:
        command.append("--force-refresh")
    return command


def build_ia_command(args: argparse.Namespace, variant: VariantConfig) -> List[str]:
    temporal_dir = Path(variant.temporal_split_output_dir)
    obo_path = Path(args.obo) if args.obo else temporal_dir / "go-basic.obo"
    return [
        args.python_bin,
        "scripts/data_generation/build_ia_weights.py",
        "--annotations",
        str(temporal_dir / "train_assigned_propagated.tsv"),
        "--obo",
        str(obo_path),
        "--output",
        str(temporal_dir / "IA.txt"),
    ]


def run_command(command: Sequence[str], *, dry_run: bool) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"[pipeline] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(list(command), cwd=repo_root(), check=True)


def require_files(directory: Path, filenames: Iterable[str]) -> None:
    missing = [name for name in filenames if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(f"{directory} is missing required files: {missing}")


def validate_temporal_split(variant: VariantConfig) -> Dict[str, Any]:
    temporal_dir = repo_root() / variant.temporal_split_output_dir
    require_files(temporal_dir, REQUIRED_TEMPORAL_SPLIT_FILES)
    summary_path = temporal_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "temporal_dir": str(temporal_dir),
        "summary_path": str(summary_path),
        "splits": {item["split"]: item["proteins"] for item in summary.get("windows", [])},
    }


def validate_reasoning_dataset(variant: VariantConfig) -> Dict[str, Any]:
    dataset_dir = repo_root() / variant.reasoning_output_dir
    summary_path = dataset_dir / "dataset_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Reasoning dataset summary is missing: {summary_path}")
    return {
        "reasoning_dir": str(dataset_dir),
        "counts": json.loads(summary_path.read_text(encoding="utf-8")),
    }


def upload_artifact(
    *,
    entity: str,
    project: str,
    name: str,
    artifact_type: str,
    path: Path,
    aliases: Sequence[str],
) -> str:
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - optional runtime feature
        raise RuntimeError("wandb is required for --upload-to-wandb") from exc

    if not entity or not project:
        raise ValueError("--wandb-entity and --wandb-project are required for upload")
    with wandb.init(entity=entity, project=project, job_type="data_generation", name=f"upload-{name}") as run:
        artifact = wandb.Artifact(name=name, type=artifact_type)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        run.log_artifact(artifact, aliases=list(aliases))
        return f"{entity}/{project}/{name}:{aliases[0] if aliases else 'latest'}"


def maybe_upload(args: argparse.Namespace, variant: VariantConfig) -> Dict[str, str]:
    if not args.upload_to_wandb:
        return {}
    temporal_dir = repo_root() / variant.temporal_split_output_dir
    reasoning_dir = repo_root() / variant.reasoning_output_dir
    aliases = variant.artifact_aliases
    return {
        "temporal_split_artifact": upload_artifact(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=f"{args.temporal_artifact_family}-{variant.benchmark_dir_name}",
            artifact_type="dataset",
            path=temporal_dir,
            aliases=aliases,
        ),
        "reasoning_dataset": upload_artifact(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=f"{args.reasoning_artifact_family}-{variant.benchmark_dir_name}",
            artifact_type="dataset",
            path=reasoning_dir,
            aliases=aliases,
        ),
        "ia_file": upload_artifact(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=f"{args.ia_artifact_family}-{variant.benchmark_dir_name}",
            artifact_type="dataset",
            path=temporal_dir / "IA.txt",
            aliases=aliases,
        ),
    }


def run_variant(args: argparse.Namespace, variant: VariantConfig) -> Dict[str, Any]:
    result: Dict[str, Any] = {"variant": asdict(variant), "commands": []}
    if not args.skip_temporal_split:
        command = build_temporal_split_command(args, variant)
        result["commands"].append(command)
        run_command(command, dry_run=args.dry_run)
    if not args.skip_reasoning_dataset:
        command = build_reasoning_dataset_command(args, variant)
        result["commands"].append(command)
        run_command(command, dry_run=args.dry_run)
    if not args.skip_ia:
        command = build_ia_command(args, variant)
        result["commands"].append(command)
        run_command(command, dry_run=args.dry_run)

    if not args.dry_run:
        result["temporal_split"] = validate_temporal_split(variant)
        if not args.skip_reasoning_dataset:
            result["reasoning_dataset"] = validate_reasoning_dataset(variant)
        result["uploads"] = maybe_upload(args, variant)
    return result


def main() -> int:
    args = parse_args()
    results = [run_variant(args, variant) for variant in resolve_variants(args.variant)]
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

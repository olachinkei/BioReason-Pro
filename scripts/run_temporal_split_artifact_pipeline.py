#!/usr/bin/env python3
"""
Prepare local storage, run the temporal split build, validate the outputs,
build datasets, and upload artifacts only when sanity checks pass.
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

try:
    import wandb
except ImportError:  # pragma: no cover - optional at runtime
    wandb = None


REQUIRED_TEMPORAL_SPLIT_FILES = [
    "summary.json",
    "report.md",
    "train_assigned_labels.tsv",
    "dev_assigned_labels.tsv",
    "test_assigned_labels.tsv",
    "train_assigned_propagated.tsv",
    "dev_assigned_propagated.tsv",
    "test_assigned_propagated.tsv",
    "train_assigned_nk_lk.tsv",
    "dev_assigned_nk_lk.tsv",
    "test_assigned_nk_lk.tsv",
    "train_assigned_nk_lk_propagated.tsv",
    "dev_assigned_nk_lk_propagated.tsv",
    "test_assigned_nk_lk_propagated.tsv",
    "nk_lk_eda.tsv",
    "earliest_split_by_protein.json",
]
REPORT_TABLE_SNIPPET = "| Split | Window | Proteins | Unique labels |"


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
    def default_reasoning_dir(self) -> str:
        return f"data/artifacts/datasets/disease_temporal_hc_reasoning_v1/{self.benchmark_dir_name}"


VARIANT_CONFIGS = {
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


def log(message: str) -> None:
    print(message, flush=True)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def parse_aliases(values: Optional[Iterable[str]]) -> List[str]:
    if values is None:
        return []

    if isinstance(values, str):
        raw_values: Iterable[str] = values.split(",")
    else:
        raw_values = values

    aliases: List[str] = []
    for raw_value in raw_values:
        alias = normalize_text(raw_value).strip()
        if alias and alias not in aliases:
            aliases.append(alias)
    return aliases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=["main", "comparison", "all"],
        default="all",
        help="Which temporal split variant(s) to process.",
    )
    parser.add_argument(
        "--temporal-split-script",
        dest="temporal_split_script",
        default="scripts/build_disease_temporal_split_artifact.py",
        help="Path to the temporal split build script.",
    )
    parser.add_argument(
        "--shortlist-mode",
        choices=["main", "high-confidence"],
        default="high-confidence",
        help="Shortlist mode forwarded to the temporal split build script.",
    )
    parser.add_argument(
        "--use-shell-filter",
        action="store_true",
        help="Forward --use-shell-filter to the temporal split build script.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Forward --force-download to the temporal split build script.",
    )
    parser.add_argument(
        "--skip-propagation",
        action="store_true",
        help="Forward --skip-propagation to the temporal split build script.",
    )
    parser.add_argument(
        "--upload-to-wandb",
        action="store_true",
        help="Upload the temporal split artifact and dataset artifacts after sanity checks pass.",
    )
    parser.add_argument(
        "--build-datasets",
        action="store_true",
        help="Build the reasoning dataset before upload.",
    )
    parser.add_argument(
        "--dataset-build-script",
        default="scripts/build_disease_benchmark_datasets.py",
        help="Path to the dataset build script.",
    )
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT"))
    parser.add_argument(
        "--temporal-split-artifact-family",
        dest="temporal_split_artifact_family",
        default="disease-temporal-split",
        help="W&B artifact family for the temporal split artifact.",
    )
    parser.add_argument(
        "--reasoning-artifact-family",
        default="disease-temporal-reasoning",
        help="W&B artifact family for reasoning datasets.",
    )
    parser.add_argument(
        "--reasoning-dir",
        default=None,
        help="Optional dataset directory to upload. Defaults to the variant-specific path.",
    )
    parser.add_argument("--source-metadata-dataset", default=os.getenv("BIOREASON_SOURCE_METADATA_DATASET"))
    parser.add_argument("--source-metadata-name", default=os.getenv("BIOREASON_SOURCE_METADATA_NAME"))
    parser.add_argument("--source-metadata-local-dir", default=os.getenv("BIOREASON_SOURCE_METADATA_LOCAL_DIR"))
    parser.add_argument("--source-metadata-cache-dir", default=os.getenv("BIOREASON_SOURCE_METADATA_CACHE_DIR"))
    parser.add_argument("--interpro-metadata-dataset", default=os.getenv("BIOREASON_INTERPRO_METADATA_DATASET"))
    parser.add_argument("--interpro-metadata-name", default=os.getenv("BIOREASON_INTERPRO_METADATA_NAME"))
    parser.add_argument("--interpro-metadata-local-dir", default=os.getenv("BIOREASON_INTERPRO_METADATA_LOCAL_DIR"))
    parser.add_argument(
        "--allow-missing-paper-context",
        action="store_true",
        help="Allow dataset upload even if GO-GPT / InterPro / PPI prompt context coverage is incomplete.",
    )
    return parser.parse_args()


def resolve_variants(selection: str) -> List[VariantConfig]:
    if selection == "all":
        return [VARIANT_CONFIGS["main"], VARIANT_CONFIGS["comparison"]]
    return [VARIANT_CONFIGS[selection]]


def run_temporal_split_command(repo_root: Path, args: argparse.Namespace, variant: VariantConfig) -> Dict[str, Any]:
    temporal_split_script = repo_root / args.temporal_split_script
    command = [
        sys.executable,
        str(temporal_split_script),
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
        "--shortlist-mode",
        args.shortlist_mode,
    ]
    if args.use_shell_filter:
        command.append("--use-shell-filter")
    if args.force_download:
        command.append("--force-download")
    if args.skip_propagation:
        command.append("--skip-propagation")

    log(f"[pipeline] running temporal split build for {variant.name}: {' '.join(command)}")
    completed = subprocess.run(command, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "completed": completed.returncode == 0,
    }


def run_dataset_build_command(repo_root: Path, args: argparse.Namespace, variant: VariantConfig) -> Dict[str, Any]:
    dataset_build_script = repo_root / args.dataset_build_script
    reasoning_dir = resolve_dataset_dir(args.reasoning_dir, variant.default_reasoning_dir, repo_root)
    command = [
        sys.executable,
        str(dataset_build_script),
        "--temporal-split-dir",
        variant.temporal_split_output_dir,
        "--reasoning-output-dir",
        str(reasoning_dir.relative_to(repo_root)),
    ]
    optional_flags = {
        "--source-metadata-dataset": normalize_text(getattr(args, "source_metadata_dataset", None)),
        "--source-metadata-name": normalize_text(getattr(args, "source_metadata_name", None)),
        "--source-metadata-local-dir": normalize_text(getattr(args, "source_metadata_local_dir", None)),
        "--source-metadata-cache-dir": normalize_text(getattr(args, "source_metadata_cache_dir", None)),
        "--interpro-metadata-dataset": normalize_text(getattr(args, "interpro_metadata_dataset", None)),
        "--interpro-metadata-name": normalize_text(getattr(args, "interpro_metadata_name", None)),
        "--interpro-metadata-local-dir": normalize_text(getattr(args, "interpro_metadata_local_dir", None)),
    }
    for flag, value in optional_flags.items():
        if value:
            command.extend([flag, value])
    if getattr(args, "allow_missing_paper_context", False):
        command.append("--allow-missing-paper-context")
    log(f"[pipeline] building datasets for {variant.name}: {' '.join(command)}")
    completed = subprocess.run(command, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "completed": completed.returncode == 0,
    }


def prepare_local_storage(repo_root: Path, variant: VariantConfig) -> Dict[str, Any]:
    directories = [
        repo_root / "data" / "artifacts",
        repo_root / "data" / "artifacts" / "eval",
        repo_root / variant.temporal_split_output_dir,
        repo_root / variant.default_reasoning_dir,
    ]
    created_dirs: List[str] = []
    for directory in directories:
        resolved = directory.resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        created_dirs.append(str(resolved))
    return {
        "ok": True,
        "created_dirs": created_dirs,
    }


def validate_temporal_split_outputs(output_dir: Path) -> Dict[str, Any]:
    missing_files = [filename for filename in REQUIRED_TEMPORAL_SPLIT_FILES if not (output_dir / filename).exists()]
    report_path = output_dir / "report.md"
    summary_path = output_dir / "summary.json"

    status: Dict[str, Any] = {
        "required_files_present": not missing_files,
        "missing_files": missing_files,
        "summary_json_present": summary_path.exists(),
        "report_present": report_path.exists(),
        "time_order_valid": False,
        "protein_disjoint_valid": False,
        "split_counts_present": False,
        "report_has_split_summary_table": False,
        "proteins_by_split": {},
        "errors": [],
    }

    summary_payload: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            status["errors"].append(f"summary.json is not valid JSON: {exc}")
        else:
            split_validation = summary_payload.get("split_validation") or {}
            status["time_order_valid"] = split_validation.get("time_order_valid") is True
            status["protein_disjoint_valid"] = split_validation.get("protein_disjoint_valid") is True

            windows = summary_payload.get("windows") or []
            proteins_by_split: Dict[str, int] = {}
            for window in windows:
                split_name = window.get("split")
                count = window.get("disease_proteins_after_assignment")
                if isinstance(split_name, str) and isinstance(count, int):
                    proteins_by_split[split_name] = count
            status["proteins_by_split"] = proteins_by_split
            status["split_counts_present"] = all(
                split in proteins_by_split for split in ("train", "dev", "test")
            )

    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
        status["report_has_split_summary_table"] = REPORT_TABLE_SNIPPET in report_text

    status["ok"] = all(
        [
            status["required_files_present"],
            status["summary_json_present"],
            status["report_present"],
            status["time_order_valid"],
            status["protein_disjoint_valid"],
            status["split_counts_present"],
            status["report_has_split_summary_table"],
        ]
    )
    return status


def build_upload_metadata(
    artifact_kind: str,
    local_dir: Path,
    variant: VariantConfig,
    temporal_split_artifact_ref: Optional[str],
) -> Dict[str, Any]:
    metadata = {
        "artifact_kind": artifact_kind,
        "benchmark_tag": variant.benchmark_tag,
        "variant": variant.name,
        "local_dir": str(local_dir),
    }
    if temporal_split_artifact_ref:
        metadata["temporal_split_artifact"] = temporal_split_artifact_ref
    if artifact_kind == "reasoning_dataset":
        metadata["dataset_source"] = "wanglab/cafa5"
        metadata["dataset_name"] = local_dir.parent.name
    return metadata


def upload_directory_artifact(
    *,
    entity: str,
    project: str,
    artifact_name: str,
    local_dir: Path,
    aliases: Sequence[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    if wandb is None:
        raise RuntimeError("wandb is not installed in the current environment")

    resolved_aliases = parse_aliases(aliases)
    run = wandb.init(
        entity=entity,
        project=project,
        job_type="data_prep",
        name=f"upload-{artifact_name}-{metadata['benchmark_tag']}",
        config=metadata,
    )
    artifact = wandb.Artifact(artifact_name, type="dataset", metadata=metadata)
    artifact.add_dir(str(local_dir))
    run.log_artifact(artifact, aliases=resolved_aliases or None)
    run.finish()
    return {
        "artifact_name": artifact_name,
        "local_dir": str(local_dir),
        "aliases": resolved_aliases,
        "uploaded": True,
    }


def resolve_dataset_dir(value: Optional[str], default_relative_path: str, repo_root: Path) -> Path:
    if value:
        return (repo_root / value).resolve()
    return (repo_root / default_relative_path).resolve()


def is_populated_directory(directory: Path) -> bool:
    return directory.is_dir() and any(directory.iterdir())


def validate_reasoning_dataset_outputs(dataset_dir: Path) -> Dict[str, Any]:
    metadata_path = dataset_dir / "build_metadata.json"
    status: Dict[str, Any] = {
        "metadata_present": metadata_path.exists(),
        "paper_context_ready": False,
        "context_coverage_by_split": {},
        "errors": [],
        "ok": False,
    }
    if not metadata_path.exists():
        status["errors"].append("build_metadata.json missing")
        return status
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        status["errors"].append(f"build_metadata.json invalid JSON: {exc}")
        return status

    coverage_by_split = payload.get("context_coverage_by_split") or {}
    status["context_coverage_by_split"] = coverage_by_split
    ready_flags = []
    for split in ("train", "validation", "test"):
        split_payload = coverage_by_split.get(split) or {}
        ready_flags.append(split_payload.get("paper_context_ready") is True)
    status["paper_context_ready"] = bool(ready_flags) and all(ready_flags)
    if not status["paper_context_ready"]:
        status["errors"].append("paper context coverage is incomplete")
    status["ok"] = status["metadata_present"] and status["paper_context_ready"]
    return status


def upload_variant_artifacts(
    repo_root: Path,
    args: argparse.Namespace,
    variant: VariantConfig,
) -> List[Dict[str, Any]]:
    entity = normalize_text(args.wandb_entity).strip()
    project = normalize_text(args.wandb_project).strip()
    if not entity or not project:
        raise ValueError("--wandb-entity and --wandb-project are required for --upload-to-wandb")

    uploads: List[Dict[str, Any]] = []
    temporal_split_dir = (repo_root / variant.temporal_split_output_dir).resolve()
    temporal_split_ref = f"{args.temporal_split_artifact_family}:{variant.benchmark_tag}"

    uploads.append(
        upload_directory_artifact(
            entity=entity,
            project=project,
            artifact_name=args.temporal_split_artifact_family,
            local_dir=temporal_split_dir,
            aliases=variant.artifact_aliases,
            metadata=build_upload_metadata("temporal_split_artifact", temporal_split_dir, variant, None),
        )
    )

    reasoning_dir = resolve_dataset_dir(args.reasoning_dir, variant.default_reasoning_dir, repo_root)
    reasoning_validation = validate_reasoning_dataset_outputs(reasoning_dir) if is_populated_directory(reasoning_dir) else None
    if is_populated_directory(reasoning_dir) and reasoning_validation and reasoning_validation["ok"]:
        uploads.append(
            upload_directory_artifact(
                entity=entity,
                project=project,
                artifact_name=args.reasoning_artifact_family,
                local_dir=reasoning_dir,
                aliases=variant.artifact_aliases,
                metadata=build_upload_metadata("reasoning_dataset", reasoning_dir, variant, temporal_split_ref),
            )
        )
    else:
        uploads.append(
            {
                "artifact_name": args.reasoning_artifact_family,
                "local_dir": str(reasoning_dir),
                "aliases": list(variant.artifact_aliases),
                "uploaded": False,
                "skip_reason": (
                    "directory_missing_or_empty"
                    if reasoning_validation is None
                    else "paper_context_validation_failed"
                ),
                "validation": reasoning_validation,
            }
        )

    return uploads


def write_pipeline_status(output_dir: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "pipeline_status.json"
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_variant_pipeline(repo_root: Path, args: argparse.Namespace, variant: VariantConfig) -> Dict[str, Any]:
    output_dir = (repo_root / variant.temporal_split_output_dir).resolve()
    pipeline_status: Dict[str, Any] = {
        "variant": asdict(variant),
        "temporal_split_output_dir": str(output_dir),
        "local_storage": {},
        "upload_requested": bool(args.upload_to_wandb),
        "temporal_split_build": {},
        "sanity_check": {},
        "dataset_build": {},
        "uploads": [],
        "ok": False,
    }

    pipeline_status["local_storage"] = prepare_local_storage(repo_root, variant)
    temporal_split_result = run_temporal_split_command(repo_root, args, variant)
    pipeline_status["temporal_split_build"] = temporal_split_result
    if not temporal_split_result["completed"]:
        pipeline_status["error"] = (
            f"Temporal split build failed with return code {temporal_split_result['returncode']}"
        )
        write_pipeline_status(output_dir, pipeline_status)
        return pipeline_status

    sanity_status = validate_temporal_split_outputs(output_dir)
    pipeline_status["sanity_check"] = sanity_status
    if not sanity_status["ok"]:
        pipeline_status["error"] = "Sanity check failed; skipping artifact upload."
        write_pipeline_status(output_dir, pipeline_status)
        return pipeline_status

    if getattr(args, "build_datasets", False):
        dataset_build_result = run_dataset_build_command(repo_root, args, variant)
        pipeline_status["dataset_build"] = dataset_build_result
        if not dataset_build_result["completed"]:
            pipeline_status["error"] = (
                f"Dataset build failed with return code {dataset_build_result['returncode']}"
            )
            write_pipeline_status(output_dir, pipeline_status)
            return pipeline_status

    if args.upload_to_wandb:
        pipeline_status["uploads"] = upload_variant_artifacts(repo_root, args, variant)

    pipeline_status["ok"] = True
    write_pipeline_status(output_dir, pipeline_status)
    return pipeline_status


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    variants = resolve_variants(args.variant)

    all_statuses = [run_variant_pipeline(repo_root, args, variant) for variant in variants]
    failures = [status for status in all_statuses if not status.get("ok")]

    for status in all_statuses:
        variant_name = status["variant"]["name"]
        log(
            f"[pipeline] {variant_name}: "
            f"temporal_split_ok={status['temporal_split_build'].get('completed')}, "
            f"sanity_ok={status['sanity_check'].get('ok')}, "
            f"dataset_build_ok={status['dataset_build'].get('completed', not getattr(args, 'build_datasets', False))}, "
            f"uploads={len(status.get('uploads', []))}"
        )

    if failures:
        log(f"[pipeline] failed variants: {[status['variant']['name'] for status in failures]}")
        return 1

    log("[pipeline] all requested variants completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

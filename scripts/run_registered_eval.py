#!/usr/bin/env python3
"""
Run W&B-registry-driven evaluation targets for the disease benchmark.

This script removes the need to manually pass local model or dataset paths:
- data lineage is resolved from a W&B Registry path manifest
- model checkpoints are resolved from W&B Registry refs
- external baseline prediction files can be evaluated from prediction artifacts
"""

import argparse
import ast
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioreason2.utils.research_registry import (
    RegistryError,
    apply_template_context,
    expand_target_group,
    load_exported_env_file,
    load_data_bundle,
    load_eval_target,
    load_eval_target_registry,
    materialize_bundle_asset,
    materialize_first_available_source,
    normalize_text,
)
DEFAULT_DATA_REGISTRY = "configs/disease_benchmark/data_registry.json"
DEFAULT_TARGET_REGISTRY = "configs/disease_benchmark/eval_target_registry.json"
DEFAULT_GO_OBO_PATH = str((ROOT / "bioreason2" / "dataset" / "go-basic.obo").resolve())
DEFAULT_STRUCTURE_DIR = str((ROOT / "data" / "structures").resolve())
DEFAULT_EVAL_OUTPUT_ROOT = "data/artifacts/eval"
DEFAULT_DATASET_CACHE_DIR = "data/artifacts/hf_cache"
DEFAULT_REGISTRY_ENV_FILE = "configs/disease_benchmark/wandb_registry_paths.env"
DEFAULT_MODEL_CACHE_DIR = "data/artifacts/cache/hf_models"
DEFAULT_PAPER_MODEL_LOCAL_DIR = "data/artifacts/models/bioreason_pro_rl_paper"
DEFAULT_TRAIN_SFT_HF_DIR = "data/artifacts/models/train_sft_output_hf"
SFT_TO_HF_CONVERTER = str((ROOT / "bioreason2" / "utils" / "save_unsloth_ckpt.py").resolve())

SAMPLE_TABLE_COLUMNS = [
    "protein_id",
    "go_aspect",
    "split",
    "model_name",
    "prompt",
    "prediction",
    "expected_output",
    "accuracy_or_match_note",
    "reasoning_excerpt",
    "reasoning_full",
    "final_answer",
    "intermediate_trace",
    "success",
    "attempt_count",
    "successful_attempt_count",
    "attempt_file_names",
    "attempt_predictions_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run disease benchmark evaluation from W&B Registry path manifests.")
    parser.add_argument("--target", type=str, default=None, help="Single evaluation target to run.")
    parser.add_argument(
        "--target-group",
        type=str,
        default=None,
        help="Named target group from the evaluation-target manifest.",
    )
    parser.add_argument(
        "--data-bundle",
        type=str,
        default=None,
        help="Named data bundle from the data-bundle manifest. Defaults to the manifest default bundle.",
    )
    parser.add_argument(
        "--data-manifest-path",
        "--data-registry-path",
        type=str,
        dest="data_manifest_path",
        default=DEFAULT_DATA_REGISTRY,
        help="Path to the data-bundle W&B Registry path manifest JSON.",
    )
    parser.add_argument(
        "--target-manifest-path",
        "--target-registry-path",
        type=str,
        dest="target_manifest_path",
        default=DEFAULT_TARGET_REGISTRY,
        help="Path to the evaluation-target W&B Registry path manifest JSON.",
    )
    parser.add_argument(
        "--registry-env-file",
        type=str,
        default=DEFAULT_REGISTRY_ENV_FILE,
        help="Optional env file with W&B Registry refs for data and model artifacts.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Evaluation split to run. Validation defaults to a 100-sample stratified subset; test defaults to the full split.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_EVAL_OUTPUT_ROOT,
        help="Root directory for temporary eval scratch outputs.",
    )
    parser.add_argument(
        "--keep-local-eval-outputs",
        action="store_true",
        help="Keep local eval scratch after W&B logging instead of cleaning it up.",
    )
    parser.add_argument("--wandb-project", type=str, default=os.getenv("WANDB_PROJECT") or None)
    parser.add_argument("--wandb-entity", type=str, default=os.getenv("WANDB_ENTITY") or None)
    parser.add_argument("--wandb-mode", type=str, default=os.getenv("WANDB_MODE") or None)
    parser.add_argument("--weave-project", type=str, default=os.getenv("WEAVE_PROJECT") or None)
    parser.add_argument(
        "--dataset-cache-dir",
        type=str,
        default=os.getenv("BIOREASON_DATASET_CACHE_DIR") or DEFAULT_DATASET_CACHE_DIR,
    )
    parser.add_argument(
        "--go-embeddings-path",
        type=str,
        default=os.getenv("BIOREASON_GO_EMBEDDINGS_PATH") or "",
    )
    parser.add_argument(
        "--go-obo-path",
        type=str,
        default=os.getenv("BIOREASON_GO_OBO_PATH") or DEFAULT_GO_OBO_PATH,
    )
    parser.add_argument(
        "--ia-file-path",
        type=str,
        default=os.getenv("BIOREASON_IA_FILE_PATH") or "",
    )
    parser.add_argument(
        "--structure-dir",
        type=str,
        default=os.getenv("BIOREASON_STRUCTURE_DIR") or DEFAULT_STRUCTURE_DIR,
    )
    parser.add_argument("--metric-threads", type=int, default=0)
    parser.add_argument("--metric-threshold-step", type=float, default=0.99)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override sample count. Defaults to 100 for validation and full split for test.",
    )
    parser.add_argument(
        "--sample-strategy",
        type=str,
        default="stratified_aspect_profile",
        choices=["stratified_aspect_profile", "shuffled_prefix"],
    )
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue when an optional target or one item in a target group fails.",
    )
    return parser.parse_args()


def resolve_runtime_paths(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "dataset_cache_dir": str((ROOT / args.dataset_cache_dir).resolve())
        if not Path(args.dataset_cache_dir).is_absolute()
        else args.dataset_cache_dir,
        "go_embeddings_path": str((ROOT / args.go_embeddings_path).resolve())
        if args.go_embeddings_path and not Path(args.go_embeddings_path).is_absolute()
        else args.go_embeddings_path,
        "go_obo_path": str((ROOT / args.go_obo_path).resolve())
        if args.go_obo_path and not Path(args.go_obo_path).is_absolute()
        else args.go_obo_path,
        "ia_file_path": str((ROOT / args.ia_file_path).resolve())
        if args.ia_file_path and not Path(args.ia_file_path).is_absolute()
        else args.ia_file_path,
        "structure_dir": str((ROOT / args.structure_dir).resolve())
        if args.structure_dir and not Path(args.structure_dir).is_absolute()
        else args.structure_dir,
        "output_root": str((ROOT / args.output_root).resolve())
        if not Path(args.output_root).is_absolute()
        else args.output_root,
    }


def normalize_bundle(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    bundle_copy = dict(bundle)
    releases = bundle_copy.get("releases", {})
    bundle_copy["benchmark_alias_dir"] = bundle_copy.get("benchmark_alias", "").replace(".", "_")
    bundle_copy["train_start_release"] = releases.get("train_start")
    bundle_copy["train_end_release"] = releases.get("train_end")
    bundle_copy["dev_end_release"] = releases.get("dev_end")
    bundle_copy["test_end_release"] = releases.get("test_end")
    return bundle_copy


def materialize_data_bundle(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    resolved = normalize_bundle(bundle)
    output = dict(resolved)
    output["temporal_split_artifact"] = materialize_bundle_asset(resolved.get("temporal_split_artifact", {}))
    output["reasoning_dataset"] = materialize_bundle_asset(resolved.get("reasoning_dataset", {}))
    return output


def with_bundle_context(target: Mapping[str, Any], bundle: Mapping[str, Any]) -> Dict[str, Any]:
    context = {
        "benchmark_alias": bundle.get("benchmark_alias", ""),
        "benchmark_alias_dir": bundle.get("benchmark_alias_dir", ""),
        "benchmark_version": bundle.get("benchmark_version", ""),
    }
    return apply_template_context(dict(target), context)


def ensure_parent(path_value: str) -> str:
    Path(path_value).parent.mkdir(parents=True, exist_ok=True)
    return path_value


def resolve_effective_max_samples(args: argparse.Namespace) -> int:
    if args.max_samples is not None:
        return args.max_samples
    if args.split == "validation":
        return 100
    return -1


def resolve_dataset_loader_source(asset: Mapping[str, Any]) -> str:
    """Prefer a downloaded local dataset artifact directory when available."""
    local_dir = normalize_text(asset.get("local_dir")).strip()
    if local_dir:
        local_path = Path(local_dir)
        if local_path.is_dir() and any(local_path.iterdir()):
            return str(local_path)
    return normalize_text(asset.get("dataset_source") or "wanglab/cafa5")


def build_run_names(target_name: str, split: str, benchmark_alias: str) -> Dict[str, str]:
    suffix = f"{target_name}-{split}-{benchmark_alias}"
    return {
        "run_name": f"eval-{suffix}",
        "artifact_name": f"eval-{suffix}",
        "weave_eval_name": f"eval-{suffix}",
    }


def wandb_can_be_source_of_truth(args: argparse.Namespace) -> bool:
    project = normalize_text(args.wandb_project).strip()
    if not project:
        return False
    mode = normalize_text(args.wandb_mode).strip().lower()
    return mode not in {"offline", "disabled", "dryrun"}


def should_cleanup_local_eval_outputs(args: argparse.Namespace) -> bool:
    return (not getattr(args, "keep_local_eval_outputs", False)) and wandb_can_be_source_of_truth(args)


def remove_local_eval_output(path_value: Path) -> None:
    if path_value.exists():
        shutil.rmtree(path_value)


def remove_empty_parent(path_value: Path) -> None:
    try:
        if path_value.exists() and path_value.is_dir() and not any(path_value.iterdir()):
            path_value.rmdir()
    except OSError:
        pass


def run_shell_command(command: Sequence[str], env: Mapping[str, str]) -> None:
    subprocess.run(list(command), cwd=str(ROOT), env=dict(env), check=True)


def prepare_clean_eval_output_dir(path_value: Path) -> Path:
    """Always start eval runs from a clean scratch directory."""
    remove_local_eval_output(path_value)
    path_value.mkdir(parents=True, exist_ok=True)
    return path_value


def select_checkpoint_for_eval(model_dir: Path) -> Optional[Path]:
    checkpoints = sorted(model_dir.rglob("*.ckpt"))
    if not checkpoints:
        return None
    preferred = [path for path in checkpoints if "-best-" in path.name]
    candidates = preferred or checkpoints
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def resolve_hf_model_dir_for_eval(
    target: Mapping[str, Any],
    resolved_model: Mapping[str, Any],
    runtime_paths: Mapping[str, str],
) -> Dict[str, Any]:
    model_dir = Path(resolved_model["local_path"])
    if (model_dir / "config.json").is_file():
        return dict(resolved_model)

    ckpt_path = select_checkpoint_for_eval(model_dir)
    if ckpt_path is None:
        raise RegistryError(
            f"Resolved model source for {target['target_name']} does not contain config.json or a .ckpt checkpoint: "
            f"{model_dir}"
        )

    paper_model_ref = normalize_text(os.environ.get("BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH")).strip()
    if not paper_model_ref:
        raise RegistryError(
            "BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH is required to convert raw SFT checkpoints into HF format "
            "for evaluation."
        )

    paper_model_source = materialize_first_available_source(
        [
            {
                "type": "wandb_artifact",
                "wandb_registry_path": paper_model_ref,
                "local_dir": str((ROOT / DEFAULT_PAPER_MODEL_LOCAL_DIR).resolve()),
                "required_paths": ["config.json"],
            }
        ],
        allow_missing=False,
    )
    paper_model_dir = Path(paper_model_source["local_path"])

    hf_model_dir = (ROOT / DEFAULT_TRAIN_SFT_HF_DIR).resolve()
    if hf_model_dir.exists():
        shutil.rmtree(hf_model_dir)
    hf_model_dir.parent.mkdir(parents=True, exist_ok=True)

    converter_command = [
        sys.executable,
        SFT_TO_HF_CONVERTER,
        "--checkpoint_path",
        str(ckpt_path),
        "--save_dir",
        str(hf_model_dir),
        "--text_model_name",
        str(paper_model_dir),
        "--protein_model_name",
        "esm3_sm_open_v1",
        "--cache_dir",
        str((ROOT / DEFAULT_MODEL_CACHE_DIR).resolve()),
        "--max_length_text",
        "4000",
        "--max_length_protein",
        "2000",
        "--lora_rank",
        "128",
        "--lora_alpha",
        "256",
        "--lora_dropout",
        "0.05",
        "--protein_embedding_layer",
        "-1",
        "--go_obo_path",
        normalize_text(runtime_paths.get("go_obo_path")),
        "--go_hidden_dim",
        "512",
        "--go_num_gat_layers",
        "3",
        "--go_num_heads",
        "8",
        "--go_num_reduced_embeddings",
        "200",
        "--go_embedding_dim",
        "2560",
        "--unified_go_encoder",
        "False",
        "--protein_model_finetune",
        "False",
    ]
    checkpoint_go_embeddings = paper_model_dir / "go_embedding.pt"
    if checkpoint_go_embeddings.is_file():
        converter_command.extend(["--precomputed_embeddings_path", str(checkpoint_go_embeddings)])

    run_shell_command(converter_command, os.environ.copy())
    if not (hf_model_dir / "config.json").is_file():
        raise RegistryError(f"Converted HF model directory is missing config.json: {hf_model_dir}")

    converted = dict(resolved_model)
    converted["local_path"] = str(hf_model_dir)
    converted["converted_from_checkpoint"] = str(ckpt_path)
    converted["source_ref"] = normalize_text(resolved_model.get("source_ref"))
    return converted


def run_protein_llm_target(
    args: argparse.Namespace,
    bundle: Mapping[str, Any],
    target: Mapping[str, Any],
    runtime_paths: Mapping[str, str],
) -> Dict[str, Any]:
    resolved_model = materialize_first_available_source(
        target.get("model_sources", []),
        allow_missing=bool(target.get("optional")),
    )
    if not resolved_model:
        raise RegistryError(f"Could not resolve any model source for {target['target_name']}.")
    resolved_model = resolve_hf_model_dir_for_eval(target, resolved_model, runtime_paths)

    output_dir = Path(runtime_paths["output_root"]) / target["target_name"] / args.split
    prepare_clean_eval_output_dir(output_dir)

    names = build_run_names(target["target_name"], args.split, bundle["benchmark_alias"])
    env = os.environ.copy()
    env.update(
        {
            "EVALS_DIR": str(output_dir),
            "EVALS_PATH": str(output_dir / "results"),
            "MODEL_PATH": resolved_model["local_path"],
            "MODEL_NAME": normalize_text(target.get("display_name") or target["target_name"]),
            "GO_OBO_PATH": runtime_paths["go_obo_path"],
            "IA_FILE_PATH": runtime_paths["ia_file_path"],
            "GO_EMBEDDINGS_PATH": runtime_paths["go_embeddings_path"],
            "DATASET_CACHE_DIR": runtime_paths["dataset_cache_dir"],
            "STRUCTURE_DIR": runtime_paths["structure_dir"],
            "CAFA5_DATASET": resolve_dataset_loader_source(bundle["reasoning_dataset"]),
            "DATASET_NAME": normalize_text(bundle["reasoning_dataset"].get("dataset_name")),
            "REASONING_DATASET_NAME": normalize_text(bundle["reasoning_dataset"].get("dataset_name")),
            "EVAL_SPLIT": args.split,
            "BENCHMARK_VERSION": normalize_text(bundle.get("benchmark_version")),
            "TEMPORAL_SPLIT_ARTIFACT": normalize_text(bundle["temporal_split_artifact"].get("wandb_registry_path")),
            "DATASET_ARTIFACT": normalize_text(bundle["reasoning_dataset"].get("wandb_registry_path")),
            "MODEL_ARTIFACT": normalize_text(resolved_model.get("source_ref")),
            "SHORTLIST_QUERY": normalize_text(bundle.get("shortlist_query")),
            "SHORTLIST_MODE": normalize_text(bundle.get("shortlist_mode")),
            "TRAIN_START_RELEASE": normalize_text(bundle.get("train_start_release")),
            "TRAIN_END_RELEASE": normalize_text(bundle.get("train_end_release")),
            "DEV_END_RELEASE": normalize_text(bundle.get("dev_end_release")),
            "TEST_END_RELEASE": normalize_text(bundle.get("test_end_release")),
            "METRIC_THREADS": str(args.metric_threads),
            "METRIC_THRESHOLD_STEP": str(args.metric_threshold_step),
            "MAX_SAMPLES": str(resolve_effective_max_samples(args)),
            "SAMPLE_STRATEGY": normalize_text(getattr(args, "sample_strategy", "stratified_aspect_profile")),
            "NUM_CHUNKS": str(args.num_chunks),
            "CHUNK_ID": str(args.chunk_id),
            "WANDB_PROJECT": normalize_text(args.wandb_project),
            "WANDB_ENTITY": normalize_text(args.wandb_entity),
            "WANDB_MODE": normalize_text(args.wandb_mode),
            "WANDB_RUN_NAME": names["run_name"],
            "WANDB_ARTIFACT_NAME": names["artifact_name"],
            "WEAVE_PROJECT": normalize_text(args.weave_project),
            "WEAVE_EVAL_NAME": names["weave_eval_name"],
            "KEEP_LOCAL_EVAL_OUTPUTS": "1" if args.keep_local_eval_outputs else "0",
        }
    )
    run_shell_command(["bash", "scripts/sh_eval.sh"], env)
    if should_cleanup_local_eval_outputs(args):
        remove_empty_parent(output_dir)

    return {
        "target_name": target["target_name"],
        "runner": target["runner"],
        "status": "completed",
        "output_dir": str(output_dir),
        "model_source": resolved_model.get("source_ref"),
        "max_samples": resolve_effective_max_samples(args),
    }


def parse_go_terms(value: Any) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(item) for item in value if item}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return set()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                return {str(item) for item in parsed if item}
        if text.startswith("GO:"):
            return {item.strip() for item in text.split() if item.startswith("GO:")}
    return set()


def load_ground_truth_split(
    dataset_source: str,
    dataset_name: str,
    split: str,
    cache_dir: str,
) -> List[Dict[str, Any]]:
    from datasets import load_dataset, load_from_disk

    dataset_path = Path(dataset_source)
    if dataset_path.is_dir() and (dataset_path / "dataset_dict.json").exists():
        dataset_dict = load_from_disk(str(dataset_path))
    else:
        dataset_dict = load_dataset(dataset_source, name=dataset_name, cache_dir=cache_dir)
    dataset_split = dataset_dict[split]
    rows: List[Dict[str, Any]] = []
    for sample in dataset_split:
        gt_terms = set()
        for column in ("go_bp", "go_mf", "go_cc"):
            gt_terms.update(parse_go_terms(sample.get(column)))
        rows.append(
            {
                "protein_id": normalize_text(sample.get("protein_id")),
                "ground_truth_terms": gt_terms,
            }
        )
    return rows


def copy_prediction_files(source_dir: str, destination_dir: str, pattern: str) -> List[str]:
    source_path = Path(source_dir)
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    copied_files: List[str] = []
    for input_file in sorted(source_path.glob(pattern)):
        target_file = dest_path / input_file.name
        shutil.copy2(input_file, target_file)
        copied_files.append(str(target_file))

    if not copied_files:
        raise RegistryError(f"No prediction files matching '{pattern}' were found in {source_dir}.")
    return copied_files


def parse_prediction_files(prediction_files: Iterable[str]) -> Dict[str, Set[str]]:
    predictions: Dict[str, Set[str]] = {}
    for file_path in prediction_files:
        with open(file_path, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if len(row) < 2:
                    continue
                protein_id = row[0].strip()
                go_term = row[1].strip()
                if not protein_id or not go_term.startswith("GO:"):
                    continue
                predictions.setdefault(protein_id, set()).add(go_term)
    return predictions


def write_cafa_ground_truth(
    rows: Sequence[Mapping[str, Any]],
    output_path: str,
) -> str:
    ensure_parent(output_path)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            protein_id = normalize_text(row.get("protein_id"))
            for go_term in sorted(row.get("ground_truth_terms", set())):
                handle.write(f"{protein_id}\t{go_term}\n")
    return output_path


def write_sample_table(
    rows: Sequence[Mapping[str, Any]],
    prediction_map: Mapping[str, Set[str]],
    *,
    split: str,
    model_name: str,
    output_path: str,
) -> str:
    ensure_parent(output_path)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_TABLE_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            protein_id = normalize_text(row.get("protein_id"))
            gt_terms = set(row.get("ground_truth_terms", set()))
            pred_terms = set(prediction_map.get(protein_id, set()))
            true_positives = len(gt_terms & pred_terms)
            writer.writerow(
                {
                    "protein_id": protein_id,
                    "go_aspect": "ALL",
                    "split": split,
                    "model_name": model_name,
                    "prompt": "",
                    "prediction": ", ".join(sorted(pred_terms)),
                    "expected_output": ", ".join(sorted(gt_terms)),
                    "accuracy_or_match_note": f"tp={true_positives}; pred={len(pred_terms)}; gt={len(gt_terms)}",
                    "reasoning_excerpt": "",
                    "reasoning_full": "",
                    "final_answer": "",
                    "intermediate_trace": "",
                    "success": bool(pred_terms),
                    "attempt_count": 1,
                    "successful_attempt_count": 1 if pred_terms else 0,
                    "attempt_file_names": "",
                    "attempt_predictions_json": json.dumps(sorted(pred_terms)),
                }
            )
    return output_path


def maybe_log_prediction_eval_to_wandb(
    *,
    args: argparse.Namespace,
    bundle: Mapping[str, Any],
    target: Mapping[str, Any],
    metrics: Mapping[str, Any],
    sample_rows: Sequence[Mapping[str, Any]],
    results_dir: str,
    run_name: str,
) -> Optional[str]:
    project = normalize_text(args.wandb_project).strip()
    if not project:
        return None

    try:
        import wandb
    except ImportError:  # pragma: no cover - depends on runtime environment
        return None

    config = {
        "job_type": "eval",
        "benchmark_version": bundle.get("benchmark_version"),
        "temporal_split_artifact": bundle.get("temporal_split_artifact", {}).get("wandb_registry_path"),
        "dataset_config": bundle.get("reasoning_dataset", {}).get("dataset_name"),
        "dataset_artifact": bundle.get("reasoning_dataset", {}).get("wandb_registry_path"),
        "shortlist_query": bundle.get("shortlist_query"),
        "shortlist_mode": bundle.get("shortlist_mode"),
        "train_start_release": bundle.get("train_start_release"),
        "train_end_release": bundle.get("train_end_release"),
        "dev_end_release": bundle.get("dev_end_release"),
        "test_end_release": bundle.get("test_end_release"),
        "model_name": target.get("display_name", target["target_name"]),
        "model_artifact": normalize_text(target.get("resolved_source_ref")),
        "local_eval_outputs_retained": bool(args.keep_local_eval_outputs),
    }

    run = wandb.init(
        project=project,
        entity=normalize_text(args.wandb_entity).strip() or None,
        dir=os.getenv("WANDB_DIR") or os.getcwd(),
        name=run_name,
        job_type="eval",
        mode=normalize_text(args.wandb_mode).strip() or None,
        config=config,
        reinit=True,
    )

    if metrics:
        run.log(dict(metrics))

    summary_columns = [
        "model_name",
        "dataset_config",
        "split",
        "benchmark_version",
        "fmax_mf",
        "fmax_bp",
        "fmax_cc",
        "macro_note",
    ]
    summary_table = wandb.Table(columns=summary_columns)
    summary_table.add_data(
        target.get("display_name", target["target_name"]),
        bundle.get("reasoning_dataset", {}).get("dataset_name"),
        args.split,
        bundle.get("benchmark_version"),
        metrics.get("fmax_mf"),
        metrics.get("fmax_bp"),
        metrics.get("fmax_cc"),
        "prediction-artifact baseline",
    )

    sample_table = wandb.Table(columns=SAMPLE_TABLE_COLUMNS)
    for row in sample_rows:
        sample_table.add_data(*(row.get(column) for column in SAMPLE_TABLE_COLUMNS))

    run.log({"eval_summary": summary_table, "eval_samples": sample_table})

    artifact = wandb.Artifact(
        f"{run_name}-results",
        type="eval-results",
        metadata=config,
    )
    artifact.add_dir(results_dir)
    run.log_artifact(artifact)
    run.finish()
    return artifact.name


def run_prediction_artifact_target(
    args: argparse.Namespace,
    bundle: Mapping[str, Any],
    target: Mapping[str, Any],
    runtime_paths: Mapping[str, str],
) -> Dict[str, Any]:
    resolved_predictions = materialize_first_available_source(
        target.get("prediction_sources", []),
        allow_missing=bool(target.get("optional")),
    )
    if not resolved_predictions:
        raise RegistryError(f"Could not resolve any prediction artifact for {target['target_name']}.")

    output_dir = Path(runtime_paths["output_root"]) / target["target_name"] / args.split
    prepare_clean_eval_output_dir(output_dir)
    results_dir = output_dir / "results"
    prediction_input_dir = results_dir / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = copy_prediction_files(
        resolved_predictions["local_path"],
        str(prediction_input_dir),
        normalize_text(target.get("prediction_glob") or "*.tsv"),
    )
    prediction_map = parse_prediction_files(prediction_files)

    ground_truth_rows = load_ground_truth_split(
        dataset_source=resolve_dataset_loader_source(bundle["reasoning_dataset"]),
        dataset_name=normalize_text(bundle["reasoning_dataset"].get("dataset_name")),
        split=args.split,
        cache_dir=runtime_paths["dataset_cache_dir"],
    )
    ground_truth_file = write_cafa_ground_truth(ground_truth_rows, str(results_dir / "ground_truth.tsv"))

    from evals import cafa_evals

    evaluation_results = cafa_evals.run_cafa_evaluation(
        runtime_paths["go_obo_path"],
        str(prediction_input_dir),
        ground_truth_file,
        ia_file_path=runtime_paths["ia_file_path"] or None,
        n_cpu=args.metric_threads,
        th_step=args.metric_threshold_step,
    )
    metrics = cafa_evals.normalize_metrics_for_logging(cafa_evals.extract_metrics_summary(evaluation_results))
    metrics_summary_path = cafa_evals.write_metrics_summary(metrics, str(results_dir / "cafa_metrics"))

    sample_table_path = write_sample_table(
        ground_truth_rows,
        prediction_map,
        split=args.split,
        model_name=normalize_text(target.get("display_name") or target["target_name"]),
        output_path=str(results_dir / "sample_results.tsv"),
    )

    run_summary = {
        "job_type": "eval",
        "target_name": target["target_name"],
        "runner": target["runner"],
        "split": args.split,
        "benchmark_version": bundle.get("benchmark_version"),
        "metrics_summary_path": metrics_summary_path,
        "prediction_source": resolved_predictions.get("source_ref"),
        "ground_truth_file": ground_truth_file,
        "sample_results_tsv": sample_table_path,
    }
    run_summary_path = results_dir / "run_summary.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2, sort_keys=True), encoding="utf-8")

    sample_rows: List[Dict[str, Any]] = []
    with open(sample_table_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        sample_rows = list(reader)

    names = build_run_names(target["target_name"], args.split, bundle["benchmark_alias"])
    target["resolved_source_ref"] = resolved_predictions.get("source_ref")
    artifact_logged = maybe_log_prediction_eval_to_wandb(
        args=args,
        bundle=bundle,
        target=target,
        metrics=metrics,
        sample_rows=sample_rows,
        results_dir=str(results_dir),
        run_name=names["run_name"],
    )
    if artifact_logged and should_cleanup_local_eval_outputs(args):
        remove_local_eval_output(output_dir)
        remove_empty_parent(output_dir.parent)

    return {
        "target_name": target["target_name"],
        "runner": target["runner"],
        "status": "completed",
        "output_dir": str(output_dir),
        "prediction_source": resolved_predictions.get("source_ref"),
    }


def run_target(
    args: argparse.Namespace,
    bundle: Mapping[str, Any],
    target: Mapping[str, Any],
    runtime_paths: Mapping[str, str],
) -> Dict[str, Any]:
    runner = normalize_text(target.get("runner")).strip()
    if runner == "protein_llm":
        return run_protein_llm_target(args, bundle, target, runtime_paths)
    if runner == "prediction_artifact":
        return run_prediction_artifact_target(args, bundle, target, runtime_paths)
    raise RegistryError(f"Unsupported target runner: {runner}")


def main() -> None:
    args = parse_args()
    registry_env_file = normalize_text(args.registry_env_file).strip()
    if registry_env_file:
        if Path(registry_env_file).is_absolute():
            load_exported_env_file(registry_env_file)
        else:
            load_exported_env_file(str((ROOT / registry_env_file).resolve()))
    runtime_paths = resolve_runtime_paths(args)
    Path(runtime_paths["dataset_cache_dir"]).mkdir(parents=True, exist_ok=True)
    Path(runtime_paths["output_root"]).mkdir(parents=True, exist_ok=True)

    bundle = materialize_data_bundle(load_data_bundle(args.data_manifest_path, args.data_bundle, ROOT))
    registry = load_eval_target_registry(args.target_manifest_path, ROOT)
    target_names = expand_target_group(registry, args.target, args.target_group)

    statuses: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for target_name in target_names:
        target = with_bundle_context(load_eval_target(registry, target_name, ROOT), bundle)
        try:
            status = run_target(args, bundle, target, runtime_paths)
        except Exception as exc:  # pragma: no cover - exercised in integration/unit tests via mocks
            status = {
                "target_name": target_name,
                "runner": target.get("runner"),
                "status": "failed",
                "error": str(exc),
            }
            is_optional = bool(target.get("optional"))
            if not (args.continue_on_error or (len(target_names) > 1 and is_optional)):
                statuses.append(status)
                if args.keep_local_eval_outputs:
                    suite_summary_path = Path(runtime_paths["output_root"]) / "suite_summary.json"
                    suite_summary_path.write_text(json.dumps({"statuses": statuses}, indent=2), encoding="utf-8")
                raise
            failures.append(status)
        statuses.append(status)

    suite_summary = {
        "bundle_name": bundle.get("bundle_name"),
        "benchmark_version": bundle.get("benchmark_version"),
        "split": args.split,
        "statuses": statuses,
        "failures": failures,
    }
    if args.keep_local_eval_outputs:
        suite_summary_path = Path(runtime_paths["output_root"]) / "suite_summary.json"
        suite_summary_path.write_text(json.dumps(suite_summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()

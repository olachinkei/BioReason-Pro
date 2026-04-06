#!/usr/bin/env python3
"""
CAFA-5 Evaluation Script

Features:
- Modular function design for maintainability
- Individual JSON file output per protein_id + go_aspect combination
- Robust error handling and logging
- Progress tracking and resumable execution
- Multi-GPU safe concurrent execution
- Professional argument parsing with grouped options

Usage:
    python eval.py --ckpt_dir /path/to/checkpoint --evals_path /path/to/results [options]
"""

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
from tqdm import tqdm
import traceback

from bioreason2.models.protein_vllm import ProteinLLMModel
from bioreason2.dataset.cafa5.load import load_cafa5_dataset
from bioreason2.utils import str2bool

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency at runtime
    wandb = None

try:
    import weave
except ImportError:  # pragma: no cover - optional dependency at runtime
    weave = None

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STOP_TOKENS = ["<|im_end|>"]
ERROR_LOG_FILE = "evaluation_errors.json"
RUN_SUMMARY_FILE = "run_summary.json"
SAMPLE_TABLE_FILE = "sample_results.tsv"
REASONING_OPEN_TAG = "<think>"
REASONING_CLOSE_TAG = "</think>"

# GO Aspect mapping for cleaner filenames
GO_ASPECT_CODES = {"molecular_function": "MF", "cellular_component": "CC", "biological_process": "BP"}

def get_go_aspect_code(go_aspect: str) -> str:
    """Convert GO aspect to short code for cleaner filenames."""
    return GO_ASPECT_CODES.get(go_aspect, go_aspect)


def _get_ground_truth(sample: Dict[str, Any]) -> str:
    """Extracts the ground truth assistant reasoning and answer from the sample."""
    prompt_data = sample.get("prompt")
    if isinstance(prompt_data, list):
        for message in prompt_data:
            if message.get("role") == "assistant":
                reasoning = message.get("reasoning_content", "")
                answer = ""
                content = message.get("content", [])
                if isinstance(content, list) and content:
                    answer = content[0].get("text", "")
                return f"{reasoning}\n\n{answer}" if reasoning and answer else reasoning or answer
    return sample.get("answer", "")


def parse_result_filename(filename: str) -> Optional[Tuple[str, str]]:
    """Parse saved eval filenames while preserving protein IDs with underscores."""
    if not filename.endswith(".json"):
        return None

    base_name = filename[:-5]
    if "_k" in base_name and base_name.rsplit("_k", 1)[1].isdigit():
        base_name = base_name.rsplit("_k", 1)[0]

    if "_" not in base_name:
        return None

    protein_id, go_aspect_code = base_name.rsplit("_", 1)
    if not protein_id or not go_aspect_code:
        return None
    return protein_id, go_aspect_code


def _normalize_text_for_match(text: str) -> str:
    """Normalize whitespace for lightweight exact-match notes."""
    return " ".join((text or "").split())


def extract_reasoning_fields(text: str) -> Dict[str, str]:
    """Split a reasoning response into trace and final-answer fields."""
    raw_text = (text or "").strip()
    if not raw_text:
        return {
            "reasoning_excerpt": "",
            "reasoning_full": "",
            "final_answer": "",
            "intermediate_trace": "",
        }

    reasoning_full = ""
    final_answer = raw_text

    if REASONING_CLOSE_TAG in raw_text:
        reasoning_part, final_part = raw_text.split(REASONING_CLOSE_TAG, 1)
        if REASONING_OPEN_TAG in reasoning_part:
            reasoning_full = reasoning_part.split(REASONING_OPEN_TAG, 1)[1].strip()
        else:
            reasoning_full = reasoning_part.strip()
        final_answer = final_part.strip()
    elif REASONING_OPEN_TAG in raw_text:
        reasoning_full = raw_text.split(REASONING_OPEN_TAG, 1)[1].strip()
        final_answer = ""

    return {
        "reasoning_excerpt": reasoning_full[:500],
        "reasoning_full": reasoning_full,
        "final_answer": final_answer,
        "intermediate_trace": reasoning_full,
    }


def initialize_model(args) -> ProteinLLMModel:
    """Initialize and return the ProteinLLMModel."""
    print(f"📥 Loading ProteinLLMModel from checkpoint: {args.ckpt_dir}...")
    model = ProteinLLMModel(
        ckpt_dir=args.ckpt_dir,
        protein_model_name=args.protein_model_name,
        protein_embedding_layer=args.protein_embedding_layer,
        go_obo_path=args.go_obo_path,
        precomputed_embeddings_path=args.precomputed_embeddings_path,
        max_length_protein=args.max_length_protein,
        max_length_text=args.max_model_len,
        max_model_len=args.max_model_len,
        unified_go_encoder=args.unified_go_encoder,
        go_hidden_dim=args.go_hidden_dim,
        go_num_gat_layers=args.go_num_gat_layers,
        go_num_heads=args.go_num_heads,
        go_num_reduced_embeddings=args.go_num_reduced_embeddings,
        go_embedding_dim=args.go_embedding_dim,
        text_model_finetune=False,
        protein_model_finetune=False,
        go_model_finetune=False,
    )
    print("Model initialized successfully.")
    return model


def select_eval_dataset(train_ds, val_ds, test_ds, eval_split: str):
    """Select the requested evaluation split from a pre-split dataset triple."""
    split_to_dataset = {
        "validation": val_ds,
        "test": test_ds,
    }
    if eval_split not in split_to_dataset:
        raise ValueError(f"Unsupported eval split: {eval_split}")

    dataset = split_to_dataset[eval_split]
    if dataset is None or len(dataset) == 0:
        raise ValueError(f"{eval_split.title()} dataset is empty or failed to load.")
    return dataset


def load_dataset(args):
    """Load and prepare the requested CAFA-5 evaluation split."""
    print(f"\n📥 Loading and preparing CAFA-5 {args.eval_split} dataset...")
    train_ds, val_ds, test_ds = load_cafa5_dataset(
        dataset=args.cafa5_dataset,
        dataset_name=args.cafa5_dataset_name,
        cache_dir=args.dataset_cache_dir,
        dataset_subset=args.cafa5_dataset_subset,
        max_length=args.max_length_protein,
        seed=args.seed,
        val_split_ratio=args.val_split_ratio,
        return_as_chat_template=True,
        split_go_aspects=args.split_go_aspects,
        structure_dir=args.structure_dir,
        include_go_defs=args.include_go_defs,
        interpro_dataset_name=args.interpro_dataset_name,
        include_protein_function_summary=args.include_protein_function_summary,
        interpro_in_prompt=args.interpro_in_prompt,
        predict_interpro=args.predict_interpro,
        ppi_in_prompt=args.ppi_in_prompt,
        reasoning_dataset_name=args.reasoning_dataset_name,
        go_gpt_predictions_column=args.go_gpt_predictions_column,
        min_go_mf_freq=args.min_go_mf_freq,
        min_go_bp_freq=args.min_go_bp_freq,
        min_go_cc_freq=args.min_go_cc_freq,
        apply_go_filtering_to_val_test=args.apply_go_filtering_to_val_test,
        add_uniprot_summary=args.add_uniprot_summary,
        debug=args.debug,
    )
    eval_ds = select_eval_dataset(train_ds, val_ds, test_ds, args.eval_split)
    eval_ds = eval_ds.shuffle(seed=args.seed)

    n_samples = len(eval_ds) if args.max_samples <= 0 else min(args.max_samples, len(eval_ds))

    # Handle chunking for multi-GPU processing
    if args.num_chunks > 1:
        chunk_size = n_samples // args.num_chunks
        start_idx = args.chunk_id * chunk_size

        if args.chunk_id == args.num_chunks - 1:
            # Last chunk gets any remaining samples
            end_idx = n_samples
        else:
            end_idx = start_idx + chunk_size

        print(
            f"Processing {args.eval_split} chunk {args.chunk_id + 1}/{args.num_chunks}: "
            f"samples {start_idx} to {end_idx-1}"
        )
        samples = eval_ds.select(range(start_idx, end_idx))
    else:
        print(f"📊 Processing full {args.eval_split} dataset (no chunking)")
        samples = eval_ds.select(range(n_samples))

    print(f"Loaded {len(samples)} samples for evaluation from split: {args.eval_split}")
    return samples


def filter_unprocessed_samples(samples, evals_path: str) -> List[Dict[str, Any]]:
    """Filter out already processed samples and return only unprocessed ones.
    
    Simplified logic: If ANY file exists for a (protein_id, go_aspect) combination,
    skip it entirely. Don't worry about whether all k iterations are complete.
    """
    os.makedirs(evals_path, exist_ok=True)
    processed_ids = set()

    if os.path.exists(evals_path):
        existing_files = os.listdir(evals_path)
        for filename in existing_files:
            if filename.endswith(".json"):
                if filename in {ERROR_LOG_FILE, RUN_SUMMARY_FILE}:
                    continue
                parsed = parse_result_filename(filename)
                if parsed is not None:
                    protein_id, go_aspect_code = parsed
                    processed_unique_id = f"{protein_id}_{go_aspect_code}"
                    processed_ids.add(processed_unique_id)
        
        print(f"🔄 Found {len(processed_ids)} samples with at least one result file.")

    # Filter out already processed samples
    print("🔍 Filtering out already processed samples...")
    unprocessed_samples = []
    for sample in samples:
        protein_id = sample.get("protein_id")
        go_aspect = sample.get("go_aspect")
        go_aspect_code = get_go_aspect_code(go_aspect)
        sample_unique_id = f"{protein_id}_{go_aspect_code}"
        
        if sample_unique_id not in processed_ids:
            unprocessed_samples.append(sample)
    
    print(f"📊 Total samples: {len(samples)}")
    print(f"Already processed: {len(samples) - len(unprocessed_samples)}")
    print(f"Remaining to process: {len(unprocessed_samples)}")

    return unprocessed_samples


def save_result(result_record: Dict[str, Any], protein_id: str, go_aspect: str, evals_path: str, k_idx: int = 0) -> None:
    """Save individual result to its own JSON file using short GO aspect codes."""
    go_aspect_code = get_go_aspect_code(go_aspect)
    result_filename = f"{protein_id}_{go_aspect_code}_k{k_idx:02d}.json"
    result_filepath = os.path.join(evals_path, result_filename)

    with open(result_filepath, "w") as f:
        json.dump(result_record, f, indent=4)


def collect_result_rows(evals_path: str) -> List[Dict[str, Any]]:
    """Collect saved per-sample JSON files into tabular rows."""
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(evals_path):
        return rows

    for filename in sorted(os.listdir(evals_path)):
        if not filename.endswith(".json"):
            continue
        if filename in {ERROR_LOG_FILE, RUN_SUMMARY_FILE}:
            continue

        filepath = os.path.join(evals_path, filename)
        try:
            with open(filepath, "r") as f:
                record = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if not isinstance(record, dict):
            continue
        if "protein_id" not in record or "go_aspect" not in record:
            continue

        rows.append(
            {
                "file_name": filename,
                "protein_id": record.get("protein_id", ""),
                "go_aspect": record.get("go_aspect", ""),
                "success": bool(record.get("success", False)),
                "sequence_length": record.get("sequence_length", 0),
                "input_prompt": record.get("input_prompt", ""),
                "ground_truth": record.get("ground_truth", ""),
                "generated_response": record.get("generated_response", ""),
            }
        )

    return rows


def build_sample_table_rows(args, result_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert saved JSON rows into richer sample rows for tracking backends."""
    rows: List[Dict[str, Any]] = []
    model_name = resolve_model_name(args)

    for row in result_rows:
        prediction_fields = extract_reasoning_fields(row.get("generated_response", ""))
        ground_truth_fields = extract_reasoning_fields(row.get("ground_truth", ""))
        predicted_final = _normalize_text_for_match(prediction_fields["final_answer"])
        expected_final = _normalize_text_for_match(ground_truth_fields["final_answer"])
        exact_match = bool(predicted_final and expected_final and predicted_final == expected_final)

        rows.append(
            {
                "protein_id": row.get("protein_id", ""),
                "go_aspect": row.get("go_aspect", ""),
                "split": args.eval_split,
                "model_name": model_name,
                "prompt": row.get("input_prompt", ""),
                "prediction": row.get("generated_response", ""),
                "expected_output": row.get("ground_truth", ""),
                "accuracy_or_match_note": f"success={bool(row.get('success', False))}; exact_match={exact_match}",
                "reasoning_excerpt": prediction_fields["reasoning_excerpt"],
                "reasoning_full": prediction_fields["reasoning_full"],
                "final_answer": prediction_fields["final_answer"],
                "intermediate_trace": prediction_fields["intermediate_trace"],
                "success": bool(row.get("success", False)),
            }
        )

    return rows


def normalize_metrics_summary(metrics_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Add stable logging aliases for metric summaries when available."""
    normalized = dict(metrics_summary or {})
    alias_map = {
        "molecular_function_f1": "fmax_mf",
        "biological_process_f1": "fmax_bp",
        "cellular_component_f1": "fmax_cc",
        "overall_mean_f1": "overall_mean_fmax",
        "molecular_function_weighted_f1": "weighted_fmax_mf",
        "biological_process_weighted_f1": "weighted_fmax_bp",
        "cellular_component_weighted_f1": "weighted_fmax_cc",
        "overall_mean_weighted_f1": "overall_mean_weighted_fmax",
    }
    for source_key, alias_key in alias_map.items():
        if source_key in normalized and alias_key not in normalized:
            normalized[alias_key] = normalized[source_key]
    return normalized


def load_metrics_summary(metrics_summary_path: Optional[str]) -> Dict[str, Any]:
    """Load an optional metrics summary JSON for downstream tracking."""
    if not metrics_summary_path:
        return {}
    if not os.path.exists(metrics_summary_path):
        print(f"⚠️  Metrics summary not found, skipping metric logging: {metrics_summary_path}")
        return {}

    try:
        with open(metrics_summary_path, "r") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"⚠️  Failed to load metrics summary {metrics_summary_path}: {exc}")
        return {}

    if not isinstance(payload, dict):
        print(f"⚠️  Metrics summary must be a JSON object: {metrics_summary_path}")
        return {}
    return normalize_metrics_summary(payload)


def resolve_model_name(args) -> str:
    """Resolve a stable model name for tracking outputs."""
    explicit_name = getattr(args, "model_name", None)
    if explicit_name:
        return explicit_name

    ckpt_dir = getattr(args, "ckpt_dir", "")
    normalized = os.path.basename(os.path.normpath(ckpt_dir))
    return normalized or "unknown_model"


def resolve_benchmark_version(args) -> str:
    """Resolve a benchmark identifier for tracking configs."""
    explicit_version = getattr(args, "benchmark_version", None)
    if explicit_version:
        return explicit_version
    return args.reasoning_dataset_name or args.cafa5_dataset_name


def build_eval_summary_row(args, run_summary: Dict[str, Any], metrics_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Build the one-row summary table payload described in the spec."""
    metrics_summary = normalize_metrics_summary(metrics_summary)
    return {
        "model_name": resolve_model_name(args),
        "dataset_config": args.cafa5_dataset_name,
        "split": args.eval_split,
        "benchmark_version": resolve_benchmark_version(args),
        "fmax_mf": metrics_summary.get("fmax_mf"),
        "fmax_bp": metrics_summary.get("fmax_bp"),
        "fmax_cc": metrics_summary.get("fmax_cc"),
        "macro_note": (
            f"loaded={run_summary['loaded_samples']}, "
            f"processed={run_summary['newly_processed_samples']}, "
            f"result_files={run_summary['result_files_total']}, "
            f"metrics_loaded={bool(metrics_summary)}"
        ),
    }


def build_tracking_config(args, run_summary: Dict[str, Any], metrics_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Build a shared config payload for W&B tracking."""
    config = {
        "job_type": "eval",
        "benchmark_version": resolve_benchmark_version(args),
        "step0_artifact": getattr(args, "step0_artifact", None),
        "dataset_config": args.cafa5_dataset_name,
        "reasoning_dataset_config": args.reasoning_dataset_name,
        "dataset_artifact": getattr(args, "dataset_artifact", None),
        "shortlist_query": getattr(args, "shortlist_query", None),
        "shortlist_mode": getattr(args, "shortlist_mode", None),
        "train_start_release": getattr(args, "train_start_release", None),
        "train_end_release": getattr(args, "train_end_release", None),
        "dev_end_release": getattr(args, "dev_end_release", None),
        "test_end_release": getattr(args, "test_end_release", None),
        "base_checkpoint": args.ckpt_dir,
        "model_artifact": getattr(args, "model_artifact", None),
        "output_dir": args.evals_path,
        "seed": args.seed,
        "eval_split": args.eval_split,
        "pass_at_k": args.pass_at_k,
        "max_samples": args.max_samples,
        "metrics_summary_path": getattr(args, "metrics_summary_path", None),
        "result_files_total": run_summary.get("result_files_total"),
        "unique_sample_keys_total": run_summary.get("unique_sample_keys_total"),
        "successful_result_files_total": run_summary.get("successful_result_files_total"),
    }
    config.update(
        {
            key: value
            for key, value in normalize_metrics_summary(metrics_summary).items()
            if isinstance(value, (int, float))
        }
    )
    return {key: value for key, value in config.items() if value not in (None, "")}


def write_sample_results_table(rows: List[Dict[str, Any]], evals_path: str) -> str:
    """Write a sample-level TSV that mirrors the saved JSON results."""
    output_path = os.path.join(evals_path, SAMPLE_TABLE_FILE)
    fieldnames = [
        "file_name",
        "protein_id",
        "go_aspect",
        "success",
        "sequence_length",
        "input_prompt",
        "ground_truth",
        "generated_response",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return output_path


def build_run_summary(
    args,
    loaded_samples: int,
    remaining_samples: int,
    newly_processed: int,
    total_time: float,
    result_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a machine-readable summary for the evaluation run."""
    unique_samples = {f"{row['protein_id']}::{row['go_aspect']}" for row in result_rows}
    successful_rows = sum(1 for row in result_rows if row.get("success"))
    return {
        "job_type": "eval",
        "eval_split": args.eval_split,
        "dataset_name": args.cafa5_dataset_name,
        "reasoning_dataset_name": args.reasoning_dataset_name,
        "checkpoint_dir": args.ckpt_dir,
        "max_samples": args.max_samples,
        "pass_at_k": args.pass_at_k,
        "loaded_samples": loaded_samples,
        "remaining_samples_before_run": remaining_samples,
        "newly_processed_samples": newly_processed,
        "result_files_total": len(result_rows),
        "unique_sample_keys_total": len(unique_samples),
        "successful_result_files_total": successful_rows,
        "duration_seconds": round(total_time, 3),
    }


def write_run_summary(summary: Dict[str, Any], evals_path: str) -> str:
    """Persist the evaluation run summary as JSON."""
    output_path = os.path.join(evals_path, RUN_SUMMARY_FILE)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4, sort_keys=True)
    return output_path


def build_wandb_table(rows: List[Dict[str, Any]]):
    """Build a W&B table if the SDK is available."""
    if wandb is None or not rows:
        return None
    columns = list(rows[0].keys())
    data = [[row.get(column) for column in columns] for row in rows]
    return wandb.Table(columns=columns, data=data)


def maybe_init_wandb_run(args, run_summary: Dict[str, Any], metrics_summary: Dict[str, Any]):
    """Initialize an optional W&B eval run."""
    if wandb is None:
        return None

    project = (getattr(args, "wandb_project", None) or os.getenv("WANDB_PROJECT") or "").strip()
    if not project:
        return None

    init_kwargs = {
        "project": project,
        "entity": (getattr(args, "wandb_entity", None) or os.getenv("WANDB_ENTITY") or None),
        "dir": args.evals_path,
        "name": getattr(args, "wandb_run_name", None) or f"eval-{resolve_model_name(args)}-{args.eval_split}",
        "job_type": "eval",
        "config": build_tracking_config(args, run_summary, metrics_summary),
        "reinit": "finish_previous",
    }
    wandb_mode = getattr(args, "wandb_mode", None)
    if wandb_mode:
        init_kwargs["mode"] = wandb_mode

    try:
        return wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"⚠️  W&B init failed, continuing without W&B tracking: {exc}")
        return None


def log_eval_outputs_to_wandb(
    run,
    args,
    run_summary: Dict[str, Any],
    metrics_summary: Dict[str, Any],
    sample_rows: List[Dict[str, Any]],
) -> bool:
    """Log eval metrics, tables, and artifacts to W&B."""
    if run is None or wandb is None:
        return False

    metrics_to_log = {
        key: value for key, value in normalize_metrics_summary(metrics_summary).items() if isinstance(value, (int, float))
    }
    if metrics_to_log:
        wandb.log(metrics_to_log)

    summary_row = build_eval_summary_row(args, run_summary, metrics_summary)
    summary_table = build_wandb_table([summary_row])
    if summary_table is not None:
        wandb.log({"eval_summary": summary_table})

    sample_table = build_wandb_table(sample_rows)
    if sample_table is not None:
        wandb.log({"eval_samples": sample_table})

    artifact_name = getattr(args, "wandb_artifact_name", None) or f"eval-results-{resolve_model_name(args)}-{args.eval_split}"
    artifact_name = artifact_name.replace("/", "-").replace(" ", "-")
    artifact = wandb.Artifact(
        artifact_name,
        type="evaluation",
        metadata={
            "job_type": "eval",
            "eval_split": args.eval_split,
            "benchmark_version": resolve_benchmark_version(args),
        },
    )
    artifact.add_dir(args.evals_path)
    run.log_artifact(artifact)
    return True


def maybe_log_eval_to_weave(
    args,
    run_summary: Dict[str, Any],
    metrics_summary: Dict[str, Any],
    sample_rows: List[Dict[str, Any]],
) -> bool:
    """Track eval rows with Weave Evaluation when configured."""
    if weave is None or not sample_rows:
        return False

    weave_project = (getattr(args, "weave_project", None) or "").strip()
    if not weave_project:
        wandb_entity = (getattr(args, "wandb_entity", None) or os.getenv("WANDB_ENTITY") or "").strip()
        wandb_project = (getattr(args, "wandb_project", None) or os.getenv("WANDB_PROJECT") or "").strip()
        if wandb_entity and wandb_project:
            weave_project = f"{wandb_entity}/{wandb_project}"
    if not weave_project:
        return False

    try:
        weave.init(weave_project)

        @weave.op
        def replay_prediction(
            prediction: str,
            reasoning_full: str = "",
            final_answer: str = "",
            intermediate_trace: str = "",
        ) -> Dict[str, str]:
            return {
                "prediction": prediction,
                "reasoning_full": reasoning_full,
                "final_answer": final_answer,
                "intermediate_trace": intermediate_trace,
            }

        @weave.op
        def exact_match_score(expected_output: str, model_output: Dict[str, str]) -> Dict[str, Any]:
            expected_final = extract_reasoning_fields(expected_output).get("final_answer", "")
            predicted_final = model_output.get("final_answer", "")
            return {
                "exact_match": _normalize_text_for_match(expected_final) == _normalize_text_for_match(predicted_final)
            }

        evaluation = weave.Evaluation(
            name=getattr(args, "weave_eval_name", None) or f"eval-{resolve_model_name(args)}-{args.eval_split}",
            dataset=sample_rows,
            scorers=[exact_match_score],
            preprocess_model_input=lambda row: {
                "prediction": row.get("prediction", ""),
                "reasoning_full": row.get("reasoning_full", ""),
                "final_answer": row.get("final_answer", ""),
                "intermediate_trace": row.get("intermediate_trace", ""),
            },
            metadata={
                "job_type": "eval",
                "benchmark_version": resolve_benchmark_version(args),
                "eval_split": args.eval_split,
                "result_files_total": run_summary.get("result_files_total"),
                "metrics_loaded": bool(metrics_summary),
            },
        )
        evaluation.evaluate(replay_prediction)
        return True
    except Exception as exc:
        print(f"⚠️  Weave eval logging failed, continuing without Weave tracking: {exc}")
        return False


def log_eval_tracking(
    args,
    run_summary: Dict[str, Any],
    result_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Log optional W&B / Weave tracking for the completed eval run."""
    metrics_summary = load_metrics_summary(getattr(args, "metrics_summary_path", None))
    sample_rows = build_sample_table_rows(args, result_rows)
    tracking_status = {
        "metrics_loaded": bool(metrics_summary),
        "wandb_logged": False,
        "weave_logged": False,
    }

    wandb_run = maybe_init_wandb_run(args, run_summary, metrics_summary)
    try:
        tracking_status["wandb_logged"] = log_eval_outputs_to_wandb(
            run=wandb_run,
            args=args,
            run_summary=run_summary,
            metrics_summary=metrics_summary,
            sample_rows=sample_rows,
        )
    finally:
        if wandb_run is not None:
            finish = getattr(wandb_run, "finish", None)
            if callable(finish):
                finish()

    tracking_status["weave_logged"] = maybe_log_eval_to_weave(
        args=args,
        run_summary=run_summary,
        metrics_summary=metrics_summary,
        sample_rows=sample_rows,
    )
    return tracking_status


def log_error(evals_path: str, error_type: str, protein_id: str, go_aspect: str, go_bp: str, go_mf: str, go_cc: str, go_bp_leaf: str, go_mf_leaf: str, go_cc_leaf: str, error_msg: str = "") -> None:
    """Log errors to a centralized JSON file."""
    os.makedirs(evals_path, exist_ok=True)
    error_log_path = os.path.join(evals_path, ERROR_LOG_FILE)
    error_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "error_type": error_type,
        "protein_id": protein_id,
        "go_aspect": go_aspect,
        "go_bp": go_bp,
        "go_mf": go_mf,
        "go_cc": go_cc,
        "go_bp_leaf": go_bp_leaf,
        "go_mf_leaf": go_mf_leaf,
        "go_cc_leaf": go_cc_leaf,
        "error_message": error_msg if error_msg else ("Out of Memory" if error_type == "oom" else "Unknown error"),
    }

    # Load existing errors or create new list
    errors = []
    if os.path.exists(error_log_path):
        try:
            with open(error_log_path, "r") as f:
                errors = json.load(f)
        except (json.JSONDecodeError, Exception):
            errors = []

    # Append new error
    errors.append(error_record)

    # Save back to file
    with open(error_log_path, "w") as f:
        json.dump(errors, f, indent=4)


def process_single_sample(
    model: ProteinLLMModel, sample: Dict[str, Any], protein_id: str, go_aspect: str, go_bp: str, go_mf: str, go_cc: str, go_bp_leaf: str, go_mf_leaf: str, go_cc_leaf: str, args
) -> Dict[str, Any]:
    """Process a single sample and return the result."""
    conversation_data = sample.get("prompt")
    if conversation_data is None:
        print(f"No prompt data for protein {protein_id}, skipping...")
        return None

    # Extract only system and user messages for generation
    # Filter out assistant messages to create proper generation prompt
    user_conversation = []
    for message in conversation_data:
        if message.get("role") in ["system", "user"]:
            user_conversation.append(message)
        elif message.get("role") == "assistant":
            # Stop at first assistant message - we only want the input
            break

    final_prompt_string = model.text_tokenizer.apply_chat_template(
        user_conversation,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,  # Avoid empty thinking injection
    )

    sequence = sample.get("sequence")
    if sequence is None:
        print(f"No sequence data for protein {protein_id}, skipping...")
        return None

    processed_inputs = model.processor(
        text=[final_prompt_string],
        batch_protein_sequences=[[sequence]],
        batch_go_aspects=[go_aspect],
        max_length_text=model.max_length_text,
        max_length_protein=model.max_length_protein,
        return_tensors="pt",
    )

    input_ids = processed_inputs.get("input_ids").to(DEVICE)
    attention_mask = processed_inputs.get("attention_mask").to(DEVICE)
    structure_coords = processed_inputs.get("structure_coords")

    # Run Inference
    with torch.inference_mode():
        generated_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            protein_sequences=[sequence],
            batch_idx_map=[0],
            go_aspects=[go_aspect],
            structure_coords=structure_coords,
            # Pass generation parameters from args
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            stop=STOP_TOKENS,
        )

    response_text = generated_outputs[0] if generated_outputs else "Error: Empty response"

    result_record = {
        "protein_id": protein_id,
        "go_aspect": go_aspect,
        "ground_truth": _get_ground_truth(sample),
        "generated_response": response_text,
        "success": True,
        "protein_sequence": sequence,
        "input_prompt": final_prompt_string,
        "sequence_length": len(sequence) if sequence else 0,
        "go_bp": go_bp,
        "go_mf": go_mf,
        "go_cc": go_cc,
        "go_bp_leaf": go_bp_leaf,
        "go_mf_leaf": go_mf_leaf,
        "go_cc_leaf": go_cc_leaf,
    }

    return result_record


def print_final_statistics(newly_processed: int, total_time: float, evals_path: str) -> None:
    """Print final evaluation statistics."""
    total_files = len(collect_result_rows(evals_path))

    print("\nEvaluation complete.")
    print(f"⏱️  Processed {newly_processed} new samples in {total_time:.2f}s")
    if newly_processed > 0:
        print(f"📈 Processing rate: {newly_processed/total_time:.2f} samples/s")
    print(f"💾 Total result files: {total_files} in directory: {evals_path}")
    print("Individual JSON files saved for each protein_id + aspect combination")


def run_local_inference(args):
    """
    Main function to orchestrate data loading, model inference, and result saving.
    """
    print("--- Starting Local CAFA-5 Inference ---")

    try:
        # Initialize model
        model = initialize_model(args)

        # Load dataset
        samples = load_dataset(args)
        loaded_samples = len(samples)

        # Filter out already processed samples
        unprocessed_samples = filter_unprocessed_samples(samples, args.evals_path)
        remaining_samples = len(unprocessed_samples)

        # Main inference loop - only process unprocessed samples
        print(f"\nStarting inference loop with pass@{args.pass_at_k}...")
        t_start = time.time()
        successfully_processed = 0

        for sample in tqdm(unprocessed_samples, desc="Processing Samples", total=len(unprocessed_samples), unit="sample"):
            protein_id = sample.get("protein_id")
            go_aspect = sample.get("go_aspect", "all")
            go_bp = sample.get("go_bp", "")
            go_mf = sample.get("go_mf", "")
            go_cc = sample.get("go_cc", "")
            go_bp_leaf = sample.get("go_bp_leaf", "")
            go_mf_leaf = sample.get("go_mf_leaf", "")
            go_cc_leaf = sample.get("go_cc_leaf", "")

            # Generate k samples for pass@k
            sample_has_success = False
            for k_idx in range(args.pass_at_k):
                try:
                    result_record = process_single_sample(model, sample, protein_id, go_aspect, go_bp, go_mf, go_cc, go_bp_leaf, go_mf_leaf, go_cc_leaf, args)
                    if result_record is not None:
                        save_result(result_record, protein_id, go_aspect, args.evals_path, k_idx=k_idx)
                        if not sample_has_success:
                            successfully_processed += 1
                            sample_has_success = True

                except torch.cuda.OutOfMemoryError:
                    print(f"CUDA Out of Memory on sample ID: {protein_id}, k={k_idx}. Skipping this k iteration.")
                    log_error(
                        args.evals_path,
                        "oom",
                        protein_id,
                        go_aspect,
                        go_bp,
                        go_mf,
                        go_cc,
                        go_bp_leaf,
                        go_mf_leaf,
                        go_cc_leaf,
                    )
                    torch.cuda.empty_cache()
                    continue

                except Exception as e:
                    print(f"Unexpected error on sample ID {protein_id}, k={k_idx}: {e}")
                    log_error(
                        args.evals_path,
                        "other",
                        protein_id,
                        go_aspect,
                        go_bp,
                        go_mf,
                        go_cc,
                        go_bp_leaf,
                        go_mf_leaf,
                        go_cc_leaf,
                        str(e),
                    )
                    traceback.print_exc()
                    continue

        # Print final statistics
        t_end = time.time()
        dt = t_end - t_start
        print_final_statistics(successfully_processed, dt, args.evals_path)
        result_rows = collect_result_rows(args.evals_path)
        sample_table_path = write_sample_results_table(result_rows, args.evals_path)
        run_summary = build_run_summary(
            args=args,
            loaded_samples=loaded_samples,
            remaining_samples=remaining_samples,
            newly_processed=successfully_processed,
            total_time=dt,
            result_rows=result_rows,
        )
        tracking_status = log_eval_tracking(args, run_summary, result_rows)
        run_summary.update(tracking_status)
        summary_path = write_run_summary(run_summary, args.evals_path)
        print(f"🧾 Sample-level TSV saved to: {sample_table_path}")
        print(f"🧾 Run summary saved to: {summary_path}")
        print(
            "🧾 Tracking status: "
            f"W&B={run_summary['wandb_logged']}, "
            f"Weave={run_summary['weave_logged']}, "
            f"metrics_loaded={run_summary['metrics_loaded']}"
        )

    except Exception as e:
        print(f"Critical Error: {e}")
        traceback.print_exc()
        return


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup and return the argument parser."""
    parser = argparse.ArgumentParser(description="Local CAFA inference with ProteinLLMModel")

    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--ckpt_dir", type=str, required=True, help="Path to the ProteinLLMModel checkpoint directory."
    )
    model_group.add_argument(
        "--protein_model_name", type=str, default="esm3_sm_open_v1", help="Name of the protein encoder model."
    )
    model_group.add_argument(
        "--protein_embedding_layer",
        type=int,
        default=-1,
        help="ESM3 layer to extract embeddings from. Use -1 for final output (default), 0-N for specific transformer layers. Only works with ESM3 models."
    )
    model_group.add_argument("--go_obo_path", type=str, required=True, help="Path to GO ontology .obo file.")
    model_group.add_argument(
        "--precomputed_embeddings_path",
        type=str,
        required=True,
        help="Path to directory with precomputed GO embeddings.",
    )
    model_group.add_argument(
        "--unified_go_encoder",
        type=str2bool,
        default=False,
        help="If True, use unified GOGraphEncoderUnified; if False, use original GOGraphEncoder.",
    )
    model_group.add_argument("--max_model_len", type=int, default=32768, help="Maximum length of the model.")
    model_group.add_argument(
        "--go_hidden_dim", type=int, default=512, help="Hidden dimension for GO GAT layers (must match training)."
    )
    model_group.add_argument(
        "--go_num_gat_layers", type=int, default=3, help="Number of GAT layers in GO encoder (must match training)."
    )
    model_group.add_argument(
        "--go_num_heads", type=int, default=8, help="Number of attention heads in GO GAT (must match training)."
    )
    model_group.add_argument(
        "--go_num_reduced_embeddings",
        type=int,
        default=200,
        help="Number of reduced embeddings per GO namespace (must match training).",
    )
    model_group.add_argument(
        "--go_embedding_dim", type=int, default=2560, help="GO embedding dimension (must match training)."
    )

    # Dataset options
    dataset_group = parser.add_argument_group("Dataset Configuration")
    dataset_group.add_argument("--cafa5_dataset", type=str, default="wanglab/cafa5")
    dataset_group.add_argument("--cafa5_dataset_name", type=str, default="cafa5_reasoning")
    dataset_group.add_argument("--cafa5_dataset_subset", type=str, default=None)
    dataset_group.add_argument("--dataset_cache_dir", type=str, default=None)
    dataset_group.add_argument(
        "--structure_dir", type=str, default=None
    )
    dataset_group.add_argument("--include_go_defs", type=str2bool, default=False)
    dataset_group.add_argument("--interpro_dataset_name", type=str, default="interpro_metadata")
    dataset_group.add_argument("--split_go_aspects", type=str2bool, default=True)
    dataset_group.add_argument("--interpro_in_prompt", type=str2bool, default=True)
    dataset_group.add_argument("--predict_interpro", type=str2bool, default=False)
    dataset_group.add_argument("--ppi_in_prompt", type=str2bool, default=True)
    dataset_group.add_argument("--include_protein_function_summary", type=str2bool, default=True)
    dataset_group.add_argument("--val_split_ratio", type=float, default=0.1)
    dataset_group.add_argument("--seed", type=int, default=23)
    dataset_group.add_argument("--debug", type=str2bool, default=False)
    dataset_group.add_argument(
        "--max_length_protein", type=int, default=2048, help="Maximum length of protein sequences."
    )
    dataset_group.add_argument("--enable_thinking", type=str2bool, default=True)
    dataset_group.add_argument(
        "--reasoning_dataset_name",
        type=str,
        default=None,
        help="Config name for reasoning traces dataset (e.g., 'experiment_data_reasoning'). If provided, uses reasoning data instead of generating assistant reasoning. Requires split_go_aspects=False since reasoning contains comprehensive analysis for all GO aspects together.",
    )
    dataset_group.add_argument(
        "--go_gpt_predictions_column",
        type=str,
        default="go_pred",
        help="Column name for GO-GPT predictions (must match training).",
    )
    dataset_group.add_argument(
        "--min_go_mf_freq",
        type=int,
        default=50,
        help="Minimum frequency for molecular function GO terms to include in dataset (must match training).",
    )
    dataset_group.add_argument(
        "--min_go_bp_freq",
        type=int,
        default=100,
        help="Minimum frequency for biological process GO terms to include in dataset (must match training).",
    )
    dataset_group.add_argument(
        "--min_go_cc_freq",
        type=int,
        default=50,
        help="Minimum frequency for cellular component GO terms to include in dataset (must match training).",
    )
    dataset_group.add_argument(
        "--apply_go_filtering_to_val_test",
        type=str2bool,
        default=False,
        help="Whether to apply GO frequency filtering to validation/test sets (must match training).",
    )
    dataset_group.add_argument("--add_uniprot_summary", type=str2bool, default=False)

    # Evaluation controls
    eval_group = parser.add_argument_group("Evaluation Configuration")
    eval_group.add_argument(
        "--eval_split",
        type=str,
        choices=["validation", "test"],
        default="validation",
        help="Dataset split to evaluate. Use validation for development and test for final reporting.",
    )
    eval_group.add_argument("--max_samples", type=int, default=-1, help="Max samples to process (-1 for all).")
    eval_group.add_argument("--max_new_tokens", type=int, default=1024)
    eval_group.add_argument("--temperature", type=float, default=0.1)
    eval_group.add_argument("--top_p", type=float, default=0.9)
    eval_group.add_argument("--repetition_penalty", type=float, default=1.0)
    eval_group.add_argument(
        "--pass_at_k", 
        type=int, 
        default=1, 
        help="Number of inference attempts per sample for pass@k evaluation (default: 1). Use temperature > 0 for diversity."
    )

    # Data chunking (optional)
    chunk_group = parser.add_argument_group("Data Chunking (Optional)")
    chunk_group.add_argument(
        "--num_chunks",
        type=int,
        default=1,
        help="Total number of chunks for distributed processing. Default: 1 (no chunking).",
    )
    chunk_group.add_argument(
        "--chunk_id", type=int, default=0, help="ID of this chunk (0-indexed). Only used when num_chunks > 1."
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--evals_path", type=str, required=True, help="Directory path to save individual evaluation results."
    )

    tracking_group = parser.add_argument_group("Tracking Configuration")
    tracking_group.add_argument("--benchmark_version", type=str, default=None)
    tracking_group.add_argument("--model_name", type=str, default=None)
    tracking_group.add_argument("--step0_artifact", type=str, default=None)
    tracking_group.add_argument("--dataset_artifact", type=str, default=None)
    tracking_group.add_argument("--model_artifact", type=str, default=None)
    tracking_group.add_argument("--shortlist_query", type=str, default=None)
    tracking_group.add_argument("--shortlist_mode", type=str, default=None)
    tracking_group.add_argument("--train_start_release", type=str, default=None)
    tracking_group.add_argument("--train_end_release", type=str, default=None)
    tracking_group.add_argument("--dev_end_release", type=str, default=None)
    tracking_group.add_argument("--test_end_release", type=str, default=None)
    tracking_group.add_argument("--metrics_summary_path", type=str, default=None)
    tracking_group.add_argument("--wandb_project", type=str, default=None)
    tracking_group.add_argument("--wandb_entity", type=str, default=None)
    tracking_group.add_argument("--wandb_run_name", type=str, default=None)
    tracking_group.add_argument("--wandb_artifact_name", type=str, default=None)
    tracking_group.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled", "shared"],
        default=None,
    )
    tracking_group.add_argument("--weave_project", type=str, default=None)
    tracking_group.add_argument("--weave_eval_name", type=str, default=None)

    return parser


if __name__ == "__main__":
    parser = setup_argument_parser()
    args = parser.parse_args()
    run_local_inference(args)

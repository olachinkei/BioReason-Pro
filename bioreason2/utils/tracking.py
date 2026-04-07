import os
from typing import Any, Dict, Iterable, List, Mapping, Optional


REASONING_OPEN_TAG = "<think>"
REASONING_CLOSE_TAG = "</think>"

SFT_SAMPLE_TABLE_COLUMNS = [
    "protein_id",
    "split",
    "input_summary",
    "reasoning",
    "final_answer",
    "expected_go_bp",
    "expected_go_mf",
    "expected_go_cc",
    "ground_truth",
    "generation",
]


def normalize_text(value: Any) -> str:
    """Return a stable string form for optional text-like values."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value if item not in (None, ""))
    return str(value)


def resolve_split_name(prefix: str) -> str:
    """Translate internal Lightning prefixes to benchmark split names."""
    if prefix == "val":
        return "validation"
    return prefix


def extract_reasoning_fields(text: str) -> Dict[str, str]:
    """Split a reasoning response into trace and final-answer fields."""
    raw_text = (text or "").strip()
    if not raw_text:
        return {"reasoning": "", "final_answer": ""}

    reasoning = ""
    final_answer = raw_text

    if REASONING_CLOSE_TAG in raw_text:
        reasoning_part, final_part = raw_text.split(REASONING_CLOSE_TAG, 1)
        if REASONING_OPEN_TAG in reasoning_part:
            reasoning = reasoning_part.split(REASONING_OPEN_TAG, 1)[1].strip()
        else:
            reasoning = reasoning_part.strip()
        final_answer = final_part.strip()
    elif REASONING_OPEN_TAG in raw_text:
        reasoning = raw_text.split(REASONING_OPEN_TAG, 1)[1].strip()
        final_answer = ""

    return {
        "reasoning": reasoning,
        "final_answer": final_answer,
    }


def _get_arg(args: Any, name: str, default: Any = None) -> Any:
    """Safely read an argparse attribute from a lightweight namespace."""
    return getattr(args, name, default)


def first_non_empty(*values: Any) -> str:
    """Return the first non-empty textual value."""
    for value in values:
        normalized = normalize_text(value).strip()
        if normalized:
            return normalized
    return ""


def build_training_tracking_config(args: Any, run_name: str, job_type: Optional[str] = None) -> Dict[str, Any]:
    """Build the common W&B config required by the disease benchmark spec."""
    resolved_job_type = job_type or _get_arg(args, "wandb_job_type", "train_sft")
    temporal_split_artifact = first_non_empty(_get_arg(args, "temporal_split_artifact"))
    dataset_config = first_non_empty(
        _get_arg(args, "dataset_config"),
        _get_arg(args, "reasoning_dataset_config"),
        _get_arg(args, "reasoning_dataset_name"),
        _get_arg(args, "cafa5_dataset_name"),
    )
    reasoning_dataset_config = first_non_empty(
        _get_arg(args, "reasoning_dataset_config"),
        _get_arg(args, "reasoning_dataset_name"),
        dataset_config,
    )
    output_model_artifact = first_non_empty(
        _get_arg(args, "model_artifact"),
        _get_arg(args, "checkpoint_artifact_name"),
        f"{run_name}-checkpoints",
    )

    return {
        "job_type": resolved_job_type,
        "benchmark_version": first_non_empty(_get_arg(args, "benchmark_version"), dataset_config),
        "temporal_split_artifact": temporal_split_artifact,
        "dataset_config": dataset_config,
        "reasoning_dataset_config": reasoning_dataset_config,
        "dataset_artifact": normalize_text(_get_arg(args, "dataset_artifact")),
        "shortlist_query": normalize_text(_get_arg(args, "shortlist_query")),
        "shortlist_mode": normalize_text(_get_arg(args, "shortlist_mode")),
        "train_start_release": _get_arg(args, "train_start_release"),
        "train_end_release": _get_arg(args, "train_end_release"),
        "dev_end_release": _get_arg(args, "dev_end_release"),
        "test_end_release": _get_arg(args, "test_end_release"),
        "base_checkpoint": first_non_empty(
            _get_arg(args, "base_checkpoint"),
            _get_arg(args, "ckpt_path"),
            _get_arg(args, "projector_checkpoint_path"),
        ),
        "model_artifact": output_model_artifact,
        "output_dir": first_non_empty(
            _get_arg(args, "checkpoint_dir"),
            _get_arg(args, "output_dir"),
        ),
        "seed": _get_arg(args, "seed"),
        "learning_rate": _get_arg(args, "learning_rate"),
        "batch_size": _get_arg(args, "batch_size"),
        "gradient_accumulation_steps": _get_arg(args, "gradient_accumulation_steps"),
        "num_train_epochs": _get_arg(args, "max_epochs"),
        "validation_subset_size": _get_arg(args, "validation_subset_size"),
        "validation_subset_strategy": normalize_text(_get_arg(args, "validation_subset_strategy")),
        "max_eval_samples": _get_arg(args, "max_eval_samples"),
        "eval_sample_strategy": normalize_text(_get_arg(args, "eval_sample_strategy")),
        "weave_project": normalize_text(_get_arg(args, "weave_project")),
        "weave_trace_budget": _get_arg(args, "weave_trace_budget"),
        "job_time_limit": normalize_text(_get_arg(args, "job_time_limit", "12:00:00")),
        "run_name": run_name,
        "training_stage": _get_arg(args, "training_stage"),
    }


def sync_run_config(run: Any, config: Mapping[str, Any]) -> bool:
    """Write config values into an already initialized W&B run when available."""
    if run is None or not hasattr(run, "config"):
        return False

    update = getattr(run.config, "update", None)
    if update is None:
        return False

    try:
        update(dict(config), allow_val_change=True)
    except TypeError:
        update(dict(config))
    return True


def parse_artifact_aliases(raw_aliases: Any) -> List[str]:
    """Parse comma-separated artifact aliases while preserving order."""
    if raw_aliases is None:
        return []
    if isinstance(raw_aliases, str):
        values: Iterable[Any] = raw_aliases.split(",")
    else:
        values = raw_aliases

    aliases: List[str] = []
    for value in values:
        alias = normalize_text(value).strip()
        if alias and alias not in aliases:
            aliases.append(alias)
    return aliases


def looks_like_wandb_artifact_ref(value: Any) -> bool:
    """Return True when the value matches entity/project/artifact:alias style refs."""
    text = normalize_text(value).strip()
    return bool(text) and ":" in text and text.count("/") >= 2


def maybe_use_artifact(
    run: Any,
    artifact_ref: Any,
    artifact_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Register an input artifact on the active W&B run when a ref is available."""
    resolved_ref = normalize_text(artifact_ref).strip()
    if run is None or not resolved_ref:
        return {
            "used": False,
            "artifact_ref": resolved_ref,
            "artifact_type": artifact_type or "",
            "reason": "missing_run_or_ref",
        }

    use_artifact = getattr(run, "use_artifact", None)
    if use_artifact is None:
        return {
            "used": False,
            "artifact_ref": resolved_ref,
            "artifact_type": artifact_type or "",
            "reason": "missing_use_artifact",
        }

    if not looks_like_wandb_artifact_ref(resolved_ref):
        return {
            "used": False,
            "artifact_ref": resolved_ref,
            "artifact_type": artifact_type or "",
            "reason": "not_wandb_artifact_ref",
        }

    try:
        if artifact_type:
            use_artifact(resolved_ref, type=artifact_type)
        else:
            use_artifact(resolved_ref)
    except TypeError:
        use_artifact(resolved_ref)

    return {
        "used": True,
        "artifact_ref": resolved_ref,
        "artifact_type": artifact_type or "",
        "reason": "",
    }


def maybe_use_artifact_refs(
    run: Any,
    artifact_refs: Mapping[str, Any],
    artifact_types: Optional[Mapping[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Register multiple input artifacts and return per-key statuses."""
    statuses: Dict[str, Dict[str, Any]] = {}
    for key, artifact_ref in artifact_refs.items():
        artifact_type = None if artifact_types is None else artifact_types.get(key)
        statuses[key] = maybe_use_artifact(run, artifact_ref, artifact_type=artifact_type)
    return statuses


def build_checkpoint_artifact_metadata(
    args: Any,
    run_name: str,
    tracking_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build artifact metadata for checkpoint uploads."""
    metadata = dict(tracking_config or {})
    metadata.update(
        {
            "run_name": run_name,
            "checkpoint_dir": first_non_empty(
                _get_arg(args, "checkpoint_dir"),
                _get_arg(args, "output_dir"),
            ),
            "training_stage": _get_arg(args, "training_stage"),
        }
    )
    return {key: value for key, value in metadata.items() if value not in (None, "")}


def maybe_log_directory_artifact(
    run: Any,
    wandb_module: Any,
    artifact_name: str,
    artifact_type: str,
    directory: str,
    aliases: Optional[Iterable[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Log a directory artifact when the run and directory are available."""
    resolved_directory = normalize_text(directory)
    resolved_name = normalize_text(artifact_name)
    resolved_aliases = parse_artifact_aliases(aliases)

    if run is None or wandb_module is None or not resolved_name or not os.path.isdir(resolved_directory):
        return {
            "logged": False,
            "artifact_name": resolved_name,
            "artifact_type": artifact_type,
            "directory": resolved_directory,
            "aliases": resolved_aliases,
        }

    artifact = wandb_module.Artifact(
        resolved_name,
        type=artifact_type,
        metadata=dict(metadata or {}),
    )
    artifact.add_dir(resolved_directory)

    log_artifact = getattr(run, "log_artifact", None)
    if log_artifact is None:
        return {
            "logged": False,
            "artifact_name": resolved_name,
            "artifact_type": artifact_type,
            "directory": resolved_directory,
            "aliases": resolved_aliases,
        }

    try:
        log_artifact(artifact, aliases=resolved_aliases or None)
    except TypeError:
        log_artifact(artifact)

    return {
        "logged": True,
        "artifact_name": resolved_name,
        "artifact_type": artifact_type,
        "directory": resolved_directory,
        "aliases": resolved_aliases,
    }


def build_sft_sample_row(batch: Mapping[str, Any], prefix: str, result: Mapping[str, Any], example_idx: int = 0) -> Dict[str, str]:
    """Create the sample row required by the SFT section of the spec."""
    generation_fields = extract_reasoning_fields(normalize_text(result.get("generation")))
    expected_final = normalize_text(result.get("ground_truth"))

    def _pick(batch_key: str) -> str:
        values = batch.get(batch_key, [])
        if isinstance(values, list) and example_idx < len(values):
            return normalize_text(values[example_idx])
        return ""

    return {
        "protein_id": _pick("protein_ids"),
        "split": _pick("sample_splits") or resolve_split_name(prefix),
        "input_summary": normalize_text(result.get("user_input")),
        "reasoning": generation_fields["reasoning"],
        "final_answer": generation_fields["final_answer"],
        "expected_go_bp": _pick("go_bp_targets"),
        "expected_go_mf": _pick("go_mf_targets"),
        "expected_go_cc": _pick("go_cc_targets"),
        "ground_truth": expected_final,
        "generation": normalize_text(result.get("generation")),
    }

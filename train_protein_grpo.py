#!/usr/bin/env python3
"""
GRPO-style RL training entry point for ProteinLLM disease benchmarking.

This script follows the high-level structure of BioReason's GRPO training flow,
but uses a custom multimodal loop so we can keep ProteinLLM's protein encoder,
GO encoder, and reasoning dataset format intact.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import importlib.util
import json
import math
import os
import random
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import weave
except ImportError:  # pragma: no cover - optional dependency at runtime
    weave = None


GO_ID_PATTERN = re.compile(r"GO:\d{7}")
THINK_TAG_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
STRUCTURAL_TAG_PATTERN = re.compile(r"</?(?:think|answer|tool_call)>")
GO_ASPECT_PATTERN = re.compile(r"(?im)^\s*(MF|BP|CC)\s*:\s*(.+)$")
GO_SUMMARY_START = "<|GO_SUMMARY_START|>"
GO_SUMMARY_END = "<|GO_SUMMARY_END|>"
FUNCTION_SUMMARY_START = "<|FUNCTION_SUMMARY_START|>"
FUNCTION_SUMMARY_END = "<|FUNCTION_SUMMARY_END|>"
GO_ASPECT_ORDER = ("MF", "BP", "CC")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value if item not in (None, ""))
    return str(value)


def resolve_attn_implementation(preferred: str) -> str:
    normalized = normalize_text(preferred).strip() or "sdpa"
    if normalized != "flash_attention_2":
        return normalized
    if importlib.util.find_spec("flash_attn") is not None:
        return normalized
    print("flash_attn is unavailable; falling back to attn_implementation=sdpa for RL")
    return "sdpa"


def maybe_parse_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_text(item).strip() for item in value if normalize_text(item).strip()]
    if isinstance(value, tuple):
        return [normalize_text(item).strip() for item in value if normalize_text(item).strip()]
    text = normalize_text(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [normalize_text(item).strip() for item in parsed if normalize_text(item).strip()]
        except json.JSONDecodeError:
            pass
        try:
            import ast

            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [normalize_text(item).strip() for item in parsed if normalize_text(item).strip()]
        except Exception:
            pass
    return [part.strip() for part in text.split(",") if part.strip()]


def extract_go_ids(text: Any) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for match in GO_ID_PATTERN.findall(normalize_text(text)):
        if match not in seen:
            seen.add(match)
            ordered.append(match)
    return ordered


def extract_tagged_block(text: Any, start_marker: str, end_marker: str) -> str:
    raw = normalize_text(text)
    if not raw:
        return ""
    start_idx = raw.find(start_marker)
    if start_idx < 0:
        return ""
    end_idx = raw.find(end_marker, start_idx + len(start_marker))
    if end_idx < 0:
        return ""
    return raw[start_idx + len(start_marker) : end_idx].strip()


def extract_go_aspect_map(text: Any) -> Dict[str, List[str]]:
    aspect_map: Dict[str, List[str]] = {}
    for match in GO_ASPECT_PATTERN.finditer(normalize_text(text)):
        aspect = match.group(1).upper()
        go_ids = extract_go_ids(match.group(2))
        if go_ids:
            aspect_map[aspect] = go_ids
    return aspect_map


def extract_reasoning_and_answer(text: Any) -> Dict[str, str]:
    raw = normalize_text(text).strip()
    if not raw:
        return {"reasoning": "", "final_answer": ""}

    reasoning = ""
    final_answer = raw

    think_match = THINK_TAG_PATTERN.search(raw)
    if think_match:
        reasoning = think_match.group(1).strip()

    answer_match = ANSWER_TAG_PATTERN.search(raw)
    if answer_match:
        final_answer = answer_match.group(1).strip()
    elif think_match:
        final_answer = THINK_TAG_PATTERN.sub("", raw).strip()

    for marker in ("<|im_end|>", "<|endoftext|>"):
        if marker in final_answer:
            final_answer = final_answer.split(marker, 1)[0].strip()

    return {"reasoning": reasoning, "final_answer": final_answer}


def count_structural_noise_tokens(text: Any) -> int:
    return len(STRUCTURAL_TAG_PATTERN.findall(normalize_text(text)))


def has_meaningful_text(text: Any) -> bool:
    return bool(re.search(r"[A-Za-z0-9]", normalize_text(text)))


def build_requested_go_aspects(sample_meta: Mapping[str, Any]) -> List[str]:
    requested = normalize_text(sample_meta.get("go_aspect")).strip().lower()
    if requested in {"mf", "bp", "cc"}:
        return [requested.upper()]

    target_aspects: List[str] = []
    for aspect, field_name in (("MF", "go_mf"), ("BP", "go_bp"), ("CC", "go_cc")):
        if extract_go_ids(sample_meta.get(field_name)):
            target_aspects.append(aspect)
    return target_aspects or list(GO_ASPECT_ORDER)


@lru_cache(maxsize=8192)
def inspect_completion_text(raw_completion: str) -> Dict[str, Any]:
    sections = extract_reasoning_and_answer(raw_completion)
    final_answer = sections["final_answer"].strip()
    go_summary = extract_tagged_block(final_answer, GO_SUMMARY_START, GO_SUMMARY_END)
    function_summary = extract_tagged_block(final_answer, FUNCTION_SUMMARY_START, FUNCTION_SUMMARY_END)
    go_summary_aspects = extract_go_aspect_map(go_summary)
    final_answer_aspects = extract_go_aspect_map(final_answer)
    structural_noise_count = count_structural_noise_tokens(final_answer)
    final_answer_clean = bool(final_answer) and structural_noise_count == 0 and has_meaningful_text(final_answer)

    prediction_source = "none"
    prediction_text = ""
    if go_summary:
        prediction_source = "go_summary"
        prediction_text = go_summary
    elif has_meaningful_text(final_answer):
        prediction_source = "final_answer"
        prediction_text = final_answer

    return {
        "reasoning": sections["reasoning"],
        "final_answer": final_answer,
        "go_summary": go_summary,
        "function_summary": function_summary,
        "go_summary_aspects": go_summary_aspects,
        "go_summary_aspect_labels": list(go_summary_aspects.keys()),
        "final_answer_aspects": final_answer_aspects,
        "final_answer_aspect_labels": list(final_answer_aspects.keys()),
        "has_go_summary": bool(go_summary),
        "has_function_summary": bool(function_summary),
        "has_complete_summary_schema": bool(go_summary and function_summary and has_meaningful_text(function_summary)),
        "final_answer_clean": final_answer_clean,
        "final_answer_has_text": has_meaningful_text(final_answer),
        "structural_noise_count": structural_noise_count,
        "prediction_source": prediction_source,
        "prediction_text": prediction_text,
        "predicted_go_ids": extract_go_ids(prediction_text),
        "final_answer_go_ids": extract_go_ids(final_answer),
        "completion_go_ids": extract_go_ids(raw_completion),
    }


def inspect_completion(completion: Any) -> Dict[str, Any]:
    return dict(inspect_completion_text(normalize_text(completion).strip()))


def build_target_go_ids(sample_meta: Mapping[str, Any]) -> List[str]:
    targets: List[str] = []
    for key in ("go_bp", "go_mf", "go_cc", "ground_truth_go_terms"):
        targets.extend(extract_go_ids(sample_meta.get(key)))

    seen = set()
    ordered: List[str] = []
    for go_id in targets:
        if go_id not in seen:
            seen.add(go_id)
            ordered.append(go_id)
    return ordered


def strict_format_reward(completion: str, _: Mapping[str, Any]) -> float:
    meta = inspect_completion(completion)
    return 1.0 if meta["reasoning"] and meta["final_answer_clean"] else 0.0


def reasoning_presence_reward(completion: str, _: Mapping[str, Any]) -> float:
    return 1.0 if inspect_completion(completion)["reasoning"] else 0.0


def concise_reasoning_reward(completion: str, _: Mapping[str, Any]) -> float:
    reasoning = inspect_completion(completion)["reasoning"]
    if not reasoning:
        return 0.0
    length = len(reasoning.split())
    if 32 <= length <= 384:
        return 1.0
    if 16 <= length <= 512:
        return 0.5
    return 0.0


def answer_nonempty_reward(completion: str, _: Mapping[str, Any]) -> float:
    return 1.0 if inspect_completion(completion)["final_answer_clean"] else 0.0


def summary_schema_reward(completion: str, sample_meta: Mapping[str, Any]) -> float:
    meta = inspect_completion(completion)
    requested_aspects = set(build_requested_go_aspects(sample_meta))
    predicted_aspects = set(meta["go_summary_aspect_labels"])
    matched_aspects = predicted_aspects & requested_aspects
    if meta["has_complete_summary_schema"] and meta["predicted_go_ids"] and matched_aspects == requested_aspects:
        return 1.0
    if meta["has_complete_summary_schema"] and meta["predicted_go_ids"] and matched_aspects:
        return 0.75
    if meta["has_go_summary"] and meta["predicted_go_ids"] and matched_aspects:
        return 0.5
    return 0.0


def structural_noise_reward(completion: str, _: Mapping[str, Any]) -> float:
    noise_count = inspect_completion(completion)["structural_noise_count"]
    if noise_count <= 0:
        return 0.0
    return -min(1.0, 0.25 * float(noise_count))


def go_presence_reward(completion: str, sample_meta: Mapping[str, Any]) -> float:
    meta = inspect_completion(completion)
    predicted_go_ids = meta["predicted_go_ids"]
    if predicted_go_ids:
        return 1.0 if meta["has_go_summary"] else 0.5

    if not build_target_go_ids(sample_meta):
        return 0.0
    if meta["final_answer_has_text"] or meta["reasoning"]:
        return -1.0
    return -0.5


def go_aspect_coverage_reward(completion: str, sample_meta: Mapping[str, Any]) -> float:
    meta = inspect_completion(completion)
    requested_aspects = set(build_requested_go_aspects(sample_meta))
    if not requested_aspects or not meta["predicted_go_ids"]:
        return 0.0

    predicted_aspects = set(meta["go_summary_aspect_labels"])
    if not predicted_aspects:
        return 0.0

    matched_aspects = predicted_aspects & requested_aspects
    if not matched_aspects:
        return 0.0
    return len(matched_aspects) / len(requested_aspects)


def go_overlap_reward(completion: str, sample_meta: Mapping[str, Any]) -> float:
    predicted = set(inspect_completion(completion)["predicted_go_ids"])
    target = set(build_target_go_ids(sample_meta))
    if not predicted or not target:
        return 0.0

    true_positive = len(predicted & target)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(target) if target else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_go_set_reward(completion: str, sample_meta: Mapping[str, Any]) -> float:
    predicted = set(inspect_completion(completion)["predicted_go_ids"])
    target = set(build_target_go_ids(sample_meta))
    return 1.0 if predicted and predicted == target else 0.0


def build_reward_registry() -> Dict[str, Any]:
    return {
        "strict_format": strict_format_reward,
        "reasoning_presence": reasoning_presence_reward,
        "concise_reasoning": concise_reasoning_reward,
        "answer_nonempty": answer_nonempty_reward,
        "summary_schema": summary_schema_reward,
        "go_presence": go_presence_reward,
        "go_aspect_coverage": go_aspect_coverage_reward,
        "structural_noise": structural_noise_reward,
        "go_overlap": go_overlap_reward,
        "exact_go_set": exact_go_set_reward,
    }


def parse_csv_items(raw: str) -> List[str]:
    return [item.strip() for item in normalize_text(raw).split(",") if item.strip()]


def parse_reward_weights(raw: str, count: int) -> List[float]:
    if not normalize_text(raw).strip():
        return [1.0] * count
    values = [float(item) for item in parse_csv_items(raw)]
    if len(values) != count:
        raise ValueError(f"Expected {count} reward weights, got {len(values)}")
    return values


def standardize_group_rewards(rewards: Sequence[float]) -> List[float]:
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(max(variance, 0.0))
    if std < 1e-8:
        return [0.0 for _ in rewards]
    return [(reward - mean) / (std + 1e-8) for reward in rewards]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "bioreasoning-pro"))
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)
    parser.add_argument("--wandb_job_type", type=str, default="train_rl", choices=["train_rl"])
    parser.add_argument("--weave_project", type=str, default=None)
    parser.add_argument("--weave_trace_budget", type=int, default=64)

    parser.add_argument("--benchmark_version", type=str, default="213 -> 221 -> 225 -> 228")
    parser.add_argument("--temporal_split_artifact", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default="disease_temporal_hc_reasoning_v1")
    parser.add_argument("--reasoning_dataset_config", type=str, default="disease_temporal_hc_reasoning_v1")
    parser.add_argument("--dataset_artifact", type=str, default=None)
    parser.add_argument("--shortlist_query", type=str, default=None)
    parser.add_argument("--shortlist_mode", type=str, default="high-confidence")
    parser.add_argument("--train_start_release", type=int, default=213)
    parser.add_argument("--train_end_release", type=int, default=221)
    parser.add_argument("--dev_end_release", type=int, default=225)
    parser.add_argument("--test_end_release", type=int, default=228)
    parser.add_argument("--base_checkpoint", type=str, default=None)
    parser.add_argument("--model_artifact", type=str, default="train-rl-output")
    parser.add_argument("--job_time_limit", type=str, default="12:00:00")

    parser.add_argument("--text_model_name", type=str, required=True, help="Local HF model directory used to initialize RL.")
    parser.add_argument("--protein_model_name", type=str, default="esm3_sm_open_v1")
    parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.cache/huggingface/hub"))
    parser.add_argument("--go_obo_path", type=str, default=None)
    parser.add_argument("--precomputed_embeddings_path", type=str, default=None)
    parser.add_argument("--structure_dir", type=str, default=None)
    parser.add_argument("--dataset_cache_dir", type=str, default=None)

    parser.add_argument("--dataset_type", type=str, default="cafa5", choices=["cafa5"])
    parser.add_argument("--cafa5_dataset", type=str, default="wanglab/cafa5")
    parser.add_argument("--cafa5_dataset_name", type=str, default="disease_temporal_hc_reasoning_v1")
    parser.add_argument("--reasoning_dataset_name", type=str, default="disease_temporal_hc_reasoning_v1")
    parser.add_argument("--interpro_dataset_name", type=str, default="interpro_metadata")
    parser.add_argument("--go_gpt_predictions_column", type=str, default="go_pred")
    parser.add_argument("--include_ground_truth_in_final_answer", type=str, default="false")
    parser.add_argument("--add_uniprot_summary", type=str, default="true")
    parser.add_argument("--is_swissprot", type=str, default="false")
    parser.add_argument("--include_go_defs", type=str, default="false")
    parser.add_argument("--interpro_in_prompt", type=str, default="true")
    parser.add_argument("--ppi_in_prompt", type=str, default="true")
    parser.add_argument("--predict_interpro", type=str, default="false")
    parser.add_argument("--include_protein_function_summary", type=str, default="true")
    parser.add_argument("--split_go_aspects", type=str, default="false")

    parser.add_argument("--max_length_text", type=int, default=10000)
    parser.add_argument("--max_length_protein", type=int, default=2000)
    parser.add_argument("--protein_embedding_layer", type=int, default=37)
    parser.add_argument("--go_hidden_dim", type=int, default=512)
    parser.add_argument("--go_num_gat_layers", type=int, default=3)
    parser.add_argument("--go_num_heads", type=int, default=8)
    parser.add_argument("--go_num_reduced_embeddings", type=int, default=200)
    parser.add_argument("--go_embedding_dim", type=int, default=2560)
    parser.add_argument("--unified_go_encoder", type=str, default="true")
    parser.add_argument("--protein_model_finetune", type=str, default="false")
    parser.add_argument("--train_projector", type=str, default="false")
    parser.add_argument("--train_go_modules", type=str, default="false")

    parser.add_argument("--use_qlora", type=str, default="false")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", type=str, default="true")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=128)
    parser.add_argument(
        "--eval_sample_strategy",
        type=str,
        default="stratified_aspect_profile",
        choices=["stratified_aspect_profile", "shuffled_prefix"],
    )
    parser.add_argument("--eval_every_n_steps", type=int, default=50)
    parser.add_argument("--save_every_n_steps", type=int, default=50)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--rotating_eval_every_n_steps", type=int, default=100)
    parser.add_argument("--rotating_eval_max_samples", type=int, default=256)
    parser.add_argument(
        "--rotating_eval_sample_strategy",
        type=str,
        default="stratified_aspect_profile",
        choices=["stratified_aspect_profile", "shuffled_prefix"],
    )
    parser.add_argument("--rotating_eval_seed_stride", type=int, default=9973)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--do_sample", type=str, default="true")
    parser.add_argument("--eval_do_sample", type=str, default="false")
    parser.add_argument("--eval_temperature", type=float, default=0.1)
    parser.add_argument("--eval_top_p", type=float, default=0.9)
    parser.add_argument("--eval_top_k", type=int, default=20)
    parser.add_argument("--kl_beta", type=float, default=0.02)
    parser.add_argument(
        "--reward_funcs",
        type=str,
        default="strict_format,summary_schema,go_presence,go_aspect_coverage,go_overlap,structural_noise",
        help="Comma-separated reward function names.",
    )
    parser.add_argument(
        "--reward_weights",
        type=str,
        default="0.25,0.75,1.5,0.5,2.5,1.0",
        help="Optional comma-separated reward weights aligned with --reward_funcs.",
    )

    parser.add_argument("--resume_from_raw_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="data/artifacts/models/train_rl_output")
    parser.add_argument("--checkpoint_artifact_name", type=str, default="train-rl-output")
    parser.add_argument("--checkpoint_artifact_aliases", type=str, default="latest")
    parser.add_argument("--ablation_from_paper_rl", type=str, default="false")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    bool_fields = [
        "include_ground_truth_in_final_answer",
        "add_uniprot_summary",
        "is_swissprot",
        "include_go_defs",
        "interpro_in_prompt",
        "ppi_in_prompt",
        "predict_interpro",
        "include_protein_function_summary",
        "split_go_aspects",
        "unified_go_encoder",
        "protein_model_finetune",
        "train_projector",
        "train_go_modules",
        "use_qlora",
        "bnb_4bit_use_double_quant",
        "do_sample",
        "eval_do_sample",
        "ablation_from_paper_rl",
    ]

    def _str2bool(raw: Any) -> bool:
        value = normalize_text(raw).strip().lower()
        return value in {"1", "true", "t", "yes", "y"}

    for field_name in bool_fields:
        setattr(args, field_name, _str2bool(getattr(args, field_name)))

    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_auxiliary_checkpoint_components(model: Any, checkpoint_dir: str) -> None:
    import torch

    checkpoint_path = Path(checkpoint_dir)
    optional_components = [
        (checkpoint_path / "protein_projection.pt", getattr(model, "protein_projection", None), "protein projection"),
        (checkpoint_path / "go_projection.pt", getattr(model, "go_projection", None), "GO projection"),
        (checkpoint_path / "go_encoder.pt", getattr(model, "go_encoder", None), "GO encoder"),
        (checkpoint_path / "protein_model" / "pytorch_model.bin", getattr(model, "protein_model", None), "protein model"),
    ]

    for weight_path, module, label in optional_components:
        if module is None or not weight_path.exists():
            continue
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, Mapping):
            module.load_state_dict(state_dict, strict=False)
            print(f"Loaded {label} weights from {weight_path}")


def configure_trainable_modules(model: Any, args: argparse.Namespace) -> None:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    from bioreason2.models.protein_llm import _get_target_modules

    for param in model.parameters():
        param.requires_grad = False

    if args.train_projector and getattr(model, "protein_projection", None) is not None:
        model.protein_projection.train()
        for param in model.protein_projection.parameters():
            param.requires_grad = True

    if args.train_go_modules:
        if getattr(model, "go_projection", None) is not None:
            model.go_projection.train()
            for param in model.go_projection.parameters():
                param.requires_grad = True
        if getattr(model, "go_encoder", None) is not None:
            model.go_encoder.train()
            for param in model.go_encoder.parameters():
                param.requires_grad = True

    target_modules = _get_target_modules(model)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
        bias="none",
        task_type="CAUSAL_LM",
    )

    if args.use_qlora:
        model.text_model = prepare_model_for_kbit_training(model.text_model)
    model.text_model = get_peft_model(model.text_model, lora_config)
    model.text_model.train()


def build_quantization_config(args: argparse.Namespace) -> Optional[Any]:
    if not args.use_qlora:
        return None

    import torch
    from transformers import BitsAndBytesConfig

    dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )


def instantiate_model(args: argparse.Namespace, trainable: bool) -> Any:
    from bioreason2.models.protein_llm import ProteinLLMModel

    quantization_config = build_quantization_config(args) if trainable else None
    attn_implementation = resolve_attn_implementation("flash_attention_2")
    precomputed_embeddings_source = args.precomputed_embeddings_path
    precomputed_go_embedding_cache_path = None
    if precomputed_embeddings_source:
        candidate = Path(precomputed_embeddings_source).expanduser()
        if candidate.is_file():
            precomputed_go_embedding_cache_path = str(candidate)
            precomputed_embeddings_source = None
    model = ProteinLLMModel(
        text_model_name=args.text_model_name,
        protein_model_name=args.protein_model_name,
        cache_dir=args.cache_dir,
        max_length_protein=args.max_length_protein,
        max_length_text=args.max_length_text,
        text_model_finetune=True,
        protein_model_finetune=args.protein_model_finetune,
        protein_embedding_layer=args.protein_embedding_layer,
        go_model_finetune=args.train_go_modules,
        attn_implementation=attn_implementation,
        go_obo_path=args.go_obo_path,
        precomputed_embeddings_path=precomputed_embeddings_source,
        go_hidden_dim=args.go_hidden_dim,
        go_num_gat_layers=args.go_num_gat_layers,
        go_num_heads=args.go_num_heads,
        go_num_reduced_embeddings=args.go_num_reduced_embeddings,
        go_embedding_dim=args.go_embedding_dim,
        quantization_config=quantization_config,
        load_in_4bit=args.use_qlora if trainable else False,
        unified_go_encoder=args.unified_go_encoder,
        use_unsloth=False,
    )
    if precomputed_go_embedding_cache_path:
        model.load_precomputed_go_embedding_cache(precomputed_go_embedding_cache_path, aspect="all")
    load_auxiliary_checkpoint_components(model, args.text_model_name)
    if trainable:
        configure_trainable_modules(model, args)
    else:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    return model


def limit_dataset(dataset: Any, max_samples: int) -> Any:
    if max_samples is None or max_samples < 0 or len(dataset) <= max_samples:
        return dataset
    return dataset.select(range(max_samples))


def select_validation_subset(
    dataset: Any,
    *,
    max_samples: int,
    seed: int,
    strategy: str,
    label: str,
) -> Tuple[Any, Dict[str, Any]]:
    from bioreason2.dataset.cafa5.subset import select_dataset_subset

    subset, subset_summary = select_dataset_subset(
        dataset,
        max_samples=max_samples,
        seed=seed,
        strategy=strategy,
    )
    print(
        f"Using RL {label} validation subset: "
        f"strategy={subset_summary['strategy']}, "
        f"seed={seed}, "
        f"requested={subset_summary['requested_samples']}, "
        f"selected={subset_summary['selected_samples']}"
    )
    if subset_summary.get("group_counts"):
        print(f"RL {label} validation group counts: {subset_summary['group_counts']}")
    if subset_summary.get("aspect_coverage"):
        print(f"RL {label} validation aspect coverage: {subset_summary['aspect_coverage']}")
    return subset, subset_summary


def load_rl_datasets(args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    from bioreason2.dataset.cafa5.load import load_cafa5_dataset

    train_dataset, full_val_dataset, _ = load_cafa5_dataset(
        dataset=args.cafa5_dataset,
        dataset_name=args.cafa5_dataset_name,
        max_length=args.max_length_protein,
        val_split_ratio=0.1,
        seed=args.seed,
        cache_dir=args.dataset_cache_dir,
        structure_dir=args.structure_dir,
        debug=False,
        include_go_defs=args.include_go_defs,
        interpro_dataset_name=args.interpro_dataset_name,
        split_go_aspects=args.split_go_aspects,
        include_protein_function_summary=args.include_protein_function_summary,
        interpro_in_prompt=args.interpro_in_prompt,
        ppi_in_prompt=args.ppi_in_prompt,
        predict_interpro=args.predict_interpro,
        reasoning_dataset_name=args.reasoning_dataset_name,
        go_gpt_predictions_column=args.go_gpt_predictions_column,
        include_ground_truth_in_final_answer=args.include_ground_truth_in_final_answer,
        add_uniprot_summary=args.add_uniprot_summary,
        is_swissprot=args.is_swissprot,
        return_as_chat_template=True,
    )
    train_dataset = limit_dataset(train_dataset, args.max_train_samples)
    fixed_val_dataset, _ = select_validation_subset(
        full_val_dataset,
        max_samples=args.max_eval_samples,
        seed=args.seed,
        strategy=args.eval_sample_strategy,
        label="fixed",
    )
    return train_dataset, fixed_val_dataset, full_val_dataset


def build_dataloader(dataset: Any, model: Any, batch_size: int, num_workers: int, shuffle: bool) -> Any:
    from functools import partial

    from torch.utils.data import DataLoader

    from bioreason2.dataset.cafa5.collate import qwen_protein_collate_fn

    collate_fn = partial(
        qwen_protein_collate_fn,
        processor=model.processor,
        max_length_text=model.max_length_text,
        max_length_protein=model.max_length_protein,
        return_answer_in_batch=False,
        inference_mode=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def extract_sample_meta_from_batch(batch: Mapping[str, Any], example_idx: int) -> Dict[str, str]:
    return {
        "protein_id": batch.get("protein_ids", [""])[example_idx] if example_idx < len(batch.get("protein_ids", [])) else "",
        "split": batch.get("sample_splits", [""])[example_idx] if example_idx < len(batch.get("sample_splits", [])) else "",
        "go_bp": batch.get("go_bp_targets", [""])[example_idx] if example_idx < len(batch.get("go_bp_targets", [])) else "",
        "go_mf": batch.get("go_mf_targets", [""])[example_idx] if example_idx < len(batch.get("go_mf_targets", [])) else "",
        "go_cc": batch.get("go_cc_targets", [""])[example_idx] if example_idx < len(batch.get("go_cc_targets", [])) else "",
        "reasoning": batch.get("reasoning_targets", [""])[example_idx]
        if example_idx < len(batch.get("reasoning_targets", []))
        else "",
        "final_answer": batch.get("final_answers", [""])[example_idx]
        if example_idx < len(batch.get("final_answers", []))
        else "",
        "prompt_preview": batch.get("prompt", [""])[example_idx] if example_idx < len(batch.get("prompt", [])) else "",
    }


def extract_example_from_batch(batch: Mapping[str, Any], example_idx: int, device: Any) -> Dict[str, Any]:
    import torch

    batch_idx_map = list(batch.get("batch_idx_map") or [])
    protein_sequences = list(batch.get("protein_sequences") or [])
    protein_indices = [i for i, mapped_idx in enumerate(batch_idx_map) if mapped_idx == example_idx]
    example_sequences = [protein_sequences[i] for i in protein_indices]

    structure_coords = batch.get("structure_coords")
    example_structure = None
    if isinstance(structure_coords, torch.Tensor):
        example_structure = structure_coords[example_idx : example_idx + 1].to(device)

    go_aspects = batch.get("batch_go_aspects") or []
    example_go_aspect = None
    if example_idx < len(go_aspects):
        example_go_aspect = go_aspects[example_idx]

    return {
        "input_ids": batch["input_ids"][example_idx : example_idx + 1].to(device),
        "attention_mask": batch["attention_mask"][example_idx : example_idx + 1].to(device),
        "protein_sequences": example_sequences,
        "batch_idx_map": [0] * len(example_sequences),
        "structure_coords": example_structure,
        "go_aspects": [example_go_aspect if example_go_aspect is not None else "all"],
        "sample_meta": extract_sample_meta_from_batch(batch, example_idx),
    }


def expand_example_for_rollouts(example: Mapping[str, Any], rollout_count: int) -> Dict[str, Any]:
    import torch

    if rollout_count <= 0:
        raise ValueError(f"rollout_count must be positive, got {rollout_count}")

    input_ids = example["input_ids"].repeat(rollout_count, 1)
    attention_mask = example["attention_mask"].repeat(rollout_count, 1)

    structure_coords = example.get("structure_coords")
    if isinstance(structure_coords, torch.Tensor):
        repeat_dims = [rollout_count] + [1] * max(structure_coords.dim() - 1, 0)
        structure_coords = structure_coords.repeat(*repeat_dims)

    base_sequences = list(example.get("protein_sequences") or [])
    protein_sequences: List[str] = []
    batch_idx_map: List[int] = []
    for rollout_idx in range(rollout_count):
        protein_sequences.extend(base_sequences)
        batch_idx_map.extend([rollout_idx] * len(base_sequences))

    base_go_aspects = list(example.get("go_aspects") or [])
    rollout_go_aspect = base_go_aspects[0] if base_go_aspects else "all"

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "protein_sequences": protein_sequences,
        "batch_idx_map": batch_idx_map,
        "structure_coords": structure_coords,
        "go_aspects": [rollout_go_aspect] * rollout_count,
    }


def decode_completion(tokenizer: Any, completion_ids: Any) -> str:
    text = tokenizer.decode(completion_ids, skip_special_tokens=False).strip()
    for marker in ("<|im_end|>", "<|endoftext|>"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


def extract_completion_ids(generated_ids: Any, prompt_input_ids: Any) -> Any:
    """Handle both prompt-inclusive and completion-only outputs from generate()."""
    import torch

    if generated_ids.dim() != 2:
        raise ValueError(f"Expected rank-2 generated ids, got shape {tuple(generated_ids.shape)}")
    if generated_ids.shape[0] != 1:
        raise ValueError(f"Expected batch size 1 during RL rollout extraction, got {generated_ids.shape[0]}")

    if prompt_input_ids.dim() == 2:
        if prompt_input_ids.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1 for prompt ids during RL rollout extraction, got {prompt_input_ids.shape[0]}"
            )
        prompt_tokens = prompt_input_ids[0]
    else:
        prompt_tokens = prompt_input_ids

    prompt_tokens = prompt_tokens.detach().to(device=generated_ids.device, dtype=generated_ids.dtype)
    generated_tokens = generated_ids[0]
    prompt_len = int(prompt_tokens.shape[0])

    if generated_tokens.shape[0] >= prompt_len and torch.equal(generated_tokens[:prompt_len], prompt_tokens):
        return generated_tokens[prompt_len:].detach()
    return generated_tokens.detach()


def extract_completion_ids_batch(generated_ids: Any, prompt_input_ids: Any) -> List[Any]:
    import torch

    if generated_ids.dim() != 2:
        raise ValueError(f"Expected rank-2 generated ids, got shape {tuple(generated_ids.shape)}")
    if prompt_input_ids.dim() != 2:
        raise ValueError(f"Expected rank-2 prompt ids for batched extraction, got shape {tuple(prompt_input_ids.shape)}")
    if generated_ids.shape[0] != prompt_input_ids.shape[0]:
        raise ValueError(
            "Generated ids and prompt ids must have the same batch size for batched extraction: "
            f"{generated_ids.shape[0]} != {prompt_input_ids.shape[0]}"
        )

    completions: List[Any] = []
    for row_idx in range(generated_ids.shape[0]):
        prompt_tokens = prompt_input_ids[row_idx].detach().to(device=generated_ids.device, dtype=generated_ids.dtype)
        generated_tokens = generated_ids[row_idx]
        prompt_len = int(prompt_tokens.shape[0])
        if generated_tokens.shape[0] >= prompt_len and torch.equal(generated_tokens[:prompt_len], prompt_tokens):
            completions.append(generated_tokens[prompt_len:].detach())
        else:
            completions.append(generated_tokens.detach())
    return completions


def build_rollout_group_inputs(
    example: Mapping[str, Any],
    completion_ids_list: Sequence[Any],
    pad_token_id: int,
) -> Dict[str, Any]:
    import torch

    rollout_count = len(completion_ids_list)
    if rollout_count <= 0:
        raise ValueError("completion_ids_list must not be empty")

    expanded_example = expand_example_for_rollouts(example, rollout_count)
    prompt_input_ids = expanded_example["input_ids"]
    prompt_attention_mask = expanded_example["attention_mask"]
    max_completion_len = max((int(completion_ids.numel()) for completion_ids in completion_ids_list), default=0)
    completion_batch = torch.full(
        (rollout_count, max_completion_len),
        int(pad_token_id),
        dtype=prompt_input_ids.dtype,
        device=prompt_input_ids.device,
    )
    completion_attention = torch.zeros(
        (rollout_count, max_completion_len),
        dtype=prompt_attention_mask.dtype,
        device=prompt_attention_mask.device,
    )

    for rollout_idx, completion_ids in enumerate(completion_ids_list):
        completion_len = int(completion_ids.numel())
        if completion_len <= 0:
            continue
        completion_tensor = completion_ids.to(device=prompt_input_ids.device, dtype=prompt_input_ids.dtype)
        completion_batch[rollout_idx, :completion_len] = completion_tensor
        completion_attention[rollout_idx, :completion_len] = 1

    combined_input_ids = torch.cat([prompt_input_ids, completion_batch], dim=1)
    combined_attention_mask = torch.cat([prompt_attention_mask, completion_attention], dim=1)
    return {
        "combined_input_ids": combined_input_ids,
        "combined_attention_mask": combined_attention_mask,
        "completion_attention": completion_attention,
        "prompt_token_len": prompt_input_ids.shape[1],
        "protein_sequences": expanded_example["protein_sequences"],
        "batch_idx_map": expanded_example["batch_idx_map"],
        "structure_coords": expanded_example["structure_coords"],
        "go_aspects": expanded_example["go_aspects"],
    }


def build_combined_inputs(prompt_input_ids: Any, prompt_attention_mask: Any, completion_ids: Any) -> Tuple[Any, Any, int]:
    import torch

    if completion_ids.dim() == 1:
        completion_ids = completion_ids.unsqueeze(0)
    prompt_len = prompt_input_ids.shape[1]
    completion_attention = torch.ones_like(completion_ids, dtype=prompt_attention_mask.dtype)
    combined_input_ids = torch.cat([prompt_input_ids, completion_ids], dim=1)
    combined_attention_mask = torch.cat([prompt_attention_mask, completion_attention], dim=1)
    return combined_input_ids, combined_attention_mask, prompt_len


def compute_completion_token_log_probs(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    prompt_len: int,
    protein_sequences: Sequence[str],
    batch_idx_map: Sequence[int],
    structure_coords: Any,
    go_aspects: Sequence[str],
) -> Any:
    import torch
    import torch.nn.functional as F

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        protein_sequences=list(protein_sequences),
        batch_idx_map=list(batch_idx_map),
        structure_coords=structure_coords,
        go_aspects=list(go_aspects),
    )
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    token_log_probs = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    start_index = max(prompt_len - 1, 0)
    return token_log_probs[:, start_index:]


def compute_batched_completion_kl(
    model: Any,
    ref_model: Any,
    prompt_input_ids: Any,
    prompt_attention_mask: Any,
    completion_ids_list: Sequence[Any],
    protein_sequences: Sequence[str],
    batch_idx_map: Sequence[int],
    structure_coords: Any,
    go_aspects: Sequence[str],
    pad_token_id: int,
) -> float:
    import torch

    if ref_model is None:
        return 0.0

    max_completion_len = max((int(completion_ids.numel()) for completion_ids in completion_ids_list), default=0)
    if max_completion_len <= 0:
        return 0.0

    completion_batch = torch.full(
        (len(completion_ids_list), max_completion_len),
        int(pad_token_id),
        dtype=prompt_input_ids.dtype,
        device=prompt_input_ids.device,
    )
    completion_attention = torch.zeros(
        (len(completion_ids_list), max_completion_len),
        dtype=prompt_attention_mask.dtype,
        device=prompt_attention_mask.device,
    )

    for example_idx, completion_ids in enumerate(completion_ids_list):
        completion_len = int(completion_ids.numel())
        if completion_len <= 0:
            continue
        completion_tensor = completion_ids.to(device=prompt_input_ids.device, dtype=prompt_input_ids.dtype)
        completion_batch[example_idx, :completion_len] = completion_tensor
        completion_attention[example_idx, :completion_len] = 1

    combined_input_ids = torch.cat([prompt_input_ids, completion_batch], dim=1)
    combined_attention_mask = torch.cat([prompt_attention_mask, completion_attention], dim=1)
    prompt_len = prompt_input_ids.shape[1]

    current_lp = compute_completion_token_log_probs(
        model,
        combined_input_ids,
        combined_attention_mask,
        prompt_len,
        protein_sequences,
        batch_idx_map,
        structure_coords,
        go_aspects,
    )
    ref_lp = compute_completion_token_log_probs(
        ref_model,
        combined_input_ids,
        combined_attention_mask,
        prompt_len,
        protein_sequences,
        batch_idx_map,
        structure_coords,
        go_aspects,
    )
    valid_mask = completion_attention.bool()
    if not torch.any(valid_mask):
        return 0.0

    kl_term = torch.exp(ref_lp - current_lp) - (ref_lp - current_lp) - 1.0
    valid_counts = valid_mask.sum(dim=1)
    per_example_kl = (kl_term * valid_mask.to(dtype=kl_term.dtype)).sum(dim=1) / valid_counts.clamp_min(1).to(
        dtype=kl_term.dtype
    )
    return float(per_example_kl[valid_counts > 0].sum().item())


def compute_group_policy_losses_batched(
    model: Any,
    ref_model: Any,
    example: Mapping[str, Any],
    completion_ids_list: Sequence[Any],
    advantages: Sequence[float],
    pad_token_id: int,
    kl_beta: float,
) -> Tuple[List[Any], List[float], bool]:
    import torch

    rollout_group = build_rollout_group_inputs(example, completion_ids_list, pad_token_id)
    valid_mask = rollout_group["completion_attention"].bool()
    nonempty_rollouts = valid_mask.any(dim=1)
    if not torch.any(nonempty_rollouts):
        return [], [], False

    current_log_probs = compute_completion_token_log_probs(
        model,
        rollout_group["combined_input_ids"],
        rollout_group["combined_attention_mask"],
        rollout_group["prompt_token_len"],
        rollout_group["protein_sequences"],
        rollout_group["batch_idx_map"],
        rollout_group["structure_coords"],
        rollout_group["go_aspects"],
    )

    with torch.no_grad():
        if ref_model is not None:
            ref_log_probs = compute_completion_token_log_probs(
                ref_model,
                rollout_group["combined_input_ids"],
                rollout_group["combined_attention_mask"],
                rollout_group["prompt_token_len"],
                rollout_group["protein_sequences"],
                rollout_group["batch_idx_map"],
                rollout_group["structure_coords"],
                rollout_group["go_aspects"],
            )
        else:
            ref_log_probs = torch.zeros_like(current_log_probs)

    token_mask = rollout_group["completion_attention"].to(
        device=current_log_probs.device,
        dtype=current_log_probs.dtype,
    )
    valid_counts = token_mask.sum(dim=1).clamp_min(1.0)
    advantage_tensor = torch.tensor(
        list(advantages),
        device=current_log_probs.device,
        dtype=current_log_probs.dtype,
    ).unsqueeze(1)
    log_ratio = current_log_probs - current_log_probs.detach()
    policy_term = torch.exp(log_ratio) * advantage_tensor
    kl_term = torch.exp(ref_log_probs - current_log_probs) - (ref_log_probs - current_log_probs) - 1.0
    per_rollout_loss = -((policy_term - kl_beta * kl_term) * token_mask).sum(dim=1) / valid_counts
    per_rollout_kl = ((kl_term * token_mask).sum(dim=1) / valid_counts)[nonempty_rollouts]
    return (
        list(per_rollout_loss[nonempty_rollouts].unbind()),
        [float(value.detach().item()) for value in per_rollout_kl],
        True,
    )


def compute_group_policy_losses_sequential(
    model: Any,
    ref_model: Any,
    example: Mapping[str, Any],
    completion_ids_list: Sequence[Any],
    advantages: Sequence[float],
    kl_beta: float,
) -> Tuple[List[Any], List[float], bool]:
    import torch

    losses: List[Any] = []
    kl_values: List[float] = []
    trainable = False
    for completion_ids, advantage in zip(completion_ids_list, advantages):
        if completion_ids.numel() <= 0:
            continue
        trainable = True
        combined_ids, combined_mask, prompt_token_len = build_combined_inputs(
            example["input_ids"],
            example["attention_mask"],
            completion_ids,
        )
        current_log_probs = compute_completion_token_log_probs(
            model,
            combined_ids,
            combined_mask,
            prompt_token_len,
            example["protein_sequences"],
            example["batch_idx_map"],
            example["structure_coords"],
            example["go_aspects"],
        )
        with torch.no_grad():
            if ref_model is not None:
                ref_log_probs = compute_completion_token_log_probs(
                    ref_model,
                    combined_ids,
                    combined_mask,
                    prompt_token_len,
                    example["protein_sequences"],
                    example["batch_idx_map"],
                    example["structure_coords"],
                    example["go_aspects"],
                )
            else:
                ref_log_probs = torch.zeros_like(current_log_probs)

        log_ratio = current_log_probs - current_log_probs.detach()
        policy_term = torch.exp(log_ratio) * float(advantage)
        kl_term = torch.exp(ref_log_probs - current_log_probs) - (ref_log_probs - current_log_probs) - 1.0
        losses.append(-(policy_term - kl_beta * kl_term).mean())
        kl_values.append(float(kl_term.mean().detach().item()))

    return losses, kl_values, trainable


def generate_rollouts_sequential(
    model: Any,
    example: Mapping[str, Any],
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    global_step: int,
    epoch_idx: int,
    prompt_len: int,
) -> Tuple[List[Any], List[str]]:
    completion_ids_list: List[Any] = []
    completions: List[str] = []
    for rollout_idx in range(args.num_generations):
        print(
            "Generating RL rollout: "
            f"global_step={global_step}, epoch={epoch_idx}, rollout_idx={rollout_idx}, "
            f"prompt_len={prompt_len}, max_new_tokens={args.max_new_tokens}"
        )
        generated_ids = model.generate(
            input_ids=example["input_ids"],
            attention_mask=example["attention_mask"],
            protein_sequences=example["protein_sequences"],
            batch_idx_map=example["batch_idx_map"],
            structure_coords=example["structure_coords"],
            go_aspects=example["go_aspects"],
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_new_tokens=args.min_new_tokens,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        completion_ids = extract_completion_ids(generated_ids, example["input_ids"])
        print(
            "Completed RL rollout generation: "
            f"global_step={global_step}, rollout_idx={rollout_idx}, completion_tokens={completion_ids.numel()}"
        )
        completion_ids_list.append(completion_ids)
        completions.append(decode_completion(tokenizer, completion_ids))
    return completion_ids_list, completions


def generate_rollouts_for_example(
    model: Any,
    example: Mapping[str, Any],
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    global_step: int,
    epoch_idx: int,
) -> Tuple[List[Any], List[str]]:
    import torch

    prompt_len = example["input_ids"].shape[1]
    rollout_batch = expand_example_for_rollouts(example, args.num_generations)
    try:
        print(
            "Generating RL rollout batch: "
            f"global_step={global_step}, epoch={epoch_idx}, num_generations={args.num_generations}, "
            f"prompt_len={prompt_len}, max_new_tokens={args.max_new_tokens}"
        )
        generated_ids = model.generate(
            input_ids=rollout_batch["input_ids"],
            attention_mask=rollout_batch["attention_mask"],
            protein_sequences=rollout_batch["protein_sequences"],
            batch_idx_map=rollout_batch["batch_idx_map"],
            structure_coords=rollout_batch["structure_coords"],
            go_aspects=rollout_batch["go_aspects"],
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_new_tokens=args.min_new_tokens,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        completion_ids_list = extract_completion_ids_batch(generated_ids, rollout_batch["input_ids"])
        print(
            "Completed RL rollout batch generation: "
            f"global_step={global_step}, num_generations={args.num_generations}, "
            f"completion_tokens={[int(completion_ids.numel()) for completion_ids in completion_ids_list]}"
        )
        return completion_ids_list, [decode_completion(tokenizer, completion_ids) for completion_ids in completion_ids_list]
    except torch.cuda.OutOfMemoryError:
        print(
            f"CUDA Out of Memory while generating {args.num_generations} RL rollouts in one batch; "
            "falling back to sequential rollout generation for this prompt."
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        print(
            f"Batched RL rollout generation failed for num_generations={args.num_generations}: {exc}. "
            "Falling back to sequential rollout generation for this prompt."
        )
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return generate_rollouts_sequential(
        model,
        example,
        tokenizer,
        args,
        global_step=global_step,
        epoch_idx=epoch_idx,
        prompt_len=prompt_len,
    )


def evaluate_policy_example(
    model: Any,
    ref_model: Any,
    example: Mapping[str, Any],
    tokenizer: Any,
    args: argparse.Namespace,
    reward_names: Sequence[str],
    reward_weights: Sequence[float],
    trace_state: Optional[Dict[str, Any]] = None,
    global_step: int = 0,
    trace_split_name: str = "validation",
) -> Dict[str, float]:
    import torch

    generated_ids = model.generate(
        input_ids=example["input_ids"],
        attention_mask=example["attention_mask"],
        protein_sequences=example["protein_sequences"],
        batch_idx_map=example["batch_idx_map"],
        structure_coords=example["structure_coords"],
        go_aspects=example["go_aspects"],
        do_sample=args.eval_do_sample,
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        top_k=args.eval_top_k,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    completion_ids = extract_completion_ids(generated_ids, example["input_ids"])
    completion_text = decode_completion(tokenizer, completion_ids)
    rewards, reward_components = compute_group_rewards([completion_text], example["sample_meta"], reward_names, reward_weights)
    maybe_trace_generation(
        trace_state,
        split=trace_split_name,
        global_step=global_step,
        sample_meta=example["sample_meta"],
        completion=completion_text,
        total_reward=rewards[0],
        advantage=None,
        reward_components={name: scores[0] for name, scores in reward_components.items()},
    )

    kl_value = 0.0
    if ref_model is not None and completion_ids.numel() > 0:
        combined_ids, combined_mask, prompt_token_len = build_combined_inputs(
            example["input_ids"],
            example["attention_mask"],
            completion_ids,
        )
        current_lp = compute_completion_token_log_probs(
            model,
            combined_ids,
            combined_mask,
            prompt_token_len,
            example["protein_sequences"],
            example["batch_idx_map"],
            example["structure_coords"],
            example["go_aspects"],
        )
        ref_lp = compute_completion_token_log_probs(
            ref_model,
            combined_ids,
            combined_mask,
            prompt_token_len,
            example["protein_sequences"],
            example["batch_idx_map"],
            example["structure_coords"],
            example["go_aspects"],
        )
        kl_term = torch.exp(ref_lp - current_lp) - (ref_lp - current_lp) - 1.0
        kl_value = float(kl_term.mean().item())

    return {
        "reward_sum": float(rewards[0]),
        "length_sum": float(completion_ids.numel()),
        "kl_sum": kl_value,
        "sample_count": 1.0,
    }


def evaluate_policy_batch(
    model: Any,
    ref_model: Any,
    batch: Mapping[str, Any],
    tokenizer: Any,
    args: argparse.Namespace,
    reward_names: Sequence[str],
    reward_weights: Sequence[float],
    device: Any,
    trace_state: Optional[Dict[str, Any]] = None,
    global_step: int = 0,
    trace_split_name: str = "validation",
) -> Dict[str, float]:
    import torch

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    batch_size = input_ids.shape[0]
    structure_coords = batch.get("structure_coords")
    if isinstance(structure_coords, torch.Tensor):
        structure_coords = structure_coords.to(device)

    protein_sequences = list(batch.get("protein_sequences") or [])
    batch_idx_map = list(batch.get("batch_idx_map") or [])
    raw_go_aspects = list(batch.get("batch_go_aspects") or [])
    go_aspects = [raw_go_aspects[idx] if idx < len(raw_go_aspects) and raw_go_aspects[idx] is not None else "all" for idx in range(batch_size)]

    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        protein_sequences=protein_sequences,
        batch_idx_map=batch_idx_map,
        structure_coords=structure_coords,
        go_aspects=go_aspects,
        do_sample=args.eval_do_sample,
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        top_k=args.eval_top_k,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    completion_ids_list = extract_completion_ids_batch(generated_ids, input_ids)

    reward_sum = 0.0
    length_sum = 0.0
    for example_idx, completion_ids in enumerate(completion_ids_list):
        completion_text = decode_completion(tokenizer, completion_ids)
        sample_meta = extract_sample_meta_from_batch(batch, example_idx)
        rewards, reward_components = compute_group_rewards([completion_text], sample_meta, reward_names, reward_weights)
        reward_sum += float(rewards[0])
        length_sum += float(completion_ids.numel())
        maybe_trace_generation(
            trace_state,
            split=trace_split_name,
            global_step=global_step,
            sample_meta=sample_meta,
            completion=completion_text,
            total_reward=rewards[0],
            advantage=None,
            reward_components={name: scores[0] for name, scores in reward_components.items()},
        )

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    kl_sum = compute_batched_completion_kl(
        model,
        ref_model,
        input_ids,
        attention_mask,
        completion_ids_list,
        protein_sequences,
        batch_idx_map,
        structure_coords,
        go_aspects,
        pad_token_id,
    )
    return {
        "reward_sum": reward_sum,
        "length_sum": length_sum,
        "kl_sum": kl_sum,
        "sample_count": float(batch_size),
    }


def compute_group_rewards(
    completions: Sequence[str],
    sample_meta: Mapping[str, Any],
    reward_funcs: Sequence[str],
    reward_weights: Sequence[float],
) -> Tuple[List[float], Dict[str, List[float]]]:
    registry = build_reward_registry()
    component_scores: Dict[str, List[float]] = {}
    total_scores = [0.0 for _ in completions]

    for reward_name, reward_weight in zip(reward_funcs, reward_weights):
        reward_fn = registry[reward_name]
        scores = [float(reward_fn(completion, sample_meta)) for completion in completions]
        component_scores[reward_name] = scores
        for idx, score in enumerate(scores):
            total_scores[idx] += reward_weight * score

    return total_scores, component_scores


def save_raw_checkpoint(model: Any, checkpoint_dir: Path, step: int, args: argparse.Namespace) -> Path:
    import torch

    step_dir = checkpoint_dir / f"checkpoint-{step}"
    step_dir.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    cpu_state_dict = {}
    for key, value in state_dict.items():
        cpu_state_dict[key] = value.detach().cpu() if hasattr(value, "detach") else value

    torch.save(cpu_state_dict, step_dir / "pytorch_model.bin")
    metadata = {
        "global_step": step,
        "benchmark_version": args.benchmark_version,
        "dataset_artifact": args.dataset_artifact,
        "base_checkpoint": args.base_checkpoint,
        "reward_funcs": parse_csv_items(args.reward_funcs),
        "reward_weights": parse_reward_weights(args.reward_weights, len(parse_csv_items(args.reward_funcs))),
        "kl_beta": args.kl_beta,
    }
    with open(step_dir / "training_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return step_dir


def export_hf_model(model: Any, save_dir: Path) -> None:
    import shutil
    import torch

    export_dir = save_dir / "exported_hf"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=False)

    if hasattr(model.text_model, "merge_and_unload"):
        model.text_model = model.text_model.merge_and_unload()

    model = model.cpu()
    model.text_model.save_pretrained(export_dir)
    model.text_tokenizer.save_pretrained(export_dir)
    torch.save(model.protein_projection.state_dict(), export_dir / "protein_projection.pt")

    if getattr(model, "go_projection", None) is not None:
        torch.save(model.go_projection.state_dict(), export_dir / "go_projection.pt")
    if getattr(model, "go_encoder", None) is not None:
        torch.save(model.go_encoder.state_dict(), export_dir / "go_encoder.pt")
    protein_model_dir = export_dir / "protein_model"
    protein_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.protein_model.state_dict(), protein_model_dir / "pytorch_model.bin")
    for item in export_dir.iterdir():
        target_path = save_dir / item.name
        if target_path.exists():
            if target_path.is_dir():
                shutil.rmtree(target_path)
            else:
                target_path.unlink()
        item.replace(target_path)
    export_dir.rmdir()


def maybe_init_wandb(args: argparse.Namespace, tracking_config: Mapping[str, Any]) -> Any:
    import wandb

    init_kwargs: Dict[str, Any] = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "job_type": args.wandb_job_type,
        "name": tracking_config.get("run_name"),
        "config": dict(tracking_config),
    }
    if normalize_text(args.wandb_mode).strip():
        init_kwargs["mode"] = args.wandb_mode
    return wandb.init(**init_kwargs)


def resolve_weave_project(args: argparse.Namespace) -> str:
    explicit = normalize_text(getattr(args, "weave_project", None)).strip()
    if explicit:
        return explicit
    entity = normalize_text(getattr(args, "wandb_entity", None)).strip()
    project = normalize_text(getattr(args, "wandb_project", None)).strip()
    if entity and project:
        return f"{entity}/{project}"
    return ""


def ensure_weave_server_cache_dir(output_dir: Path) -> str:
    cache_dir = os.environ.get("WEAVE_SERVER_CACHE_DIR")
    if normalize_text(cache_dir).strip():
        resolved = Path(cache_dir).expanduser()
    else:
        resolved = output_dir / "weave_server_cache"
        os.environ["WEAVE_SERVER_CACHE_DIR"] = str(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def maybe_init_weave(args: argparse.Namespace, output_dir: Path) -> Optional[Dict[str, Any]]:
    if weave is None:
        print("⚠️  Weave is unavailable; RL generation traces will be skipped.")
        return None

    weave_project = resolve_weave_project(args)
    if not weave_project:
        print("⚠️  Weave project is not set; RL generation traces will be skipped.")
        return None

    cache_dir = ensure_weave_server_cache_dir(output_dir)
    print(f"Initializing Weave for RL tracing: project={weave_project}, cache_dir={cache_dir}")
    client = weave.init(weave_project)
    print("Weave initialization for RL tracing completed.")

    @weave.op(name="train_rl_generation_trace")
    def trace_generation(payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload

    return {
        "client": client,
        "trace_generation": trace_generation,
        "remaining_budget": max(int(getattr(args, "weave_trace_budget", 0)), 0),
        "project": weave_project,
        "logged": 0,
    }


def maybe_trace_generation(
    trace_state: Optional[Dict[str, Any]],
    *,
    split: str,
    global_step: int,
    sample_meta: Mapping[str, Any],
    completion: str,
    total_reward: float,
    advantage: Optional[float] = None,
    reward_components: Optional[Mapping[str, float]] = None,
) -> None:
    if not trace_state or trace_state.get("remaining_budget", 0) <= 0:
        return

    trace_state["remaining_budget"] -= 1
    meta = inspect_completion(completion)
    payload = {
        "split": split,
        "global_step": int(global_step),
        "protein_id": normalize_text(sample_meta.get("protein_id")),
        "go_aspect": normalize_text(sample_meta.get("go_aspect")),
        "prompt_preview": normalize_text(sample_meta.get("prompt_preview"))[:512],
        "completion": completion,
        "reasoning": meta["reasoning"],
        "final_answer": meta["final_answer"],
        "predicted_go_ids": meta["predicted_go_ids"],
        "final_answer_go_ids": meta["final_answer_go_ids"],
        "completion_go_ids": meta["completion_go_ids"],
        "prediction_source": meta["prediction_source"],
        "go_summary_aspect_labels": meta["go_summary_aspect_labels"],
        "final_answer_aspect_labels": meta["final_answer_aspect_labels"],
        "has_go_summary": bool(meta["has_go_summary"]),
        "has_function_summary": bool(meta["has_function_summary"]),
        "has_complete_summary_schema": bool(meta["has_complete_summary_schema"]),
        "final_answer_clean": bool(meta["final_answer_clean"]),
        "structural_noise_count": int(meta["structural_noise_count"]),
        "predicted_go_id_count": int(len(meta["predicted_go_ids"])),
        "requested_go_aspects": build_requested_go_aspects(sample_meta),
        "target_go_ids": build_target_go_ids(sample_meta),
        "reward_total": float(total_reward),
        "advantage": None if advantage is None else float(advantage),
        "reward_components": dict(reward_components or {}),
    }
    trace_state["trace_generation"](payload)
    trace_state["logged"] += 1


def evaluate_policy(
    model: Any,
    ref_model: Any,
    dataloader: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    reward_names: Sequence[str],
    reward_weights: Sequence[float],
    device: Any,
    trace_state: Optional[Dict[str, Any]] = None,
    global_step: int = 0,
    trace_split_name: str = "validation",
) -> Dict[str, float]:
    import torch

    model.eval()
    total_reward = 0.0
    total_kl = 0.0
    total_length = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if args.max_eval_batches > 0 and batch_idx >= args.max_eval_batches:
                break
            batch_size = batch["input_ids"].shape[0]
            batch_metrics: Optional[Dict[str, float]] = None
            if batch_size > 1:
                try:
                    print(
                        f"Running batched RL validation eval: batch_idx={batch_idx}, "
                        f"batch_size={batch_size}, global_step={global_step}."
                    )
                    batch_metrics = evaluate_policy_batch(
                        model=model,
                        ref_model=ref_model,
                        batch=batch,
                        tokenizer=tokenizer,
                        args=args,
                        reward_names=reward_names,
                        reward_weights=reward_weights,
                        device=device,
                        trace_state=trace_state,
                        global_step=global_step,
                        trace_split_name=trace_split_name,
                    )
                except torch.cuda.OutOfMemoryError:
                    print(
                        f"CUDA Out of Memory during RL validation eval batch of size {batch_size}; "
                        "falling back to single-sample validation for this batch."
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as exc:
                    print(
                        f"Batched RL validation eval failed for batch size {batch_size}: {exc}. "
                        "Falling back to single-sample validation for this batch."
                    )
                    traceback.print_exc()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                batch_metrics = evaluate_policy_batch(
                    model=model,
                    ref_model=ref_model,
                    batch=batch,
                    tokenizer=tokenizer,
                    args=args,
                    reward_names=reward_names,
                    reward_weights=reward_weights,
                    device=device,
                    trace_state=trace_state,
                    global_step=global_step,
                    trace_split_name=trace_split_name,
                )

            if batch_metrics is not None:
                total_reward += batch_metrics["reward_sum"]
                total_length += batch_metrics["length_sum"]
                total_kl += batch_metrics["kl_sum"]
                sample_count += int(batch_metrics["sample_count"])
                continue

            for example_idx in range(batch_size):
                example = extract_example_from_batch(batch, example_idx, device)
                example_metrics = evaluate_policy_example(
                    model=model,
                    ref_model=ref_model,
                    example=example,
                    tokenizer=tokenizer,
                    args=args,
                    reward_names=reward_names,
                    reward_weights=reward_weights,
                    trace_state=trace_state,
                    global_step=global_step,
                    trace_split_name=trace_split_name,
                )
                total_reward += example_metrics["reward_sum"]
                total_length += example_metrics["length_sum"]
                total_kl += example_metrics["kl_sum"]
                sample_count += int(example_metrics["sample_count"])

    model.train()
    if sample_count == 0:
        return {
            "eval_reward": 0.0,
            "eval_completion_length": 0.0,
            "eval_loss_kl_div": 0.0,
            "eval_data_step_num_datums": 0.0,
        }
    return {
        "eval_reward": total_reward / sample_count,
        "eval_completion_length": total_length / sample_count,
        "eval_loss_kl_div": total_kl / sample_count if sample_count else 0.0,
        "eval_data_step_num_datums": float(sample_count),
    }


def maybe_run_rotating_validation_eval(
    model: Any,
    ref_model: Any,
    full_val_dataset: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    reward_names: Sequence[str],
    reward_weights: Sequence[float],
    device: Any,
    trace_state: Optional[Dict[str, Any]],
    global_step: int,
    wandb_run: Any,
) -> Optional[Dict[str, float]]:
    if args.rotating_eval_every_n_steps <= 0 or global_step % args.rotating_eval_every_n_steps != 0:
        return None

    rotating_seed = args.seed + global_step * max(args.rotating_eval_seed_stride, 1)
    rotating_dataset, _ = select_validation_subset(
        full_val_dataset,
        max_samples=args.rotating_eval_max_samples,
        seed=rotating_seed,
        strategy=args.rotating_eval_sample_strategy,
        label="rotating",
    )
    rotating_loader = build_dataloader(rotating_dataset, model, args.eval_batch_size, args.num_workers, shuffle=False)
    print(f"Starting rotating RL validation eval at global_step={global_step} with seed={rotating_seed}.")
    rotating_metrics = evaluate_policy(
        model=model,
        ref_model=ref_model,
        dataloader=rotating_loader,
        tokenizer=tokenizer,
        args=args,
        reward_names=reward_names,
        reward_weights=reward_weights,
        device=device,
        trace_state=trace_state,
        global_step=global_step,
        trace_split_name="validation_rotating",
    )
    rotating_metrics = {f"rotating_{key}": value for key, value in rotating_metrics.items()}
    rotating_metrics["rotating_eval_subset_seed"] = float(rotating_seed)
    rotating_metrics["rotating_eval_selected_samples"] = float(len(rotating_dataset))
    if wandb_run is not None:
        wandb_run.log(rotating_metrics, step=global_step)
        print(f"Rotating RL validation metrics logged at global_step={global_step}.")
    return rotating_metrics


def train(args: argparse.Namespace) -> None:
    import torch
    from torch.nn.utils import clip_grad_norm_
    from torch.optim import AdamW

    from bioreason2.utils import (
        build_checkpoint_artifact_metadata,
        build_training_tracking_config,
        maybe_log_directory_artifact,
        maybe_use_artifact_refs,
        parse_artifact_aliases,
        prepare_model_artifact_directory,
        sync_run_config,
    )

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("train_protein_grpo.py requires CUDA. Run it on the CoreWeave GPU cluster.")

    run_name = args.run_name or f"train-rl-{int(time.time())}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_checkpoint_dir = output_dir / "raw_checkpoints"
    raw_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tracking_config = build_training_tracking_config(args=args, run_name=run_name, job_type="train_rl")
    tracking_config["model_artifact"] = args.checkpoint_artifact_name or args.model_artifact

    model = instantiate_model(args, trainable=True).to(device)
    ref_model = instantiate_model(args, trainable=False).to(device) if args.kl_beta > 0 else None

    train_dataset, fixed_val_dataset, full_val_dataset = load_rl_datasets(args)
    train_loader = build_dataloader(train_dataset, model, args.train_batch_size, args.num_workers, shuffle=True)
    val_loader = build_dataloader(fixed_val_dataset, model, args.eval_batch_size, args.num_workers, shuffle=False)

    tokenizer = model.text_tokenizer
    reward_names = parse_csv_items(args.reward_funcs)
    reward_weights = parse_reward_weights(args.reward_weights, len(reward_names))

    if args.resume_from_raw_checkpoint:
        checkpoint = torch.load(args.resume_from_raw_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Resumed RL model weights from {args.resume_from_raw_checkpoint}")

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    ensure_weave_server_cache_dir(output_dir)
    wandb_run = maybe_init_wandb(args, tracking_config)

    if wandb_run is not None:
        sync_run_config(wandb_run, tracking_config)
        sync_run_config(
            wandb_run,
            {
                "dataset_train_size": len(train_dataset),
                "dataset_validation_size": len(fixed_val_dataset),
                "dataset_validation_full_size": len(full_val_dataset),
            },
        )
        maybe_use_artifact_refs(
            wandb_run,
            {
                "temporal_split_artifact": args.temporal_split_artifact,
                "dataset_artifact": args.dataset_artifact,
                "base_checkpoint": args.base_checkpoint,
            },
        )
    weave_trace_state = maybe_init_weave(args, output_dir)
    if wandb_run is not None:
        sync_run_config(
            wandb_run,
            {
                "weave_trace_enabled": bool(weave_trace_state is not None),
                "weave_trace_budget": int(getattr(args, "weave_trace_budget", 0)),
            },
        )

    best_val_reward = float("-inf")
    global_step = 0
    last_eval_step: Optional[int] = None
    optimizer.zero_grad(set_to_none=True)

    for epoch_idx in range(args.max_epochs):
        if global_step >= args.max_steps:
            break

        for batch in train_loader:
            model.train()
            print(f"Starting RL train batch at global_step={global_step}, epoch={epoch_idx}.")
            batch_size = batch["input_ids"].shape[0]
            sample_losses: List[Any] = []
            reward_totals: List[float] = []
            reward_stds: List[float] = []
            reward_component_scores: Dict[str, List[float]] = {reward_name: [] for reward_name in reward_names}
            completion_lengths: List[int] = []
            kl_values: List[float] = []
            trainable_group_count = 0
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

            for example_idx in range(batch_size):
                example = extract_example_from_batch(batch, example_idx, device)

                model.eval()
                completion_ids_list, completions = generate_rollouts_for_example(
                    model,
                    example,
                    tokenizer,
                    args,
                    global_step=global_step,
                    epoch_idx=epoch_idx,
                )
                model.train()
                completion_lengths.extend(int(completion_ids.numel()) for completion_ids in completion_ids_list)

                total_rewards, reward_components = compute_group_rewards(
                    completions,
                    example["sample_meta"],
                    reward_names,
                    reward_weights,
                )
                for reward_name, scores in reward_components.items():
                    reward_component_scores.setdefault(reward_name, []).extend(scores)
                advantages = standardize_group_rewards(total_rewards)
                reward_totals.extend(total_rewards)
                if len(total_rewards) > 1:
                    reward_stds.append(float(torch.tensor(total_rewards, dtype=torch.float32).std(unbiased=False).item()))
                else:
                    reward_stds.append(0.0)

                for generation_idx, (completion_text, total_reward, advantage) in enumerate(
                    zip(completions, total_rewards, advantages)
                ):
                    if generation_idx != 0:
                        continue
                    maybe_trace_generation(
                        weave_trace_state,
                        split="train",
                        global_step=global_step,
                        sample_meta=example["sample_meta"],
                        completion=completion_text,
                        total_reward=total_reward,
                        advantage=advantage,
                        reward_components={
                            reward_name: reward_components[reward_name][generation_idx]
                            for reward_name in reward_components
                        },
                    )

                try:
                    rollout_losses, rollout_kls, has_trainable_rollout = compute_group_policy_losses_batched(
                        model,
                        ref_model,
                        example,
                        completion_ids_list,
                        advantages,
                        pad_token_id,
                        args.kl_beta,
                    )
                except torch.cuda.OutOfMemoryError:
                    print(
                        "CUDA Out of Memory during batched RL loss computation; "
                        "falling back to sequential rollout loss computation for this prompt."
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    rollout_losses, rollout_kls, has_trainable_rollout = compute_group_policy_losses_sequential(
                        model,
                        ref_model,
                        example,
                        completion_ids_list,
                        advantages,
                        args.kl_beta,
                    )

                if has_trainable_rollout:
                    trainable_group_count += 1
                    sample_losses.extend(rollout_losses)
                    kl_values.extend(rollout_kls)

            if not sample_losses:
                global_step += 1
                skipped_payload = {
                    "loss_learning_rate": optimizer.param_groups[0]["lr"],
                    "data_step_num_groups_submitted": float(batch_size),
                    "data_step_num_groups_trainable": float(trainable_group_count),
                    "data_step_num_trajectories": float(len(reward_totals)),
                    "data_step_num_datums": float(batch_size),
                    "data_step_trainer_tokens": float(sum(completion_lengths)),
                    "train_skipped_update": 1.0,
                }
                if wandb_run is not None:
                    print(f"Logging RL skipped-update metrics at global_step={global_step}.")
                    wandb_run.log(skipped_payload, step=global_step)
                    print(f"RL skipped-update metrics logged at global_step={global_step}.")

                if args.eval_every_n_steps > 0 and global_step % args.eval_every_n_steps == 0:
                    print(f"Starting RL validation eval at global_step={global_step} after skipped update.")
                    val_metrics = evaluate_policy(
                        model=model,
                        ref_model=ref_model,
                        dataloader=val_loader,
                        tokenizer=tokenizer,
                        args=args,
                        reward_names=reward_names,
                        reward_weights=reward_weights,
                        device=device,
                        trace_state=weave_trace_state,
                        global_step=global_step,
                        trace_split_name="validation_fixed",
                    )
                    if wandb_run is not None:
                        wandb_run.log(val_metrics, step=global_step)
                        print(f"RL validation metrics logged at global_step={global_step}.")
                    last_eval_step = global_step
                    if val_metrics["eval_reward"] > best_val_reward:
                        best_val_reward = val_metrics["eval_reward"]
                        save_raw_checkpoint(model, raw_checkpoint_dir / "best", global_step, args)
                    maybe_run_rotating_validation_eval(
                        model=model,
                        ref_model=ref_model,
                        full_val_dataset=full_val_dataset,
                        tokenizer=tokenizer,
                        args=args,
                        reward_names=reward_names,
                        reward_weights=reward_weights,
                        device=device,
                        trace_state=weave_trace_state,
                        global_step=global_step,
                        wandb_run=wandb_run,
                    )

                if args.save_every_n_steps > 0 and global_step % args.save_every_n_steps == 0:
                    save_raw_checkpoint(model, raw_checkpoint_dir, global_step, args)

                if global_step >= args.max_steps:
                    break
                continue

            loss = torch.stack(sample_losses).mean() / max(args.gradient_accumulation_steps, 1)
            loss.backward()

            should_step = ((global_step + 1) % max(args.gradient_accumulation_steps, 1)) == 0
            grad_norm_value = 0.0
            if should_step:
                grad_norm = clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    args.max_grad_norm,
                )
                grad_norm_value = float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

            log_payload = {
                "loss_train": float(loss.detach().item() * max(args.gradient_accumulation_steps, 1)),
                "reward": sum(reward_totals) / max(len(reward_totals), 1),
                "reward_std_dev": sum(reward_stds) / max(len(reward_stds), 1),
                "loss_kl_div": sum(kl_values) / max(len(kl_values), 1) if kl_values else 0.0,
                "loss_learning_rate": optimizer.param_groups[0]["lr"],
                "loss_grad_norm": grad_norm_value,
                "data_step_num_groups_submitted": float(batch_size),
                "data_step_num_groups_trainable": float(trainable_group_count),
                "data_step_num_trajectories": float(len(reward_totals)),
                "data_step_num_datums": float(batch_size),
                "data_step_trainer_tokens": float(sum(completion_lengths)),
                "train_skipped_update": 0.0,
            }
            for reward_name, scores in reward_component_scores.items():
                if not scores:
                    continue
                log_payload[f"reward_component/{reward_name}"] = sum(scores) / len(scores)
            if wandb_run is not None:
                print(f"Logging RL train metrics at global_step={global_step}.")
                wandb_run.log(log_payload, step=global_step)
                print(f"RL train metrics logged at global_step={global_step}.")

            if args.eval_every_n_steps > 0 and global_step % args.eval_every_n_steps == 0:
                print(f"Starting RL validation eval at global_step={global_step}.")
                val_metrics = evaluate_policy(
                    model=model,
                    ref_model=ref_model,
                    dataloader=val_loader,
                    tokenizer=tokenizer,
                    args=args,
                    reward_names=reward_names,
                    reward_weights=reward_weights,
                    device=device,
                    trace_state=weave_trace_state,
                    global_step=global_step,
                    trace_split_name="validation_fixed",
                )
                if wandb_run is not None:
                    wandb_run.log(val_metrics, step=global_step)
                    print(f"RL validation metrics logged at global_step={global_step}.")
                last_eval_step = global_step
                if val_metrics["eval_reward"] > best_val_reward:
                    best_val_reward = val_metrics["eval_reward"]
                    save_raw_checkpoint(model, raw_checkpoint_dir / "best", global_step, args)
                maybe_run_rotating_validation_eval(
                    model=model,
                    ref_model=ref_model,
                    full_val_dataset=full_val_dataset,
                    tokenizer=tokenizer,
                    args=args,
                    reward_names=reward_names,
                    reward_weights=reward_weights,
                    device=device,
                    trace_state=weave_trace_state,
                    global_step=global_step,
                    wandb_run=wandb_run,
                )

            if args.save_every_n_steps > 0 and global_step % args.save_every_n_steps == 0:
                save_raw_checkpoint(model, raw_checkpoint_dir, global_step, args)

            if global_step >= args.max_steps:
                break

    if global_step > 0 and last_eval_step != global_step:
        print(f"Starting RL final validation eval at global_step={global_step}.")
        final_val_metrics = evaluate_policy(
            model=model,
            ref_model=ref_model,
            dataloader=val_loader,
            tokenizer=tokenizer,
            args=args,
            reward_names=reward_names,
            reward_weights=reward_weights,
            device=device,
            trace_state=weave_trace_state,
            global_step=global_step,
            trace_split_name="validation_final",
        )
        final_val_metrics = {f"final/{key}": value for key, value in final_val_metrics.items()}
        if wandb_run is not None:
            wandb_run.log(final_val_metrics, step=global_step)
            print(f"RL final validation metrics logged at global_step={global_step}.")

    final_checkpoint_dir = save_raw_checkpoint(model, raw_checkpoint_dir, global_step, args)
    export_hf_model(model, output_dir)

    if wandb_run is not None:
        try:
            import wandb

            artifact_export_dir = output_dir / "_artifact_export"
            artifact_manifest = prepare_model_artifact_directory(
                source_dir=str(output_dir),
                export_dir=str(artifact_export_dir),
            )
            artifact_directory = (
                artifact_manifest["export_dir"]
                if artifact_manifest.get("prepared")
                else str(output_dir)
            )
            checkpoint_status = maybe_log_directory_artifact(
                run=wandb_run,
                wandb_module=wandb,
                artifact_name=args.checkpoint_artifact_name,
                artifact_type="model",
                directory=artifact_directory,
                aliases=parse_artifact_aliases(args.checkpoint_artifact_aliases),
                metadata=build_checkpoint_artifact_metadata(
                    args,
                    run_name,
                    tracking_config={
                        **tracking_config,
                        "artifact_export_mode": artifact_manifest.get("mode"),
                        "artifact_selected_checkpoint": artifact_manifest.get("selected_checkpoint"),
                    },
                ),
            )
            if checkpoint_status["logged"]:
                sync_run_config(
                    wandb_run,
                    {
                        "model_artifact": checkpoint_status["artifact_name"],
                        "model_artifact_aliases": checkpoint_status["aliases"],
                        "last_raw_checkpoint": str(final_checkpoint_dir),
                        "weave_trace_project": weave_trace_state["project"] if weave_trace_state else None,
                        "weave_trace_count": weave_trace_state["logged"] if weave_trace_state else 0,
                    },
                )
        finally:
            if weave_trace_state and weave_trace_state.get("client") is not None:
                flush = getattr(weave_trace_state["client"], "flush", None)
                if callable(flush):
                    flush()
            wandb_run.finish()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GRPO-style RL training entry point for ProteinLLM disease benchmarking.

This script follows the high-level structure of BioReason's GRPO training flow,
but uses a custom multimodal loop so we can keep ProteinLLM's protein encoder,
GO encoder, and reasoning dataset format intact.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import weave
except ImportError:  # pragma: no cover - optional dependency at runtime
    weave = None


GO_ID_PATTERN = re.compile(r"GO:\d{7}")
THINK_TAG_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
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
    has_think = "<think>" in completion and "</think>" in completion
    sections = extract_reasoning_and_answer(completion)
    return 1.0 if has_think and sections["final_answer"] else 0.0


def reasoning_presence_reward(completion: str, _: Mapping[str, Any]) -> float:
    reasoning = extract_reasoning_and_answer(completion)["reasoning"]
    return 1.0 if reasoning else 0.0


def concise_reasoning_reward(completion: str, _: Mapping[str, Any]) -> float:
    reasoning = extract_reasoning_and_answer(completion)["reasoning"]
    if not reasoning:
        return 0.0
    length = len(reasoning.split())
    if 32 <= length <= 384:
        return 1.0
    if 16 <= length <= 512:
        return 0.5
    return 0.0


def answer_nonempty_reward(completion: str, _: Mapping[str, Any]) -> float:
    final_answer = extract_reasoning_and_answer(completion)["final_answer"]
    return 1.0 if final_answer else 0.0


def go_overlap_reward(completion: str, sample_meta: Mapping[str, Any]) -> float:
    predicted = set(extract_go_ids(completion))
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
    predicted = set(extract_go_ids(completion))
    target = set(build_target_go_ids(sample_meta))
    return 1.0 if predicted and predicted == target else 0.0


def build_reward_registry() -> Dict[str, Any]:
    return {
        "strict_format": strict_format_reward,
        "reasoning_presence": reasoning_presence_reward,
        "concise_reasoning": concise_reasoning_reward,
        "answer_nonempty": answer_nonempty_reward,
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
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "bioreason-pro-custom"))
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
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=100)
    parser.add_argument(
        "--eval_sample_strategy",
        type=str,
        default="stratified_aspect_profile",
        choices=["stratified_aspect_profile", "shuffled_prefix"],
    )
    parser.add_argument("--eval_every_n_steps", type=int, default=25)
    parser.add_argument("--save_every_n_steps", type=int, default=50)
    parser.add_argument("--max_eval_batches", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", type=str, default="true")
    parser.add_argument("--kl_beta", type=float, default=0.02)
    parser.add_argument(
        "--reward_funcs",
        type=str,
        default="strict_format,reasoning_presence,go_overlap,answer_nonempty",
        help="Comma-separated reward function names.",
    )
    parser.add_argument(
        "--reward_weights",
        type=str,
        default="",
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
        precomputed_embeddings_path=args.precomputed_embeddings_path,
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


def load_rl_datasets(args: argparse.Namespace) -> Tuple[Any, Any]:
    from bioreason2.dataset.cafa5.load import load_cafa5_dataset
    from bioreason2.dataset.cafa5.subset import select_dataset_subset

    train_dataset, val_dataset, _ = load_cafa5_dataset(
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
    val_dataset, subset_summary = select_dataset_subset(
        val_dataset,
        max_samples=args.max_eval_samples,
        seed=args.seed,
        strategy=args.eval_sample_strategy,
    )
    print(
        "Using RL validation subset: "
        f"strategy={subset_summary['strategy']}, "
        f"requested={subset_summary['requested_samples']}, "
        f"selected={subset_summary['selected_samples']}"
    )
    if subset_summary.get("group_counts"):
        print(f"RL validation subset group counts: {subset_summary['group_counts']}")
    return train_dataset, val_dataset


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

    sample_meta = {
        "protein_id": batch.get("protein_ids", [""])[example_idx] if example_idx < len(batch.get("protein_ids", [])) else "",
        "split": batch.get("sample_splits", [""])[example_idx] if example_idx < len(batch.get("sample_splits", [])) else "",
        "go_bp": batch.get("go_bp_targets", [""])[example_idx] if example_idx < len(batch.get("go_bp_targets", [])) else "",
        "go_mf": batch.get("go_mf_targets", [""])[example_idx] if example_idx < len(batch.get("go_mf_targets", [])) else "",
        "go_cc": batch.get("go_cc_targets", [""])[example_idx] if example_idx < len(batch.get("go_cc_targets", [])) else "",
        "reasoning": batch.get("reasoning_targets", [""])[example_idx] if example_idx < len(batch.get("reasoning_targets", [])) else "",
        "final_answer": batch.get("final_answers", [""])[example_idx] if example_idx < len(batch.get("final_answers", [])) else "",
        "prompt_preview": batch.get("prompt", [""])[example_idx] if example_idx < len(batch.get("prompt", [])) else "",
    }

    return {
        "input_ids": batch["input_ids"][example_idx : example_idx + 1].to(device),
        "attention_mask": batch["attention_mask"][example_idx : example_idx + 1].to(device),
        "protein_sequences": example_sequences,
        "batch_idx_map": [0] * len(example_sequences),
        "structure_coords": example_structure,
        "go_aspects": [example_go_aspect if example_go_aspect is not None else "all"],
        "sample_meta": sample_meta,
    }


def decode_completion(tokenizer: Any, completion_ids: Any) -> str:
    text = tokenizer.decode(completion_ids, skip_special_tokens=False).strip()
    for marker in ("<|im_end|>", "<|endoftext|>"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


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
    if getattr(model, "protein_model", None) is not None:
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
) -> None:
    if not trace_state or trace_state.get("remaining_budget", 0) <= 0:
        return

    trace_state["remaining_budget"] -= 1
    sections = extract_reasoning_and_answer(completion)
    payload = {
        "split": split,
        "global_step": int(global_step),
        "protein_id": normalize_text(sample_meta.get("protein_id")),
        "go_aspect": normalize_text(sample_meta.get("go_aspect")),
        "prompt_preview": normalize_text(sample_meta.get("prompt_preview"))[:512],
        "completion": completion,
        "reasoning": sections["reasoning"],
        "final_answer": sections["final_answer"],
        "predicted_go_ids": extract_go_ids(completion),
        "target_go_ids": build_target_go_ids(sample_meta),
        "reward_total": float(total_reward),
        "advantage": None if advantage is None else float(advantage),
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
            for example_idx in range(batch_size):
                example = extract_example_from_batch(batch, example_idx, device)
                prompt_len = example["input_ids"].shape[1]
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
                    min_new_tokens=args.min_new_tokens,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                completion_ids = generated_ids[0, prompt_len:]
                completion_text = decode_completion(tokenizer, completion_ids)
                rewards, _ = compute_group_rewards([completion_text], example["sample_meta"], reward_names, reward_weights)
                total_reward += rewards[0]
                total_length += float(completion_ids.numel())
                maybe_trace_generation(
                    trace_state,
                    split="validation",
                    global_step=global_step,
                    sample_meta=example["sample_meta"],
                    completion=completion_text,
                    total_reward=rewards[0],
                    advantage=None,
                )
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
                    total_kl += float(kl_term.mean().item())
                sample_count += 1

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

    train_dataset, val_dataset = load_rl_datasets(args)
    train_loader = build_dataloader(train_dataset, model, args.train_batch_size, args.num_workers, shuffle=True)
    val_loader = build_dataloader(val_dataset, model, args.eval_batch_size, args.num_workers, shuffle=False)

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
                "dataset_validation_size": len(val_dataset),
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
            completion_lengths: List[int] = []
            kl_values: List[float] = []
            trainable_group_count = 0

            for example_idx in range(batch_size):
                example = extract_example_from_batch(batch, example_idx, device)
                prompt_len = example["input_ids"].shape[1]

                model.eval()
                completions: List[str] = []
                completion_ids_list: List[Any] = []
                for _ in range(args.num_generations):
                    print(
                        "Generating RL rollout: "
                        f"global_step={global_step}, epoch={epoch_idx}, prompt_len={prompt_len}, "
                        f"max_new_tokens={args.max_new_tokens}"
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
                        min_new_tokens=args.min_new_tokens,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    completion_ids = generated_ids[0, prompt_len:].detach()
                    print(
                        "Completed RL rollout generation: "
                        f"global_step={global_step}, completion_tokens={completion_ids.numel()}"
                    )
                    completion_ids_list.append(completion_ids)
                    completions.append(decode_completion(tokenizer, completion_ids))
                    completion_lengths.append(int(completion_ids.numel()))
                model.train()

                total_rewards, _ = compute_group_rewards(
                    completions,
                    example["sample_meta"],
                    reward_names,
                    reward_weights,
                )
                advantages = standardize_group_rewards(total_rewards)
                reward_totals.extend(total_rewards)
                if len(total_rewards) > 1:
                    reward_stds.append(float(torch.tensor(total_rewards, dtype=torch.float32).std(unbiased=False).item()))
                else:
                    reward_stds.append(0.0)

                for generation_idx, (completion_ids, completion_text, total_reward, advantage) in enumerate(
                    zip(completion_ids_list, completions, total_rewards, advantages)
                ):
                    if generation_idx == 0:
                        maybe_trace_generation(
                            weave_trace_state,
                            split="train",
                            global_step=global_step,
                            sample_meta=example["sample_meta"],
                            completion=completion_text,
                            total_reward=total_reward,
                            advantage=advantage,
                        )
                    if completion_ids.numel() == 0:
                        continue

                    if generation_idx == 0:
                        trainable_group_count += 1

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
                    sample_loss = -(policy_term - args.kl_beta * kl_term).mean()
                    sample_losses.append(sample_loss)
                    kl_values.append(float(kl_term.mean().detach().item()))

            if not sample_losses:
                global_step += 1
                skipped_payload = {
                    "loss_train": 0.0,
                    "reward": sum(reward_totals) / max(len(reward_totals), 1) if reward_totals else 0.0,
                    "reward_std_dev": sum(reward_stds) / max(len(reward_stds), 1) if reward_stds else 0.0,
                    "loss_kl_div": 0.0,
                    "loss_learning_rate": optimizer.param_groups[0]["lr"],
                    "loss_grad_norm": 0.0,
                    "data_step_num_groups_submitted": float(batch_size),
                    "data_step_num_groups_trainable": float(trainable_group_count),
                    "data_step_num_trajectories": float(len(reward_totals)),
                    "data_step_num_datums": float(batch_size),
                    "data_step_trainer_tokens": float(sum(completion_lengths)),
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
                    )
                    if wandb_run is not None:
                        wandb_run.log(val_metrics, step=global_step)
                        print(f"RL validation metrics logged at global_step={global_step}.")
                    last_eval_step = global_step
                    if val_metrics["eval_reward"] > best_val_reward:
                        best_val_reward = val_metrics["eval_reward"]
                        save_raw_checkpoint(model, raw_checkpoint_dir / "best", global_step, args)

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
            }
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
                )
                if wandb_run is not None:
                    wandb_run.log(val_metrics, step=global_step)
                    print(f"RL validation metrics logged at global_step={global_step}.")
                last_eval_step = global_step
                if val_metrics["eval_reward"] > best_val_reward:
                    best_val_reward = val_metrics["eval_reward"]
                    save_raw_checkpoint(model, raw_checkpoint_dir / "best", global_step, args)

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

            checkpoint_status = maybe_log_directory_artifact(
                run=wandb_run,
                wandb_module=wandb,
                artifact_name=args.checkpoint_artifact_name,
                artifact_type="model",
                directory=str(output_dir),
                aliases=parse_artifact_aliases(args.checkpoint_artifact_aliases),
                metadata=build_checkpoint_artifact_metadata(args, run_name, tracking_config=tracking_config),
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

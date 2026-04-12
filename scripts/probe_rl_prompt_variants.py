#!/usr/bin/env python3
"""Probe prompt / decoding variants against the current RL checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_protein_grpo import (
    PreparedQuery,
    build_query_sample_meta,
    build_completion_format_summary,
    build_single_example_batch,
    build_target_go_ids,
    create_vllm_rollout_model,
    extract_go_terms_from_completion,
    instantiate_policy_model_from_source,
    load_reasoning_datasets,
    normalize_text,
    parse_args,
    cleanup_policy_model,
    cleanup_vllm_rollout_model,
)


ASSISTANT_MARKER = "<|im_end|>\n<|im_start|>assistant\n"
LEGACY_STOP_MARKER = "- Hypothesized Interaction Partners"


@dataclass(frozen=True)
class ProbeVariant:
    name: str
    backend: str
    assistant_prefill: str
    do_sample: bool
    max_new_tokens: int
    include_legacy_stop: bool = False
    prefer_original_generate: bool = False
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0


def build_probe_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--protein-id", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--probe-max-new-tokens", type=int, default=768)
    parser.add_argument("--include-hf", type=str, default="true")
    return parser


def parse_probe_args() -> tuple[argparse.Namespace, argparse.Namespace]:
    probe_parser = build_probe_arg_parser()
    probe_args, remaining = probe_parser.parse_known_args()
    train_args = parse_args(remaining)
    train_args.debug_single_process = True
    return probe_args, train_args


def extract_assistant_prefix(prompt_text: str) -> str:
    prompt_text = normalize_text(prompt_text)
    marker_index = prompt_text.rfind(ASSISTANT_MARKER)
    if marker_index < 0:
        return ""
    return prompt_text[marker_index + len(ASSISTANT_MARKER):]


def rewrite_prompt_prefill(prompt_text: str, assistant_prefill: str) -> str:
    prompt_text = normalize_text(prompt_text)
    marker_index = prompt_text.rfind(ASSISTANT_MARKER)
    if marker_index < 0:
        raise ValueError("Assistant marker not found in prompt.")
    return prompt_text[: marker_index + len(ASSISTANT_MARKER)] + assistant_prefill


def clone_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    return value


def rebuild_batch_with_prompt(batch: Mapping[str, Any], model: Any, prompt_text: str) -> Dict[str, Any]:
    tokenizer = getattr(model, "processor").tokenizer
    marker_ids = tokenizer.encode(ASSISTANT_MARKER, add_special_tokens=False)
    prefix_ids = batch["input_ids"][0][batch["attention_mask"][0].bool()]

    marker_end: Optional[int] = None
    for index in range(0, prefix_ids.numel() - len(marker_ids) + 1):
        if torch.equal(prefix_ids[index : index + len(marker_ids)], torch.tensor(marker_ids, device=prefix_ids.device)):
            marker_end = index + len(marker_ids)
            break
    if marker_end is None:
        raise ValueError("Assistant marker token sequence not found in batch input_ids.")

    assistant_prefill = extract_assistant_prefix(prompt_text)
    prefill_ids = tokenizer.encode(assistant_prefill, add_special_tokens=False)
    new_ids = prefix_ids[:marker_end].tolist() + list(prefill_ids)
    input_ids = torch.tensor([new_ids], dtype=batch["input_ids"].dtype, device=batch["input_ids"].device)
    attention_mask = torch.ones_like(input_ids, dtype=batch["attention_mask"].dtype)

    rebuilt = {key: clone_tensor(value) for key, value in batch.items()}
    rebuilt["input_ids"] = input_ids
    rebuilt["attention_mask"] = attention_mask
    rebuilt["prompt"] = [prompt_text]
    if "labels" in rebuilt:
        rebuilt["labels"] = torch.full_like(input_ids, -100)
    return rebuilt


def choose_example(dataset: Any, protein_id: str, index: int) -> tuple[int, Mapping[str, Any]]:
    if protein_id:
        normalized_target = normalize_text(protein_id).strip()
        for sample_index in range(len(dataset)):
            sample = dataset[int(sample_index)]
            if normalize_text(sample.get("protein_id")).strip() == normalized_target:
                return sample_index, sample
        raise ValueError(f"protein_id {protein_id!r} was not found in the selected split.")
    if index < 0 or index >= len(dataset):
        raise IndexError(f"index out of range for split: {index} not in [0, {len(dataset)})")
    return int(index), dataset[int(index)]


def extract_probe_query(batch: Mapping[str, Any], device: Any) -> PreparedQuery:
    prompt_mask = batch["attention_mask"][0].bool()
    input_ids = batch["input_ids"][0][prompt_mask].unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    structure_coords = batch.get("structure_coords")
    if isinstance(structure_coords, torch.Tensor):
        structure_coords = structure_coords[0:1].to(device)
    protein_sequences = list(batch.get("protein_sequences") or [])
    return PreparedQuery(
        input_ids=input_ids,
        attention_mask=attention_mask,
        protein_sequences=protein_sequences,
        batch_idx_map=[0 for _ in protein_sequences],
        structure_coords=structure_coords,
        go_aspects=[normalize_text((batch.get("batch_go_aspects") or ["all"])[0]).strip() or "all"],
        sample_meta=build_query_sample_meta(batch),
        prompt_text=normalize_text((batch.get("prompt") or [""])[0]),
        multimodal_cache=None,
    )


def build_variants(max_new_tokens: int, include_hf: bool) -> List[ProbeVariant]:
    variants: List[ProbeVariant] = [
        ProbeVariant(
            name="vllm_reasoning_sample_legacy_stop",
            backend="vllm",
            assistant_prefill="<|REASONING|>\n",
            do_sample=True,
            temperature=1.0,
            top_k=20,
            top_p=0.95,
            min_p=0.0,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            include_legacy_stop=True,
        ),
        ProbeVariant(
            name="vllm_reasoning_greedy_legacy_stop",
            backend="vllm",
            assistant_prefill="<|REASONING|>\n",
            do_sample=False,
            temperature=0.0,
            top_k=-1,
            top_p=1.0,
            min_p=0.0,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            include_legacy_stop=True,
        ),
        ProbeVariant(
            name="vllm_reasoning_greedy_no_legacy_stop",
            backend="vllm",
            assistant_prefill="<|REASONING|>\n",
            do_sample=False,
            temperature=0.0,
            top_k=-1,
            top_p=1.0,
            min_p=0.0,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            include_legacy_stop=False,
        ),
        ProbeVariant(
            name="vllm_no_prefill_greedy_no_legacy_stop",
            backend="vllm",
            assistant_prefill="",
            do_sample=False,
            temperature=0.0,
            top_k=-1,
            top_p=1.0,
            min_p=0.0,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            include_legacy_stop=False,
        ),
        ProbeVariant(
            name="vllm_final_answer_prefill_greedy_no_legacy_stop",
            backend="vllm",
            assistant_prefill="<|FINAL_ANSWER|>\n",
            do_sample=False,
            temperature=0.0,
            top_k=-1,
            top_p=1.0,
            min_p=0.0,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            include_legacy_stop=False,
        ),
        ProbeVariant(
            name="vllm_closed_reasoning_to_final_greedy_no_legacy_stop",
            backend="vllm",
            assistant_prefill="<|REASONING|>\n<|/REASONING|>\n<|FINAL_ANSWER|>\n",
            do_sample=False,
            temperature=0.0,
            top_k=-1,
            top_p=1.0,
            min_p=0.0,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            include_legacy_stop=False,
        ),
    ]
    if include_hf:
        variants.extend(
            [
                ProbeVariant(
                    name="hf_reasoning_greedy",
                    backend="hf",
                    assistant_prefill="<|REASONING|>\n",
                    do_sample=False,
                    prefer_original_generate=True,
                    temperature=0.0,
                    top_k=-1,
                    top_p=1.0,
                    max_new_tokens=max_new_tokens,
                ),
                ProbeVariant(
                    name="hf_final_answer_prefill_greedy",
                    backend="hf",
                    assistant_prefill="<|FINAL_ANSWER|>\n",
                    do_sample=False,
                    prefer_original_generate=True,
                    temperature=0.0,
                    top_k=-1,
                    top_p=1.0,
                    max_new_tokens=max_new_tokens,
                ),
            ]
        )
    return variants


def compact_tail(text: str, limit: int = 800) -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    return text[-limit:]


def compact_head(text: str, limit: int = 400) -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    return text[:limit]


def build_effective_response(prompt_text: str, completion: str) -> str:
    return extract_assistant_prefix(prompt_text) + normalize_text(completion)


def stop_markers(include_legacy_stop: bool) -> List[str]:
    markers = ["<|/FINAL_ANSWER|>", "<|GO_SUMMARY_END|>", "<|im_end|>", "<|endoftext|>"]
    if include_legacy_stop:
        markers.append(LEGACY_STOP_MARKER)
    return markers


def run_vllm_variant(
    *,
    model: Any,
    batch: Mapping[str, Any],
    variant: ProbeVariant,
    checkpoint_dir: Path,
) -> Dict[str, Any]:
    query = extract_probe_query(batch, model.device)
    outputs = model.generate(
        input_ids=query.input_ids,
        attention_mask=query.attention_mask,
        protein_sequences=query.protein_sequences,
        batch_idx_map=query.batch_idx_map,
        structure_coords=query.structure_coords,
        go_aspects=query.go_aspects,
        multimodal_cache=query.multimodal_cache,
        temperature=float(variant.temperature),
        top_k=int(variant.top_k),
        top_p=float(variant.top_p),
        min_p=float(variant.min_p),
        repetition_penalty=float(variant.repetition_penalty),
        max_new_tokens=int(variant.max_new_tokens),
        seed=123,
        stop=stop_markers(variant.include_legacy_stop),
    )
    completion = normalize_text(outputs[0]).strip()
    return summarize_variant_result(
        variant=variant,
        checkpoint_dir=checkpoint_dir,
        prompt_text=normalize_text((batch.get("prompt") or [""])[0]),
        sample_meta=query.sample_meta,
        completion=completion,
    )


def run_hf_variant(
    *,
    model: Any,
    batch: Mapping[str, Any],
    variant: ProbeVariant,
    checkpoint_dir: Path,
) -> Dict[str, Any]:
    query = extract_probe_query(batch, next(model.parameters()).device)
    tokenizer = model.text_tokenizer
    outputs = model.generate(
        input_ids=query.input_ids,
        attention_mask=query.attention_mask,
        protein_sequences=query.protein_sequences,
        batch_idx_map=query.batch_idx_map,
        structure_coords=query.structure_coords,
        go_aspects=query.go_aspects,
        multimodal_cache=query.multimodal_cache,
        prepared_inputs_embeds=None,
        do_sample=bool(variant.do_sample),
        temperature=float(variant.temperature),
        top_k=int(variant.top_k),
        top_p=float(variant.top_p),
        min_p=float(variant.min_p),
        repetition_penalty=float(variant.repetition_penalty),
        max_new_tokens=int(variant.max_new_tokens),
        prefer_original_generate=bool(variant.prefer_original_generate),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[:, query.input_ids.shape[1] :]
    completion = normalize_text(
        tokenizer.batch_decode(generated, skip_special_tokens=False)[0]
    ).strip()
    return summarize_variant_result(
        variant=variant,
        checkpoint_dir=checkpoint_dir,
        prompt_text=normalize_text((batch.get("prompt") or [""])[0]),
        sample_meta=query.sample_meta,
        completion=completion,
    )


def summarize_variant_result(
    *,
    variant: ProbeVariant,
    checkpoint_dir: Path,
    prompt_text: str,
    sample_meta: Mapping[str, Any],
    completion: str,
) -> Dict[str, Any]:
    effective_response = build_effective_response(prompt_text, completion)
    raw_summary = build_completion_format_summary(completion)
    effective_summary = build_completion_format_summary(effective_response)
    return {
        "variant": asdict(variant),
        "checkpoint_dir": str(checkpoint_dir),
        "protein_id": normalize_text(sample_meta.get("protein_id")).strip(),
        "target_go_ids": build_target_go_ids(sample_meta),
        "assistant_prefill": extract_assistant_prefix(prompt_text),
        "prompt_head": compact_head(prompt_text),
        "prompt_tail": compact_tail(prompt_text),
        "completion_head": compact_head(completion),
        "completion_tail": compact_tail(completion),
        "raw_completion_length_chars": len(completion),
        "raw_completion_go_ids": list(extract_go_terms_from_completion(completion) or []),
        "raw_format_summary": raw_summary,
        "effective_response_length_chars": len(effective_response),
        "effective_response_go_ids": list(extract_go_terms_from_completion(effective_response) or []),
        "effective_format_summary": effective_summary,
    }


def print_human_summary(result: Mapping[str, Any]) -> None:
    raw_summary = result["raw_format_summary"]
    effective_summary = result["effective_format_summary"]
    print(
        json.dumps(
            {
                "variant": result["variant"]["name"],
                "backend": result["variant"]["backend"],
                "protein_id": result["protein_id"],
                "assistant_prefill": result["assistant_prefill"],
                "target_go_ids": result["target_go_ids"],
                "raw_format_valid": raw_summary["format_valid"],
                "effective_format_valid": effective_summary["format_valid"],
                "raw_has_final_answer_tag": raw_summary["has_final_answer_tag"],
                "effective_has_final_answer_tag": effective_summary["has_final_answer_tag"],
                "raw_go_ids": result["raw_completion_go_ids"],
                "effective_go_ids": result["effective_response_go_ids"],
                "completion_tail": result["completion_tail"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    probe_args, train_args = parse_probe_args()
    train_args.trace_rollouts_to_weave = False
    train_args.wandb_mode = "disabled"
    train_args.debug_single_process = True

    runtime = type("ProbeRuntime", (), {"enabled": False, "rank": 0})()
    train_dataset, validation_dataset = load_reasoning_datasets(train_args, runtime)
    dataset = train_dataset if probe_args.split == "train" else validation_dataset
    example_index, example = choose_example(dataset, probe_args.protein_id, probe_args.index)

    checkpoint_dir = Path(normalize_text(train_args.text_model_name).strip()).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"text_model_name path does not exist: {checkpoint_dir}")

    variants = build_variants(
        max_new_tokens=int(probe_args.probe_max_new_tokens),
        include_hf=normalize_text(str(probe_args.include_hf)).strip().lower() not in {"0", "false", "no"},
    )

    print(
        json.dumps(
            {
                "selected_split": probe_args.split,
                "selected_index": int(example_index),
                "selected_protein_id": normalize_text(example.get("protein_id")).strip(),
                "reasoning_dataset_name": train_args.reasoning_dataset_name,
                "reasoning_prompt_style": train_args.reasoning_prompt_style,
                "checkpoint_dir": str(checkpoint_dir),
                "variants": [variant.name for variant in variants],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    results: List[Dict[str, Any]] = []

    vllm_model = create_vllm_rollout_model(train_args, checkpoint_dir)
    try:
        base_batch = build_single_example_batch(example, vllm_model)
        for variant in variants:
            if variant.backend != "vllm":
                continue
            variant_prompt = rewrite_prompt_prefill(
                normalize_text((base_batch.get("prompt") or [""])[0]),
                variant.assistant_prefill,
            )
            variant_batch = rebuild_batch_with_prompt(base_batch, vllm_model, variant_prompt)
            result = run_vllm_variant(
                model=vllm_model,
                batch=variant_batch,
                variant=variant,
                checkpoint_dir=checkpoint_dir,
            )
            results.append(result)
            print_human_summary(result)
    finally:
        cleanup_vllm_rollout_model(vllm_model)

    if any(variant.backend == "hf" for variant in variants):
        hf_model = instantiate_policy_model_from_source(train_args, text_model_name=str(checkpoint_dir), trainable=False)
        hf_model.to("cuda")
        try:
            hf_base_batch = build_single_example_batch(example, hf_model)
            for variant in variants:
                if variant.backend != "hf":
                    continue
                variant_prompt = rewrite_prompt_prefill(
                    normalize_text((hf_base_batch.get("prompt") or [""])[0]),
                    variant.assistant_prefill,
                )
                variant_batch = rebuild_batch_with_prompt(hf_base_batch, hf_model, variant_prompt)
                result = run_hf_variant(
                    model=hf_model,
                    batch=variant_batch,
                    variant=variant,
                    checkpoint_dir=checkpoint_dir,
                )
                results.append(result)
                print_human_summary(result)
        finally:
            cleanup_policy_model(hf_model)

    payload = {
        "probe": {
            "split": probe_args.split,
            "index": int(example_index),
            "protein_id": normalize_text(example.get("protein_id")).strip(),
            "reasoning_dataset_name": train_args.reasoning_dataset_name,
            "reasoning_prompt_style": train_args.reasoning_prompt_style,
            "checkpoint_dir": str(checkpoint_dir),
        },
        "results": results,
    }

    if probe_args.output_json:
        output_path = Path(probe_args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote probe results to {output_path}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GRPO-style RL training entry point for ProteinLLM disease benchmarking.

This script follows the high-level structure of BioReason's GRPO training flow,
but uses a custom multimodal loop so we can keep ProteinLLM's protein encoder,
GO encoder, and reasoning dataset format intact.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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
STRUCTURAL_TAG_PATTERN = re.compile(r"</?tool_call>")
GO_ASPECT_PATTERN = re.compile(r"(?im)^\s*(MF|BP|CC)\s*:\s*(.+)$")
GO_SUMMARY_START = "<|GO_SUMMARY_START|>"
GO_SUMMARY_END = "<|GO_SUMMARY_END|>"
FUNCTION_SUMMARY_START = "<|FUNCTION_SUMMARY_START|>"
FUNCTION_SUMMARY_END = "<|FUNCTION_SUMMARY_END|>"
GO_ASPECT_ORDER = ("MF", "BP", "CC")
GO_NAMESPACE_TO_ASPECT = {
    "molecular_function": "MF",
    "biological_process": "BP",
    "cellular_component": "CC",
}
TERMINAL_SUMMARY_MARKERS = (
    GO_SUMMARY_END,
    "</answer>",
    "<|im_end|>",
    "<|endoftext|>",
)
DIAGNOSTIC_REWARD_NAMES = (
    "strict_format",
    "summary_schema",
    "go_presence",
    "go_aspect_coverage",
    "go_overlap",
    "truncation_penalty",
    "structural_noise",
)
WANDB_BOOTSTRAP_METRICS = (
    "loss_train",
    "reward",
    "reward_std_dev",
    "loss_kl_div",
    "loss_policy_ratio_mean",
    "loss_policy_ratio_max",
    "loss_learning_rate",
    "loss_grad_norm",
    "eval_reward",
    "eval_completion_length",
    "eval_loss_kl_div",
    "eval_data_step_num_datums",
    "data_step_num_groups_submitted",
    "data_step_num_groups_trainable",
    "data_step_num_trajectories",
    "data_step_num_datums",
    "data_step_trainer_tokens",
    "data_step_num_update_passes",
    "train_skipped_update",
)
DEFAULT_GO_OBO_PATH = str((Path(__file__).resolve().parent / "bioreason2" / "dataset" / "go-basic.obo").resolve())
REWARD_CONTEXT: Dict[str, Any] = {
    "go_obo_path": DEFAULT_GO_OBO_PATH if os.path.exists(DEFAULT_GO_OBO_PATH) else "",
    "ia_file_path": "",
    "reward_final_answer_only": False,
    "reward_prediction_source": "reasoning_trace",
}


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

    structured_final_answer = extract_structured_final_answer(raw)
    if structured_final_answer:
        final_answer = structured_final_answer

    return {"reasoning": reasoning, "final_answer": final_answer}


def extract_structured_final_answer(text: Any) -> str:
    raw = normalize_text(text).strip()
    if not raw:
        return ""

    answer_match = ANSWER_TAG_PATTERN.search(raw)
    if answer_match:
        answer_scope = answer_match.group(1).strip()
    elif "</think>" in raw:
        answer_scope = raw.split("</think>", 1)[1].strip()
    else:
        answer_scope = raw

    go_summary = extract_tagged_block(answer_scope, GO_SUMMARY_START, GO_SUMMARY_END)
    function_summary = extract_tagged_block(answer_scope, FUNCTION_SUMMARY_START, FUNCTION_SUMMARY_END)

    structured_blocks: List[str] = []
    if go_summary:
        structured_blocks.append(f"{GO_SUMMARY_START}\n{go_summary}\n{GO_SUMMARY_END}")
    if function_summary:
        structured_blocks.append(f"{FUNCTION_SUMMARY_START}\n{function_summary}\n{FUNCTION_SUMMARY_END}")
    if structured_blocks:
        return "\n\n".join(structured_blocks).strip()
    return ""


def extract_reward_prediction_text(text: Any) -> str:
    raw = normalize_text(text).strip()
    if not raw:
        return ""
    reward_context = resolve_reward_context()
    prediction_source = normalize_text(reward_context.get("reward_prediction_source")).strip().lower()
    if not prediction_source:
        prediction_source = "structured_go_summary" if reward_context.get("reward_final_answer_only", False) else "reasoning_trace"

    if prediction_source == "reasoning_trace":
        # The paper describes regex extraction from the generated reasoning trace.
        # In our formatting variants that means the whole generated completion,
        # not only the text inside <think> tags.
        return raw
    if prediction_source == "final_answer":
        sections = extract_reasoning_and_answer(raw)
        final_answer = sections.get("final_answer", "").strip()
        if final_answer:
            return final_answer
        if "</think>" in raw:
            return raw.split("</think>", 1)[1].strip()
        return raw

    structured_final_answer = extract_structured_final_answer(raw)
    if not structured_final_answer:
        return ""
    return extract_tagged_block(structured_final_answer, GO_SUMMARY_START, GO_SUMMARY_END)


def require_training_ia_file(args: argparse.Namespace, reward_names: Sequence[str]) -> str:
    if "ia_weighted_f1" not in reward_names:
        return ""
    ia_file_path = normalize_text(getattr(args, "ia_file_path", None)).strip()
    require_ia_file = bool(getattr(args, "require_ia_file", True))
    if ia_file_path and os.path.exists(ia_file_path):
        return ia_file_path
    if require_ia_file:
        raise FileNotFoundError(
            "IA-weighted RL reward requires a valid --ia_file_path. "
            f"Resolved path: {ia_file_path or '<empty>'}"
        )
    return ""


def count_structural_noise_tokens(text: Any) -> int:
    return len(STRUCTURAL_TAG_PATTERN.findall(normalize_text(text)))


def has_meaningful_text(text: Any) -> bool:
    return bool(re.search(r"[A-Za-z0-9]", normalize_text(text)))


def count_words(text: Any) -> int:
    return len(normalize_text(text).split())


def has_terminal_summary_marker(text: Any) -> bool:
    raw = normalize_text(text)
    return any(marker in raw for marker in TERMINAL_SUMMARY_MARKERS)


def build_requested_go_aspects(sample_meta: Mapping[str, Any]) -> List[str]:
    requested = normalize_text(sample_meta.get("go_aspect")).strip().lower()
    if requested in {"mf", "bp", "cc"}:
        return [requested.upper()]

    target_aspects: List[str] = []
    for aspect, field_name in (("MF", "go_mf"), ("BP", "go_bp"), ("CC", "go_cc")):
        if extract_go_ids(sample_meta.get(field_name)):
            target_aspects.append(aspect)
    return target_aspects or list(GO_ASPECT_ORDER)


def configure_reward_context(args: argparse.Namespace) -> None:
    REWARD_CONTEXT["go_obo_path"] = normalize_text(getattr(args, "go_obo_path", None)).strip() or (
        DEFAULT_GO_OBO_PATH if os.path.exists(DEFAULT_GO_OBO_PATH) else ""
    )
    REWARD_CONTEXT["ia_file_path"] = normalize_text(getattr(args, "ia_file_path", None)).strip()
    REWARD_CONTEXT["reward_final_answer_only"] = bool(getattr(args, "reward_final_answer_only", False))
    REWARD_CONTEXT["reward_prediction_source"] = (
        normalize_text(getattr(args, "reward_prediction_source", None)).strip().lower()
        or ("structured_go_summary" if REWARD_CONTEXT["reward_final_answer_only"] else "reasoning_trace")
    )
    inspect_completion_text.cache_clear()


def resolve_reward_context() -> Dict[str, Any]:
    return dict(REWARD_CONTEXT)


@lru_cache(maxsize=2)
def load_go_term_metadata(obo_path: str) -> Dict[str, Dict[str, Any]]:
    resolved_path = normalize_text(obo_path).strip()
    if not resolved_path or not os.path.exists(resolved_path):
        return {}

    metadata: Dict[str, Dict[str, Any]] = {}
    current_id = ""
    current_namespace = ""
    current_parents: List[str] = []
    current_obsolete = False
    in_term = False

    def finalize_term() -> None:
        nonlocal current_id, current_namespace, current_parents, current_obsolete
        if current_id and not current_obsolete:
            ordered_parents: List[str] = []
            seen = set()
            for parent in current_parents:
                if parent and parent not in seen:
                    seen.add(parent)
                    ordered_parents.append(parent)
            metadata[current_id] = {
                "aspect": GO_NAMESPACE_TO_ASPECT.get(current_namespace, ""),
                "parents": tuple(ordered_parents),
            }
        current_id = ""
        current_namespace = ""
        current_parents = []
        current_obsolete = False

    with open(resolved_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "[Term]":
                finalize_term()
                in_term = True
                continue
            if line.startswith("[") and line != "[Term]":
                finalize_term()
                in_term = False
                continue
            if not in_term or not line:
                continue
            if line.startswith("id: "):
                match = GO_ID_PATTERN.search(line)
                if match:
                    current_id = match.group(0)
            elif line.startswith("namespace: "):
                current_namespace = line.split(":", 1)[1].strip()
            elif line.startswith("is_obsolete: "):
                current_obsolete = line.split(":", 1)[1].strip().lower() == "true"
            elif line.startswith("is_a: "):
                match = GO_ID_PATTERN.search(line)
                if match:
                    current_parents.append(match.group(0))
            elif line.startswith("relationship: part_of "):
                match = GO_ID_PATTERN.search(line)
                if match:
                    current_parents.append(match.group(0))
    finalize_term()
    return metadata


@lru_cache(maxsize=131072)
def get_go_ancestors(go_id: str, obo_path: str) -> Tuple[str, ...]:
    metadata = load_go_term_metadata(obo_path)
    if not go_id:
        return tuple()
    if go_id not in metadata:
        return (go_id,)

    ordered: List[str] = []
    seen = set()

    def visit(term_id: str) -> None:
        if term_id in seen:
            return
        seen.add(term_id)
        ordered.append(term_id)
        for parent_id in metadata.get(term_id, {}).get("parents", ()):
            visit(parent_id)

    visit(go_id)
    return tuple(ordered)


def resolve_go_aspect(go_id: str, obo_path: str) -> str:
    return normalize_text(load_go_term_metadata(obo_path).get(go_id, {}).get("aspect")).strip().upper()


def propagate_go_ids(go_ids: Iterable[str], obo_path: str, allowed_aspects: Optional[Iterable[str]] = None) -> List[str]:
    allowed = {normalize_text(aspect).strip().upper() for aspect in (allowed_aspects or []) if normalize_text(aspect).strip()}
    ordered: List[str] = []
    seen = set()
    for go_id in go_ids:
        for ancestor_id in get_go_ancestors(go_id, obo_path):
            if allowed:
                aspect = resolve_go_aspect(ancestor_id, obo_path)
                if aspect and aspect not in allowed:
                    continue
            if ancestor_id not in seen:
                seen.add(ancestor_id)
                ordered.append(ancestor_id)
    return ordered


@lru_cache(maxsize=2)
def load_ia_weights(ia_file_path: str) -> Dict[str, float]:
    resolved_path = normalize_text(ia_file_path).strip()
    if not resolved_path or not os.path.exists(resolved_path):
        return {}

    weights: Dict[str, float] = {}
    with open(resolved_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2 or not GO_ID_PATTERN.fullmatch(parts[0]):
                continue
            try:
                weights[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return weights


def compute_weighted_go_f1(
    predicted_go_ids: Iterable[str],
    target_go_ids: Iterable[str],
    *,
    ia_weights: Optional[Mapping[str, float]] = None,
) -> float:
    predicted = list(predicted_go_ids)
    target = list(target_go_ids)
    if not predicted or not target:
        return 0.0

    weight_lookup = ia_weights or {}
    predicted_set = set(predicted)
    target_set = set(target)
    intersection = predicted_set & target_set
    if not intersection:
        return 0.0

    def _weight(go_id: str) -> float:
        return float(weight_lookup.get(go_id, 1.0))

    precision_numerator = sum(_weight(go_id) for go_id in intersection)
    precision_denominator = sum(_weight(go_id) for go_id in predicted_set)
    recall_denominator = sum(_weight(go_id) for go_id in target_set)
    if precision_denominator <= 0.0 or recall_denominator <= 0.0:
        return 0.0
    precision = precision_numerator / precision_denominator
    recall = precision_numerator / recall_denominator
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


@lru_cache(maxsize=8192)
def inspect_completion_text(raw_completion: str) -> Dict[str, Any]:
    sections = extract_reasoning_and_answer(raw_completion)
    final_answer = sections["final_answer"].strip()
    reward_prediction_text = extract_reward_prediction_text(raw_completion)
    reward_context = resolve_reward_context()
    reward_prediction_source = normalize_text(reward_context.get("reward_prediction_source")).strip().lower()
    go_summary = extract_tagged_block(final_answer, GO_SUMMARY_START, GO_SUMMARY_END)
    function_summary = extract_tagged_block(final_answer, FUNCTION_SUMMARY_START, FUNCTION_SUMMARY_END)
    go_summary_aspects = extract_go_aspect_map(go_summary)
    final_answer_aspects = extract_go_aspect_map(final_answer)
    structural_noise_count = count_structural_noise_tokens(final_answer)
    final_answer_clean = bool(final_answer) and structural_noise_count == 0 and has_meaningful_text(final_answer)
    reward_prediction_aspects = extract_go_aspect_map(reward_prediction_text)
    has_closed_reasoning = "</think>" in raw_completion
    has_answer_tag = bool(ANSWER_TAG_PATTERN.search(raw_completion))

    prediction_source = "none"
    if has_meaningful_text(reward_prediction_text):
        if reward_prediction_source == "reasoning_trace":
            prediction_source = "reasoning_trace"
        elif reward_prediction_source == "final_answer":
            prediction_source = "final_answer"
        elif go_summary:
            prediction_source = "structured_final_answer"
        elif has_answer_tag:
            prediction_source = "answer_tag"
        elif has_closed_reasoning:
            prediction_source = "post_think"

    return {
        "reasoning": sections["reasoning"],
        "final_answer": final_answer,
        "reward_prediction_text": reward_prediction_text,
        "go_summary": go_summary,
        "function_summary": function_summary,
        "go_summary_aspects": go_summary_aspects,
        "go_summary_aspect_labels": list(go_summary_aspects.keys()),
        "final_answer_aspects": final_answer_aspects,
        "final_answer_aspect_labels": list(final_answer_aspects.keys()),
        "reward_prediction_aspects": reward_prediction_aspects,
        "reward_prediction_aspect_labels": list(reward_prediction_aspects.keys()),
        "has_go_summary": bool(go_summary),
        "has_function_summary": bool(function_summary),
        # We keep GO_SUMMARY schema checks even when reward is extracted from the
        # reasoning trace, because the paper still expects a structured GO output.
        "has_complete_summary_schema": bool(go_summary and go_summary_aspects),
        "has_closed_reasoning": has_closed_reasoning,
        "has_answer_tag": has_answer_tag,
        "final_answer_clean": final_answer_clean,
        "final_answer_has_text": has_meaningful_text(final_answer),
        "structural_noise_count": structural_noise_count,
        "prediction_source": prediction_source,
        "prediction_text": reward_prediction_text,
        "predicted_go_ids": extract_go_ids(reward_prediction_text),
        "go_summary_go_ids": extract_go_ids(go_summary),
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
    if meta["has_closed_reasoning"] and meta["final_answer_clean"]:
        return 1.0
    if meta["has_closed_reasoning"] and meta["predicted_go_ids"]:
        return 0.5
    return 0.0


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
    if meta["has_complete_summary_schema"] and meta["go_summary_go_ids"] and matched_aspects == requested_aspects:
        return 1.0
    if meta["has_complete_summary_schema"] and meta["go_summary_go_ids"] and matched_aspects:
        return 0.5
    if meta["has_go_summary"] and meta["go_summary_go_ids"] and matched_aspects:
        return 0.25
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
        if meta["has_complete_summary_schema"]:
            return 1.0
        if meta["has_go_summary"] or meta["has_closed_reasoning"]:
            return 0.75
        return 0.5

    if not build_target_go_ids(sample_meta):
        return 0.0
    if meta["prediction_text"]:
        return -1.0
    return -0.5


def go_aspect_coverage_reward(completion: str, sample_meta: Mapping[str, Any]) -> float:
    meta = inspect_completion(completion)
    requested_aspects = set(build_requested_go_aspects(sample_meta))
    if not requested_aspects or not meta["predicted_go_ids"]:
        return 0.0

    predicted_go_ids = set(meta["predicted_go_ids"])
    covered_aspects = set()
    for aspect, field_name in (("MF", "go_mf"), ("BP", "go_bp"), ("CC", "go_cc")):
        if aspect not in requested_aspects:
            continue
        target_ids = set(extract_go_ids(sample_meta.get(field_name)))
        if target_ids and predicted_go_ids & target_ids:
            covered_aspects.add(aspect)
    if not covered_aspects:
        return 0.0
    return len(covered_aspects) / len(requested_aspects)


def ia_weighted_f1_reward(completion: str, sample_meta: Mapping[str, Any]) -> float:
    meta = inspect_completion(completion)
    predicted = set(meta["predicted_go_ids"])
    target = set(build_target_go_ids(sample_meta))
    if not predicted or not target:
        return 0.0

    reward_context = resolve_reward_context()
    go_obo_path = normalize_text(reward_context.get("go_obo_path")).strip()
    ia_file_path = normalize_text(reward_context.get("ia_file_path")).strip()
    requested_aspects = set(build_requested_go_aspects(sample_meta))

    if go_obo_path and os.path.exists(go_obo_path):
        predicted = set(propagate_go_ids(predicted, go_obo_path, requested_aspects))
        target = set(propagate_go_ids(target, go_obo_path, requested_aspects))

    ia_weights = load_ia_weights(ia_file_path) if ia_file_path else {}
    return compute_weighted_go_f1(predicted, target, ia_weights=ia_weights)


def go_overlap_reward(completion: str, sample_meta: Mapping[str, Any]) -> float:
    meta = inspect_completion(completion)
    predicted = set(meta["predicted_go_ids"])
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


def truncation_penalty_reward(completion: str, _: Mapping[str, Any]) -> float:
    text = normalize_text(completion)
    word_count = count_words(text)
    predicted_go_ids = inspect_completion(completion)["predicted_go_ids"]
    if has_terminal_summary_marker(text):
        if word_count <= 192:
            return 0.5
        if word_count <= 320:
            return 0.25
        return 0.0
    if predicted_go_ids:
        if word_count <= 256:
            return 0.25
        if word_count >= 480:
            return -0.25
        return 0.0
    if word_count >= 320:
        return -1.0
    if word_count >= 220:
        return -0.5
    return 0.0


def build_reward_registry() -> Dict[str, Any]:
    return {
        "ia_weighted_f1": ia_weighted_f1_reward,
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
        "truncation_penalty": truncation_penalty_reward,
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


def compute_batch_relative_advantages(
    grouped_rewards: Sequence[Sequence[float]],
    *,
    epsilon_std: float = 1e-6,
    reward_scaling: str = "batch",
    distributed_device: Any = None,
) -> Tuple[List[List[float]], float]:
    if not grouped_rewards:
        return [], 0.0

    if reward_scaling != "batch":
        return [standardize_group_rewards(group_rewards) for group_rewards in grouped_rewards], 0.0

    flat_rewards = [float(reward) for group_rewards in grouped_rewards for reward in group_rewards]
    if not flat_rewards:
        return [[] for _ in grouped_rewards], 0.0

    flat_sum = float(sum(flat_rewards))
    flat_sq_sum = float(sum(reward * reward for reward in flat_rewards))
    flat_count = float(len(flat_rewards))

    if distributed_device is not None and is_distributed_enabled():
        import torch

        stats = torch.tensor(
            [flat_sum, flat_sq_sum, flat_count],
            device=distributed_device,
            dtype=torch.float64,
        )
        distributed_reduce_tensor(stats, op="sum")
        flat_sum = float(stats[0].item())
        flat_sq_sum = float(stats[1].item())
        flat_count = float(stats[2].item())

    flat_mean = flat_sum / flat_count
    variance = (flat_sq_sum / flat_count) - (flat_mean * flat_mean)
    global_std = math.sqrt(max(variance, 0.0))
    if global_std < epsilon_std:
        return [[0.0 for _ in group_rewards] for group_rewards in grouped_rewards], global_std

    normalized_groups: List[List[float]] = []
    denominator = global_std + epsilon_std
    for group_rewards in grouped_rewards:
        if not group_rewards:
            normalized_groups.append([])
            continue
        group_mean = sum(group_rewards) / len(group_rewards)
        normalized_groups.append([(float(reward) - group_mean) / denominator for reward in group_rewards])
    return normalized_groups, global_std


def build_generation_stopping_criteria(tokenizer: Any) -> Any:
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        return None
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except ImportError:
        return None

    stop_sequences: List[List[int]] = []
    for marker in TERMINAL_SUMMARY_MARKERS:
        token_ids = encode(marker, add_special_tokens=False)
        if token_ids:
            stop_sequences.append(token_ids)
    if not stop_sequences:
        return None

    class StopOnTokenSequences(StoppingCriteria):
        def __init__(self, sequences: Sequence[Sequence[int]]):
            self.sequences = [list(sequence) for sequence in sequences if sequence]

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            import torch

            done = torch.zeros((input_ids.shape[0],), dtype=torch.bool, device=input_ids.device)
            for sequence in self.sequences:
                sequence_tensor = torch.tensor(sequence, dtype=input_ids.dtype, device=input_ids.device)
                if input_ids.shape[1] < sequence_tensor.numel():
                    continue
                done |= (input_ids[:, -sequence_tensor.numel() :] == sequence_tensor).all(dim=1)
            return done

    return StoppingCriteriaList([StopOnTokenSequences(stop_sequences)])


def build_generation_kwargs(
    args: argparse.Namespace,
    tokenizer: Any,
    *,
    for_eval: bool,
) -> Dict[str, Any]:
    generation_kwargs = {
        "do_sample": args.eval_do_sample if for_eval else args.do_sample,
        "temperature": args.eval_temperature if for_eval else args.temperature,
        "top_p": args.eval_top_p if for_eval else args.top_p,
        "top_k": args.eval_top_k if for_eval else args.top_k,
        "min_new_tokens": args.min_new_tokens,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if getattr(args, "min_p", 0.0) > 0.0:
        generation_kwargs["min_p"] = args.min_p
    stopping_criteria = build_generation_stopping_criteria(tokenizer)
    if stopping_criteria is not None:
        generation_kwargs["stopping_criteria"] = stopping_criteria
    return generation_kwargs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "bioreasoning-pro"))
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)
    parser.add_argument("--wandb_job_type", type=str, default="train_rl", choices=["train_rl"])
    parser.add_argument("--weave_project", type=str, default=None)
    parser.add_argument("--weave_trace_budget", type=int, default=64)
    parser.add_argument("--weave_trace_full_group_count", type=int, default=4)
    parser.add_argument("--weave_trace_full_rollouts_per_group", type=int, default=24)

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
    parser.add_argument("--ia_file_path", type=str, default=os.environ.get("BIOREASON_IA_FILE_PATH", ""))
    parser.add_argument("--require_ia_file", type=str, default="true")
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
    parser.add_argument("--add_uniprot_summary", type=str, default="false")
    parser.add_argument("--is_swissprot", type=str, default="false")
    parser.add_argument("--include_go_defs", type=str, default="false")
    parser.add_argument("--interpro_in_prompt", type=str, default="true")
    parser.add_argument("--ppi_in_prompt", type=str, default="true")
    parser.add_argument("--predict_interpro", type=str, default="false")
    parser.add_argument("--include_protein_function_summary", type=str, default="true")
    parser.add_argument("--split_go_aspects", type=str, default="false")
    parser.add_argument(
        "--reasoning_prompt_style",
        type=str,
        default="paper_compact",
        choices=["verbose", "paper_compact"],
    )
    parser.add_argument("--compact_interpro_limit", type=int, default=12)
    parser.add_argument("--compact_ppi_limit", type=int, default=10)
    parser.add_argument("--compact_go_speculation_limit", type=int, default=8)

    parser.add_argument("--max_length_text", type=int, default=512)
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
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--gradient_checkpointing", type=str, default="true")
    parser.add_argument("--disable_model_dropout", type=str, default="true")

    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--train_batch_size",
        "--per_device_train_batch_size",
        dest="train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--eval_batch_size",
        "--per_device_eval_batch_size",
        dest="eval_batch_size",
        type=int,
        default=4,
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=200)
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

    parser.add_argument("--loss_type", type=str, default="dr_grpo", choices=["dr_grpo"])
    parser.add_argument("--steps_per_generation", type=int, default=2)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=24)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--rollout_logprob_microbatch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--do_sample", type=str, default="true")
    parser.add_argument("--eval_do_sample", type=str, default="false")
    parser.add_argument("--eval_temperature", type=float, default=0.1)
    parser.add_argument("--eval_top_p", type=float, default=0.9)
    parser.add_argument("--eval_top_k", type=int, default=20)
    parser.add_argument("--clip_epsilon_low", type=float, default=7e-4)
    parser.add_argument("--clip_epsilon_high", type=float, default=9e-4)
    parser.add_argument("--reward_scaling", type=str, default="batch", choices=["batch", "group"])
    parser.add_argument("--advantage_epsilon_std", type=float, default=1e-6)
    parser.add_argument("--importance_sampling_level", type=str, default="sequence", choices=["sequence"])
    parser.add_argument("--importance_sampling_cap", type=float, default=2.0)
    parser.add_argument("--reward_final_answer_only", type=str, default="false")
    parser.add_argument(
        "--reward_prediction_source",
        type=str,
        default="reasoning_trace",
        choices=["reasoning_trace", "final_answer", "structured_go_summary"],
    )
    parser.add_argument("--kl_beta", type=float, default=1e-4)
    parser.add_argument(
        "--reward_funcs",
        type=str,
        default="ia_weighted_f1",
        help="Comma-separated reward function names.",
    )
    parser.add_argument(
        "--reward_weights",
        type=str,
        default="1.0",
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
        "gradient_checkpointing",
        "disable_model_dropout",
        "do_sample",
        "eval_do_sample",
        "reward_final_answer_only",
        "require_ia_file",
        "ablation_from_paper_rl",
    ]

    def _str2bool(raw: Any) -> bool:
        value = normalize_text(raw).strip().lower()
        return value in {"1", "true", "t", "yes", "y"}

    for field_name in bool_fields:
        setattr(args, field_name, _str2bool(getattr(args, field_name)))

    return args


def can_cache_multimodal_prefix(args: argparse.Namespace) -> bool:
    return not any(
        (
            getattr(args, "protein_model_finetune", False),
            getattr(args, "train_projector", False),
            getattr(args, "train_go_modules", False),
        )
    )


def build_batch_semantics(args: argparse.Namespace, world_size: int) -> Dict[str, float]:
    per_device_train_batch_size = max(int(getattr(args, "train_batch_size", 1)), 1)
    per_device_eval_batch_size = max(int(getattr(args, "eval_batch_size", 1)), 1)
    num_generations = max(int(getattr(args, "num_generations", 1)), 1)
    global_unique_proteins_per_step = int(per_device_train_batch_size * max(int(world_size), 1))
    return {
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "world_size": max(int(world_size), 1),
        "global_unique_proteins_per_step": global_unique_proteins_per_step,
        "global_num_trajectories_per_step": global_unique_proteins_per_step * num_generations,
        "global_unique_proteins_target": global_unique_proteins_per_step,
    }


def unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def is_distributed_enabled() -> bool:
    try:
        import torch.distributed as dist
    except ImportError:
        return False
    return dist.is_available() and dist.is_initialized()


def get_distributed_rank() -> int:
    if not is_distributed_enabled():
        return 0
    import torch.distributed as dist

    return int(dist.get_rank())


def get_distributed_world_size() -> int:
    if not is_distributed_enabled():
        return 1
    import torch.distributed as dist

    return int(dist.get_world_size())


def is_main_process() -> bool:
    return get_distributed_rank() == 0


def distributed_barrier() -> None:
    if not is_distributed_enabled():
        return
    import torch.distributed as dist

    dist.barrier()


def distributed_reduce_tensor(tensor: Any, *, op: str = "sum") -> Any:
    if not is_distributed_enabled():
        return tensor
    import torch.distributed as dist

    if op == "sum":
        reduce_op = dist.ReduceOp.SUM
    elif op == "max":
        reduce_op = dist.ReduceOp.MAX
    else:
        raise ValueError(f"Unsupported distributed reduce op: {op}")
    dist.all_reduce(tensor, op=reduce_op)
    return tensor


def distributed_sum_scalar(value: float, device: Any) -> float:
    import torch

    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    distributed_reduce_tensor(tensor, op="sum")
    return float(tensor.item())


def distributed_max_scalar(value: float, device: Any) -> float:
    import torch

    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    distributed_reduce_tensor(tensor, op="max")
    return float(tensor.item())


def init_distributed_runtime() -> Dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("train_protein_grpo.py requires CUDA. Run it on the CoreWeave GPU cluster.")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank_env = os.environ.get("LOCAL_RANK")
    local_rank = int(local_rank_env) if local_rank_env is not None else 0

    runtime = {
        "enabled": world_size > 1,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "backend": None,
    }

    if world_size > 1:
        import torch.distributed as dist

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        runtime["backend"] = "nccl"
        runtime["device"] = torch.device("cuda", local_rank)
    else:
        torch.cuda.set_device(0)
        runtime["device"] = torch.device("cuda", 0)

    return runtime


def cleanup_distributed_runtime() -> None:
    if not is_distributed_enabled():
        return
    import torch.distributed as dist

    dist.destroy_process_group()


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


def disable_model_dropout_modules(module: Any) -> None:
    import torch

    for child in module.modules():
        if isinstance(child, torch.nn.Dropout):
            child.p = 0.0


def maybe_enable_gradient_checkpointing(model: Any, args: argparse.Namespace) -> None:
    if not getattr(args, "gradient_checkpointing", False):
        return

    enable_input_require_grads = getattr(model.text_model, "enable_input_require_grads", None)
    if callable(enable_input_require_grads):
        enable_input_require_grads()

    gradient_checkpointing_enable = getattr(model.text_model, "gradient_checkpointing_enable", None)
    if callable(gradient_checkpointing_enable):
        try:
            gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            gradient_checkpointing_enable()

    config = getattr(model.text_model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False


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
        if getattr(args, "disable_model_dropout", False):
            disable_model_dropout_modules(model.text_model)
        maybe_enable_gradient_checkpointing(model, args)
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
        reasoning_prompt_style=args.reasoning_prompt_style,
        compact_interpro_limit=args.compact_interpro_limit,
        compact_ppi_limit=args.compact_ppi_limit,
        compact_go_speculation_limit=args.compact_go_speculation_limit,
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


def build_dataloader(
    dataset: Any,
    model: Any,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    *,
    distributed: bool = False,
    seed: int = 42,
) -> Any:
    from functools import partial

    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    from bioreason2.dataset.cafa5.collate import qwen_protein_collate_fn

    base_model = unwrap_model(model)
    collate_fn = partial(
        qwen_protein_collate_fn,
        processor=base_model.processor,
        max_length_text=base_model.max_length_text,
        max_length_protein=base_model.max_length_protein,
        return_answer_in_batch=False,
        inference_mode=True,
    )
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_distributed_world_size(),
            rank=get_distributed_rank(),
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
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


def expand_single_example_multimodal_cache(model: Any, cache: Optional[Dict[str, Any]], rollout_count: int) -> Optional[Dict[str, Any]]:
    if cache is None:
        return None
    expand = getattr(model, "expand_multimodal_cache", None)
    if not callable(expand):
        return None
    return expand(cache, rollout_count)


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
    *,
    multimodal_cache: Optional[Dict[str, Any]] = None,
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
        "multimodal_cache": multimodal_cache,
    }


def slice_rollout_group(
    rollout_group: Mapping[str, Any],
    start_idx: int,
    end_idx: int,
) -> Dict[str, Any]:
    total_rollouts = int(rollout_group["combined_input_ids"].shape[0])
    if start_idx < 0 or end_idx > total_rollouts or start_idx >= end_idx:
        raise ValueError(f"Invalid rollout slice [{start_idx}, {end_idx}) for {total_rollouts} rollouts")

    protein_sequences = list(rollout_group.get("protein_sequences") or [])
    batch_idx_map = list(rollout_group.get("batch_idx_map") or [])
    proteins_per_rollout = 0
    if total_rollouts > 0 and protein_sequences:
        if len(protein_sequences) % total_rollouts != 0:
            raise ValueError("protein_sequences length must divide evenly across rollout count")
        proteins_per_rollout = len(protein_sequences) // total_rollouts
    elif total_rollouts > 0 and batch_idx_map:
        if len(batch_idx_map) % total_rollouts != 0:
            raise ValueError("batch_idx_map length must divide evenly across rollout count")
        proteins_per_rollout = len(batch_idx_map) // total_rollouts

    protein_start = start_idx * proteins_per_rollout
    protein_end = end_idx * proteins_per_rollout
    sliced_sequences = protein_sequences[protein_start:protein_end]
    sliced_batch_idx_map: List[int] = []
    if proteins_per_rollout > 0:
        for local_idx in range(end_idx - start_idx):
            sliced_batch_idx_map.extend([local_idx] * proteins_per_rollout)

    structure_coords = rollout_group.get("structure_coords")
    if hasattr(structure_coords, "shape") and getattr(structure_coords, "shape", None):
        if int(structure_coords.shape[0]) == total_rollouts:
            structure_coords = structure_coords[start_idx:end_idx]
    elif isinstance(structure_coords, list) and len(structure_coords) == total_rollouts:
        structure_coords = structure_coords[start_idx:end_idx]

    multimodal_cache = rollout_group.get("multimodal_cache")
    sliced_multimodal_cache = None
    if isinstance(multimodal_cache, Mapping):
        sliced_multimodal_cache = {
            "batch_size": end_idx - start_idx,
            "protein_embeddings": None,
            "go_embeddings": None,
        }
        protein_embeddings = multimodal_cache.get("protein_embeddings")
        if isinstance(protein_embeddings, list):
            sliced_multimodal_cache["protein_embeddings"] = protein_embeddings[start_idx:end_idx]
        go_embeddings = multimodal_cache.get("go_embeddings")
        if isinstance(go_embeddings, list):
            sliced_multimodal_cache["go_embeddings"] = go_embeddings[start_idx:end_idx]

    return {
        "combined_input_ids": rollout_group["combined_input_ids"][start_idx:end_idx],
        "combined_attention_mask": rollout_group["combined_attention_mask"][start_idx:end_idx],
        "completion_attention": rollout_group["completion_attention"][start_idx:end_idx],
        "prompt_token_len": rollout_group["prompt_token_len"],
        "protein_sequences": sliced_sequences,
        "batch_idx_map": sliced_batch_idx_map,
        "structure_coords": structure_coords,
        "go_aspects": list(rollout_group["go_aspects"][start_idx:end_idx]),
        "multimodal_cache": sliced_multimodal_cache,
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


def maybe_build_example_multimodal_cache(
    model: Any,
    example: Mapping[str, Any],
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    if not can_cache_multimodal_prefix(args):
        return None
    base_model = unwrap_model(model)
    build_cache = getattr(base_model, "build_multimodal_cache", None)
    if not callable(build_cache):
        return None
    return build_cache(
        protein_sequences=list(example.get("protein_sequences") or []),
        batch_idx_map=list(example.get("batch_idx_map") or []),
        batch_size=int(example["input_ids"].shape[0]),
        structure_coords=example.get("structure_coords"),
        go_aspects=list(example.get("go_aspects") or []),
    )


def compute_completion_token_log_probs(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    prompt_len: int,
    protein_sequences: Sequence[str],
    batch_idx_map: Sequence[int],
    structure_coords: Any,
    go_aspects: Sequence[str],
    multimodal_cache: Optional[Dict[str, Any]] = None,
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
        multimodal_cache=multimodal_cache,
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


def compute_rollout_policy_statistics(
    model: Any,
    rollout_group: Mapping[str, Any],
    microbatch_size: int = 0,
) -> Tuple[Any, Any, Any]:
    import torch

    def _compute(subgroup: Mapping[str, Any]) -> Tuple[Any, Any, Any]:
        token_log_probs = compute_completion_token_log_probs(
            model,
            subgroup["combined_input_ids"],
            subgroup["combined_attention_mask"],
            subgroup["prompt_token_len"],
            subgroup["protein_sequences"],
            subgroup["batch_idx_map"],
            subgroup["structure_coords"],
            subgroup["go_aspects"],
            subgroup.get("multimodal_cache"),
        )
        token_mask = subgroup["completion_attention"].to(device=token_log_probs.device, dtype=token_log_probs.dtype)
        sequence_log_probs = (token_log_probs * token_mask).sum(dim=1)
        valid_counts = token_mask.sum(dim=1).clamp_min(1.0)
        return token_log_probs, sequence_log_probs, valid_counts

    total_rollouts = int(rollout_group["combined_input_ids"].shape[0])
    if microbatch_size <= 0 or total_rollouts <= microbatch_size:
        return _compute(rollout_group)

    token_chunks: List[Any] = []
    sequence_chunks: List[Any] = []
    count_chunks: List[Any] = []
    for start_idx in range(0, total_rollouts, microbatch_size):
        subgroup = slice_rollout_group(rollout_group, start_idx, min(total_rollouts, start_idx + microbatch_size))
        token_log_probs, sequence_log_probs, valid_counts = _compute(subgroup)
        token_chunks.append(token_log_probs)
        sequence_chunks.append(sequence_log_probs)
        count_chunks.append(valid_counts)

    return (
        torch.cat(token_chunks, dim=0),
        torch.cat(sequence_chunks, dim=0),
        torch.cat(count_chunks, dim=0),
    )


def compute_old_policy_sequence_log_probs(
    model: Any,
    example: Mapping[str, Any],
    completion_ids_list: Sequence[Any],
    pad_token_id: int,
    microbatch_size: int = 0,
    multimodal_cache: Optional[Dict[str, Any]] = None,
) -> List[float]:
    import torch

    rollout_group = build_rollout_group_inputs(
        example,
        completion_ids_list,
        pad_token_id,
        multimodal_cache=multimodal_cache,
    )
    valid_mask = rollout_group["completion_attention"].bool()
    if not torch.any(valid_mask.any(dim=1)):
        return [0.0 for _ in completion_ids_list]

    with torch.no_grad():
        _, sequence_log_probs, _ = compute_rollout_policy_statistics(
            model,
            rollout_group,
            microbatch_size=microbatch_size,
        )
    return [float(value.detach().item()) for value in sequence_log_probs]


def precompute_ref_policy_log_probs(
    ref_model: Any,
    rollout_group: Mapping[str, Any],
    *,
    microbatch_size: int = 0,
) -> Optional[Any]:
    import torch

    if ref_model is None:
        return None
    valid_mask = rollout_group["completion_attention"].bool()
    if not torch.any(valid_mask.any(dim=1)):
        return None
    with torch.no_grad():
        ref_log_probs, _, _ = compute_rollout_policy_statistics(
            ref_model,
            rollout_group,
            microbatch_size=microbatch_size,
        )
    return ref_log_probs


def compute_group_policy_losses_batched(
    model: Any,
    ref_model: Any,
    example: Mapping[str, Any],
    completion_ids_list: Sequence[Any],
    advantages: Sequence[float],
    old_sequence_log_probs: Sequence[float],
    pad_token_id: int,
    args: argparse.Namespace,
    rollout_group: Optional[Mapping[str, Any]] = None,
    cached_ref_log_probs: Optional[Any] = None,
) -> Tuple[Any, Dict[str, float], bool]:
    import torch

    if rollout_group is None:
        rollout_group = build_rollout_group_inputs(example, completion_ids_list, pad_token_id)
    valid_mask = rollout_group["completion_attention"].bool()
    nonempty_rollouts = valid_mask.any(dim=1)
    if not torch.any(nonempty_rollouts):
        return torch.tensor(0.0, device=example["input_ids"].device), {}, False

    current_log_probs, current_sequence_log_probs, valid_counts = compute_rollout_policy_statistics(
        model,
        rollout_group,
        microbatch_size=args.rollout_logprob_microbatch_size,
    )

    with torch.no_grad():
        if cached_ref_log_probs is not None:
            ref_log_probs = cached_ref_log_probs.to(device=current_log_probs.device, dtype=current_log_probs.dtype)
        elif ref_model is not None:
            ref_log_probs, _, _ = compute_rollout_policy_statistics(
                ref_model,
                rollout_group,
                microbatch_size=args.rollout_logprob_microbatch_size,
            )
        else:
            ref_log_probs = torch.zeros_like(current_log_probs)

    token_mask = rollout_group["completion_attention"].to(
        device=current_log_probs.device,
        dtype=current_log_probs.dtype,
    )
    advantage_tensor = torch.tensor(
        list(advantages),
        device=current_log_probs.device,
        dtype=current_log_probs.dtype,
    )
    old_sequence_tensor = torch.tensor(
        list(old_sequence_log_probs),
        device=current_log_probs.device,
        dtype=current_log_probs.dtype,
    )
    log_ratio = current_sequence_log_probs - old_sequence_tensor
    ratio = torch.exp(log_ratio)
    if args.importance_sampling_cap > 0:
        ratio = ratio.clamp(max=float(args.importance_sampling_cap))
    clipped_ratio = ratio.clamp(
        min=1.0 - float(args.clip_epsilon_low),
        max=1.0 + float(args.clip_epsilon_high),
    )
    surrogate = torch.minimum(ratio * advantage_tensor, clipped_ratio * advantage_tensor)
    kl_term = torch.exp(ref_log_probs - current_log_probs) - (ref_log_probs - current_log_probs) - 1.0
    sequence_kl = (kl_term * token_mask).sum(dim=1) / valid_counts
    per_rollout_loss = -(surrogate - float(args.kl_beta) * sequence_kl) / max(float(args.max_new_tokens), 1.0)
    nonempty_losses = per_rollout_loss[nonempty_rollouts]
    metrics = {
        "kl_mean": float(sequence_kl[nonempty_rollouts].mean().detach().item()),
        "ratio_mean": float(ratio[nonempty_rollouts].mean().detach().item()),
        "ratio_max": float(ratio[nonempty_rollouts].max().detach().item()),
        "valid_rollouts": float(nonempty_rollouts.sum().item()),
    }
    return nonempty_losses.mean(), metrics, True


def compute_group_policy_losses_sequential(
    model: Any,
    ref_model: Any,
    example: Mapping[str, Any],
    completion_ids_list: Sequence[Any],
    advantages: Sequence[float],
    old_sequence_log_probs: Sequence[float],
    args: argparse.Namespace,
    multimodal_cache: Optional[Dict[str, Any]] = None,
    cached_ref_log_probs: Optional[Any] = None,
) -> Tuple[Any, Dict[str, float], bool]:
    import torch

    losses: List[Any] = []
    kl_values: List[float] = []
    ratio_values: List[float] = []
    trainable = False
    for rollout_idx, (completion_ids, advantage, old_sequence_log_prob) in enumerate(
        zip(completion_ids_list, advantages, old_sequence_log_probs)
    ):
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
            multimodal_cache,
        )
        with torch.no_grad():
            if cached_ref_log_probs is not None:
                ref_log_probs = cached_ref_log_probs[rollout_idx : rollout_idx + 1].to(
                    device=current_log_probs.device,
                    dtype=current_log_probs.dtype,
                )
            elif ref_model is not None:
                ref_log_probs = compute_completion_token_log_probs(
                    ref_model,
                    combined_ids,
                    combined_mask,
                    prompt_token_len,
                    example["protein_sequences"],
                    example["batch_idx_map"],
                    example["structure_coords"],
                    example["go_aspects"],
                    multimodal_cache,
                )
            else:
                ref_log_probs = torch.zeros_like(current_log_probs)

        sequence_log_prob = current_log_probs.sum(dim=1).squeeze(0)
        ratio = torch.exp(sequence_log_prob - float(old_sequence_log_prob))
        if args.importance_sampling_cap > 0:
            ratio = ratio.clamp(max=float(args.importance_sampling_cap))
        clipped_ratio = ratio.clamp(
            min=1.0 - float(args.clip_epsilon_low),
            max=1.0 + float(args.clip_epsilon_high),
        )
        surrogate = torch.minimum(ratio * float(advantage), clipped_ratio * float(advantage))
        kl_term = torch.exp(ref_log_probs - current_log_probs) - (ref_log_probs - current_log_probs) - 1.0
        sequence_kl = kl_term.mean()
        losses.append(-(surrogate - float(args.kl_beta) * sequence_kl) / max(float(args.max_new_tokens), 1.0))
        kl_values.append(float(kl_term.mean().detach().item()))
        ratio_values.append(float(ratio.detach().item()))

    if not losses:
        return torch.tensor(0.0, device=example["input_ids"].device), {}, False
    metrics = {
        "kl_mean": sum(kl_values) / len(kl_values),
        "ratio_mean": sum(ratio_values) / len(ratio_values),
        "ratio_max": max(ratio_values),
        "valid_rollouts": float(len(losses)),
    }
    return torch.stack(losses).mean(), metrics, trainable


def generate_rollouts_sequential(
    model: Any,
    example: Mapping[str, Any],
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    global_step: int,
    epoch_idx: int,
    prompt_len: int,
    example_multimodal_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Any], List[str]]:
    completion_ids_list: List[Any] = []
    completions: List[str] = []
    generation_kwargs = build_generation_kwargs(args, tokenizer, for_eval=False)
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
            multimodal_cache=example_multimodal_cache,
            **generation_kwargs,
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
    example_multimodal_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Any], List[str]]:
    import torch

    prompt_len = example["input_ids"].shape[1]
    rollout_batch = expand_example_for_rollouts(example, args.num_generations)
    rollout_multimodal_cache = expand_single_example_multimodal_cache(model, example_multimodal_cache, args.num_generations)
    generation_kwargs = build_generation_kwargs(args, tokenizer, for_eval=False)
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
            multimodal_cache=rollout_multimodal_cache,
            **generation_kwargs,
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
        example_multimodal_cache=example_multimodal_cache,
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

    generation_kwargs = build_generation_kwargs(args, tokenizer, for_eval=True)
    generated_ids = model.generate(
        input_ids=example["input_ids"],
        attention_mask=example["attention_mask"],
        protein_sequences=example["protein_sequences"],
        batch_idx_map=example["batch_idx_map"],
        structure_coords=example["structure_coords"],
        go_aspects=example["go_aspects"],
        **generation_kwargs,
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
    generation_kwargs = build_generation_kwargs(args, tokenizer, for_eval=True)

    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        protein_sequences=protein_sequences,
        batch_idx_map=batch_idx_map,
        structure_coords=structure_coords,
        go_aspects=go_aspects,
        **generation_kwargs,
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


def compute_reward_diagnostics(
    completions: Sequence[str],
    sample_meta: Mapping[str, Any],
) -> Dict[str, List[float]]:
    registry = build_reward_registry()
    diagnostics: Dict[str, List[float]] = {}
    for reward_name in DIAGNOSTIC_REWARD_NAMES:
        reward_fn = registry.get(reward_name)
        if reward_fn is None:
            continue
        diagnostics[reward_name] = [float(reward_fn(completion, sample_meta)) for completion in completions]
    return diagnostics


def save_raw_checkpoint(model: Any, checkpoint_dir: Path, step: int, args: argparse.Namespace) -> Path:
    import torch

    model = unwrap_model(model)
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
        "loss_type": args.loss_type,
        "steps_per_generation": args.steps_per_generation,
        "num_iterations": args.num_iterations,
        "reward_funcs": parse_csv_items(args.reward_funcs),
        "reward_weights": parse_reward_weights(args.reward_weights, len(parse_csv_items(args.reward_funcs))),
        "reward_scaling": args.reward_scaling,
        "importance_sampling_level": args.importance_sampling_level,
        "importance_sampling_cap": args.importance_sampling_cap,
        "clip_epsilon_low": args.clip_epsilon_low,
        "clip_epsilon_high": args.clip_epsilon_high,
        "ia_file_path": args.ia_file_path,
        "kl_beta": args.kl_beta,
    }
    with open(step_dir / "training_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return step_dir


def export_hf_model(model: Any, save_dir: Path) -> None:
    import shutil
    import torch

    model = unwrap_model(model)
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


def attach_global_step(payload: Mapping[str, Any], global_step: int) -> Dict[str, Any]:
    enriched = dict(payload)
    enriched["global_step"] = float(global_step)
    return enriched


def maybe_bootstrap_wandb_history(
    run: Any,
    *,
    reward_names: Sequence[str],
) -> bool:
    if run is None:
        return False

    define_metric = getattr(run, "define_metric", None)
    if callable(define_metric):
        define_metric("global_step")
        for metric_name in WANDB_BOOTSTRAP_METRICS:
            define_metric(metric_name, step_metric="global_step")
        for reward_name in reward_names:
            define_metric(f"reward_component/{reward_name}", step_metric="global_step")
        for reward_name in DIAGNOSTIC_REWARD_NAMES:
            define_metric(f"diagnostic/{reward_name}", step_metric="global_step")

    bootstrap_payload: Dict[str, Any] = {
        "global_step": 0.0,
        "status/initialized": 1.0,
    }
    for metric_name in WANDB_BOOTSTRAP_METRICS:
        bootstrap_payload[metric_name] = 0.0
    for reward_name in reward_names:
        bootstrap_payload[f"reward_component/{reward_name}"] = 0.0
    for reward_name in DIAGNOSTIC_REWARD_NAMES:
        bootstrap_payload[f"diagnostic/{reward_name}"] = 0.0
    run.log(bootstrap_payload, step=0)
    return True


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
        "full_group_budget": max(int(getattr(args, "weave_trace_full_group_count", 0)), 0),
        "full_group_rollouts": max(int(getattr(args, "weave_trace_full_rollouts_per_group", 0)), 0),
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
        "has_closed_reasoning": bool(meta["has_closed_reasoning"]),
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
        wandb_run.log(attach_global_step(rotating_metrics, global_step), step=global_step)
        print(f"Rotating RL validation metrics logged at global_step={global_step}.")
    return rotating_metrics


def build_zero_connected_loss(model: Any) -> Any:
    import torch

    reference_param = None
    for param in model.parameters():
        if param.requires_grad:
            reference_param = param
            break
    if reference_param is None:
        device = next(model.parameters()).device
        return torch.zeros((), device=device, dtype=torch.float32)
    return reference_param.reshape(-1)[0] * 0.0


def aggregate_mean_metric(local_sum: float, local_count: float, device: Any) -> float:
    global_sum = distributed_sum_scalar(local_sum, device)
    global_count = distributed_sum_scalar(local_count, device)
    if global_count <= 0.0:
        return 0.0
    return global_sum / global_count


def train(args: argparse.Namespace) -> None:
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP
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
    distributed = init_distributed_runtime()
    device = distributed["device"]
    rank = distributed["rank"]
    world_size = distributed["world_size"]
    is_main = rank == 0

    set_seed(args.seed + rank)
    configure_reward_context(args)
    torch.set_float32_matmul_precision("high")
    batch_semantics = build_batch_semantics(args, world_size)

    run_name = args.run_name or f"train-rl-{int(time.time())}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_checkpoint_dir = output_dir / "raw_checkpoints"
    raw_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tracking_config = build_training_tracking_config(args=args, run_name=run_name, job_type="train_rl")
    tracking_config["model_artifact"] = args.checkpoint_artifact_name or args.model_artifact
    tracking_config.update(batch_semantics)
    tracking_config["distributed_enabled"] = distributed["enabled"]
    tracking_config["distributed_strategy"] = "single_node_ddp" if distributed["enabled"] else "single_gpu"
    tracking_config["multimodal_cache_enabled"] = can_cache_multimodal_prefix(args)
    tracking_config["ref_logprob_cache_enabled"] = bool(args.kl_beta > 0)

    wandb_run = None
    weave_trace_state = None
    final_checkpoint_dir: Optional[Path] = None

    try:
        base_model = instantiate_model(args, trainable=True).to(device)
        if args.resume_from_raw_checkpoint:
            checkpoint = torch.load(args.resume_from_raw_checkpoint, map_location="cpu", weights_only=False)
            base_model.load_state_dict(checkpoint, strict=False)
            if is_main:
                print(f"Resumed RL model weights from {args.resume_from_raw_checkpoint}")

        model = base_model
        if distributed["enabled"]:
            model = DDP(
                base_model,
                device_ids=[distributed["local_rank"]],
                output_device=distributed["local_rank"],
                find_unused_parameters=True,
            )
        ref_model = instantiate_model(args, trainable=False).to(device) if args.kl_beta > 0 else None

        train_dataset, fixed_val_dataset, full_val_dataset = load_rl_datasets(args)
        train_loader = build_dataloader(
            train_dataset,
            model,
            int(batch_semantics["per_device_train_batch_size"]),
            args.num_workers,
            shuffle=True,
            distributed=distributed["enabled"],
            seed=args.seed,
        )
        val_loader = None
        if is_main:
            val_loader = build_dataloader(
                fixed_val_dataset,
                model,
                int(batch_semantics["per_device_eval_batch_size"]),
                args.num_workers,
                shuffle=False,
                distributed=False,
                seed=args.seed,
            )

        tokenizer = base_model.text_tokenizer
        multimodal_cache_enabled = can_cache_multimodal_prefix(args)
        reward_names = parse_csv_items(args.reward_funcs)
        reward_weights = parse_reward_weights(args.reward_weights, len(reward_names))
        resolved_ia_file_path = require_training_ia_file(args, reward_names)
        if resolved_ia_file_path and is_main:
            print(f"Using IA-weighted RL reward with IA file: {resolved_ia_file_path}")

        optimizer = AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )
        total_update_passes = max(args.max_steps, 1) * max(args.steps_per_generation, 1) * max(args.num_iterations, 1)
        optimizer_step_budget = max(math.ceil(total_update_passes / max(args.gradient_accumulation_steps, 1)), 1)
        warmup_steps = int(optimizer_step_budget * max(args.warmup_ratio, 0.0))
        if args.lr_scheduler_type == "cosine":
            from transformers import get_cosine_schedule_with_warmup

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=optimizer_step_budget,
            )
        else:
            scheduler = None

        if is_main:
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
                        **batch_semantics,
                        "distributed_enabled": distributed["enabled"],
                        "distributed_world_size": world_size,
                        "distributed_strategy": "single_node_ddp" if distributed["enabled"] else "single_gpu",
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
                maybe_bootstrap_wandb_history(
                    wandb_run,
                    reward_names=reward_names,
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
        optimizer_substep = 0
        optimizer.zero_grad(set_to_none=True)

        def maybe_run_eval_and_save(current_step: int) -> None:
            nonlocal best_val_reward, last_eval_step

            should_eval = args.eval_every_n_steps > 0 and current_step % args.eval_every_n_steps == 0
            should_save = args.save_every_n_steps > 0 and current_step % args.save_every_n_steps == 0
            if not should_eval and not should_save:
                return

            distributed_barrier()
            if should_eval and is_main and val_loader is not None:
                print(f"Starting RL validation eval at global_step={current_step}.")
                val_metrics = evaluate_policy(
                    model=base_model,
                    ref_model=ref_model,
                    dataloader=val_loader,
                    tokenizer=tokenizer,
                    args=args,
                    reward_names=reward_names,
                    reward_weights=reward_weights,
                    device=device,
                    trace_state=weave_trace_state,
                    global_step=current_step,
                    trace_split_name="validation_fixed",
                )
                if wandb_run is not None:
                    wandb_run.log(attach_global_step(val_metrics, current_step), step=current_step)
                    print(f"RL validation metrics logged at global_step={current_step}.")
                last_eval_step = current_step
                if val_metrics["eval_reward"] > best_val_reward:
                    best_val_reward = val_metrics["eval_reward"]
                    save_raw_checkpoint(base_model, raw_checkpoint_dir / "best", current_step, args)
                maybe_run_rotating_validation_eval(
                    model=base_model,
                    ref_model=ref_model,
                    full_val_dataset=full_val_dataset,
                    tokenizer=tokenizer,
                    args=args,
                    reward_names=reward_names,
                    reward_weights=reward_weights,
                    device=device,
                    trace_state=weave_trace_state,
                    global_step=current_step,
                    wandb_run=wandb_run,
                )
            distributed_barrier()

            if should_save and is_main:
                save_raw_checkpoint(base_model, raw_checkpoint_dir, current_step, args)
            distributed_barrier()

        for epoch_idx in range(args.max_epochs):
            if global_step >= args.max_steps:
                break
            sampler = getattr(train_loader, "sampler", None)
            if distributed["enabled"] and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch_idx)

            for batch in train_loader:
                model.train()
                if is_main:
                    print(f"Starting RL train batch at global_step={global_step}, epoch={epoch_idx}.")
                local_batch_size = batch["input_ids"].shape[0]
                reward_totals: List[float] = []
                reward_component_scores: Dict[str, List[float]] = {reward_name: [] for reward_name in reward_names}
                diagnostic_component_scores: Dict[str, List[float]] = {
                    reward_name: [] for reward_name in DIAGNOSTIC_REWARD_NAMES
                }
                completion_lengths: List[int] = []
                kl_values: List[float] = []
                ratio_means: List[float] = []
                ratio_maxes: List[float] = []
                local_trainable_group_count = 0
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                grouped_rewards: List[List[float]] = []
                rollout_records: List[Dict[str, Any]] = []

                for example_idx in range(local_batch_size):
                    example = extract_example_from_batch(batch, example_idx, device)
                    example_multimodal_cache = (
                        maybe_build_example_multimodal_cache(base_model, example, args)
                        if multimodal_cache_enabled
                        else None
                    )
                    rollout_multimodal_cache = (
                        expand_single_example_multimodal_cache(base_model, example_multimodal_cache, args.num_generations)
                        if example_multimodal_cache is not None
                        else None
                    )

                    base_model.eval()
                    completion_ids_list, completions = generate_rollouts_for_example(
                        base_model,
                        example,
                        tokenizer,
                        args,
                        global_step=global_step,
                        epoch_idx=epoch_idx,
                        example_multimodal_cache=example_multimodal_cache,
                    )
                    old_sequence_log_probs = compute_old_policy_sequence_log_probs(
                        base_model,
                        example,
                        completion_ids_list,
                        pad_token_id,
                        microbatch_size=args.rollout_logprob_microbatch_size,
                        multimodal_cache=rollout_multimodal_cache,
                    )
                    rollout_group = build_rollout_group_inputs(
                        example,
                        completion_ids_list,
                        pad_token_id,
                        multimodal_cache=rollout_multimodal_cache,
                    )
                    cached_ref_log_probs = precompute_ref_policy_log_probs(
                        ref_model,
                        rollout_group,
                        microbatch_size=args.rollout_logprob_microbatch_size,
                    )
                    model.train()
                    completion_lengths.extend(int(completion_ids.numel()) for completion_ids in completion_ids_list)

                    total_rewards, reward_components = compute_group_rewards(
                        completions,
                        example["sample_meta"],
                        reward_names,
                        reward_weights,
                    )
                    diagnostic_components = compute_reward_diagnostics(completions, example["sample_meta"])
                    grouped_rewards.append(total_rewards)
                    rollout_records.append(
                        {
                            "example": example,
                            "completion_ids_list": completion_ids_list,
                            "completions": completions,
                            "total_rewards": total_rewards,
                            "reward_components": reward_components,
                            "diagnostic_components": diagnostic_components,
                            "old_sequence_log_probs": old_sequence_log_probs,
                            "rollout_group": rollout_group,
                            "example_multimodal_cache": example_multimodal_cache,
                            "rollout_multimodal_cache": rollout_multimodal_cache,
                            "cached_ref_log_probs": cached_ref_log_probs,
                        }
                    )
                    for reward_name, scores in reward_components.items():
                        reward_component_scores.setdefault(reward_name, []).extend(scores)
                    for reward_name, scores in diagnostic_components.items():
                        diagnostic_component_scores.setdefault(reward_name, []).extend(scores)
                    reward_totals.extend(total_rewards)

                grouped_advantages, global_reward_std = compute_batch_relative_advantages(
                    grouped_rewards,
                    epsilon_std=args.advantage_epsilon_std,
                    reward_scaling=args.reward_scaling,
                    distributed_device=device,
                )
                for record, advantages in zip(rollout_records, grouped_advantages):
                    record["advantages"] = advantages
                    if any(completion_ids.numel() > 0 for completion_ids in record["completion_ids_list"]):
                        local_trainable_group_count += 1
                    if not record["completions"]:
                        continue
                    trace_rollout_count = 1
                    if weave_trace_state and weave_trace_state.get("full_group_budget", 0) > 0:
                        trace_rollout_count = min(
                            len(record["completions"]),
                            max(int(weave_trace_state.get("full_group_rollouts", 0)), 1),
                        )
                        weave_trace_state["full_group_budget"] -= 1
                    for rollout_idx in range(trace_rollout_count):
                        maybe_trace_generation(
                            weave_trace_state,
                            split="train",
                            global_step=global_step,
                            sample_meta=record["example"]["sample_meta"],
                            completion=record["completions"][rollout_idx],
                            total_reward=record["total_rewards"][rollout_idx] if record["total_rewards"] else 0.0,
                            advantage=advantages[rollout_idx] if advantages else None,
                            reward_components={
                                reward_name: record["reward_components"][reward_name][rollout_idx]
                                for reward_name in record["reward_components"]
                            },
                        )

                global_trainable_group_count = int(round(distributed_sum_scalar(float(local_trainable_group_count), device)))
                local_groups_submitted = float(local_batch_size)
                local_trajectories = float(len(reward_totals))
                local_tokens = float(sum(completion_lengths))
                global_groups_submitted_rank_sum = distributed_sum_scalar(local_groups_submitted, device)
                global_trajectories_rank_sum = distributed_sum_scalar(local_trajectories, device)
                global_datums_rank_sum = distributed_sum_scalar(float(local_batch_size), device)
                global_tokens_rank_sum = distributed_sum_scalar(local_tokens, device)
                global_groups_target = float(batch_semantics["global_unique_proteins_target"])
                global_trajectories_target = float(batch_semantics["global_num_trajectories_per_step"])

                if global_trainable_group_count == 0:
                    global_step += 1
                    skipped_payload = {
                        "loss_learning_rate": optimizer.param_groups[0]["lr"],
                        "data_step_num_groups_submitted": global_groups_target,
                        "data_step_num_groups_submitted_rank_sum": global_groups_submitted_rank_sum,
                        "data_step_num_groups_trainable": float(global_trainable_group_count),
                        "data_step_num_trajectories": global_trajectories_target,
                        "data_step_num_trajectories_rank_sum": global_trajectories_rank_sum,
                        "data_step_num_datums": global_groups_target,
                        "data_step_num_datums_rank_sum": global_datums_rank_sum,
                        "data_step_trainer_tokens": global_tokens_rank_sum,
                        "reward": aggregate_mean_metric(sum(reward_totals), float(len(reward_totals)), device),
                        "reward_std_dev": float(global_reward_std),
                        "train_skipped_update": 1.0,
                    }
                    for reward_name, scores in reward_component_scores.items():
                        if scores:
                            skipped_payload[f"reward_component/{reward_name}"] = aggregate_mean_metric(
                                sum(scores),
                                float(len(scores)),
                                device,
                            )
                    for reward_name, scores in diagnostic_component_scores.items():
                        if scores:
                            skipped_payload[f"diagnostic/{reward_name}"] = aggregate_mean_metric(
                                sum(scores),
                                float(len(scores)),
                                device,
                            )
                    if wandb_run is not None and is_main:
                        print(f"Logging RL skipped-update metrics at global_step={global_step}.")
                        wandb_run.log(attach_global_step(skipped_payload, global_step), step=global_step)
                        print(f"RL skipped-update metrics logged at global_step={global_step}.")

                    maybe_run_eval_and_save(global_step)
                    if global_step >= args.max_steps:
                        break
                    continue

                grad_norm_value = 0.0
                optimizer.zero_grad(set_to_none=True)
                local_update_loss_sums: List[float] = []
                local_update_loss_counts: List[float] = []
                total_update_passes = max(args.steps_per_generation, 1) * max(args.num_iterations, 1)
                for update_idx in range(total_update_passes):
                    optimizer_substep += 1
                    should_step = (
                        optimizer_substep % max(args.gradient_accumulation_steps, 1) == 0
                        or update_idx == total_update_passes - 1
                    )
                    sync_context = nullcontext()
                    if distributed["enabled"] and not should_step:
                        sync_context = model.no_sync()
                    with sync_context:
                        local_backward_terms = []
                        update_loss_accumulator = 0.0
                        update_trainable_groups = 0
                        for record in rollout_records:
                            try:
                                group_loss, group_metrics, has_trainable_rollout = compute_group_policy_losses_batched(
                                    model,
                                    ref_model,
                                    record["example"],
                                    record["completion_ids_list"],
                                    record["advantages"],
                                    record["old_sequence_log_probs"],
                                    pad_token_id,
                                    args,
                                    rollout_group=record.get("rollout_group"),
                                    cached_ref_log_probs=record.get("cached_ref_log_probs"),
                                )
                            except torch.cuda.OutOfMemoryError:
                                if is_main:
                                    print(
                                        "CUDA Out of Memory during batched RL loss computation; "
                                        "falling back to sequential rollout loss computation for this prompt."
                                    )
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                group_loss, group_metrics, has_trainable_rollout = compute_group_policy_losses_sequential(
                                    model,
                                    ref_model,
                                    record["example"],
                                    record["completion_ids_list"],
                                    record["advantages"],
                                    record["old_sequence_log_probs"],
                                    args,
                                    multimodal_cache=record.get("example_multimodal_cache"),
                                    cached_ref_log_probs=record.get("cached_ref_log_probs"),
                                )

                            if not has_trainable_rollout:
                                continue

                            local_backward_terms.append(group_loss / max(float(global_trainable_group_count), 1.0))
                            update_loss_accumulator += float(group_loss.detach().item())
                            update_trainable_groups += 1
                            if group_metrics:
                                if "kl_mean" in group_metrics:
                                    kl_values.append(float(group_metrics["kl_mean"]))
                                if "ratio_mean" in group_metrics:
                                    ratio_means.append(float(group_metrics["ratio_mean"]))
                                if "ratio_max" in group_metrics:
                                    ratio_maxes.append(float(group_metrics["ratio_max"]))

                        backward_loss = (
                            torch.stack(local_backward_terms).sum()
                            if local_backward_terms
                            else build_zero_connected_loss(base_model)
                        )
                        backward_loss.backward()

                    if should_step:
                        grad_norm = clip_grad_norm_(
                            [parameter for parameter in model.parameters() if parameter.requires_grad],
                            args.max_grad_norm,
                        )
                        grad_norm_value = float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    local_update_loss_sums.append(update_loss_accumulator)
                    local_update_loss_counts.append(float(update_trainable_groups))

                global_step += 1
                loss_train_sum = distributed_sum_scalar(sum(local_update_loss_sums), device)
                loss_train_count = distributed_sum_scalar(sum(local_update_loss_counts), device)
                log_payload = {
                    "loss_train": (loss_train_sum / loss_train_count) if loss_train_count > 0 else 0.0,
                    "reward": aggregate_mean_metric(sum(reward_totals), float(len(reward_totals)), device),
                    "reward_std_dev": float(global_reward_std),
                    "loss_kl_div": aggregate_mean_metric(sum(kl_values), float(len(kl_values)), device),
                    "loss_policy_ratio_mean": aggregate_mean_metric(sum(ratio_means), float(len(ratio_means)), device),
                    "loss_policy_ratio_max": distributed_max_scalar(max(ratio_maxes) if ratio_maxes else 0.0, device),
                    "loss_learning_rate": optimizer.param_groups[0]["lr"],
                    "loss_grad_norm": grad_norm_value,
                    "data_step_num_groups_submitted": global_groups_target,
                    "data_step_num_groups_submitted_rank_sum": global_groups_submitted_rank_sum,
                    "data_step_num_groups_trainable": float(global_trainable_group_count),
                    "data_step_num_trajectories": global_trajectories_target,
                    "data_step_num_trajectories_rank_sum": global_trajectories_rank_sum,
                    "data_step_num_datums": global_groups_target,
                    "data_step_num_datums_rank_sum": global_datums_rank_sum,
                    "data_step_trainer_tokens": global_tokens_rank_sum,
                    "data_step_num_update_passes": float(total_update_passes),
                    "train_skipped_update": 0.0,
                }
                for reward_name, scores in reward_component_scores.items():
                    if not scores:
                        continue
                    log_payload[f"reward_component/{reward_name}"] = aggregate_mean_metric(
                        sum(scores),
                        float(len(scores)),
                        device,
                    )
                for reward_name, scores in diagnostic_component_scores.items():
                    if not scores:
                        continue
                    log_payload[f"diagnostic/{reward_name}"] = aggregate_mean_metric(
                        sum(scores),
                        float(len(scores)),
                        device,
                    )
                if wandb_run is not None and is_main:
                    print(f"Logging RL train metrics at global_step={global_step}.")
                    wandb_run.log(attach_global_step(log_payload, global_step), step=global_step)
                    print(f"RL train metrics logged at global_step={global_step}.")

                maybe_run_eval_and_save(global_step)
                if global_step >= args.max_steps:
                    break

        distributed_barrier()
        if global_step > 0 and last_eval_step != global_step and is_main and val_loader is not None:
            print(f"Starting RL final validation eval at global_step={global_step}.")
            final_val_metrics = evaluate_policy(
                model=base_model,
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
                wandb_run.log(attach_global_step(final_val_metrics, global_step), step=global_step)
                print(f"RL final validation metrics logged at global_step={global_step}.")
        distributed_barrier()

        if is_main:
            final_checkpoint_dir = save_raw_checkpoint(base_model, raw_checkpoint_dir, global_step, args)
            export_hf_model(base_model, output_dir)

        distributed_barrier()

        if wandb_run is not None and is_main:
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
                        "last_raw_checkpoint": str(final_checkpoint_dir) if final_checkpoint_dir else "",
                        "weave_trace_project": weave_trace_state["project"] if weave_trace_state else None,
                        "weave_trace_count": weave_trace_state["logged"] if weave_trace_state else 0,
                    },
                )
    finally:
        if wandb_run is not None and is_main:
            if weave_trace_state and weave_trace_state.get("client") is not None:
                flush = getattr(weave_trace_state["client"], "flush", None)
                if callable(flush):
                    flush()
            wandb_run.finish()
        cleanup_distributed_runtime()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Specification-first DR-GRPO trainer for BioReason-Pro.

This is a zero-based rewrite of the RL entry point. It intentionally does not
reuse the previous trainer's DDP/CoreWeave launcher flow or its
`model.generate()`-centric control path. Instead it follows the paper-facing
specification directly:

- rollout generation is owned by a separate vLLM-backed rollout worker
- scoring / optimization is owned by a DeepSpeed-backed policy engine
- reward extraction accepts the repo's structured final answer blocks
- the canonical paper batch is 8 proteins x 24 rollouts = 192 trajectories
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - handled at runtime
    torch = None
    F = None

try:
    import weave
except ImportError:  # pragma: no cover - optional dependency
    weave = None


GO_ID_PATTERN = re.compile(r"GO:\d{7}")
FINAL_ANSWER_OPEN_TAG = "<|FINAL_ANSWER|>"
FINAL_ANSWER_CLOSE_TAG = "<|/FINAL_ANSWER|>"
ALT_FINAL_ANSWER_CLOSE_TAG = "</|FINAL_ANSWER|>"
ROLLOUT_STOP_MARKERS = [
    FINAL_ANSWER_CLOSE_TAG,
    ALT_FINAL_ANSWER_CLOSE_TAG,
    "<|GO_SUMMARY_END|>",
    "<|im_end|>",
    "<|endoftext|>",
    "</think>",
]
FINAL_ANSWER_PATTERN = re.compile(
    r"<\|FINAL_ANSWER\|>\s*(.*?)\s*(?:<\|/FINAL_ANSWER\|>|</\|FINAL_ANSWER\|>)",
    re.DOTALL,
)
GO_SUMMARY_PATTERN = re.compile(
    r"<\|GO_SUMMARY_START\|>\s*(.*?)\s*<\|GO_SUMMARY_END\|>",
    re.DOTALL,
)
ROLLOUT_TRACE_SAMPLE_META_KEYS = ("protein_id", "split")
PAPER_TARGET_QUERIES_PER_STEP = 8
PAPER_TARGET_ROLLOUTS_PER_QUERY = 24
PAPER_TARGET_TOTAL_TRAJECTORIES = 192
PAPER_TARGET_WORLD_SIZE = 8
PAPER_TARGET_MAX_NEW_TOKENS = 10_000
PAPER_TARGET_STEPS_PER_GENERATION = 2
PAPER_TARGET_NUM_ITERATIONS = 1
PAPER_TARGET_RUNTIME_STACK = "deepspeed_vllm_colocate"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value if item not in (None, ""))
    return str(value)


def normalize_path_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "/".join(str(item) for item in value if item not in (None, ""))
    return normalize_text(value)


def normalize_structured_response_text(text: Any) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    return normalized.replace(ALT_FINAL_ANSWER_CLOSE_TAG, FINAL_ANSWER_CLOSE_TAG)


def extract_final_answer_fallback_content(text: Any) -> Optional[str]:
    normalized = normalize_structured_response_text(text)
    start_index = normalized.find(FINAL_ANSWER_OPEN_TAG)
    if start_index < 0:
        return None
    content = normalized[start_index + len(FINAL_ANSWER_OPEN_TAG) :]
    stop_candidates: List[int] = []
    for marker in ("<|REASONING|>", "<|GO_SUMMARY_START|>", "<|im_end|>", "<|endoftext|>", "</think>"):
        marker_index = content.find(marker)
        if marker_index >= 0:
            stop_candidates.append(marker_index)
    if stop_candidates:
        content = content[: min(stop_candidates)]
    content = content.strip()
    return content or None


def parse_wandb_artifact_project(value: Any) -> Tuple[str, str]:
    text = normalize_text(value).strip()
    if not text or ":" not in text or text.count("/") < 2:
        return "", ""
    parts = text.split("/", 2)
    entity = parts[0].strip()
    project = parts[1].strip()
    if not entity or not project:
        return "", ""
    return entity, project


def parse_wandb_project_ref(value: Any) -> Tuple[str, str]:
    text = normalize_text(value).strip()
    if not text or text.count("/") != 1:
        return "", ""
    entity, project = [item.strip() for item in text.split("/", 1)]
    if not entity or not project:
        return "", ""
    return entity, project


def resolve_wandb_identity(args: Any, config: Optional[Mapping[str, Any]] = None) -> Tuple[str, str]:
    explicit_entity = normalize_text(getattr(args, "wandb_entity", None)).strip()
    explicit_project = normalize_text(getattr(args, "wandb_project", None)).strip()
    if explicit_entity and explicit_project:
        return explicit_entity, explicit_project

    candidate_sources: List[Any] = [
        getattr(args, "weave_project", None),
        getattr(args, "base_checkpoint", None),
        getattr(args, "dataset_artifact", None),
        getattr(args, "temporal_split_artifact", None),
    ]
    if config is not None:
        candidate_sources.extend(
            [
                config.get("weave_project"),
                config.get("base_checkpoint"),
                config.get("dataset_artifact"),
                config.get("temporal_split_artifact"),
            ]
        )

    inferred_entity = explicit_entity
    inferred_project = explicit_project
    for raw_value in candidate_sources:
        entity, project = parse_wandb_project_ref(raw_value)
        if not entity or not project:
            entity, project = parse_wandb_artifact_project(raw_value)
        if entity and not inferred_entity:
            inferred_entity = entity
        if project and not inferred_project:
            inferred_project = project
        if inferred_entity and inferred_project:
            break
    return inferred_entity, inferred_project


def build_wandb_run_url(entity: str, project: str, run_id: str) -> str:
    if not entity or not project or not run_id:
        return ""
    return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"


def paper_runtime_deviation_summary(
    algorithm: "AlgorithmSpec",
    runtime_spec: "RuntimeSpec",
    runtime: "DistributedRuntime",
) -> Dict[str, float]:
    return {
        "paper_target_queries_per_step": float(PAPER_TARGET_QUERIES_PER_STEP),
        "paper_target_rollouts_per_query": float(PAPER_TARGET_ROLLOUTS_PER_QUERY),
        "paper_target_total_trajectories_per_step": float(PAPER_TARGET_TOTAL_TRAJECTORIES),
        "paper_target_world_size": float(PAPER_TARGET_WORLD_SIZE),
        "paper_target_max_new_tokens": float(PAPER_TARGET_MAX_NEW_TOKENS),
        "paper_target_steps_per_generation": float(PAPER_TARGET_STEPS_PER_GENERATION),
        "paper_target_num_iterations": float(PAPER_TARGET_NUM_ITERATIONS),
        "paper_deviation_queries_per_step": float(algorithm.queries_per_step != PAPER_TARGET_QUERIES_PER_STEP),
        "paper_deviation_rollouts_per_query": float(algorithm.rollouts_per_query != PAPER_TARGET_ROLLOUTS_PER_QUERY),
        "paper_deviation_total_trajectories_per_step": float(algorithm.total_trajectories != PAPER_TARGET_TOTAL_TRAJECTORIES),
        "paper_deviation_world_size": float(runtime.world_size != PAPER_TARGET_WORLD_SIZE),
        "paper_deviation_max_new_tokens": float(algorithm.max_new_tokens != PAPER_TARGET_MAX_NEW_TOKENS),
        "paper_deviation_steps_per_generation": float(
            algorithm.steps_per_generation != PAPER_TARGET_STEPS_PER_GENERATION
        ),
        "paper_deviation_num_iterations": float(algorithm.num_iterations != PAPER_TARGET_NUM_ITERATIONS),
        "paper_deviation_runtime_stack": float(runtime_spec.runtime_stack != PAPER_TARGET_RUNTIME_STACK),
        "paper_runtime_deviation_from_spec": float(
            algorithm.queries_per_step != PAPER_TARGET_QUERIES_PER_STEP
            or algorithm.rollouts_per_query != PAPER_TARGET_ROLLOUTS_PER_QUERY
            or algorithm.total_trajectories != PAPER_TARGET_TOTAL_TRAJECTORIES
            or runtime.world_size != PAPER_TARGET_WORLD_SIZE
            or algorithm.max_new_tokens != PAPER_TARGET_MAX_NEW_TOKENS
            or algorithm.steps_per_generation != PAPER_TARGET_STEPS_PER_GENERATION
            or algorithm.num_iterations != PAPER_TARGET_NUM_ITERATIONS
            or runtime_spec.runtime_stack != PAPER_TARGET_RUNTIME_STACK
        ),
    }


def resolve_weave_project(args: Any) -> str:
    explicit = normalize_text(getattr(args, "weave_project", None)).strip()
    if explicit:
        return explicit
    entity = normalize_text(getattr(args, "wandb_entity", None)).strip()
    project = normalize_text(getattr(args, "wandb_project", None)).strip()
    if entity and project:
        return f"{entity}/{project}"
    return ""


def ensure_weave_server_cache_dir(output_dir: Path) -> str:
    configured_dir = normalize_text(os.getenv("WEAVE_SERVER_CACHE_DIR")).strip()
    if configured_dir:
        cache_dir = Path(configured_dir).expanduser()
    else:
        cache_dir = (output_dir / "weave_server_cache").resolve()
        os.environ["WEAVE_SERVER_CACHE_DIR"] = str(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir.resolve())


def traceable_sample_meta(sample_meta: Mapping[str, Any], *, allowed_keys: Sequence[str]) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for key in allowed_keys:
        normalized = normalize_text(sample_meta.get(key)).strip()
        if normalized:
            payload[str(key)] = normalized
    return payload


def parse_bool(raw: Any) -> bool:
    return normalize_text(raw).strip().lower() in {"1", "true", "t", "yes", "y"}


def env_int(name: str, default: int) -> int:
    raw = normalize_text(os.environ.get(name)).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("train_protein_grpo.py requires torch to be installed.")


def require_module(module_name: str, install_hint: Optional[str] = None) -> None:
    if importlib.util.find_spec(module_name) is not None:
        return
    hint = f" Install {install_hint}." if install_hint else ""
    raise RuntimeError(f"Missing required dependency: {module_name}.{hint}")


def module_is_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def is_probably_local_path(value: Any) -> bool:
    text = normalize_text(value).strip()
    if not text:
        return False
    return text.startswith(("/", ".", "~"))


@dataclass(frozen=True)
class AlgorithmSpec:
    queries_per_step: int = 8
    rollouts_per_query: int = 24
    steps_per_generation: int = 2
    num_iterations: int = 1
    clip_epsilon_low: float = 7e-4
    clip_epsilon_high: float = 9e-4
    importance_sampling_cap: float = 2.0
    kl_beta: float = 1e-4
    max_new_tokens: int = 10_000
    reward_std_epsilon: float = 1e-6

    @property
    def total_trajectories(self) -> int:
        return self.queries_per_step * self.rollouts_per_query

    @property
    def policy_denominator(self) -> float:
        return float(self.total_trajectories * self.max_new_tokens)

    @property
    def kl_denominator(self) -> float:
        return float(self.total_trajectories)


@dataclass(frozen=True)
class RuntimeSpec:
    optimizer_micro_batch_size_per_gpu: int = 6
    gradient_accumulation_steps: int = 2
    target_num_nodes: int = 2
    target_gpus_per_node: int = 8
    zero_stage: int = 2
    bf16: bool = True
    runtime_stack: str = "deepspeed_vllm_colocate"

    @property
    def target_world_size(self) -> int:
        return self.target_num_nodes * self.target_gpus_per_node

    @property
    def local_trajectories_per_rank(self) -> int:
        return self.optimizer_micro_batch_size_per_gpu * self.gradient_accumulation_steps


@dataclass(frozen=True)
class SamplingSpec:
    temperature: float = 1.0
    top_k: int = 20
    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 10_000


@dataclass(frozen=True)
class EvalSpec:
    validation_num_proteins: int = 200
    validation_every_n_steps: int = 50
    save_every_n_steps: int = 50


@dataclass
class DistributedRuntime:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: Any
    query_parallel_degree: int = 1
    query_group_index: int = 0
    query_rank_in_group: int = 0
    query_group_ranks: Tuple[int, ...] = field(default_factory=tuple)
    query_process_group: Any = None


@dataclass
class PreparedQuery:
    input_ids: Any
    attention_mask: Any
    protein_sequences: List[str]
    batch_idx_map: List[int]
    structure_coords: Optional[Any]
    go_aspects: List[str]
    sample_meta: Dict[str, str]
    prompt_text: str
    multimodal_cache: Optional[Dict[str, Any]] = None


@dataclass
class RolloutGroup:
    query: PreparedQuery
    completions: List[str]
    completion_ids: List[Any]
    rewards: List[float]
    selected_completion_ids: Optional[List[Any]] = None
    filtered_rollouts: float = 0.0
    advantages: Optional[Any] = None
    old_log_probs: Optional[Any] = None
    ref_log_probs: Optional[Any] = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_job_type", type=str, default="train_rl", choices=["train_rl"])
    parser.add_argument("--benchmark_version", type=str, default="213 -> 221 -> 225 -> 228")
    parser.add_argument("--temporal_split_artifact", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default="disease_temporal_hc_reasoning_v2")
    parser.add_argument("--reasoning_dataset_config", type=str, default="disease_temporal_hc_reasoning_v2")
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

    parser.add_argument("--text_model_name", type=str, required=True, help="HF checkpoint used as the SFT RL init policy.")
    parser.add_argument("--protein_model_name", type=str, default="esm3_sm_open_v1")
    parser.add_argument("--attn_implementation", type=str, default=os.environ.get("BIOREASON_ATTN_IMPLEMENTATION", "auto"))
    parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.cache/huggingface/hub"))
    parser.add_argument("--go_obo_path", type=str, default="bioreason2/dataset/go-basic.obo")
    parser.add_argument("--ia_file_path", type=str, default=os.environ.get("BIOREASON_IA_FILE_PATH", ""))
    parser.add_argument("--precomputed_embeddings_path", type=str, default=None)

    parser.add_argument("--cafa5_dataset", type=str, default="wanglab/cafa5")
    parser.add_argument("--reasoning_dataset_name", type=str, default="disease_temporal_hc_reasoning_v2")
    parser.add_argument("--interpro_dataset_name", type=str, default="interpro_metadata")
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--structure_dir", type=str, default=None)
    parser.add_argument("--go_gpt_predictions_column", type=str, default="go_pred")
    parser.add_argument("--dataset_num_proc", type=int, default=env_int("BIOREASON_DATASET_NUM_PROC", 4))
    parser.add_argument(
        "--reasoning_prompt_style",
        type=str,
        default="paper_native_tight",
        choices=["paper_native", "paper_native_tight", "paper_compact"],
    )

    parser.add_argument("--max_length_text", type=int, default=512)
    parser.add_argument("--max_length_protein", type=int, default=2000)
    parser.add_argument("--protein_embedding_layer", type=int, default=37)
    parser.add_argument("--go_hidden_dim", type=int, default=512)
    parser.add_argument("--go_num_gat_layers", type=int, default=3)
    parser.add_argument("--go_num_heads", type=int, default=8)
    parser.add_argument("--go_num_reduced_embeddings", type=int, default=200)
    parser.add_argument("--go_embedding_dim", type=int, default=2560)
    parser.add_argument("--unified_go_encoder", type=str, default="true")

    parser.add_argument("--queries_per_step", type=int, default=8)
    parser.add_argument("--rollouts_per_query", type=int, default=24)
    parser.add_argument("--steps_per_generation", type=int, default=2)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1200)

    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--optimizer_micro_batch_size_per_gpu", type=int, default=6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--target_num_nodes", type=int, default=2)
    parser.add_argument("--target_gpus_per_node", type=int, default=8)
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--runtime_stack", type=str, default="deepspeed_vllm_colocate")

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--gradient_checkpointing", type=str, default="true")
    parser.add_argument("--disable_model_dropout", type=str, default="true")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=10_000)
    parser.add_argument("--rollout_max_new_tokens", type=int, default=0)
    parser.add_argument("--max_loss_completion_tokens", type=int, default=0)
    parser.add_argument("--rollout_logprob_microbatch_size", type=int, default=4)
    parser.add_argument("--clip_epsilon_low", type=float, default=7e-4)
    parser.add_argument("--clip_epsilon_high", type=float, default=9e-4)
    parser.add_argument("--importance_sampling_cap", type=float, default=2.0)
    parser.add_argument("--kl_beta", type=float, default=1e-4)
    parser.add_argument("--reward_std_epsilon", type=float, default=1e-6)

    parser.add_argument("--validation_num_proteins", type=int, default=200)
    parser.add_argument("--validation_every_n_steps", type=int, default=50)
    parser.add_argument("--save_every_n_steps", type=int, default=50)

    parser.add_argument("--output_dir", type=str, default="data/artifacts/models/train_rl_output")
    parser.add_argument("--checkpoint_artifact_name", type=str, default="train-rl-output")
    parser.add_argument("--checkpoint_artifact_aliases", type=str, default="latest")

    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "bioreasoning-pro"))
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)
    parser.add_argument("--weave_project", type=str, default=None)
    parser.add_argument("--trace_rollouts_to_weave", type=str, default="true")
    parser.add_argument("--trace_jsonl_name", type=str, default="rollout_traces.jsonl")
    parser.add_argument("--weave_trace_budget", type=int, default=64)
    parser.add_argument("--weave_trace_full_group_count", type=int, default=4)
    parser.add_argument("--weave_trace_full_rollouts_per_group", type=int, default=24)

    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.35)
    parser.add_argument("--vllm_max_model_len", type=int, default=32768)
    parser.add_argument("--vllm_max_num_seqs", type=int, default=32)
    parser.add_argument("--vllm_cpu_offload_gb", type=float, default=0.0)
    parser.add_argument("--vllm_swap_space_gb", type=float, default=4.0)
    parser.add_argument("--vllm_enforce_eager", type=str, default="true")
    parser.add_argument("--vllm_enable_sleep_mode", type=str, default="false")
    parser.add_argument("--vllm_sleep_level", type=int, default=1)
    parser.add_argument(
        "--vllm_attention_backend",
        type=str,
        default=os.environ.get("BIOREASON_VLLM_ATTENTION_BACKEND", os.environ.get("VLLM_ATTENTION_BACKEND", "XFORMERS")),
    )
    parser.add_argument(
        "--vllm_worker_multiproc_method",
        type=str,
        default=os.environ.get(
            "BIOREASON_VLLM_WORKER_MULTIPROC_METHOD",
            os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "spawn"),
        ),
        choices=["spawn", "forkserver", "fork"],
    )
    parser.add_argument(
        "--vllm_use_v1",
        type=str,
        default=os.environ.get("BIOREASON_VLLM_USE_V1", os.environ.get("VLLM_USE_V1", "false")),
    )
    parser.add_argument("--rollout_backend", type=str, default="subprocess", choices=["subprocess", "inprocess"])
    parser.add_argument("--rollout_worker_start_method", type=str, default="spawn", choices=["spawn", "forkserver", "fork"])
    parser.add_argument("--rollout_worker_generate_timeout_s", type=float, default=900.0)
    parser.add_argument("--rollout_worker_vllm_port_base", type=int, default=39000)
    parser.add_argument("--rollout_worker_vllm_port_stride", type=int, default=32)
    parser.add_argument("--rollout_worker_vllm_host_ip", type=str, default="127.0.0.1")

    # DeepSpeed injects this flag into worker processes; accept it even though the
    # runtime primarily reads LOCAL_RANK from the environment.
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--debug_single_process", type=str, default="false")
    parser.add_argument("--preflight_only", type=str, default="false")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    for name in (
        "unified_go_encoder",
        "gradient_checkpointing",
        "disable_model_dropout",
        "trace_rollouts_to_weave",
        "vllm_enforce_eager",
        "vllm_enable_sleep_mode",
        "vllm_use_v1",
        "debug_single_process",
        "preflight_only",
    ):
        setattr(args, name, parse_bool(getattr(args, name)))
    return args


def build_algorithm_spec(args: argparse.Namespace) -> AlgorithmSpec:
    return AlgorithmSpec(
        queries_per_step=int(args.queries_per_step),
        rollouts_per_query=int(args.rollouts_per_query),
        steps_per_generation=int(args.steps_per_generation),
        num_iterations=int(args.num_iterations),
        clip_epsilon_low=float(args.clip_epsilon_low),
        clip_epsilon_high=float(args.clip_epsilon_high),
        importance_sampling_cap=float(args.importance_sampling_cap),
        kl_beta=float(args.kl_beta),
        max_new_tokens=int(args.max_new_tokens),
        reward_std_epsilon=float(args.reward_std_epsilon),
    )


def build_runtime_spec(args: argparse.Namespace) -> RuntimeSpec:
    return RuntimeSpec(
        optimizer_micro_batch_size_per_gpu=int(args.optimizer_micro_batch_size_per_gpu),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        target_num_nodes=int(args.target_num_nodes),
        target_gpus_per_node=int(args.target_gpus_per_node),
        zero_stage=int(args.zero_stage),
        runtime_stack=normalize_text(args.runtime_stack).strip() or "deepspeed_vllm_colocate",
    )


def build_sampling_spec(args: argparse.Namespace) -> SamplingSpec:
    rollout_max_new_tokens = int(getattr(args, "rollout_max_new_tokens", 0) or 0)
    if rollout_max_new_tokens <= 0:
        rollout_max_new_tokens = int(args.max_new_tokens)
    return SamplingSpec(
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        min_p=float(args.min_p),
        repetition_penalty=float(args.repetition_penalty),
        max_new_tokens=rollout_max_new_tokens,
    )


def build_eval_spec(args: argparse.Namespace) -> EvalSpec:
    return EvalSpec(
        validation_num_proteins=int(args.validation_num_proteins),
        validation_every_n_steps=int(args.validation_every_n_steps),
        save_every_n_steps=int(args.save_every_n_steps),
    )


def resolve_effective_vllm_sleep_mode(args: argparse.Namespace) -> bool:
    requested = bool(getattr(args, "vllm_enable_sleep_mode", False))
    backend = normalize_text(getattr(args, "rollout_backend", "subprocess")).strip() or "subprocess"
    if requested and backend == "subprocess":
        return False
    return requested


def resolve_dataset_num_proc(value: Any) -> Optional[int]:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def resolve_effective_dataset_num_proc(value: Any, *, distributed: bool) -> Optional[int]:
    resolved = resolve_dataset_num_proc(value)
    if distributed and resolved is not None and resolved > 1:
        return 1
    return resolved


def validate_spec_inputs(args: argparse.Namespace) -> None:
    text_model_name = normalize_text(args.text_model_name).strip()
    if not text_model_name:
        raise ValueError("Spec-first DR-GRPO requires text_model_name to point to a resolved local model bundle.")
    text_model_path = Path(text_model_name).expanduser()
    if not text_model_path.exists():
        raise ValueError(f"text_model_name points to a missing local path: {text_model_name}")
    if not text_model_path.is_dir():
        raise ValueError(f"text_model_name must point to a local model directory: {text_model_name}")
    validate_model_bundle_dir(text_model_path)

    dataset_path = normalize_text(args.cafa5_dataset).strip()
    if not dataset_path:
        raise ValueError("Spec-first DR-GRPO requires cafa5_dataset to point to a materialized local dataset directory.")
    resolved_dataset_path = Path(dataset_path).expanduser()
    if not resolved_dataset_path.exists() or not resolved_dataset_path.is_dir():
        raise ValueError(
            "Spec-first DR-GRPO requires cafa5_dataset to point to a materialized local dataset directory. "
            f"Got cafa5_dataset={args.cafa5_dataset!r}."
        )

    ia_path = normalize_text(args.ia_file_path).strip()
    if not ia_path or not os.path.exists(ia_path):
        raise ValueError(
            "Spec-first DR-GRPO requires a valid IA file because the reward is IA-weighted F1. "
            f"Got ia_file_path={args.ia_file_path!r}."
        )
    go_obo_path = normalize_text(args.go_obo_path).strip()
    if not go_obo_path or not os.path.exists(go_obo_path):
        raise ValueError(
            "Spec-first DR-GRPO requires a valid GO ontology OBO file for ancestor propagation. "
            f"Got go_obo_path={args.go_obo_path!r}."
        )


def validate_runtime_dependencies() -> None:
    require_module("torch", install_hint="torch")
    require_module("deepspeed", install_hint="deepspeed")
    require_module("peft", install_hint="peft")
    require_module("transformers", install_hint="transformers")
    require_module("vllm", install_hint="vllm")


def validate_model_bundle_dir(model_dir: Path) -> None:
    required_files = (
        "config.json",
        "tokenizer_config.json",
        "protein_projection.pt",
        "protein_model/pytorch_model.bin",
    )
    missing = [relative_path for relative_path in required_files if not (model_dir / relative_path).exists()]
    if missing:
        raise ValueError(
            "Spec-first DR-GRPO requires a materialized local model bundle with the expected files. "
            f"Missing from {model_dir}: {', '.join(missing)}"
        )

    tokenizer_candidates = (
        "tokenizer.json",
        "tokenizer.model",
        "spiece.model",
        "vocab.json",
    )
    if not any((model_dir / candidate).exists() for candidate in tokenizer_candidates):
        raise ValueError(
            "Spec-first DR-GRPO requires tokenizer files in the resolved model bundle. "
            f"Expected one of {', '.join(tokenizer_candidates)} under {model_dir}."
        )


def collect_runtime_dependency_statuses() -> Dict[str, bool]:
    return {
        "torch": module_is_available("torch"),
        "deepspeed": module_is_available("deepspeed"),
        "peft": module_is_available("peft"),
        "transformers": module_is_available("transformers"),
        "vllm": module_is_available("vllm"),
    }


def run_preflight(args: argparse.Namespace) -> bool:
    algorithm = build_algorithm_spec(args)
    runtime_spec = build_runtime_spec(args)
    effective_dataset_num_proc = resolve_effective_dataset_num_proc(
        args.dataset_num_proc,
        distributed=runtime_spec.target_world_size > 1,
    )
    dependency_statuses = collect_runtime_dependency_statuses()
    missing_dependencies = [name for name, present in dependency_statuses.items() if not present]
    failures: List[str] = []
    warnings: List[str] = []

    try:
        validate_algorithm_runtime_contract(algorithm, runtime_spec)
    except Exception as exc:
        failures.append(str(exc))

    try:
        validate_spec_inputs(args)
    except Exception as exc:
        failures.append(str(exc))

    if runtime_spec.target_world_size > 1 and normalize_text(getattr(args, "rollout_backend", "subprocess")).strip() != "subprocess":
        failures.append(
            "Spec-first distributed launches require rollout_backend=subprocess for the colocated vLLM worker."
        )
    if bool(args.vllm_enable_sleep_mode) and not resolve_effective_vllm_sleep_mode(args):
        warnings.append(
            "vLLM sleep mode is automatically disabled when rollout_backend=subprocess because vLLM refreshes "
            "cannot safely instantiate multiple sleep-enabled engines in the same worker process."
        )
    if effective_dataset_num_proc != resolve_dataset_num_proc(args.dataset_num_proc):
        warnings.append(
            "dataset_num_proc is automatically reduced to 1 for distributed launches to avoid pyarrow mmap worker "
            "bus errors during concurrent dataset.map() preprocessing."
        )

    if missing_dependencies:
        failures.append(
            "Missing runtime dependencies: " + ", ".join(missing_dependencies)
        )

    text_model_name = normalize_text(args.text_model_name).strip()
    resolved_text_model_path = str(Path(text_model_name).expanduser()) if text_model_name else ""
    dataset_path = normalize_text(args.cafa5_dataset).strip()
    resolved_dataset_path = str(Path(dataset_path).expanduser()) if dataset_path else ""
    ia_path = normalize_text(args.ia_file_path).strip()
    resolved_ia_path = str(Path(ia_path).expanduser()) if ia_path else ""
    go_obo_path = normalize_text(args.go_obo_path).strip()
    resolved_go_obo_path = str(Path(go_obo_path).expanduser()) if go_obo_path else ""

    base_checkpoint = normalize_text(args.base_checkpoint).strip()
    if base_checkpoint and is_probably_local_path(base_checkpoint) and not Path(base_checkpoint).expanduser().exists():
        failures.append(f"base_checkpoint points to a missing local path: {base_checkpoint}")

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if env_world_size > 1 and env_world_size != runtime_spec.target_world_size:
        failures.append(
            "Current WORLD_SIZE does not match the configured distributed runtime shape: "
            f"WORLD_SIZE={env_world_size}, required={runtime_spec.target_world_size}."
        )
    if env_world_size == 1 and not args.debug_single_process:
        warnings.append(
            "WORLD_SIZE is not set for a distributed launch in this shell. "
            f"That is fine for preflight, but the real run should use deepspeed with {runtime_spec.target_world_size} ranks."
        )

    preflight_plan = {
        "run_name": normalize_text(args.run_name).strip() or "<auto>",
        "text_model_name": text_model_name,
        "base_checkpoint": base_checkpoint or text_model_name,
        "queries_per_step": algorithm.queries_per_step,
        "rollouts_per_query": algorithm.rollouts_per_query,
        "total_trajectories": algorithm.total_trajectories,
        "steps_per_generation": algorithm.steps_per_generation,
        "target_world_size": runtime_spec.target_world_size,
        "query_parallel_degree": resolve_query_parallel_degree(runtime_spec.target_world_size, algorithm.queries_per_step),
        "runtime_stack": runtime_spec.runtime_stack,
        "rollout_backend": normalize_text(args.rollout_backend).strip(),
        "attn_implementation": normalize_text(args.attn_implementation).strip() or "auto",
        "dataset_num_proc": effective_dataset_num_proc,
        "vllm_attention_backend": normalize_text(args.vllm_attention_backend).strip() or "<auto>",
        "vllm_worker_multiproc_method": normalize_text(args.vllm_worker_multiproc_method).strip() or "spawn",
        "vllm_use_v1": bool(args.vllm_use_v1),
        "debug_single_process": bool(args.debug_single_process),
        "dependencies": dependency_statuses,
    }
    resolved_paths = {
        "text_model_name": resolved_text_model_path,
        "cafa5_dataset": resolved_dataset_path,
        "ia_file_path": resolved_ia_path,
        "go_obo_path": resolved_go_obo_path,
    }
    artifact_refs = {
        "base_checkpoint": base_checkpoint or text_model_name,
        "temporal_split_artifact": normalize_text(args.temporal_split_artifact).strip(),
        "dataset_artifact": normalize_text(args.dataset_artifact).strip(),
    }
    launch_contract = {
        "runtime_stack": runtime_spec.runtime_stack,
        "target_num_nodes": runtime_spec.target_num_nodes,
        "target_gpus_per_node": runtime_spec.target_gpus_per_node,
        "target_world_size": runtime_spec.target_world_size,
        "queries_per_step": algorithm.queries_per_step,
        "rollouts_per_query": algorithm.rollouts_per_query,
        "query_parallel_degree": resolve_query_parallel_degree(runtime_spec.target_world_size, algorithm.queries_per_step),
        "local_rollouts_per_rank": resolve_local_rollouts_per_rank(algorithm, runtime_spec.target_world_size),
        "local_trajectories_per_rank": runtime_spec.local_trajectories_per_rank,
        "optimizer_micro_batch_size_per_gpu": runtime_spec.optimizer_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": runtime_spec.gradient_accumulation_steps,
        "rollout_logprob_microbatch_size": int(args.rollout_logprob_microbatch_size),
        "max_loss_completion_tokens": int(args.max_loss_completion_tokens),
        "attn_implementation": normalize_text(args.attn_implementation).strip() or "auto",
        "dataset_num_proc": effective_dataset_num_proc,
        "vllm_attention_backend": normalize_text(args.vllm_attention_backend).strip() or "<auto>",
        "vllm_worker_multiproc_method": normalize_text(args.vllm_worker_multiproc_method).strip() or "spawn",
        "vllm_use_v1": bool(args.vllm_use_v1),
        "vllm_enable_sleep_mode": resolve_effective_vllm_sleep_mode(args),
        "world_size_env": env_world_size,
    }
    print(
        json.dumps(
            {
                "artifact_refs": artifact_refs,
                "failures": failures,
                "launch_contract": launch_contract,
                "preflight": preflight_plan,
                "resolved_paths": resolved_paths,
                "warnings": warnings,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return not failures


def is_distributed_initialized() -> bool:
    require_torch()
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def initialize_runtime(args: argparse.Namespace) -> DistributedRuntime:
    require_torch()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(getattr(args, "local_rank", 0))))
    enabled = world_size > 1

    if enabled:
        import deepspeed

        deepspeed.init_distributed(dist_backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Spec-first DR-GRPO training requires CUDA.")
        import deepspeed

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(env_int("MASTER_PORT", 29500)))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        if not is_distributed_initialized():
            deepspeed.init_distributed(dist_backend="nccl")
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)

    return DistributedRuntime(
        enabled=enabled,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def resolve_query_parallel_degree(world_size: int, queries_per_step: int) -> int:
    if world_size <= 1 or queries_per_step <= 0:
        return 1
    if world_size % queries_per_step != 0:
        raise ValueError(
            "Configured WORLD_SIZE must be an integer multiple of queries_per_step. "
            f"Got world_size={world_size}, queries_per_step={queries_per_step}."
        )
    return max(world_size // queries_per_step, 1)


def resolve_local_rollouts_per_rank(algorithm: AlgorithmSpec, world_size: int) -> int:
    query_parallel_degree = resolve_query_parallel_degree(world_size, algorithm.queries_per_step)
    if algorithm.rollouts_per_query % query_parallel_degree != 0:
        raise ValueError(
            "rollouts_per_query must be divisible by query_parallel_degree for distributed rollout sharding. "
            f"Got rollouts_per_query={algorithm.rollouts_per_query}, query_parallel_degree={query_parallel_degree}."
        )
    return max(algorithm.rollouts_per_query // query_parallel_degree, 1)


def resolve_effective_vllm_max_num_seqs(args: argparse.Namespace) -> int:
    configured = max(int(args.vllm_max_num_seqs), 1)
    target_world_size = max(int(args.target_num_nodes) * int(args.target_gpus_per_node), 1)
    queries_per_step = max(int(args.queries_per_step), 1)
    rollouts_per_query = max(int(args.rollouts_per_query), 1)
    query_parallel_degree = resolve_query_parallel_degree(target_world_size, queries_per_step)
    if rollouts_per_query % query_parallel_degree != 0:
        raise ValueError(
            "rollouts_per_query must be divisible by query_parallel_degree when resolving "
            "the effective vLLM max_num_seqs. "
            f"Got rollouts_per_query={rollouts_per_query}, "
            f"query_parallel_degree={query_parallel_degree}."
        )
    local_rollouts = max(rollouts_per_query // query_parallel_degree, 1)
    return max(configured, local_rollouts)


def resolve_rollout_worker_vllm_port(args: argparse.Namespace, runtime_rank: int) -> int:
    base = max(int(getattr(args, "rollout_worker_vllm_port_base", 39000)), 1024)
    stride = max(int(getattr(args, "rollout_worker_vllm_port_stride", 32)), 1)
    return base + (max(int(runtime_rank), 0) * stride)


def configure_query_parallel_runtime(runtime: DistributedRuntime, algorithm: AlgorithmSpec) -> None:
    if not runtime.enabled:
        runtime.query_parallel_degree = 1
        runtime.query_group_index = 0
        runtime.query_rank_in_group = 0
        runtime.query_group_ranks = tuple()
        runtime.query_process_group = None
        return
    query_parallel_degree = resolve_query_parallel_degree(runtime.world_size, algorithm.queries_per_step)
    runtime.query_parallel_degree = query_parallel_degree
    runtime.query_group_index = runtime.rank // query_parallel_degree
    runtime.query_rank_in_group = runtime.rank % query_parallel_degree
    group_start = runtime.query_group_index * query_parallel_degree
    runtime.query_group_ranks = tuple(range(group_start, group_start + query_parallel_degree))
    if query_parallel_degree > 1:
        runtime.query_process_group = torch.distributed.new_group(ranks=list(runtime.query_group_ranks))
    else:
        runtime.query_process_group = None


def rank0_print(runtime: DistributedRuntime, message: str) -> None:
    if runtime.rank == 0:
        print(message, flush=True)


def rank_print(runtime: DistributedRuntime, message: str) -> None:
    print(f"[rank {runtime.rank}] {message}", flush=True)


def maybe_stagger_startup_model_load(runtime: DistributedRuntime) -> None:
    if not runtime.enabled:
        return
    local_rank = max(int(runtime.local_rank), 0)
    if local_rank <= 0:
        return
    time.sleep(min(float(local_rank) * 2.0, 14.0))


def destroy_torch_distributed_process_group(log_prefix: Optional[str] = None) -> None:
    if torch is None:
        return
    if not torch.distributed.is_available():
        return
    if not torch.distributed.is_initialized():
        return
    try:
        if log_prefix:
            print(f"{log_prefix} destroying torch.distributed process group", flush=True)
        torch.distributed.destroy_process_group()
    except Exception as exc:
        if log_prefix:
            print(f"{log_prefix} destroy_process_group failed: {exc}", flush=True)


def shutdown_runtime(runtime: DistributedRuntime) -> None:
    if torch is None:
        return
    if not is_distributed_initialized():
        return
    destroy_torch_distributed_process_group(log_prefix=f"[rank {runtime.rank}]")


def all_reduce_sum_scalar(value: float, runtime: DistributedRuntime, process_group: Any = None) -> float:
    if torch is None:
        if runtime.enabled:
            raise RuntimeError("Distributed scalar reduction requires torch.")
        return float(value)
    tensor = torch.tensor(float(value), device=runtime.device, dtype=torch.float64)
    if runtime.enabled:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=process_group)
    return float(tensor.item())


def all_reduce_max_scalar(value: float, runtime: DistributedRuntime, process_group: Any = None) -> float:
    if torch is None:
        if runtime.enabled:
            raise RuntimeError("Distributed scalar reduction requires torch.")
        return float(value)
    tensor = torch.tensor(float(value), device=runtime.device, dtype=torch.float64)
    if runtime.enabled:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX, group=process_group)
    return float(tensor.item())


def barrier(runtime: DistributedRuntime) -> None:
    if runtime.enabled and is_distributed_initialized():
        torch.distributed.barrier()


def broadcast_indices(indices: List[int], runtime: DistributedRuntime) -> List[int]:
    require_torch()
    tensor = torch.tensor(indices if runtime.rank == 0 else [0] * len(indices), device=runtime.device, dtype=torch.long)
    if runtime.enabled:
        torch.distributed.broadcast(tensor, src=0)
    return [int(item) for item in tensor.cpu().tolist()]


def sample_query_indices(dataset_length: int, queries_per_step: int, seed: int, step: int, runtime: DistributedRuntime) -> List[int]:
    if runtime.rank == 0:
        rng = random.Random(seed + step)
        if dataset_length < queries_per_step:
            raise ValueError(
                f"Train dataset has only {dataset_length} items, but the specification requires {queries_per_step} queries per step."
            )
        indices = rng.sample(range(dataset_length), queries_per_step)
    else:
        indices = [0] * queries_per_step
    return broadcast_indices(indices, runtime)


def partition_queries_for_rank(global_indices: Sequence[int], rank: int, world_size: int, queries_per_step: int) -> List[int]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if world_size == 1:
        return [int(item) for item in global_indices]
    query_parallel_degree = resolve_query_parallel_degree(world_size, queries_per_step)
    query_index = rank // query_parallel_degree
    if query_index < 0 or query_index >= len(global_indices):
        raise ValueError(
            f"Rank {rank} resolved to query_index={query_index}, but only {len(global_indices)} queries were sampled."
        )
    return [int(global_indices[query_index])]


def resolve_local_cuda_visible_device(local_rank: int, cuda_visible_devices: Optional[str] = None) -> str:
    visible = normalize_text(cuda_visible_devices if cuda_visible_devices is not None else os.environ.get("CUDA_VISIBLE_DEVICES")).strip()
    if not visible:
        return str(local_rank)
    device_tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if not device_tokens:
        return str(local_rank)
    if len(device_tokens) == 1:
        return device_tokens[0]
    if local_rank < 0 or local_rank >= len(device_tokens):
        raise ValueError(
            "local_rank is out of range for CUDA_VISIBLE_DEVICES. "
            f"Got local_rank={local_rank}, CUDA_VISIBLE_DEVICES={visible!r}."
        )
    return device_tokens[local_rank]


def select_rollout_indices_for_loss(
    completion_ids_list: Sequence[Any],
    *,
    max_loss_completion_tokens: int,
) -> List[int]:
    if max_loss_completion_tokens <= 0:
        return list(range(len(completion_ids_list)))
    selected_indices: List[int] = []
    for rollout_idx, completion_ids in enumerate(completion_ids_list):
        if int(completion_ids.numel()) <= int(max_loss_completion_tokens):
            selected_indices.append(rollout_idx)
    return selected_indices


def validate_runtime_shape(
    runtime: DistributedRuntime,
    algorithm: AlgorithmSpec,
    runtime_spec: RuntimeSpec,
    args: argparse.Namespace,
) -> None:
    validate_algorithm_runtime_contract(algorithm, runtime_spec)
    if runtime.enabled and runtime.world_size != runtime_spec.target_world_size:
        raise ValueError(
            "Spec-first distributed training requires world_size == target_num_nodes * target_gpus_per_node. "
            f"Got world_size={runtime.world_size}, target_world_size={runtime_spec.target_world_size}."
        )
    if (not runtime.enabled) and (not args.debug_single_process):
        raise ValueError(
            "Spec-first trainer expects a distributed DeepSpeed launch. "
            f"Use deepspeed to launch {runtime_spec.target_world_size} ranks, or pass --debug_single_process true for a non-paper-faithful debug run."
        )
    if runtime.enabled and normalize_text(getattr(args, "rollout_backend", "subprocess")).strip() != "subprocess":
        raise ValueError(
            "Distributed spec-first training requires rollout_backend=subprocess so the vLLM worker runs in a clean "
            "single-GPU subprocess per rank."
        )


def validate_algorithm_runtime_contract(
    algorithm: AlgorithmSpec,
    runtime_spec: RuntimeSpec,
) -> None:
    if runtime_spec.runtime_stack != "deepspeed_vllm_colocate":
        raise ValueError(
            f"Spec-first trainer only supports runtime_stack=deepspeed_vllm_colocate, got {runtime_spec.runtime_stack!r}."
        )
    query_parallel_degree = resolve_query_parallel_degree(runtime_spec.target_world_size, algorithm.queries_per_step)
    expected_local_trajectories = resolve_local_rollouts_per_rank(algorithm, runtime_spec.target_world_size)
    if runtime_spec.local_trajectories_per_rank != expected_local_trajectories:
        raise ValueError(
            "The configured runtime requires each rank to process the per-query rollout shard owned by its query-parallel group. "
            f"Got optimizer_micro_batch_size_per_gpu={runtime_spec.optimizer_micro_batch_size_per_gpu}, "
            f"gradient_accumulation_steps={runtime_spec.gradient_accumulation_steps}, "
            f"which yields {runtime_spec.local_trajectories_per_rank} local trajectories, "
            f"but expected {expected_local_trajectories} for rollouts_per_query={algorithm.rollouts_per_query} "
            f"and query_parallel_degree={query_parallel_degree}."
        )
    if runtime_spec.target_world_size < algorithm.queries_per_step:
        raise ValueError(
            "The configured distributed runtime shape must provide at least one rank per query. "
            f"Got target_world_size={runtime_spec.target_world_size}, queries_per_step={algorithm.queries_per_step}."
        )


def extract_go_terms_from_final_answer(text: str) -> Optional[List[str]]:
    normalized = normalize_structured_response_text(text)
    match = FINAL_ANSWER_PATTERN.search(normalized)
    if match is None:
        fallback_content = extract_final_answer_fallback_content(normalized)
        if fallback_content is None:
            return None
        content = fallback_content
    else:
        content = match.group(1)
    seen = set()
    ordered: List[str] = []
    for go_id in GO_ID_PATTERN.findall(content):
        if go_id not in seen:
            seen.add(go_id)
            ordered.append(go_id)
    return ordered


def extract_go_terms_from_go_summary(text: str) -> Optional[List[str]]:
    match = GO_SUMMARY_PATTERN.search(normalize_structured_response_text(text))
    if match is None:
        return None
    seen = set()
    ordered: List[str] = []
    for go_id in GO_ID_PATTERN.findall(match.group(1)):
        if go_id not in seen:
            seen.add(go_id)
            ordered.append(go_id)
    return ordered


def extract_go_terms_from_completion(text: str) -> Optional[List[str]]:
    return extract_go_terms_from_final_answer(text)


def completion_has_final_answer_tag(text: str) -> bool:
    normalized = normalize_structured_response_text(text)
    return (FINAL_ANSWER_OPEN_TAG in normalized) and (
        FINAL_ANSWER_PATTERN.search(normalized) is not None or extract_final_answer_fallback_content(normalized) is not None
    )


def completion_has_go_summary_block(text: str) -> bool:
    return GO_SUMMARY_PATTERN.search(normalize_structured_response_text(text)) is not None


def completion_uses_alt_final_answer_close_tag(text: str) -> bool:
    return ALT_FINAL_ANSWER_CLOSE_TAG in normalize_text(text)


def completion_has_unclosed_final_answer_tag(text: str) -> bool:
    normalized = normalize_structured_response_text(text)
    if FINAL_ANSWER_OPEN_TAG not in normalized:
        return False
    return FINAL_ANSWER_PATTERN.search(normalized) is None


def completion_has_repeated_final_answer_open_tag(text: str) -> bool:
    normalized = normalize_structured_response_text(text)
    return normalized.count(FINAL_ANSWER_OPEN_TAG) > 1


def completion_has_tool_call_residue(text: str) -> bool:
    normalized = normalize_text(text)
    return "<tool_call>" in normalized or "</tool_call>" in normalized


def completion_has_think_residue(text: str) -> bool:
    normalized = normalize_text(text)
    return "<think>" in normalized or "</think>" in normalized


def build_completion_format_summary(text: str) -> Dict[str, Any]:
    parsed_go_ids = extract_go_terms_from_completion(text) or []
    return {
        "has_final_answer_tag": completion_has_final_answer_tag(text),
        "has_go_summary_block": completion_has_go_summary_block(text),
        "uses_alt_final_answer_close_tag": completion_uses_alt_final_answer_close_tag(text),
        "has_unclosed_final_answer_tag": completion_has_unclosed_final_answer_tag(text),
        "has_repeated_final_answer_open_tag": completion_has_repeated_final_answer_open_tag(text),
        "has_tool_call_residue": completion_has_tool_call_residue(text),
        "has_think_residue": completion_has_think_residue(text),
        "parsed_go_ids": list(parsed_go_ids),
        "parsed_go_count": len(parsed_go_ids),
        "format_valid": bool(parsed_go_ids),
    }


def build_query_sample_meta(batch: Mapping[str, Any]) -> Dict[str, str]:
    return {
        "protein_id": normalize_text((batch.get("protein_ids") or [""])[0]),
        "split": normalize_text((batch.get("sample_splits") or [""])[0]),
        "go_bp": normalize_text((batch.get("go_bp_targets") or [""])[0]),
        "go_mf": normalize_text((batch.get("go_mf_targets") or [""])[0]),
        "go_cc": normalize_text((batch.get("go_cc_targets") or [""])[0]),
    }


def build_target_go_ids(sample_meta: Mapping[str, Any]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for key in ("go_bp", "go_mf", "go_cc"):
        for go_id in GO_ID_PATTERN.findall(normalize_text(sample_meta.get(key))):
            if go_id not in seen:
                seen.add(go_id)
                ordered.append(go_id)
    return ordered


def load_go_term_graph(go_obo_path: str) -> Dict[str, Tuple[str, ...]]:
    if not go_obo_path or not os.path.exists(go_obo_path):
        return {}
    parents: Dict[str, List[str]] = {}
    current_id = ""
    current_parents: List[str] = []
    current_obsolete = False
    in_term = False

    def finalize() -> None:
        nonlocal current_id, current_parents, current_obsolete
        if current_id and not current_obsolete:
            parents[current_id] = list(dict.fromkeys(parent for parent in current_parents if parent))
        current_id = ""
        current_parents = []
        current_obsolete = False

    with open(go_obo_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "[Term]":
                finalize()
                in_term = True
                continue
            if line.startswith("[") and line != "[Term]":
                finalize()
                in_term = False
                continue
            if not in_term or not line:
                continue
            if line.startswith("id: "):
                current_id = normalize_text(line.split(":", 1)[1]).strip()
            elif line.startswith("is_a: "):
                parent = line.split("!", 1)[0].split()[1].strip()
                if GO_ID_PATTERN.fullmatch(parent):
                    current_parents.append(parent)
            elif line.startswith("relationship: part_of "):
                candidate = line.split("relationship: part_of ", 1)[1].split()[0].strip()
                if GO_ID_PATTERN.fullmatch(candidate):
                    current_parents.append(candidate)
            elif line.startswith("is_obsolete: "):
                current_obsolete = line.split(":", 1)[1].strip().lower() == "true"
    finalize()
    return {key: tuple(value) for key, value in parents.items()}


def propagate_go_ids(go_ids: Iterable[str], graph: Mapping[str, Tuple[str, ...]]) -> List[str]:
    seen = set()
    ordered: List[str] = []

    def visit(go_id: str) -> None:
        if go_id in seen:
            return
        seen.add(go_id)
        ordered.append(go_id)
        for parent_id in graph.get(go_id, ()):
            visit(parent_id)

    for go_id in go_ids:
        visit(go_id)
    return ordered


def load_ia_weights(ia_file_path: str) -> Dict[str, float]:
    if not ia_file_path or not os.path.exists(ia_file_path):
        return {}
    weights: Dict[str, float] = {}
    with open(ia_file_path, "r", encoding="utf-8") as handle:
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


def compute_weighted_f1(predicted: Iterable[str], target: Iterable[str], ia_weights: Mapping[str, float]) -> float:
    predicted_set = set(predicted)
    target_set = set(target)
    if not predicted_set or not target_set:
        return 0.0
    intersection = predicted_set & target_set
    if not intersection:
        return 0.0

    def weight(go_id: str) -> float:
        return float(ia_weights.get(go_id, 1.0))

    precision_num = sum(weight(go_id) for go_id in intersection)
    precision_den = sum(weight(go_id) for go_id in predicted_set)
    recall_den = sum(weight(go_id) for go_id in target_set)
    if precision_den <= 0.0 or recall_den <= 0.0:
        return 0.0
    precision = precision_num / precision_den
    recall = precision_num / recall_den
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_group_rewards(
    completions: Sequence[str],
    sample_meta: Mapping[str, Any],
    go_graph: Mapping[str, Tuple[str, ...]],
    ia_weights: Mapping[str, float],
) -> List[float]:
    target_go_ids = build_target_go_ids(sample_meta)
    propagated_target = propagate_go_ids(target_go_ids, go_graph) if go_graph else target_go_ids
    rewards: List[float] = []
    for completion in completions:
        predicted_go_ids = extract_go_terms_from_completion(completion)
        if predicted_go_ids is None:
            rewards.append(0.0)
            continue
        propagated_pred = propagate_go_ids(predicted_go_ids, go_graph) if go_graph else predicted_go_ids
        rewards.append(compute_weighted_f1(propagated_pred, propagated_target, ia_weights))
    return rewards


def compute_global_reward_std(local_group_rewards: Sequence[Sequence[float]], runtime: DistributedRuntime, epsilon: float) -> float:
    flat_rewards = [float(reward) for group in local_group_rewards for reward in group]
    local_sum = sum(flat_rewards)
    local_sq_sum = sum(reward * reward for reward in flat_rewards)
    local_count = float(len(flat_rewards))

    total_sum = all_reduce_sum_scalar(local_sum, runtime)
    total_sq_sum = all_reduce_sum_scalar(local_sq_sum, runtime)
    total_count = all_reduce_sum_scalar(local_count, runtime)
    if total_count <= 0:
        return epsilon

    mean = total_sum / total_count
    variance = max((total_sq_sum / total_count) - (mean * mean), 0.0)
    return math.sqrt(variance) + epsilon


def compute_group_advantages(group_rewards: Sequence[float], global_std: float, group_mean: Optional[float] = None) -> List[float]:
    if not group_rewards:
        return []
    resolved_group_mean = float(group_mean) if group_mean is not None else (sum(group_rewards) / len(group_rewards))
    return [(float(reward) - resolved_group_mean) / global_std for reward in group_rewards]


def compute_query_group_mean(group_rewards: Sequence[float], runtime: DistributedRuntime) -> float:
    if not group_rewards:
        return 0.0
    if (not runtime.enabled) or runtime.query_parallel_degree <= 1:
        return sum(float(reward) for reward in group_rewards) / float(len(group_rewards))
    local_sum = sum(float(reward) for reward in group_rewards)
    local_count = float(len(group_rewards))
    total_sum = all_reduce_sum_scalar(local_sum, runtime, process_group=runtime.query_process_group)
    total_count = all_reduce_sum_scalar(local_count, runtime, process_group=runtime.query_process_group)
    if total_count <= 0.0:
        return 0.0
    return total_sum / total_count


def build_tracking_config(
    args: argparse.Namespace,
    algorithm: AlgorithmSpec,
    runtime_spec: RuntimeSpec,
    runtime: DistributedRuntime,
    run_name: str,
) -> Dict[str, Any]:
    from bioreason2.utils.tracking import build_training_tracking_config

    tracking_args = argparse.Namespace(**vars(args))
    tracking_args.run_name = run_name
    tracking_args.dataset_config = normalize_text(args.dataset_config).strip() or args.reasoning_dataset_name
    tracking_args.reasoning_dataset_config = (
        normalize_text(args.reasoning_dataset_config).strip() or args.reasoning_dataset_name
    )
    tracking_args.base_checkpoint = normalize_text(args.base_checkpoint).strip() or args.text_model_name
    tracking_args.model_artifact = normalize_text(args.model_artifact).strip() or args.checkpoint_artifact_name
    tracking_args.batch_size = algorithm.total_trajectories
    tracking_args.train_batch_size = algorithm.total_trajectories
    tracking_args.eval_batch_size = 1
    tracking_args.per_device_train_batch_size = runtime_spec.optimizer_micro_batch_size_per_gpu
    tracking_args.per_device_eval_batch_size = 1
    tracking_args.max_epochs = None
    tracking_args.eval_every_n_steps = args.validation_every_n_steps
    tracking_args.save_every_n_steps = args.save_every_n_steps
    tracking_args.rollout_execution_mode = "batch_first"
    tracking_args.rollout_query_batch_size = algorithm.queries_per_step
    tracking_args.rollout_group_size = algorithm.rollouts_per_query
    tracking_args.rollout_total_trajectories_target = algorithm.total_trajectories
    tracking_args.target_global_world_size = runtime_spec.target_world_size
    tracking_args.query_parallel_degree = resolve_query_parallel_degree(runtime_spec.target_world_size, algorithm.queries_per_step)
    tracking_args.local_rollouts_per_rank = resolve_local_rollouts_per_rank(algorithm, runtime_spec.target_world_size)
    tracking_args.actual_rollout_group_size = algorithm.rollouts_per_query
    tracking_args.actual_global_unique_proteins_per_step = algorithm.queries_per_step
    tracking_args.actual_global_num_trajectories_per_step = algorithm.total_trajectories
    tracking_args.global_unique_proteins_per_step = algorithm.queries_per_step
    tracking_args.global_num_trajectories_per_step = algorithm.total_trajectories
    tracking_args.paper_faithful_batch_shape = algorithm.total_trajectories == 192
    tracking_args.paper_faithful_hardware_shape = runtime_spec.target_world_size == 8
    tracking_args.paper_faithful_runtime_stack = runtime_spec.runtime_stack == "deepspeed_vllm_colocate"
    tracking_args.paper_faithful_execution_mode = True
    tracking_args.paper_faithful_ready = float(
        tracking_args.paper_faithful_batch_shape
        and tracking_args.paper_faithful_hardware_shape
        and tracking_args.paper_faithful_runtime_stack
        and tracking_args.paper_faithful_execution_mode
    )
    tracking_args.loss_type = "dr_grpo"
    tracking_args.num_generations = algorithm.rollouts_per_query
    tracking_args.reward_funcs = "ia_weighted_f1"
    tracking_args.reward_weights = "1.0"
    tracking_args.reward_scaling = "batch"
    tracking_args.reward_final_answer_only = True
    tracking_args.reward_prediction_source = "final_answer_block"
    tracking_args.require_ia_file = False
    tracking_args.advantage_epsilon_std = algorithm.reward_std_epsilon
    tracking_args.importance_sampling_level = "sequence"
    tracking_args.max_eval_samples = args.validation_num_proteins
    tracking_args.eval_sample_strategy = "full_validation_split"
    tracking_args.distributed_enabled = runtime.enabled
    tracking_args.distributed_strategy = "deepspeed" if runtime.enabled else "single_gpu_debug"
    tracking_args.multimodal_cache_enabled = True
    tracking_args.ref_logprob_cache_enabled = True
    tracking_args.max_steps = args.max_steps
    tracking_args.rollout_backend = args.rollout_backend
    tracking_args.rollout_logprob_microbatch_size = int(args.rollout_logprob_microbatch_size)
    tracking_args.max_loss_completion_tokens = int(args.max_loss_completion_tokens)
    tracking_args.vllm_enable_sleep_mode = resolve_effective_vllm_sleep_mode(args)
    tracking_args.vllm_sleep_level = int(args.vllm_sleep_level)
    tracking_args.vllm_attention_backend = normalize_text(args.vllm_attention_backend).strip() or ""
    tracking_args.vllm_worker_multiproc_method = normalize_text(args.vllm_worker_multiproc_method).strip() or "spawn"
    tracking_args.vllm_use_v1 = parse_bool(args.vllm_use_v1)

    config = build_training_tracking_config(tracking_args, run_name=run_name, job_type="train_rl")
    config.update(paper_runtime_deviation_summary(algorithm, runtime_spec, runtime))
    config.update(
        {
            "algorithm": "DR-GRPO",
            "queries_per_step": algorithm.queries_per_step,
            "rollouts_per_query": algorithm.rollouts_per_query,
            "total_trajectories_per_step": algorithm.total_trajectories,
            "query_parallel_degree": resolve_query_parallel_degree(runtime_spec.target_world_size, algorithm.queries_per_step),
            "local_rollouts_per_rank": resolve_local_rollouts_per_rank(algorithm, runtime_spec.target_world_size),
            "steps_per_generation": algorithm.steps_per_generation,
            "num_iterations": algorithm.num_iterations,
            "clip_epsilon_low": algorithm.clip_epsilon_low,
            "clip_epsilon_high": algorithm.clip_epsilon_high,
            "importance_sampling_cap": algorithm.importance_sampling_cap,
            "kl_beta": algorithm.kl_beta,
            "max_new_tokens": algorithm.max_new_tokens,
            "rollout_max_new_tokens": int(getattr(args, "rollout_max_new_tokens", 0) or 0),
            "optimizer_micro_batch_size_per_gpu": runtime_spec.optimizer_micro_batch_size_per_gpu,
            "target_world_size": runtime_spec.target_world_size,
            "world_size": runtime.world_size,
            "zero_stage": runtime_spec.zero_stage,
            "reward_extraction": "final_answer_only",
            "reasoning_prompt_style": normalize_text(args.reasoning_prompt_style).strip() or "paper_native",
            "wandb_project": normalize_text(args.wandb_project).strip(),
            "wandb_entity": normalize_text(args.wandb_entity).strip(),
            "weave_project": resolve_weave_project(args),
            "rollout_backend": args.rollout_backend,
            "rollout_logprob_microbatch_size": int(args.rollout_logprob_microbatch_size),
            "max_loss_completion_tokens": int(args.max_loss_completion_tokens),
            "vllm_enable_sleep_mode": resolve_effective_vllm_sleep_mode(args),
            "vllm_sleep_level": int(args.vllm_sleep_level),
            "vllm_cpu_offload_gb": float(args.vllm_cpu_offload_gb),
            "vllm_swap_space_gb": float(args.vllm_swap_space_gb),
            "vllm_enforce_eager": parse_bool(args.vllm_enforce_eager),
            "rollout_worker_generate_timeout_s": float(getattr(args, "rollout_worker_generate_timeout_s", 900.0)),
            "rollout_worker_vllm_port_base": int(getattr(args, "rollout_worker_vllm_port_base", 39000)),
            "rollout_worker_vllm_port_stride": int(getattr(args, "rollout_worker_vllm_port_stride", 32)),
            "rollout_worker_vllm_host_ip": normalize_text(getattr(args, "rollout_worker_vllm_host_ip", "127.0.0.1")).strip() or "127.0.0.1",
            "vllm_attention_backend": normalize_text(args.vllm_attention_backend).strip() or "<auto>",
            "vllm_worker_multiproc_method": normalize_text(args.vllm_worker_multiproc_method).strip() or "spawn",
            "vllm_use_v1": parse_bool(args.vllm_use_v1),
        }
    )
    return config


def build_trace_path(output_dir: Path, trace_jsonl_name: str, runtime: DistributedRuntime) -> Path:
    trace_name = normalize_text(trace_jsonl_name).strip() or "rollout_traces.jsonl"
    base = Path(trace_name)
    if runtime.world_size <= 1:
        return output_dir / base
    stem = base.stem or "rollout_traces"
    suffix = base.suffix or ".jsonl"
    return output_dir / f"{stem}.rank{runtime.rank:02d}{suffix}"


def summarize_length_values(lengths: Sequence[int]) -> Dict[str, Any]:
    if not lengths:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "max": 0, "mean": 0.0}
    ordered = sorted(int(length) for length in lengths)

    def percentile(fraction: float) -> int:
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
        return int(ordered[index])

    return {
        "count": len(ordered),
        "min": int(ordered[0]),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "max": int(ordered[-1]),
        "mean": round(sum(ordered) / len(ordered), 2),
    }


def summarize_boolean_values(values: Sequence[bool]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "true_count": 0, "false_count": 0, "true_rate": 0.0}
    true_count = sum(1 for value in values if bool(value))
    count = len(values)
    return {
        "count": count,
        "true_count": true_count,
        "false_count": count - true_count,
        "true_rate": round(true_count / count, 4),
    }


def build_rollout_trace_result(outputs: Sequence[str], tokenizer: Any = None) -> Dict[str, Any]:
    normalized_outputs = [normalize_text(output).strip() for output in outputs]
    char_lengths = [len(output) for output in normalized_outputs]
    word_lengths = [len(output.split()) for output in normalized_outputs]
    format_summaries = [build_completion_format_summary(output) for output in normalized_outputs]
    final_answer_flags = [bool(summary["has_final_answer_tag"]) for summary in format_summaries]
    go_summary_flags = [bool(summary["has_go_summary_block"]) for summary in format_summaries]
    format_valid_flags = [bool(summary["format_valid"]) for summary in format_summaries]
    alt_close_flags = [bool(summary["uses_alt_final_answer_close_tag"]) for summary in format_summaries]
    unclosed_final_answer_flags = [bool(summary["has_unclosed_final_answer_tag"]) for summary in format_summaries]
    repeated_final_answer_open_flags = [
        bool(summary["has_repeated_final_answer_open_tag"]) for summary in format_summaries
    ]
    tool_call_residue_flags = [bool(summary["has_tool_call_residue"]) for summary in format_summaries]
    think_residue_flags = [bool(summary["has_think_residue"]) for summary in format_summaries]
    parsed_go_counts = [int(summary["parsed_go_count"]) for summary in format_summaries]
    token_lengths: List[int] = []
    if tokenizer is not None:
        try:
            token_lengths = [len(tokenizer.encode(output, add_special_tokens=False)) for output in normalized_outputs]
        except Exception:
            token_lengths = []
    result = {
        "outputs": normalized_outputs,
        "output_count": len(normalized_outputs),
        "output_char_lengths": char_lengths,
        "output_word_lengths": word_lengths,
        "output_has_final_answer_tag": final_answer_flags,
        "output_has_go_summary_block": go_summary_flags,
        "output_format_valid": format_valid_flags,
        "output_uses_alt_final_answer_close_tag": alt_close_flags,
        "output_has_unclosed_final_answer_tag": unclosed_final_answer_flags,
        "output_has_repeated_final_answer_open_tag": repeated_final_answer_open_flags,
        "output_has_tool_call_residue": tool_call_residue_flags,
        "output_has_think_residue": think_residue_flags,
        "output_parsed_go_counts": parsed_go_counts,
        "output_length_summary": {
            "chars": summarize_length_values(char_lengths),
            "words": summarize_length_values(word_lengths),
        },
        "output_format_summary": {
            "final_answer_tag": summarize_boolean_values(final_answer_flags),
            "go_summary_block": summarize_boolean_values(go_summary_flags),
            "format_valid": summarize_boolean_values(format_valid_flags),
            "alt_final_answer_close_tag": summarize_boolean_values(alt_close_flags),
            "unclosed_final_answer_tag": summarize_boolean_values(unclosed_final_answer_flags),
            "repeated_final_answer_open_tag": summarize_boolean_values(repeated_final_answer_open_flags),
            "tool_call_residue": summarize_boolean_values(tool_call_residue_flags),
            "think_residue": summarize_boolean_values(think_residue_flags),
            "parsed_go_counts": summarize_length_values(parsed_go_counts),
        },
    }
    if token_lengths:
        result["output_token_lengths"] = token_lengths
        result["output_length_summary"]["tokens"] = summarize_length_values(token_lengths)
    return result


class RunTracker:
    def __init__(self, args: argparse.Namespace, config: Mapping[str, Any], output_dir: Path, runtime: DistributedRuntime) -> None:
        self.args = args
        self.config = dict(config)
        self.output_dir = output_dir
        self.runtime = runtime
        self.trace_path = build_trace_path(output_dir, args.trace_jsonl_name, runtime)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.wandb_run = None
        self.weave_client = None
        self.weave_project = ""
        self.weave_prompt_refs: Dict[str, str] = {}
        self.selected_weave_prompt_ref = ""
        self.weave_trace_fn = None
        self.weave_reward_trace_fn = None
        self.weave_scoring_trace_fn = None
        self.weave_update_trace_fn = None
        self._weave_stage_callbacks: Dict[str, Any] = {}
        self.wandb_entity = ""
        self.wandb_project = ""
        self.wandb_run_id = ""
        self.wandb_run_path = ""
        self.wandb_run_url = ""
        self.wandb_dir = (output_dir / "wandb").resolve()
        self.weave_remaining_budget = max(int(getattr(args, "weave_trace_budget", 0)), 0)
        self.weave_full_group_budget = max(int(getattr(args, "weave_trace_full_group_count", 0)), 0)
        self.weave_full_group_rollouts = max(int(getattr(args, "weave_trace_full_rollouts_per_group", 0)), 0)
        self.weave_groups_logged = 0
        self.weave_rollouts_logged = 0

        self.trace_path.write_text("", encoding="utf-8")
        self._configure_weave_cache_dir()

        if runtime.rank == 0:
            self.wandb_run = self._maybe_init_wandb(config)
            if self.wandb_run is not None:
                self._maybe_register_input_artifacts()
        self.weave_trace_fn = self._maybe_init_weave()

    def _configure_weave_cache_dir(self) -> Optional[str]:
        project = resolve_weave_project(self.args)
        if not project:
            return None
        cache_dir = Path(ensure_weave_server_cache_dir(self.output_dir))
        if self.runtime.world_size > 1:
            cache_dir = cache_dir / f"rank{self.runtime.rank:02d}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["WEAVE_SERVER_CACHE_DIR"] = str(cache_dir.resolve())
        return str(cache_dir.resolve())

    def _maybe_init_wandb(self, config: Mapping[str, Any]) -> Any:
        try:
            import wandb
        except ImportError:
            return None

        resolved_entity, resolved_project = resolve_wandb_identity(self.args, config)
        if resolved_entity:
            self.args.wandb_entity = resolved_entity
        if resolved_project:
            self.args.wandb_project = resolved_project
        self.wandb_entity = normalize_text(getattr(self.args, "wandb_entity", None)).strip()
        self.wandb_project = normalize_text(getattr(self.args, "wandb_project", None)).strip()
        self.wandb_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs = {
            "project": self.wandb_project,
            "config": dict(config),
            "name": self.args.run_name,
            "job_type": "train_rl",
            "dir": str(self.wandb_dir),
        }
        if self.wandb_entity:
            init_kwargs["entity"] = self.wandb_entity
        if normalize_text(self.args.wandb_mode).strip():
            init_kwargs["mode"] = self.args.wandb_mode
        run = wandb.init(**init_kwargs)
        self.wandb_entity = normalize_text(getattr(run, "entity", None)).strip() or self.wandb_entity
        self.wandb_project = normalize_text(getattr(run, "project", None)).strip() or self.wandb_project
        self.wandb_run_id = normalize_text(getattr(run, "id", None)).strip()
        self.wandb_run_path = normalize_path_value(getattr(run, "path", None)).strip()
        if not self.wandb_run_path and self.wandb_entity and self.wandb_project and self.wandb_run_id:
            self.wandb_run_path = f"{self.wandb_entity}/{self.wandb_project}/{self.wandb_run_id}"
        self.wandb_run_url = normalize_text(getattr(run, "url", None)).strip() or build_wandb_run_url(
            self.wandb_entity,
            self.wandb_project,
            self.wandb_run_id,
        )
        self._define_wandb_metrics(run, wandb)
        self._sync_wandb_run_config(run, config)
        self._write_wandb_run_info()
        startup_metrics = self._build_wandb_startup_metrics()
        if startup_metrics:
            run.log(self._augment_metrics_for_wandb(startup_metrics, step=0), step=0)
        summary = self.wandb_run_path or self.wandb_run_url or "<unknown>"
        print(f"[wandb] tracking train_rl run at {summary}", flush=True)
        return run

    def _define_wandb_metrics(self, run: Any, wandb_module: Any) -> None:
        define_metric = getattr(run, "define_metric", None)
        if not callable(define_metric):
            define_metric = getattr(wandb_module, "define_metric", None)
        if not callable(define_metric):
            return
        metric_specs = [
            ("train/global_step", {}),
            ("train/*", {"step_metric": "train/global_step"}),
            ("validation/*", {"step_metric": "train/global_step"}),
            ("timing/*", {"step_metric": "train/global_step"}),
            ("system/*", {"step_metric": "train/global_step"}),
        ]
        for metric_name, kwargs in metric_specs:
            try:
                define_metric(metric_name, **kwargs)
            except TypeError:
                define_metric(metric_name)
            except Exception:
                continue

    def _sync_wandb_run_config(self, run: Any, config: Mapping[str, Any]) -> None:
        from bioreason2.utils.tracking import sync_run_config

        resolved_config = dict(config)
        resolved_config.update(
            {
                "wandb_entity": self.wandb_entity,
                "wandb_project": self.wandb_project,
                "wandb_run_id": self.wandb_run_id,
                "wandb_run_path": self.wandb_run_path,
                "wandb_run_url": self.wandb_run_url,
                "wandb_dir": str(self.wandb_dir),
            }
        )
        sync_run_config(run, resolved_config)

    def _write_wandb_run_info(self) -> None:
        if not self.wandb_run_id and not self.wandb_run_path and not self.wandb_run_url:
            return
        payload = {
            "entity": self.wandb_entity,
            "project": self.wandb_project,
            "run_id": self.wandb_run_id,
            "run_path": self.wandb_run_path,
            "run_url": self.wandb_run_url,
            "run_name": normalize_text(self.args.run_name).strip() or "<auto>",
            "wandb_dir": str(self.wandb_dir),
        }
        save_json(self.output_dir / "wandb_run_info.json", payload)

    def _build_wandb_startup_metrics(self) -> Dict[str, float]:
        return {
            "system_wandb_initialized": 1.0,
            "system_weave_enabled": float(parse_bool(self.args.trace_rollouts_to_weave) and bool(self.weave_project)),
        }

    def _augment_metrics_for_wandb(self, metrics: Mapping[str, Any], step: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"train/global_step": int(step)}
        for key, value in metrics.items():
            if key == "step":
                continue
            if "/" in key:
                payload[key] = value
                continue
            if key.startswith("validation_"):
                payload[f"validation/{key[len('validation_'):]}"] = value
            elif key.startswith("timing_"):
                payload[f"timing/{key[len('timing_'):]}"] = value
            elif key.startswith("system_"):
                payload[f"system/{key[len('system_'):]}"] = value
            else:
                payload[f"train/{key}"] = value
        return payload

    def _maybe_init_weave(self) -> Any:
        if weave is None or not self.args.trace_rollouts_to_weave:
            return None
        if self.runtime.rank != 0:
            return None
        project = resolve_weave_project(self.args)
        if not project:
            return None
        try:
            from bioreason2.dataset.prompts.cafa5 import publish_cafa5_reasoning_prompts_to_weave

            self._configure_weave_cache_dir()
            global_attributes = {
                "job_type": "train_rl",
                "run_name": normalize_text(self.args.run_name).strip() or "<auto>",
                "rank": int(self.runtime.rank),
                "world_size": int(self.runtime.world_size),
            }
            self.weave_project = project
            self.weave_client = weave.init(project, global_attributes=global_attributes)
            selected_variant = normalize_text(getattr(self.args, "reasoning_prompt_style", None)).strip() or "paper_native_tight"
            self.weave_prompt_refs = publish_cafa5_reasoning_prompts_to_weave(
                weave,
                variants=[selected_variant],
            )
            self.selected_weave_prompt_ref = self.weave_prompt_refs.get(selected_variant, "")
            if self.wandb_run is not None and self.weave_prompt_refs:
                self._sync_wandb_run_config(
                    self.wandb_run,
                    {
                        **self.config,
                        "weave_prompt_refs_json": json.dumps(self.weave_prompt_refs, sort_keys=True),
                        "selected_weave_prompt_ref": self.selected_weave_prompt_ref,
                    },
                )

            def invoke_stage(stage_name: str) -> Any:
                callback = self._weave_stage_callbacks.get(stage_name)
                if not callable(callback):
                    raise RuntimeError(f"Weave stage tracing expected an active callback for stage={stage_name!r}.")
                return callback()

            @weave.op(name="train_rl_rollout_generate")
            def trace_generation(payload: Dict[str, Any]) -> Dict[str, Any]:
                rollout_result = dict(invoke_stage("rollout") or {})
                return {
                    "run_name": payload.get("run_name"),
                    "job_type": "train_rl",
                    "stage": "rollout",
                    "step": payload.get("step"),
                    "rank": payload.get("rank"),
                    "split": payload.get("split"),
                    "protein_id": payload.get("protein_id"),
                    "repeat_count": payload.get("repeat_count"),
                    "query": payload.get("query"),
                    "sampling": payload.get("sampling"),
                    **rollout_result,
                }

            @weave.op(name="train_rl_reward_score")
            def trace_reward(payload: Dict[str, Any]) -> Dict[str, Any]:
                rewards = [float(item) for item in invoke_stage("reward")]
                return {
                    "run_name": payload.get("run_name"),
                    "job_type": "train_rl",
                    "stage": "reward",
                    "step": payload.get("step"),
                    "rank": payload.get("rank"),
                    "split": payload.get("split"),
                    "protein_id": payload.get("protein_id"),
                    "target_go_ids": list(payload.get("target_go_ids") or []),
                    "completions": list(payload.get("completions") or []),
                    "rewards": rewards,
                }

            @weave.op(name="train_rl_old_ref_score")
            def trace_scoring(payload: Dict[str, Any]) -> Dict[str, Any]:
                summary = dict(invoke_stage("scoring") or {})
                return {
                    "run_name": payload.get("run_name"),
                    "job_type": "train_rl",
                    "stage": "old_ref_scoring",
                    "step": payload.get("step"),
                    "rank": payload.get("rank"),
                    "split": payload.get("split"),
                    "protein_id": payload.get("protein_id"),
                    **summary,
                }

            @weave.op(name="train_rl_policy_update")
            def trace_update(payload: Dict[str, Any]) -> Dict[str, Any]:
                summary = dict(invoke_stage("policy_update") or {})
                return {
                    "run_name": payload.get("run_name"),
                    "job_type": "train_rl",
                    "stage": "policy_update",
                    "step": payload.get("step"),
                    "rank": payload.get("rank"),
                    "split": payload.get("split"),
                    **summary,
                }

            self.weave_reward_trace_fn = trace_reward
            self.weave_scoring_trace_fn = trace_scoring
            self.weave_update_trace_fn = trace_update
            return trace_generation
        except Exception as exc:
            print(f"⚠️  Weave init failed for RL tracing: {exc}")
            self.weave_client = None
            self.weave_project = ""
            return None

    def _maybe_register_input_artifacts(self) -> None:
        from bioreason2.utils.tracking import maybe_use_artifact_refs

        maybe_use_artifact_refs(
            self.wandb_run,
            {
                "temporal_split_artifact": self.args.temporal_split_artifact,
                "dataset_artifact": self.args.dataset_artifact,
                "base_checkpoint": normalize_text(self.args.base_checkpoint).strip() or self.args.text_model_name,
            },
        )

    def log_metrics(self, metrics: Mapping[str, Any], step: int) -> None:
        if self.runtime.rank != 0:
            return
        if self.wandb_run is not None:
            self.wandb_run.log(self._augment_metrics_for_wandb(metrics, step=step), step=step)

    def claim_full_group_trace(self) -> bool:
        if not callable(self.weave_trace_fn) or self.weave_remaining_budget <= 0 or self.weave_full_group_budget <= 0:
            return False
        self.weave_full_group_budget -= 1
        self.weave_groups_logged += 1
        return True

    def _build_weave_attributes(
        self,
        *,
        stage: str,
        step: int,
        split: str,
        protein_id: str,
    ) -> Dict[str, Any]:
        return {
            "job_type": "train_rl",
            "stage": normalize_text(stage).strip(),
            "run_name": normalize_text(self.args.run_name).strip() or "<auto>",
            "step": int(step),
            "rank": int(self.runtime.rank),
            "split": normalize_text(split).strip() or "train",
            "protein_id": normalize_text(protein_id).strip(),
            "reasoning_prompt_style": normalize_text(getattr(self.args, "reasoning_prompt_style", None)).strip()
            or "paper_native_tight",
            "selected_weave_prompt_ref": self.selected_weave_prompt_ref,
        }

    def _trace_weave_stage(
        self,
        *,
        stage_name: str,
        weave_fn: Any,
        payload: Mapping[str, Any],
        callback: Any,
        attributes: Mapping[str, Any],
        decrement_rollout_budget: bool = False,
    ) -> Any:
        if not callable(weave_fn) or (decrement_rollout_budget and self.weave_remaining_budget <= 0):
            return callback()
        if self.runtime.world_size > 1 and stage_name in {"scoring", "policy_update"}:
            # Multi-node policy/scoring tracing has been unstable in practice and is
            # not required for prompt / output-format observability. Keep rollout
            # and reward traces in Weave, but run the heavier stages directly.
            return callback()
        weave_attributes = getattr(weave, "attributes", None)
        attribute_context = weave_attributes(dict(attributes)) if callable(weave_attributes) else nullcontext()
        self._weave_stage_callbacks[stage_name] = callback
        try:
            with attribute_context:
                result = weave_fn(dict(payload))
        finally:
            self._weave_stage_callbacks.pop(stage_name, None)
        if decrement_rollout_budget:
            self.weave_remaining_budget -= 1
            self.weave_rollouts_logged += 1
        return result

    def trace_rollout_call(
        self,
        *,
        step: int,
        split: str,
        query: PreparedQuery,
        repeat_count: int,
        sampling: SamplingSpec,
        generator: Any,
        tokenizer: Any = None,
    ) -> List[str]:
        if not callable(self.weave_trace_fn) or self.weave_remaining_budget <= 0:
            return build_rollout_trace_result(generator(), tokenizer=tokenizer)["outputs"]

        protein_id = normalize_text(query.sample_meta.get("protein_id")).strip()
        call_payload = {
            "run_name": normalize_text(self.args.run_name).strip() or "<auto>",
            "job_type": "train_rl",
            "stage": "rollout",
            "step": int(step),
            "rank": int(self.runtime.rank),
            "split": normalize_text(split).strip() or "train",
            "protein_id": protein_id,
            "repeat_count": int(repeat_count),
            "query": {
                "prompt_text": normalize_text(query.prompt_text),
                "go_aspects": [normalize_text(aspect) for aspect in query.go_aspects],
                "sample_meta": traceable_sample_meta(
                    query.sample_meta,
                    allowed_keys=ROLLOUT_TRACE_SAMPLE_META_KEYS,
                ),
            },
            "sampling": asdict(sampling),
        }
        result = self._trace_weave_stage(
            stage_name="rollout",
            weave_fn=self.weave_trace_fn,
            payload=call_payload,
            callback=lambda: build_rollout_trace_result(generator(), tokenizer=tokenizer),
            attributes=self._build_weave_attributes(
                stage="rollout",
                step=step,
                split=split,
                protein_id=protein_id,
            ),
            decrement_rollout_budget=True,
        )
        return [normalize_text(output).strip() for output in result.get("outputs", [])]

    def trace_reward_call(
        self,
        *,
        step: int,
        split: str,
        query: PreparedQuery,
        completions: Sequence[str],
        callback: Any,
    ) -> List[float]:
        protein_id = normalize_text(query.sample_meta.get("protein_id")).strip()
        payload = {
            "run_name": normalize_text(self.args.run_name).strip() or "<auto>",
            "job_type": "train_rl",
            "stage": "reward",
            "step": int(step),
            "rank": int(self.runtime.rank),
            "split": normalize_text(split).strip() or "train",
            "protein_id": protein_id,
            "target_go_ids": build_target_go_ids(query.sample_meta),
            "completions": [normalize_text(completion).strip() for completion in completions],
        }
        result = self._trace_weave_stage(
            stage_name="reward",
            weave_fn=self.weave_reward_trace_fn,
            payload=payload,
            callback=callback,
            attributes=self._build_weave_attributes(
                stage="reward",
                step=step,
                split=split,
                protein_id=protein_id,
            ),
        )
        if isinstance(result, Mapping):
            return [float(item) for item in result.get("rewards", [])]
        return [float(item) for item in result]

    def trace_scoring_call(
        self,
        *,
        step: int,
        split: str,
        query: PreparedQuery,
        payload: Mapping[str, Any],
        callback: Any,
    ) -> Mapping[str, Any]:
        protein_id = normalize_text(query.sample_meta.get("protein_id")).strip()
        stage_payload = {
            "run_name": normalize_text(self.args.run_name).strip() or "<auto>",
            "job_type": "train_rl",
            "stage": "old_ref_scoring",
            "step": int(step),
            "rank": int(self.runtime.rank),
            "split": normalize_text(split).strip() or "train",
            "protein_id": protein_id,
            **dict(payload),
        }
        result = self._trace_weave_stage(
            stage_name="scoring",
            weave_fn=self.weave_scoring_trace_fn,
            payload=stage_payload,
            callback=callback,
            attributes=self._build_weave_attributes(
                stage="old_ref_scoring",
                step=step,
                split=split,
                protein_id=protein_id,
            ),
        )
        return dict(result or {})

    def trace_policy_update_call(
        self,
        *,
        step: int,
        split: str,
        callback: Any,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        stage_payload = {
            "run_name": normalize_text(self.args.run_name).strip() or "<auto>",
            "job_type": "train_rl",
            "stage": "policy_update",
            "step": int(step),
            "rank": int(self.runtime.rank),
            "split": normalize_text(split).strip() or "train",
            **dict(payload),
        }
        result = self._trace_weave_stage(
            stage_name="policy_update",
            weave_fn=self.weave_update_trace_fn,
            payload=stage_payload,
            callback=callback,
            attributes=self._build_weave_attributes(
                stage="policy_update",
                step=step,
                split=split,
                protein_id="",
            ),
        )
        return dict(result or {})

    def log_rollout_trace(self, payload: Mapping[str, Any], trace_to_weave: bool = True) -> None:
        payload_dict = dict(payload)
        payload_dict.setdefault("run_name", normalize_text(self.args.run_name).strip() or "<auto>")
        payload_dict.setdefault("job_type", "train_rl")
        if self.weave_project:
            payload_dict.setdefault("weave_project", self.weave_project)
        if self.weave_prompt_refs:
            payload_dict.setdefault("weave_prompt_refs", dict(self.weave_prompt_refs))
        if self.selected_weave_prompt_ref:
            payload_dict.setdefault("selected_weave_prompt_ref", self.selected_weave_prompt_ref)
        line = json.dumps(payload_dict, ensure_ascii=True)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        _ = trace_to_weave

    def log_checkpoint_artifact(self, checkpoint_dir: Path, aliases: Sequence[str], metadata: Mapping[str, Any]) -> None:
        if self.runtime.rank != 0 or self.wandb_run is None:
            return
        try:
            import wandb
        except ImportError:
            return
        artifact = wandb.Artifact(
            normalize_text(self.args.checkpoint_artifact_name).strip() or "train-rl-output",
            type="model",
            metadata={
                **dict(metadata),
                "wandb_run_id": self.wandb_run_id,
                "wandb_run_path": self.wandb_run_path,
                "wandb_run_url": self.wandb_run_url,
            },
        )
        artifact.add_dir(str(checkpoint_dir))
        self.wandb_run.log_artifact(artifact, aliases=list(aliases))

    def finish(self) -> None:
        if self.weave_client is not None:
            flush = getattr(self.weave_client, "flush", None)
            if callable(flush):
                try:
                    flush()
                except Exception:
                    pass
        if self.runtime.rank == 0 and self.wandb_run is not None:
            self.wandb_run.finish()


def load_reasoning_datasets(args: argparse.Namespace, runtime: DistributedRuntime) -> Tuple[Any, Any]:
    from bioreason2.dataset.cafa5.load import load_cafa5_dataset

    requested_num_proc = resolve_dataset_num_proc(args.dataset_num_proc)
    effective_num_proc = resolve_effective_dataset_num_proc(args.dataset_num_proc, distributed=runtime.enabled)
    if runtime.rank == 0 and effective_num_proc != requested_num_proc:
        print(
            "Reducing dataset_num_proc to 1 for distributed preprocessing to avoid pyarrow mmap worker bus errors.",
            flush=True,
        )

    train_dataset, validation_dataset, _ = load_cafa5_dataset(
        dataset=args.cafa5_dataset,
        dataset_name=args.reasoning_dataset_name,
        cache_dir=args.dataset_cache_dir,
        structure_dir=args.structure_dir,
        num_proc=effective_num_proc,
        return_as_chat_template=True,
        include_go_defs=False,
        interpro_dataset_name=args.interpro_dataset_name,
        split_go_aspects=False,
        include_protein_function_summary=True,
        interpro_in_prompt=True,
        ppi_in_prompt=True,
        predict_interpro=False,
        reasoning_dataset_name=args.reasoning_dataset_name,
        go_gpt_predictions_column=args.go_gpt_predictions_column,
        include_ground_truth_in_final_answer=False,
        add_uniprot_summary=False,
        is_swissprot=False,
        reasoning_prompt_style=normalize_text(args.reasoning_prompt_style).strip() or "paper_native",
    )
    return train_dataset, validation_dataset


def disable_model_dropout(module: Any) -> None:
    require_torch()
    for child in module.modules():
        if isinstance(child, torch.nn.Dropout):
            child.p = 0.0


def apply_lora_to_text_model(model: Any, args: argparse.Namespace, trainable: bool) -> None:
    from peft import LoraConfig, get_peft_model
    from bioreason2.models.protein_llm import _get_target_modules

    lora_config = LoraConfig(
        r=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=_get_target_modules(model),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.text_model = get_peft_model(model.text_model, lora_config)
    if trainable:
        model.text_model.train()
    else:
        model.text_model.eval()
        for param in model.text_model.parameters():
            param.requires_grad = False


def instantiate_policy_model_from_source(
    args: argparse.Namespace,
    *,
    text_model_name: str,
    trainable: bool,
) -> Any:
    from bioreason2.models.protein_llm import ProteinLLMModel

    model = ProteinLLMModel(
        text_model_name=text_model_name,
        protein_model_name=args.protein_model_name,
        cache_dir=args.cache_dir,
        max_length_protein=int(args.max_length_protein),
        max_length_text=int(args.max_length_text),
        text_model_finetune=True,
        protein_model_finetune=False,
        protein_embedding_layer=int(args.protein_embedding_layer),
        go_model_finetune=False,
        attn_implementation=normalize_text(args.attn_implementation).strip() or "auto",
        go_obo_path=args.go_obo_path,
        precomputed_embeddings_path=args.precomputed_embeddings_path,
        go_hidden_dim=int(args.go_hidden_dim),
        go_num_gat_layers=int(args.go_num_gat_layers),
        go_num_heads=int(args.go_num_heads),
        go_num_reduced_embeddings=int(args.go_num_reduced_embeddings),
        go_embedding_dim=int(args.go_embedding_dim),
        quantization_config=None,
        load_in_4bit=False,
        unified_go_encoder=bool(args.unified_go_encoder),
        use_unsloth=False,
        lazy_protein_encoder=True,
    )
    apply_lora_to_text_model(model, args, trainable=trainable)
    if args.disable_model_dropout:
        disable_model_dropout(model.text_model)
    if args.gradient_checkpointing and trainable:
        enable_gc = getattr(model.text_model, "gradient_checkpointing_enable", None)
        if callable(enable_gc):
            try:
                enable_gc(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                enable_gc()
        config = getattr(model.text_model, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
    if not trainable:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    return model


def instantiate_policy_model(args: argparse.Namespace, trainable: bool) -> Any:
    return instantiate_policy_model_from_source(
        args,
        text_model_name=args.text_model_name,
        trainable=trainable,
    )


def build_deepspeed_config(args: argparse.Namespace, runtime_spec: RuntimeSpec) -> Dict[str, Any]:
    return {
        "train_micro_batch_size_per_gpu": runtime_spec.optimizer_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": runtime_spec.gradient_accumulation_steps,
        "gradient_clipping": float(args.max_grad_norm),
        "bf16": {"enabled": bool(runtime_spec.bf16)},
        "zero_optimization": {
            "stage": int(runtime_spec.zero_stage),
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "steps_per_print": 1,
    }


@dataclass
class PolicyStack:
    engine: Any
    tokenizer: Any
    pad_token_id: int
    reference_checkpoint_dir: Path
    rollout_checkpoint_dir: Path


def resolve_checkpoint_dir(value: Any) -> Path:
    checkpoint_dir = Path(normalize_text(value).strip()).expanduser()
    return checkpoint_dir.resolve()


def initialize_policy_stack(
    args: argparse.Namespace,
    runtime_spec: RuntimeSpec,
    runtime: DistributedRuntime,
) -> PolicyStack:
    require_torch()
    import deepspeed
    from transformers import get_cosine_schedule_with_warmup

    maybe_stagger_startup_model_load(runtime)
    current_model = instantiate_policy_model(args, trainable=True)

    trainable_parameters = [param for param in current_model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
        eps=float(args.adam_epsilon),
    )
    total_optimizer_steps = int(args.max_steps) * int(args.steps_per_generation)
    warmup_steps = max(int(total_optimizer_steps * float(args.warmup_ratio)), 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_optimizer_steps, 1),
    )
    engine, _, _, _ = deepspeed.initialize(
        model=current_model,
        model_parameters=trainable_parameters,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=build_deepspeed_config(args, runtime_spec),
        dist_init_required=bool(runtime.enabled),
    )
    tokenizer = engine.module.text_tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    base_checkpoint_dir = resolve_checkpoint_dir(args.text_model_name)
    return PolicyStack(
        engine=engine,
        tokenizer=tokenizer,
        pad_token_id=int(pad_token_id),
        reference_checkpoint_dir=base_checkpoint_dir,
        rollout_checkpoint_dir=base_checkpoint_dir,
    )


def unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)


def move_model_to_device(model: Any, device: Any) -> Any:
    if hasattr(model, "to"):
        model.to(device)
    return model


def offload_model_to_cpu(model: Any) -> Any:
    if hasattr(model, "to"):
        model.to("cpu")
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def cleanup_policy_model(model: Any) -> None:
    if model is None:
        return
    offload_model_to_cpu(model)
    del model
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_frozen_scoring_model(args: argparse.Namespace, checkpoint_dir: Path, runtime: DistributedRuntime) -> Any:
    model = instantiate_policy_model_from_source(
        args,
        text_model_name=str(checkpoint_dir),
        trainable=False,
    )
    return move_model_to_device(model, runtime.device)


def build_single_example_batch(example: Mapping[str, Any], model: Any) -> Dict[str, Any]:
    from bioreason2.dataset.cafa5.collate import qwen_protein_collate_fn

    return qwen_protein_collate_fn(
        [dict(example)],
        processor=unwrap_model(model).processor,
        max_length_text=int(unwrap_model(model).max_length_text),
        max_length_protein=int(unwrap_model(model).max_length_protein),
        return_answer_in_batch=False,
        inference_mode=True,
    )


def extract_single_query(batch: Mapping[str, Any], model: Any, device: Any) -> PreparedQuery:
    require_torch()
    prompt_mask = batch["attention_mask"][0].bool()
    input_ids = batch["input_ids"][0][prompt_mask].unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    structure_coords = batch.get("structure_coords")
    if isinstance(structure_coords, torch.Tensor):
        structure_coords = structure_coords[0:1].to(device)
    query = PreparedQuery(
        input_ids=input_ids,
        attention_mask=attention_mask,
        protein_sequences=list(batch.get("protein_sequences") or []),
        batch_idx_map=[0 for _ in list(batch.get("protein_sequences") or [])],
        structure_coords=structure_coords,
        go_aspects=[normalize_text((batch.get("batch_go_aspects") or ["all"])[0]).strip() or "all"],
        sample_meta=build_query_sample_meta(batch),
        prompt_text=normalize_text((batch.get("prompt") or [""])[0]),
    )
    query.multimodal_cache = unwrap_model(model).build_multimodal_cache(
        protein_sequences=query.protein_sequences,
        batch_idx_map=query.batch_idx_map,
        batch_size=1,
        structure_coords=query.structure_coords,
        go_aspects=query.go_aspects,
    )
    query.multimodal_cache = align_multimodal_cache_to_input_ids(
        query.multimodal_cache,
        query.input_ids,
        unwrap_model(model),
    )
    return query


def repeat_multimodal_cache(cache: Optional[Dict[str, Any]], repeat_count: int) -> Optional[Dict[str, Any]]:
    if cache is None:
        return None
    if repeat_count <= 0:
        raise ValueError(f"repeat_count must be positive, got {repeat_count}")
    protein_embeddings = cache.get("protein_embeddings")
    go_embeddings = cache.get("go_embeddings")
    expanded: Dict[str, Any] = {
        "batch_size": repeat_count,
        "protein_embeddings": None,
        "go_embeddings": None,
    }
    if protein_embeddings:
        expanded["protein_embeddings"] = [protein_embeddings[0] for _ in range(repeat_count)]
    if go_embeddings:
        expanded["go_embeddings"] = [go_embeddings[0] for _ in range(repeat_count)]
    return expanded


def move_multimodal_cache_to_cpu(cache: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if cache is None:
        return None

    def move_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, list):
            return [move_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move_value(item) for item in value)
        if isinstance(value, dict):
            return {key: move_value(item) for key, item in value.items()}
        return value

    return {key: move_value(value) for key, value in cache.items()}


def align_multimodal_cache_to_input_ids(
    cache: Optional[Dict[str, Any]],
    input_ids: Any,
    model: Any,
) -> Optional[Dict[str, Any]]:
    require_torch()
    if cache is None:
        return None
    protein_token_id = getattr(model, "protein_token_id", None)
    go_token_id = getattr(model, "go_token_id", None)
    aligned = dict(cache)

    protein_embeddings = cache.get("protein_embeddings")
    if protein_embeddings is not None and protein_token_id is not None:
        protein_counts = (input_ids == int(protein_token_id)).sum(dim=1).tolist()
        trimmed_embeddings: List[Any] = []
        for emb, count in zip(protein_embeddings, protein_counts):
            if int(count) <= 0:
                trimmed_embeddings.append(emb[:0])
            else:
                trimmed_embeddings.append(emb[: int(count)])
        aligned["protein_embeddings"] = trimmed_embeddings

    go_embeddings = cache.get("go_embeddings")
    if go_embeddings is not None and go_token_id is not None:
        go_counts = (input_ids == int(go_token_id)).sum(dim=1).tolist()
        trimmed_go_embeddings: List[Any] = []
        for emb, count in zip(go_embeddings, go_counts):
            if int(count) <= 0:
                trimmed_go_embeddings.append(emb[:0])
            else:
                trimmed_go_embeddings.append(emb[: int(count)])
        aligned["go_embeddings"] = trimmed_go_embeddings

    return aligned


def repeat_query_for_rollouts(query: PreparedQuery, repeat_count: int, device: Any) -> Dict[str, Any]:
    require_torch()
    input_ids = query.input_ids.repeat(repeat_count, 1).to(device)
    attention_mask = query.attention_mask.repeat(repeat_count, 1).to(device)

    structure_coords = query.structure_coords
    if isinstance(structure_coords, torch.Tensor):
        repeat_dims = [repeat_count] + [1] * max(structure_coords.dim() - 1, 0)
        structure_coords = structure_coords.repeat(*repeat_dims).to(device)

    protein_sequences: List[str] = []
    batch_idx_map: List[int] = []
    for rollout_idx in range(repeat_count):
        protein_sequences.extend(query.protein_sequences)
        batch_idx_map.extend([rollout_idx] * len(query.protein_sequences))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "protein_sequences": protein_sequences,
        "batch_idx_map": batch_idx_map,
        "structure_coords": structure_coords,
        "go_aspects": query.go_aspects * repeat_count,
        "multimodal_cache": repeat_multimodal_cache(query.multimodal_cache, repeat_count),
    }


def build_rollout_multimodal_cache(model: Any, query: PreparedQuery, repeat_count: int) -> Optional[Dict[str, Any]]:
    if query.multimodal_cache is not None:
        repeated_cache = repeat_multimodal_cache(query.multimodal_cache, repeat_count)
        repeated_input_ids = query.input_ids.repeat(repeat_count, 1)
        return align_multimodal_cache_to_input_ids(repeated_cache, repeated_input_ids, model)
    build_cache = getattr(model, "build_multimodal_cache", None)
    if not callable(build_cache):
        repeated_cache = repeat_multimodal_cache(query.multimodal_cache, repeat_count)
        repeated_input_ids = query.input_ids.repeat(repeat_count, 1)
        return align_multimodal_cache_to_input_ids(repeated_cache, repeated_input_ids, model)
    single_cache = build_cache(
        protein_sequences=query.protein_sequences,
        batch_idx_map=query.batch_idx_map,
        batch_size=1,
        structure_coords=query.structure_coords,
        go_aspects=query.go_aspects,
    )
    repeated_cache = repeat_multimodal_cache(single_cache, repeat_count)
    repeated_input_ids = query.input_ids.repeat(repeat_count, 1)
    return align_multimodal_cache_to_input_ids(repeated_cache, repeated_input_ids, model)


def tokenize_completion_texts(tokenizer: Any, completions: Sequence[str], device: Any) -> List[Any]:
    require_torch()
    encoded: List[Any] = []
    for completion in completions:
        token_ids = tokenizer.encode(normalize_text(completion), add_special_tokens=False)
        encoded.append(torch.tensor(token_ids, dtype=torch.long, device=device))
    return encoded


def build_scoring_batch(
    query: PreparedQuery,
    completion_ids: Sequence[Any],
    pad_token_id: int,
    device: Any,
    model: Any,
) -> Dict[str, Any]:
    require_torch()

    prompt_ids = query.input_ids[0].to(device)
    prompt_len = int(prompt_ids.numel())
    max_completion_len = max((int(item.numel()) for item in completion_ids), default=0)
    total_len = prompt_len + max_completion_len

    batch_size = len(completion_ids)
    input_ids = torch.full((batch_size, total_len), int(pad_token_id), dtype=prompt_ids.dtype, device=device)
    attention_mask = torch.zeros((batch_size, total_len), dtype=query.attention_mask.dtype, device=device)
    completion_mask = torch.zeros((batch_size, total_len - 1), dtype=torch.float32, device=device)

    for row_idx, completion in enumerate(completion_ids):
        completion = completion.to(device)
        row_len = prompt_len + int(completion.numel())
        input_ids[row_idx, :prompt_len] = prompt_ids
        attention_mask[row_idx, :row_len] = 1
        if completion.numel() > 0:
            input_ids[row_idx, prompt_len:row_len] = completion
            start_idx = max(prompt_len - 1, 0)
            end_idx = start_idx + int(completion.numel())
            completion_mask[row_idx, start_idx:end_idx] = 1.0

    structure_coords = query.structure_coords
    if isinstance(structure_coords, torch.Tensor):
        repeat_dims = [batch_size] + [1] * max(structure_coords.dim() - 1, 0)
        structure_coords = structure_coords.repeat(*repeat_dims).to(device)

    protein_sequences: List[str] = []
    batch_idx_map: List[int] = []
    for row_idx in range(batch_size):
        protein_sequences.extend(query.protein_sequences)
        batch_idx_map.extend([row_idx] * len(query.protein_sequences))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "protein_sequences": protein_sequences,
        "batch_idx_map": batch_idx_map,
        "structure_coords": structure_coords,
        "go_aspects": query.go_aspects * batch_size,
        "multimodal_cache": align_multimodal_cache_to_input_ids(
            repeat_multimodal_cache(query.multimodal_cache, batch_size),
            input_ids,
            model,
        ),
    }


def compute_sequence_log_probs(
    model: Any,
    query: PreparedQuery,
    completion_ids: Sequence[Any],
    pad_token_id: int,
    device: Any,
    microbatch_size: int = 0,
) -> Any:
    require_torch()

    def _compute(subgroup_completion_ids: Sequence[Any]) -> Any:
        scoring_batch = build_scoring_batch(query, subgroup_completion_ids, pad_token_id, device, model)
        outputs = model(
            input_ids=scoring_batch["input_ids"],
            attention_mask=scoring_batch["attention_mask"],
            protein_sequences=scoring_batch["protein_sequences"],
            batch_idx_map=scoring_batch["batch_idx_map"],
            structure_coords=scoring_batch["structure_coords"],
            go_aspects=scoring_batch["go_aspects"],
            multimodal_cache=scoring_batch["multimodal_cache"],
        )
        logits = outputs.logits[:, :-1, :]
        targets = scoring_batch["input_ids"][:, 1:]
        token_log_probs = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return (token_log_probs * scoring_batch["completion_mask"]).sum(dim=1)

    total_rollouts = len(completion_ids)
    if total_rollouts == 0:
        return torch.zeros((0,), dtype=torch.float32, device=device)
    if microbatch_size <= 0 or total_rollouts <= microbatch_size:
        return _compute(completion_ids)

    sequence_chunks: List[Any] = []
    for start_idx in range(0, total_rollouts, microbatch_size):
        end_idx = min(total_rollouts, start_idx + microbatch_size)
        sequence_chunks.append(_compute(completion_ids[start_idx:end_idx]))
    return torch.cat(sequence_chunks, dim=0)


def compute_chunk_loss(
    current_model: Any,
    query: PreparedQuery,
    completion_ids: Sequence[Any],
    advantages: Any,
    old_log_probs: Any,
    ref_log_probs: Any,
    algorithm: AlgorithmSpec,
    pad_token_id: int,
    device: Any,
    logprob_microbatch_size: int = 0,
) -> Tuple[Any, Dict[str, float]]:
    require_torch()
    current_log_probs = compute_sequence_log_probs(
        current_model,
        query,
        completion_ids,
        pad_token_id,
        device,
        microbatch_size=logprob_microbatch_size,
    )
    ratios = torch.exp(current_log_probs - old_log_probs)
    if algorithm.importance_sampling_cap > 0:
        ratios = ratios.clamp(max=float(algorithm.importance_sampling_cap))
    clipped_ratios = ratios.clamp(
        min=1.0 - float(algorithm.clip_epsilon_low),
        max=1.0 + float(algorithm.clip_epsilon_high),
    )
    surrogate = torch.minimum(ratios * advantages, clipped_ratios * advantages)
    kl = current_log_probs - ref_log_probs
    loss = (
        -(surrogate.sum() / algorithm.policy_denominator)
        + (float(algorithm.kl_beta) * kl.sum() / algorithm.kl_denominator)
    )
    metrics = {
        "ratio_mean": float(ratios.detach().mean().item()),
        "ratio_max": float(ratios.detach().max().item()),
        "kl_mean": float(kl.detach().mean().item()),
        "policy_objective_mean": float(surrogate.detach().mean().item()),
    }
    return loss, metrics


def prepare_group_for_loss(
    group: RolloutGroup,
    *,
    global_reward_std: float,
    query_group_mean: float,
    runtime_device: Any,
    max_loss_completion_tokens: int,
) -> Dict[str, Any]:
    require_torch()
    full_advantages = torch.tensor(
        compute_group_advantages(group.rewards, global_reward_std, group_mean=query_group_mean),
        dtype=torch.float32,
        device=runtime_device,
    )
    selected_indices = select_rollout_indices_for_loss(
        group.completion_ids,
        max_loss_completion_tokens=max_loss_completion_tokens,
    )
    group.filtered_rollouts = float(len(group.completion_ids) - len(selected_indices))
    if not selected_indices:
        group.selected_completion_ids = []
        group.advantages = torch.zeros((0,), dtype=torch.float32, device=runtime_device)
        group.old_log_probs = torch.zeros((0,), dtype=torch.float32, device=runtime_device)
        group.ref_log_probs = torch.zeros((0,), dtype=torch.float32, device=runtime_device)
        return {
            "raw_rollout_count": len(group.completion_ids),
            "selected_rollout_count": 0,
            "filtered_rollout_count": float(group.filtered_rollouts),
        }

    index_tensor = torch.tensor(selected_indices, dtype=torch.long, device=runtime_device)
    group.selected_completion_ids = [group.completion_ids[idx] for idx in selected_indices]
    group.advantages = full_advantages.index_select(0, index_tensor)
    group.old_log_probs = torch.zeros((len(group.selected_completion_ids),), dtype=torch.float32, device=runtime_device)
    group.ref_log_probs = torch.zeros((len(group.selected_completion_ids),), dtype=torch.float32, device=runtime_device)
    return {
        "raw_rollout_count": len(group.completion_ids),
        "selected_rollout_count": len(group.selected_completion_ids),
        "filtered_rollout_count": float(group.filtered_rollouts),
    }


def score_group_log_probs(
    *,
    policy_model: Any,
    group: RolloutGroup,
    pad_token_id: int,
    device: Any,
    logprob_microbatch_size: int,
    policy_role: str,
) -> Dict[str, Any]:
    require_torch()
    selected_completion_ids = list(group.selected_completion_ids or [])
    if not selected_completion_ids:
        return {
            "policy_role": normalize_text(policy_role).strip() or "unknown",
            "selected_rollout_count": 0,
            "log_prob_mean": 0.0,
        }
    with torch.no_grad():
        log_probs = compute_sequence_log_probs(
            policy_model,
            group.query,
            selected_completion_ids,
            pad_token_id,
            device,
            microbatch_size=logprob_microbatch_size,
        ).detach()
    if normalize_text(policy_role).strip() == "old":
        group.old_log_probs = log_probs
    else:
        group.ref_log_probs = log_probs
    return {
        "policy_role": normalize_text(policy_role).strip() or "unknown",
        "selected_rollout_count": len(selected_completion_ids),
        "log_prob_mean": float(log_probs.mean().item()),
    }


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)


def export_inference_checkpoint(model: Any, export_dir: Path) -> None:
    require_torch()
    export_dir = export_dir.resolve()
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=False)

    base_model = copy.deepcopy(unwrap_model(model)).cpu()
    text_model = base_model.text_model
    if hasattr(text_model, "merge_and_unload"):
        text_model = text_model.merge_and_unload()
        base_model.text_model = text_model

    base_model.text_model.save_pretrained(export_dir, safe_serialization=True)
    base_model.text_tokenizer.save_pretrained(export_dir)
    torch.save(base_model.protein_projection.state_dict(), export_dir / "protein_projection.pt")
    if getattr(base_model, "go_projection", None) is not None:
        torch.save(base_model.go_projection.state_dict(), export_dir / "go_projection.pt")
    if getattr(base_model, "go_encoder", None) is not None:
        torch.save(base_model.go_encoder.state_dict(), export_dir / "go_encoder.pt")
    protein_model_dir = export_dir / "protein_model"
    protein_model_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(base_model, "copy_or_save_frozen_protein_model"):
        base_model.copy_or_save_frozen_protein_model(protein_model_dir)
    else:
        torch.save(base_model.protein_model.state_dict(), protein_model_dir / "pytorch_model.bin")
    del base_model
    gc.collect()


def build_rollout_query_payload(query: PreparedQuery) -> Dict[str, Any]:
    structure_coords = query.structure_coords
    if isinstance(structure_coords, torch.Tensor):
        structure_coords = structure_coords.detach().cpu()
    multimodal_cache = query.multimodal_cache
    if multimodal_cache is not None:
        multimodal_cache = move_multimodal_cache_to_cpu(multimodal_cache)
    return {
        "input_ids": query.input_ids.detach().cpu(),
        "attention_mask": query.attention_mask.detach().cpu(),
        "protein_sequences": list(query.protein_sequences),
        "batch_idx_map": list(query.batch_idx_map),
        "structure_coords": structure_coords,
        "go_aspects": list(query.go_aspects),
        "multimodal_cache": multimodal_cache,
    }


def query_from_rollout_payload(payload: Mapping[str, Any]) -> PreparedQuery:
    return PreparedQuery(
        input_ids=payload["input_ids"],
        attention_mask=payload["attention_mask"],
        protein_sequences=list(payload.get("protein_sequences") or []),
        batch_idx_map=list(payload.get("batch_idx_map") or []),
        structure_coords=payload.get("structure_coords"),
        go_aspects=list(payload.get("go_aspects") or []),
        sample_meta={},
        prompt_text="",
        multimodal_cache=payload.get("multimodal_cache"),
    )


def create_vllm_rollout_model(args: argparse.Namespace, checkpoint_dir: Path) -> Any:
    from bioreason2.models.protein_vllm import ProteinLLMModel as VLLMProteinLLMModel
    effective_max_num_seqs = resolve_effective_vllm_max_num_seqs(args)

    return VLLMProteinLLMModel(
        ckpt_dir=str(checkpoint_dir),
        text_model_name=str(checkpoint_dir),
        protein_model_name=args.protein_model_name,
        cache_dir=args.cache_dir,
        max_length_protein=int(args.max_length_protein),
        max_length_text=int(args.max_length_text),
        protein_embedding_layer=int(args.protein_embedding_layer),
        go_obo_path=args.go_obo_path,
        precomputed_embeddings_path=args.precomputed_embeddings_path,
        go_hidden_dim=int(args.go_hidden_dim),
        go_num_gat_layers=int(args.go_num_gat_layers),
        go_num_heads=int(args.go_num_heads),
        go_num_reduced_embeddings=int(args.go_num_reduced_embeddings),
        go_embedding_dim=int(args.go_embedding_dim),
        unified_go_encoder=bool(args.unified_go_encoder),
        gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
        max_model_len=int(args.vllm_max_model_len),
        max_num_seqs=effective_max_num_seqs,
        cpu_offload_gb=float(args.vllm_cpu_offload_gb),
        swap_space=float(args.vllm_swap_space_gb),
        enforce_eager=bool(args.vllm_enforce_eager),
        enable_sleep_mode=resolve_effective_vllm_sleep_mode(args),
        tensor_parallel_size=1,
        attention_backend=normalize_text(args.vllm_attention_backend).strip() or None,
        worker_multiproc_method=normalize_text(args.vllm_worker_multiproc_method).strip() or "spawn",
        use_v1=bool(args.vllm_use_v1),
    )


def cleanup_vllm_rollout_model(model: Any) -> None:
    if model is None:
        return
    try:
        shutdown = getattr(model, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        pass
    del model
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def rollout_worker_process_main(connection: Any, bootstrap: Mapping[str, Any]) -> None:
    model = None
    sleeping = False
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = normalize_text(bootstrap.get("cuda_visible_device")).strip()
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        args = argparse.Namespace(**dict(bootstrap.get("args", {})))
        runtime_rank = int(bootstrap.get("runtime_rank", 0))
        checkpoint_dir = Path(bootstrap["checkpoint_dir"])
        vllm_port = resolve_rollout_worker_vllm_port(args, runtime_rank)
        os.environ["VLLM_PORT"] = str(vllm_port)
        vllm_host_ip = normalize_text(getattr(args, "rollout_worker_vllm_host_ip", "127.0.0.1")).strip() or "127.0.0.1"
        os.environ["VLLM_HOST_IP"] = vllm_host_ip
        print(
            (
                "[rollout-worker bootstrap] "
                f"rank={runtime_rank} cuda={normalize_text(bootstrap.get('cuda_visible_device')).strip() or 'unknown'} "
                f"VLLM_PORT={vllm_port} VLLM_HOST_IP={vllm_host_ip}"
            ),
            flush=True,
        )

        def load_model(checkpoint_path: Path) -> Any:
            nonlocal model, sleeping
            cleanup_vllm_rollout_model(model)
            model = create_vllm_rollout_model(args, checkpoint_path)
            sleeping = False
            return model

        load_model(checkpoint_dir)
        connection.send({"status": "ready"})

        while True:
            message = connection.recv()
            command = normalize_text(message.get("cmd")).strip()
            if command == "generate":
                if sleeping and hasattr(model, "wake_up"):
                    model.wake_up()
                    sleeping = False
                query = query_from_rollout_payload(message["query"])
                sampling = SamplingSpec(**dict(message["sampling"]))
                rollout_batch = repeat_query_for_rollouts(query, int(message["repeat_count"]), query.input_ids.device)
                rollout_multimodal_cache = build_rollout_multimodal_cache(
                    model,
                    query,
                    int(message["repeat_count"]),
                )
                outputs = model.generate(
                    input_ids=rollout_batch["input_ids"],
                    attention_mask=rollout_batch["attention_mask"],
                    protein_sequences=None,
                    batch_idx_map=None,
                    structure_coords=None,
                    go_aspects=None,
                    multimodal_cache=rollout_multimodal_cache,
                    temperature=float(sampling.temperature),
                    top_k=int(sampling.top_k),
                    top_p=float(sampling.top_p),
                    min_p=float(sampling.min_p),
                    repetition_penalty=float(sampling.repetition_penalty),
                    max_new_tokens=int(sampling.max_new_tokens),
                    seed=int(message.get("seed", getattr(args, "seed", 0))),
                    stop=ROLLOUT_STOP_MARKERS,
                )
                if resolve_effective_vllm_sleep_mode(args) and hasattr(model, "sleep"):
                    model.sleep(level=int(args.vllm_sleep_level))
                    sleeping = True
                connection.send({"status": "ok", "outputs": [normalize_text(output).strip() for output in outputs]})
                continue
            if command == "refresh":
                load_model(Path(message["checkpoint_dir"]))
                connection.send({"status": "ok"})
                continue
            if command == "sleep":
                if model is not None and hasattr(model, "sleep"):
                    model.sleep(level=int(message.get("level", getattr(args, "vllm_sleep_level", 1))))
                    sleeping = True
                connection.send({"status": "ok"})
                continue
            if command == "close":
                connection.send({"status": "ok"})
                break
            raise RuntimeError(f"Unsupported rollout worker command: {command!r}")
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        try:
            connection.send(
                {
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
        raise
    finally:
        cleanup_vllm_rollout_model(model)
        destroy_torch_distributed_process_group(
            log_prefix=f"[rollout-worker cuda={normalize_text(bootstrap.get('cuda_visible_device')).strip() or 'unknown'}]"
        )
        connection.close()


class VLLMRolloutWorker:
    def __init__(self, args: argparse.Namespace, checkpoint_dir: Path, runtime: DistributedRuntime) -> None:
        self.args = args
        self.checkpoint_dir = checkpoint_dir
        self.runtime = runtime
        self.backend = normalize_text(getattr(args, "rollout_backend", "subprocess")).strip() or "subprocess"
        self.model = None
        self._connection = None
        self._process = None
        self._generation_counter = 0
        if self.backend != "subprocess":
            self._load(checkpoint_dir)

    def _recv_response(self, expected_status: str = "ok", timeout_s: Optional[float] = None) -> Mapping[str, Any]:
        if self._connection is None:
            raise RuntimeError("Rollout worker subprocess is not initialized.")
        if timeout_s is not None:
            poll = getattr(self._connection, "poll", None)
            if callable(poll) and not poll(float(timeout_s)):
                exitcode = self._process.exitcode if self._process is not None else None
                raise TimeoutError(
                    "Timed out waiting for rollout worker subprocess response "
                    f"after {float(timeout_s):.1f}s (exitcode={exitcode})."
                )
        try:
            response = self._connection.recv()
        except EOFError as exc:
            exitcode = self._process.exitcode if self._process is not None else None
            raise RuntimeError(f"Rollout worker subprocess exited unexpectedly with exitcode={exitcode}.") from exc
        if normalize_text(response.get("status")).strip() == "error":
            raise RuntimeError(
                "Rollout worker subprocess failed: "
                f"{normalize_text(response.get('error')).strip()}\n{normalize_text(response.get('traceback')).strip()}"
            )
        if normalize_text(response.get("status")).strip() != expected_status:
            raise RuntimeError(f"Unexpected rollout worker response: {response}")
        return response

    def _start_subprocess(self, checkpoint_dir: Path) -> None:
        ctx = mp.get_context(normalize_text(self.args.rollout_worker_start_method).strip() or "spawn")
        parent_conn, child_conn = ctx.Pipe()
        bootstrap = {
            "checkpoint_dir": str(checkpoint_dir),
            "cuda_visible_device": resolve_local_cuda_visible_device(self.runtime.local_rank),
            "args": dict(vars(self.args)),
            "runtime_rank": int(self.runtime.rank),
            "runtime_local_rank": int(self.runtime.local_rank),
        }
        process = ctx.Process(
            target=rollout_worker_process_main,
            args=(child_conn, bootstrap),
            daemon=False,
        )
        process.start()
        child_conn.close()
        self._connection = parent_conn
        self._process = process
        self._recv_response(expected_status="ready")

    def _stop_subprocess(self) -> None:
        close_timeout_s = max(float(getattr(self.args, "rollout_worker_close_timeout_s", 10.0)), 0.0)
        join_timeout_s = max(float(getattr(self.args, "rollout_worker_join_timeout_s", 10.0)), 0.0)
        terminate_timeout_s = max(float(getattr(self.args, "rollout_worker_terminate_timeout_s", 5.0)), 0.0)
        if self._connection is not None:
            try:
                self._connection.send({"cmd": "close"})
                self._recv_response(timeout_s=close_timeout_s)
            except TimeoutError:
                rank_print(
                    self.runtime,
                    (
                        "rollout worker close timed out; forcing subprocess termination "
                        f"after {close_timeout_s:.1f}s"
                    ),
                )
            except Exception:
                pass
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None
        if self._process is not None:
            self._process.join(timeout=join_timeout_s)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=terminate_timeout_s)
            if self._process.is_alive() and hasattr(self._process, "kill"):
                self._process.kill()
                self._process.join(timeout=terminate_timeout_s)
            self._process = None

    def unload(self) -> None:
        if self.backend == "subprocess":
            if self._connection is not None and resolve_effective_vllm_sleep_mode(self.args):
                self._connection.send({"cmd": "sleep", "level": int(self.args.vllm_sleep_level)})
                self._recv_response()
            else:
                self._stop_subprocess()
            return
        if self.model is not None:
            cleanup_vllm_rollout_model(self.model)
            self.model = None

    def _load(self, checkpoint_dir: Path) -> None:
        self.unload()
        self.model = create_vllm_rollout_model(self.args, checkpoint_dir)

    def refresh(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        if self.backend == "subprocess":
            self._stop_subprocess()
            return
        self._load(checkpoint_dir)

    def generate_group(self, query: PreparedQuery, repeat_count: int, sampling: SamplingSpec) -> List[str]:
        generation_seed = int(self.args.seed) + (int(self.runtime.rank) * 100003) + self._generation_counter
        self._generation_counter += 1
        if self.backend == "subprocess":
            generate_timeout_s = max(float(getattr(self.args, "rollout_worker_generate_timeout_s", 900.0)), 0.0)
            if self._connection is None:
                self._start_subprocess(self.checkpoint_dir)
            self._connection.send(
                {
                    "cmd": "generate",
                    "query": build_rollout_query_payload(query),
                    "repeat_count": int(repeat_count),
                    "sampling": asdict(sampling),
                    "seed": generation_seed,
                }
            )
            try:
                response = self._recv_response(timeout_s=generate_timeout_s)
            except TimeoutError as exc:
                protein_id = normalize_text(query.sample_meta.get("protein_id", "")).strip() or "<unknown>"
                rank_print(
                    self.runtime,
                    (
                        "rollout worker generate timed out "
                        f"(protein_id={protein_id}, repeat_count={int(repeat_count)}, timeout_s={generate_timeout_s:.1f})"
                    ),
                )
                self._stop_subprocess()
                raise RuntimeError(
                    "Rollout worker generate timed out "
                    f"after {generate_timeout_s:.1f}s for protein_id={protein_id}."
                ) from exc
            return [normalize_text(output).strip() for output in response.get("outputs", [])]
        if self.model is None:
            self._load(self.checkpoint_dir)
        rollout_batch = repeat_query_for_rollouts(query, repeat_count, query.input_ids.device)
        rollout_multimodal_cache = build_rollout_multimodal_cache(self.model, query, repeat_count)
        outputs = self.model.generate(
            input_ids=rollout_batch["input_ids"],
            attention_mask=rollout_batch["attention_mask"],
            protein_sequences=None,
            batch_idx_map=None,
            structure_coords=None,
            go_aspects=None,
            multimodal_cache=rollout_multimodal_cache,
            temperature=float(sampling.temperature),
            top_k=int(sampling.top_k),
            top_p=float(sampling.top_p),
            min_p=float(sampling.min_p),
            repetition_penalty=float(sampling.repetition_penalty),
            max_new_tokens=int(sampling.max_new_tokens),
            seed=generation_seed,
            stop=ROLLOUT_STOP_MARKERS,
        )
        return [normalize_text(output).strip() for output in outputs]

    def close(self) -> None:
        if self.backend == "subprocess":
            self._stop_subprocess()
            return
        self.unload()


def maybe_trace_group(
    tracker: RunTracker,
    runtime: DistributedRuntime,
    step: int,
    group: RolloutGroup,
) -> None:
    trace_full_group = tracker.claim_full_group_trace()
    for rollout_idx, (completion, reward) in enumerate(zip(group.completions, group.rewards)):
        format_summary = build_completion_format_summary(completion)
        trace_to_weave = (
            trace_full_group and rollout_idx < max(tracker.weave_full_group_rollouts, 0)
        ) or (
            (not trace_full_group) and rollout_idx == 0
        )
        tracker.log_rollout_trace(
            {
                "step": step,
                "rank": runtime.rank,
                "run_name": normalize_text(tracker.args.run_name).strip() or "<auto>",
                "split": "train",
                "protein_id": group.query.sample_meta.get("protein_id", ""),
                "rollout_idx": rollout_idx,
                "reward": float(reward),
                "completion": completion,
                "target_go_ids": build_target_go_ids(group.query.sample_meta),
                "predicted_go_ids": list(format_summary["parsed_go_ids"]),
                "format_summary": dict(format_summary),
                "has_final_answer_tag": bool(format_summary["has_final_answer_tag"]),
                "has_go_summary_block": bool(format_summary["has_go_summary_block"]),
                "uses_alt_final_answer_close_tag": bool(format_summary["uses_alt_final_answer_close_tag"]),
                "has_unclosed_final_answer_tag": bool(format_summary["has_unclosed_final_answer_tag"]),
                "has_repeated_final_answer_open_tag": bool(format_summary["has_repeated_final_answer_open_tag"]),
                "has_tool_call_residue": bool(format_summary["has_tool_call_residue"]),
                "has_think_residue": bool(format_summary["has_think_residue"]),
                "parsed_go_count": int(format_summary["parsed_go_count"]),
                "format_valid": bool(format_summary["format_valid"]),
            },
            trace_to_weave=trace_to_weave,
        )


def mean_or_zero(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def evaluate_validation_subset(
    validation_dataset: Any,
    policy_worker: VLLMRolloutWorker,
    policy_model: Any,
    ia_weights: Mapping[str, float],
    go_graph: Mapping[str, Tuple[str, ...]],
    eval_spec: EvalSpec,
    runtime: DistributedRuntime,
    max_new_tokens: int,
) -> Dict[str, float]:
    if runtime.rank != 0:
        return {}
    limit = min(len(validation_dataset), int(eval_spec.validation_num_proteins))
    deterministic_sampling = SamplingSpec(
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
        max_new_tokens=int(max_new_tokens),
    )
    rewards: List[float] = []
    for idx in range(limit):
        batch = build_single_example_batch(validation_dataset[int(idx)], policy_model)
        query = extract_single_query(batch, policy_model, runtime.device)
        completions = policy_worker.generate_group(query, repeat_count=1, sampling=deterministic_sampling)
        rewards.extend(compute_group_rewards(completions, query.sample_meta, go_graph, ia_weights))
    return {
        "validation_reward_mean": mean_or_zero(rewards),
        "validation_reward_nonzero_rate": mean_or_zero([1.0 if reward > 0.0 else 0.0 for reward in rewards]),
        "validation_num_proteins": float(limit),
    }


def save_training_checkpoint(
    policy_stack: PolicyStack,
    args: argparse.Namespace,
    step: int,
    tracker: RunTracker,
    runtime: DistributedRuntime,
) -> None:
    checkpoint_root = Path(args.output_dir) / "checkpoints" / f"step-{step:06d}"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    policy_stack.engine.save_checkpoint(str(checkpoint_root / "deepspeed"))
    barrier(runtime)
    if runtime.rank == 0:
        export_dir = checkpoint_root / "inference_export"
        export_inference_checkpoint(policy_stack.engine.module, export_dir)
        metadata = {
            "global_step": step,
            "checkpoint_artifact_name": args.checkpoint_artifact_name,
            "base_checkpoint": normalize_text(args.base_checkpoint).strip() or args.text_model_name,
            "benchmark_version": args.benchmark_version,
            "dataset_artifact": args.dataset_artifact,
            "runtime_stack": args.runtime_stack,
        }
        save_json(checkpoint_root / "training_metadata.json", metadata)
        aliases = [item.strip() for item in normalize_text(args.checkpoint_artifact_aliases).split(",") if item.strip()]
        tracker.log_checkpoint_artifact(checkpoint_root, aliases=aliases or ["latest"], metadata=metadata)
    barrier(runtime)


def refresh_old_policy_and_rollout_worker(
    policy_stack: PolicyStack,
    rollout_worker: VLLMRolloutWorker,
    output_dir: Path,
    runtime: DistributedRuntime,
    step: int,
) -> None:
    rollout_dir = output_dir / f"rank{runtime.rank:02d}" / "rollout_policy" / f"step-{step:06d}"
    export_inference_checkpoint(policy_stack.engine.module, rollout_dir)
    policy_stack.rollout_checkpoint_dir = rollout_dir.resolve()
    rollout_worker.refresh(rollout_dir)


def train(args: argparse.Namespace) -> None:
    require_torch()
    validate_runtime_dependencies()
    run_name = normalize_text(args.run_name).strip() or f"train-rl-{int(time.time())}"
    args.run_name = run_name
    if not normalize_text(args.base_checkpoint).strip():
        args.base_checkpoint = args.text_model_name
    validate_spec_inputs(args)

    algorithm = build_algorithm_spec(args)
    runtime_spec = build_runtime_spec(args)
    sampling = build_sampling_spec(args)
    eval_spec = build_eval_spec(args)
    runtime = initialize_runtime(args)
    validate_runtime_shape(runtime, algorithm, runtime_spec, args)
    configure_query_parallel_runtime(runtime, algorithm)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rank0_print(runtime, "Loading train / validation datasets for spec-first DR-GRPO.")
    train_dataset, validation_dataset = load_reasoning_datasets(args, runtime)

    rank0_print(runtime, "Initializing DeepSpeed policy stack.")
    policy_stack = initialize_policy_stack(args, runtime_spec, runtime)
    go_graph = load_go_term_graph(normalize_text(args.go_obo_path).strip())
    ia_weights = load_ia_weights(normalize_text(args.ia_file_path).strip())
    tracker = RunTracker(
        args=args,
        config=build_tracking_config(args, algorithm, runtime_spec, runtime, run_name=run_name),
        output_dir=output_dir,
        runtime=runtime,
    )

    rank0_print(runtime, "Initializing the vLLM rollout worker from the canonical base checkpoint.")
    rollout_worker = VLLMRolloutWorker(args, policy_stack.rollout_checkpoint_dir, runtime)

    try:
        for step in range(int(args.max_steps)):
            step_started_at = time.perf_counter()
            rollout_seconds = 0.0
            reward_seconds = 0.0
            scoring_seconds = 0.0
            policy_update_seconds = 0.0
            refresh_seconds = 0.0
            validation_seconds = 0.0
            checkpoint_seconds = 0.0
            global_query_indices = sample_query_indices(
                dataset_length=len(train_dataset),
                queries_per_step=algorithm.queries_per_step,
                seed=int(args.seed),
                step=step,
                runtime=runtime,
            )
            local_query_indices = partition_queries_for_rank(
                global_query_indices,
                runtime.rank,
                max(runtime.world_size, 1),
                algorithm.queries_per_step,
            )
            local_queries: List[PreparedQuery] = []
            for dataset_idx in local_query_indices:
                batch = build_single_example_batch(train_dataset[int(dataset_idx)], policy_stack.engine.module)
                local_queries.append(extract_single_query(batch, policy_stack.engine.module, runtime.device))

            if runtime.enabled and len(local_queries) != 1:
                raise RuntimeError(
                    f"Spec-first distributed mode expects exactly one query per rank, got {len(local_queries)} on rank {runtime.rank}."
                )

            local_groups: List[RolloutGroup] = []
            for query in local_queries:
                rollout_started_at = time.perf_counter()
                protein_id = normalize_text(query.sample_meta.get("protein_id", "")).strip() or "<unknown>"
                local_repeat_count = resolve_local_rollouts_per_rank(algorithm, runtime.world_size)
                rank_print(
                    runtime,
                    (
                        f"step {step + 1}: starting rollout generation "
                        f"(protein_id={protein_id}, local_repeat_count={local_repeat_count})"
                    ),
                )
                completions = tracker.trace_rollout_call(
                    step=step + 1,
                    split="train",
                    query=query,
                    repeat_count=local_repeat_count,
                    sampling=sampling,
                    generator=lambda current_query=query: rollout_worker.generate_group(
                        current_query,
                        local_repeat_count,
                        sampling,
                    ),
                    tokenizer=policy_stack.tokenizer,
                )
                rollout_duration = time.perf_counter() - rollout_started_at
                rollout_seconds += rollout_duration
                rank_print(
                    runtime,
                    (
                        f"step {step + 1}: finished rollout generation "
                        f"(protein_id={protein_id}, local_repeat_count={len(completions)}, duration_s={rollout_duration:.2f})"
                    ),
                )
                completion_ids = tokenize_completion_texts(policy_stack.tokenizer, completions, runtime.device)
                reward_started_at = time.perf_counter()
                rewards = tracker.trace_reward_call(
                    step=step + 1,
                    split="train",
                    query=query,
                    completions=completions,
                    callback=lambda current_completions=list(completions), current_query=query: compute_group_rewards(
                        current_completions,
                        current_query.sample_meta,
                        go_graph,
                        ia_weights,
                    ),
                )
                reward_seconds += time.perf_counter() - reward_started_at
                group = RolloutGroup(
                    query=query,
                    completions=completions,
                    completion_ids=completion_ids,
                    rewards=rewards,
                )
                local_groups.append(group)
                maybe_trace_group(tracker, runtime, step, group)

            rank_print(
                runtime,
                (
                    f"step {step + 1}: rollout generation complete "
                    f"(local_queries={len(local_queries)}, local_rollouts={sum(len(group.completions) for group in local_groups)})"
                ),
            )
            rank_print(runtime, f"step {step + 1}: unloading rollout worker before reward reduction")
            rollout_worker.unload()
            rank_print(runtime, f"step {step + 1}: rollout worker unload complete")
            global_reward_std = compute_global_reward_std(
                [group.rewards for group in local_groups],
                runtime=runtime,
                epsilon=algorithm.reward_std_epsilon,
            )

            for group in local_groups:
                query_group_mean = compute_query_group_mean(group.rewards, runtime)
                prepare_group_for_loss(
                    group,
                    global_reward_std=global_reward_std,
                    query_group_mean=query_group_mean,
                    runtime_device=runtime.device,
                    max_loss_completion_tokens=int(args.max_loss_completion_tokens),
                )

            rank_print(runtime, f"step {step + 1}: starting old/ref log-prob scoring")
            scoring_started_at = time.perf_counter()
            if any(group.selected_completion_ids for group in local_groups):
                scoring_specs = [
                    ("old", policy_stack.rollout_checkpoint_dir),
                    ("ref", policy_stack.reference_checkpoint_dir),
                ]
                for policy_role, checkpoint_dir in scoring_specs:
                    scoring_model = load_frozen_scoring_model(args, checkpoint_dir, runtime)
                    try:
                        for group in local_groups:
                            completion_token_lengths = [int(item.numel()) for item in group.completion_ids]
                            tracker.trace_scoring_call(
                                step=step + 1,
                                split="train",
                                query=group.query,
                                payload={
                                    "policy_role": policy_role,
                                    "checkpoint_dir": str(checkpoint_dir),
                                    "raw_rollout_count": len(group.completion_ids),
                                    "completion_token_lengths": completion_token_lengths,
                                    "max_loss_completion_tokens": int(args.max_loss_completion_tokens),
                                },
                                callback=lambda current_group=group, current_role=policy_role, current_model=scoring_model: score_group_log_probs(
                                    policy_model=current_model,
                                    group=current_group,
                                    pad_token_id=policy_stack.pad_token_id,
                                    device=runtime.device,
                                    logprob_microbatch_size=int(args.rollout_logprob_microbatch_size),
                                    policy_role=current_role,
                                ),
                            )
                    finally:
                        cleanup_policy_model(scoring_model)
            scoring_seconds = time.perf_counter() - scoring_started_at

            rank_print(
                runtime,
                (
                    f"step {step + 1}: finished old/ref scoring "
                    f"(valid_rollouts={sum(len(group.selected_completion_ids or []) for group in local_groups)}, "
                    f"filtered_rollouts={sum(float(group.filtered_rollouts) for group in local_groups):.0f})"
                ),
            )
            policy_loss_values: List[float] = []
            kl_values: List[float] = []
            ratio_means: List[float] = []
            ratio_maxes: List[float] = []
            filtered_rollout_counts = [float(group.filtered_rollouts) for group in local_groups]
            valid_rollout_counts = [float(len(group.selected_completion_ids or [])) for group in local_groups]

            rank_print(runtime, f"step {step + 1}: starting policy updates")

            def run_policy_updates() -> Dict[str, Any]:
                chunk_count = 0
                for _ in range(algorithm.steps_per_generation):
                    for group in local_groups:
                        selected_completion_ids = list(group.selected_completion_ids or [])
                        if not selected_completion_ids:
                            continue
                        policy_stack.engine.zero_grad()
                        chunk_size = int(runtime_spec.optimizer_micro_batch_size_per_gpu)
                        total_rollouts = len(selected_completion_ids)
                        for start_idx in range(0, total_rollouts, chunk_size):
                            end_idx = min(total_rollouts, start_idx + chunk_size)
                            chunk_completion_ids = selected_completion_ids[start_idx:end_idx]
                            chunk_advantages = group.advantages[start_idx:end_idx]
                            chunk_old_log_probs = group.old_log_probs[start_idx:end_idx]
                            chunk_ref_log_probs = group.ref_log_probs[start_idx:end_idx]
                            loss, chunk_metrics = compute_chunk_loss(
                                current_model=policy_stack.engine.module,
                                query=group.query,
                                completion_ids=chunk_completion_ids,
                                advantages=chunk_advantages,
                                old_log_probs=chunk_old_log_probs,
                                ref_log_probs=chunk_ref_log_probs,
                                algorithm=algorithm,
                                pad_token_id=policy_stack.pad_token_id,
                                device=runtime.device,
                                logprob_microbatch_size=int(args.rollout_logprob_microbatch_size),
                            )
                            policy_stack.engine.backward(loss)
                            policy_stack.engine.step()
                            policy_loss_values.append(float(loss.detach().item()))
                            kl_values.append(float(chunk_metrics["kl_mean"]))
                            ratio_means.append(float(chunk_metrics["ratio_mean"]))
                            ratio_maxes.append(float(chunk_metrics["ratio_max"]))
                            chunk_count += 1
                return {
                    "update_chunk_count": chunk_count,
                    "valid_rollout_count": int(sum(len(group.selected_completion_ids or []) for group in local_groups)),
                    "policy_loss_mean": mean_or_zero(policy_loss_values),
                    "kl_mean": mean_or_zero(kl_values),
                    "ratio_mean": mean_or_zero(ratio_means),
                    "ratio_max": max(ratio_maxes) if ratio_maxes else 0.0,
                }

            policy_update_started_at = time.perf_counter()
            tracker.trace_policy_update_call(
                step=step + 1,
                split="train",
                callback=run_policy_updates,
                payload={
                    "steps_per_generation": int(algorithm.steps_per_generation),
                    "optimizer_micro_batch_size_per_gpu": int(runtime_spec.optimizer_micro_batch_size_per_gpu),
                    "valid_rollout_counts": [int(value) for value in valid_rollout_counts],
                    "filtered_rollout_counts": [int(value) for value in filtered_rollout_counts],
                },
            )
            policy_update_seconds = time.perf_counter() - policy_update_started_at

            rank_print(runtime, f"step {step + 1}: policy updates complete, refreshing rollout worker")
            refresh_started_at = time.perf_counter()
            refresh_old_policy_and_rollout_worker(
                policy_stack=policy_stack,
                rollout_worker=rollout_worker,
                output_dir=output_dir,
                runtime=runtime,
                step=step + 1,
            )
            rank_print(runtime, f"step {step + 1}: waiting at post-refresh barrier")
            barrier(runtime)
            rank_print(runtime, f"step {step + 1}: passed post-refresh barrier")
            refresh_seconds = time.perf_counter() - refresh_started_at

            local_rewards = [reward for group in local_groups for reward in group.rewards]
            local_format_summaries = [
                build_completion_format_summary(completion)
                for group in local_groups
                for completion in group.completions
            ]
            metrics = {
                "reward_mean": mean_or_zero(local_rewards),
                "reward_nonzero_rate": mean_or_zero([1.0 if reward > 0.0 else 0.0 for reward in local_rewards]),
                "reward_std": float(global_reward_std),
                "format_valid_rate": mean_or_zero(
                    [1.0 if summary["format_valid"] else 0.0 for summary in local_format_summaries]
                ),
                "final_answer_tag_rate": mean_or_zero(
                    [1.0 if summary["has_final_answer_tag"] else 0.0 for summary in local_format_summaries]
                ),
                "go_summary_block_rate": mean_or_zero(
                    [1.0 if summary["has_go_summary_block"] else 0.0 for summary in local_format_summaries]
                ),
                "alt_final_answer_close_tag_rate": mean_or_zero(
                    [1.0 if summary["uses_alt_final_answer_close_tag"] else 0.0 for summary in local_format_summaries]
                ),
                "unclosed_final_answer_tag_rate": mean_or_zero(
                    [1.0 if summary["has_unclosed_final_answer_tag"] else 0.0 for summary in local_format_summaries]
                ),
                "repeated_final_answer_open_tag_rate": mean_or_zero(
                    [1.0 if summary["has_repeated_final_answer_open_tag"] else 0.0 for summary in local_format_summaries]
                ),
                "tool_call_residue_rate": mean_or_zero(
                    [1.0 if summary["has_tool_call_residue"] else 0.0 for summary in local_format_summaries]
                ),
                "think_residue_rate": mean_or_zero(
                    [1.0 if summary["has_think_residue"] else 0.0 for summary in local_format_summaries]
                ),
                "parsed_go_count_mean": mean_or_zero(
                    [float(summary["parsed_go_count"]) for summary in local_format_summaries]
                ),
                "loss_mean": mean_or_zero(policy_loss_values),
                "kl_mean": mean_or_zero(kl_values),
                "ratio_mean": mean_or_zero(ratio_means),
                "ratio_max": max(ratio_maxes) if ratio_maxes else 0.0,
                "filtered_rollouts": sum(filtered_rollout_counts),
                "valid_rollouts": sum(valid_rollout_counts),
                "learning_rate": float(policy_stack.engine.optimizer.param_groups[0]["lr"]),
                "step": float(step + 1),
                "timing_rollout_seconds": rollout_seconds,
                "timing_reward_seconds": reward_seconds,
                "timing_scoring_seconds": scoring_seconds,
                "timing_policy_update_seconds": policy_update_seconds,
                "timing_refresh_seconds": refresh_seconds,
            }

            aggregated_metrics = {
                "reward_mean": all_reduce_sum_scalar(metrics["reward_mean"], runtime) / float(runtime.world_size),
                "reward_nonzero_rate": all_reduce_sum_scalar(metrics["reward_nonzero_rate"], runtime) / float(runtime.world_size),
                "reward_std": metrics["reward_std"],
                "format_valid_rate": all_reduce_sum_scalar(metrics["format_valid_rate"], runtime) / float(runtime.world_size),
                "final_answer_tag_rate": all_reduce_sum_scalar(metrics["final_answer_tag_rate"], runtime) / float(runtime.world_size),
                "go_summary_block_rate": all_reduce_sum_scalar(metrics["go_summary_block_rate"], runtime) / float(runtime.world_size),
                "alt_final_answer_close_tag_rate": all_reduce_sum_scalar(
                    metrics["alt_final_answer_close_tag_rate"], runtime
                )
                / float(runtime.world_size),
                "unclosed_final_answer_tag_rate": all_reduce_sum_scalar(
                    metrics["unclosed_final_answer_tag_rate"], runtime
                )
                / float(runtime.world_size),
                "repeated_final_answer_open_tag_rate": all_reduce_sum_scalar(
                    metrics["repeated_final_answer_open_tag_rate"], runtime
                )
                / float(runtime.world_size),
                "tool_call_residue_rate": all_reduce_sum_scalar(metrics["tool_call_residue_rate"], runtime)
                / float(runtime.world_size),
                "think_residue_rate": all_reduce_sum_scalar(metrics["think_residue_rate"], runtime)
                / float(runtime.world_size),
                "parsed_go_count_mean": all_reduce_sum_scalar(metrics["parsed_go_count_mean"], runtime) / float(runtime.world_size),
                "loss_mean": all_reduce_sum_scalar(metrics["loss_mean"], runtime) / float(runtime.world_size),
                "kl_mean": all_reduce_sum_scalar(metrics["kl_mean"], runtime) / float(runtime.world_size),
                "ratio_mean": all_reduce_sum_scalar(metrics["ratio_mean"], runtime) / float(runtime.world_size),
                "ratio_max": all_reduce_max_scalar(metrics["ratio_max"], runtime),
                "filtered_rollouts": all_reduce_sum_scalar(metrics["filtered_rollouts"], runtime),
                "valid_rollouts": all_reduce_sum_scalar(metrics["valid_rollouts"], runtime),
                "learning_rate": metrics["learning_rate"],
                "timing_rollout_seconds": all_reduce_max_scalar(metrics["timing_rollout_seconds"], runtime),
                "timing_reward_seconds": all_reduce_max_scalar(metrics["timing_reward_seconds"], runtime),
                "timing_scoring_seconds": all_reduce_max_scalar(metrics["timing_scoring_seconds"], runtime),
                "timing_policy_update_seconds": all_reduce_max_scalar(metrics["timing_policy_update_seconds"], runtime),
                "timing_refresh_seconds": all_reduce_max_scalar(metrics["timing_refresh_seconds"], runtime),
            }
            tracker.log_metrics(aggregated_metrics, step=step + 1)
            rank0_print(
                runtime,
                (
                    f"[step {step + 1:04d}] reward={aggregated_metrics['reward_mean']:.4f} "
                    f"reward_std={aggregated_metrics['reward_std']:.4f} "
                    f"loss={aggregated_metrics['loss_mean']:.6f}"
                ),
            )

            if eval_spec.validation_every_n_steps > 0 and (step + 1) % eval_spec.validation_every_n_steps == 0:
                validation_started_at = time.perf_counter()
                validation_metrics = evaluate_validation_subset(
                    validation_dataset=validation_dataset,
                    policy_worker=rollout_worker,
                    policy_model=policy_stack.engine.module,
                    ia_weights=ia_weights,
                    go_graph=go_graph,
                    eval_spec=eval_spec,
                    runtime=runtime,
                    max_new_tokens=int(args.max_new_tokens),
                )
                validation_seconds = time.perf_counter() - validation_started_at
                if validation_metrics:
                    validation_metrics["timing_validation_seconds"] = validation_seconds
                    tracker.log_metrics(validation_metrics, step=step + 1)

            if eval_spec.save_every_n_steps > 0 and (step + 1) % eval_spec.save_every_n_steps == 0:
                checkpoint_started_at = time.perf_counter()
                save_training_checkpoint(
                    policy_stack=policy_stack,
                    args=args,
                    step=step + 1,
                    tracker=tracker,
                    runtime=runtime,
                )
                checkpoint_seconds = time.perf_counter() - checkpoint_started_at
                tracker.log_metrics(
                    {
                        "timing_checkpoint_seconds": checkpoint_seconds,
                    },
                    step=step + 1,
                )

            tracker.log_metrics(
                {
                    "timing_step_seconds": all_reduce_max_scalar(time.perf_counter() - step_started_at, runtime),
                },
                step=step + 1,
            )
    finally:
        rollout_worker.close()
        tracker.finish()
        shutdown_runtime(runtime)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.preflight_only:
        raise SystemExit(0 if run_preflight(args) else 1)
    train(args)


if __name__ == "__main__":
    main()

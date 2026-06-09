#!/usr/bin/env python3
"""Render or apply direct CoreWeave CKS/SUNK training jobs.

This helper is training-only: it launches BioReason train.py jobs on CKS while
Slurm places the Pods through SUNK. It does not run the upstream wandb/senpai
advisor/student GitHub PR control plane. Use
scripts/launch_wandb_senpai_control_plane.sh for the real Senpai loop.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "senpai.yaml"
STUDENT_TEMPLATE = ROOT / "k8s" / "student-job.yaml"

DEFAULT_STUDENT_NAMES = (
    "frieren",
    "fern",
    "tanjiro",
    "nezuko",
    "alphonse",
    "edward",
    "thorfinn",
    "askeladd",
)


@dataclass
class LaunchConfig:
    target_repo_url: str = "https://github.com/olachinkei/BioReason-Pro.git"
    target_repo_branch: str = "main"
    advisor_branch: str = "senpai-bioreason"
    image: str = "bioreason-pro-senpai:cuda12.6"
    image_pull_policy: str = "IfNotPresent"
    n_students: int = 1
    student_names: str = "frieren"
    gpus_per_student: int = 8
    cpu_per_student: int = 64
    memory_request_gi: int = 900
    memory_limit_gi: int = 960
    rdma_per_student: int = 1
    gpu_model: str = "H100_NVLINK_80GB"
    sunk_scheduler_name: str = "tenant-slurm-staging-slurm-scheduler"
    sunk_account: str = "root"
    sunk_partition: str = "h100"
    sunk_exclusive: str = "none"
    sunk_timeout_minutes: int = 720
    termination_grace_period_seconds: int = 10
    pvc_claim_name: str = "bioreason-runtime-pvc"
    runtime_mount_path: str = "/mnt/data"
    runtime_root: str = "/mnt/data/bioreason/BioReason-Pro"
    secret_name: str = "senpai-secrets"
    wandb_entity: str = "wandb-healthcare"
    wandb_project: str = "bioreasoning-pro-senpai"
    wandb_mode: str = "online"
    mode: str = "gate"
    gate_steps: int = 5
    continue_steps: int = 15
    max_val_samples: int = 100
    senpai_max_new_tokens: int = 10000
    senpai_vllm_max_model_len: int = 32768
    senpai_vllm_max_num_seqs: int = 4
    extra_train_args: str = ""


def parse_scalar(value: str) -> object:
    value = value.strip()
    if not value:
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return value.strip('"').strip("'")


def load_config(path: Path) -> LaunchConfig:
    values: dict[str, object] = {}
    field_names = {f.name for f in fields(LaunchConfig)}
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line or ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            key = key.strip().replace("-", "_")
            if key in field_names:
                values[key] = parse_scalar(raw_value)
    return LaunchConfig(**values)


def add_config_overrides(parser: argparse.ArgumentParser) -> None:
    for f in fields(LaunchConfig):
        value_type = int if isinstance(f.default, int) else str
        parser.add_argument(f"--{f.name}", dest=f.name, type=value_type, default=None)


def student_names(config: LaunchConfig, explicit_names: str) -> list[str]:
    raw = explicit_names or config.student_names
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        names = list(DEFAULT_STUDENT_NAMES)
    while len(names) < config.n_students:
        names.append(DEFAULT_STUDENT_NAMES[len(names) % len(DEFAULT_STUDENT_NAMES)])
    return names[: config.n_students]


def k8s_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9.-]+", "-", value)
    value = value.strip(".-")
    return value[:63].strip(".-") or "bioreason-senpai"


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_train_args(config: LaunchConfig, student_name: str, tag: str) -> str:
    parts = [
        f"--mode {shell_quote(config.mode)}",
        f"--wandb_name {shell_quote(f'{student_name}/{tag}')}",
        f"--wandb_group {shell_quote(tag)}",
        f"--wandb_entity {shell_quote(config.wandb_entity)}",
        f"--wandb_project {shell_quote(config.wandb_project)}",
        f"--wandb_mode {shell_quote(config.wandb_mode)}",
        f"--runtime_root {shell_quote(config.runtime_root)}",
        f"--gate_steps {config.gate_steps}",
        f"--continue_steps {config.continue_steps}",
        f"--max_val_samples {config.max_val_samples}",
    ]
    if config.extra_train_args:
        parts.append(config.extra_train_args)
    return " ".join(parts)


def render_template(template: str, replacements: dict[str, str]) -> str:
    out = template
    for key, value in replacements.items():
        out = out.replace(f"{{{{{key}}}}}", str(value))
    return out


def render_configmap(name: str, tag: str, config: LaunchConfig) -> str:
    runtime_root = config.runtime_root.rstrip("/")
    data = {
        "RESEARCH_TAG": tag,
        "TARGET_REPO_URL": config.target_repo_url,
        "TARGET_REPO_BRANCH": config.target_repo_branch,
        "ADVISOR_BRANCH": config.advisor_branch,
        "WANDB_ENTITY": config.wandb_entity,
        "WANDB_PROJECT": config.wandb_project,
        "WANDB_MODE": config.wandb_mode,
        "BIOREASON_RUNTIME_ROOT": runtime_root,
        "BIOREASON_ARTIFACTS_ROOT": f"{runtime_root}/data/artifacts",
        "BIOREASON_CACHE_ROOT": f"{runtime_root}/cache",
        "WANDB_DIR": f"{runtime_root}/wandb",
        "WEAVE_SERVER_CACHE_DIR": f"{runtime_root}/wandb/weave_server_cache",
        "HF_HOME": f"{runtime_root}/cache/huggingface",
        "TRANSFORMERS_CACHE": f"{runtime_root}/cache/huggingface/transformers",
        "HF_DATASETS_CACHE": f"{runtime_root}/cache/huggingface/datasets",
        "XDG_CACHE_HOME": f"{runtime_root}/cache/xdg",
        "TRITON_CACHE_DIR": f"{runtime_root}/cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{runtime_root}/cache/torch_inductor",
        "TORCHINDUCTOR_COMPILE_THREADS": "4",
        "TMPDIR": f"{runtime_root}/tmp",
        "VLLM_ATTENTION_BACKEND": "XFORMERS",
        "VLLM_USE_V1": "0",
        "BIOREASON_VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "SENPAI_MAX_NEW_TOKENS": str(config.senpai_max_new_tokens),
        "SENPAI_VLLM_MAX_MODEL_LEN": str(config.senpai_vllm_max_model_len),
        "SENPAI_VLLM_MAX_NUM_SEQS": str(config.senpai_vllm_max_num_seqs),
    }
    lines = [
        "apiVersion: v1",
        "kind: ConfigMap",
        "metadata:",
        f"  name: {name}",
        "  labels:",
        "    app: bioreason-senpai",
        f"    research-tag: {tag}",
        "data:",
    ]
    for key, value in data.items():
        lines.append(f"  {key}: {yaml_quote(value)}")
    return "\n".join(lines)


def render_student_job(
    template: str,
    configmap_name: str,
    tag: str,
    student_name: str,
    config: LaunchConfig,
) -> str:
    job_name = k8s_name(f"bioreason-{tag}-{student_name}")
    replacements = {
        "JOB_NAME": job_name,
        "CONFIGMAP_NAME": configmap_name,
        "STUDENT_NAME": student_name,
        "RESEARCH_TAG": tag,
        "IMAGE": config.image,
        "IMAGE_PULL_POLICY": config.image_pull_policy,
        "TRAIN_ARGS": build_train_args(config, student_name, tag),
        "SECRET_NAME": config.secret_name,
        "SUNK_SCHEDULER_NAME": config.sunk_scheduler_name,
        "SUNK_ACCOUNT": config.sunk_account,
        "SUNK_PARTITION": config.sunk_partition,
        "SUNK_EXCLUSIVE": config.sunk_exclusive,
        "SUNK_TIMEOUT_MINUTES": config.sunk_timeout_minutes,
        "TERMINATION_GRACE_PERIOD_SECONDS": config.termination_grace_period_seconds,
        "CPU_PER_STUDENT": config.cpu_per_student,
        "MEMORY_REQUEST_GI": config.memory_request_gi,
        "MEMORY_LIMIT_GI": config.memory_limit_gi,
        "GPUS_PER_STUDENT": config.gpus_per_student,
        "RDMA_PER_STUDENT": config.rdma_per_student,
        "RUNTIME_MOUNT_PATH": config.runtime_mount_path,
        "PVC_CLAIM_NAME": config.pvc_claim_name,
        "GPU_MODEL": config.gpu_model,
    }
    return render_template(template, replacements)


def render_manifests(config: LaunchConfig, tag: str, names: Iterable[str]) -> str:
    configmap_name = k8s_name(f"bioreason-senpai-{tag}")
    template = STUDENT_TEMPLATE.read_text(encoding="utf-8")
    docs = [render_configmap(configmap_name, tag, config)]
    for name in names:
        docs.append(render_student_job(template, configmap_name, tag, name, config))
    return "\n---\n".join(docs) + "\n"


def apply_manifest(manifest: str) -> None:
    subprocess.run(["kubectl", "apply", "-f", "-"], input=manifest, text=True, check=True)


def apply_env_defaults(config: LaunchConfig) -> None:
    wandb_project = os.environ.get("WANDB_PROJECT", "").strip()
    if wandb_project:
        config.wandb_project = wandb_project
    int_env_fields = {
        "SENPAI_MAX_NEW_TOKENS": "senpai_max_new_tokens",
        "SENPAI_VLLM_MAX_MODEL_LEN": "senpai_vllm_max_model_len",
        "SENPAI_VLLM_MAX_NUM_SEQS": "senpai_vllm_max_num_seqs",
    }
    for env_name, field_name in int_env_fields.items():
        raw_value = os.environ.get(env_name, "").strip()
        if raw_value:
            setattr(config, field_name, int(raw_value))


def main(argv: list[str] | None = None) -> int:
    config = load_config(CONFIG_PATH)
    apply_env_defaults(config)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Research tag used in names, labels, and W&B groups.")
    parser.add_argument("--names", default="", help="Comma-separated student names; overrides student_names.")
    parser.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true")
    add_config_overrides(parser)
    args = parser.parse_args(argv)

    for f in fields(LaunchConfig):
        value = getattr(args, f.name)
        if value is not None:
            setattr(config, f.name, value)

    if config.sunk_exclusive not in {"none", "ok", "user", "true", "false"}:
        sys.exit("ERROR: --sunk_exclusive must be one of: none, ok, user, true, false")
    if config.termination_grace_period_seconds >= 25:
        sys.exit("ERROR: --termination_grace_period_seconds must normally be below SUNK kill-wait minus 5s; use <25.")
    if config.memory_request_gi > config.memory_limit_gi:
        sys.exit("ERROR: --memory_request_gi must be <= --memory_limit_gi")
    if min(config.senpai_max_new_tokens, config.senpai_vllm_max_model_len, config.senpai_vllm_max_num_seqs) < 1:
        sys.exit("ERROR: Senpai token and vLLM sequence settings must all be at least 1")
    if str(config.wandb_mode).strip().lower() in {"offline", "dryrun", "disabled"}:
        sys.exit("ERROR: Senpai runs require online W&B; fix wandb/weave or credentials instead of using offline mode.")

    names = student_names(config, args.names)
    manifest = render_manifests(config, k8s_name(args.tag), names)
    if args.dry_run:
        print(manifest, end="")
    else:
        apply_manifest(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

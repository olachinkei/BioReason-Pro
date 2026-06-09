#!/usr/bin/env python3
"""Guarded B-run loop for live training plus external validation.

This script is intentionally lightweight: it runs on the login node, submits
Slurm jobs, waits for completion, reads local JSON summaries, and writes a
small diagnosis report for Codex/user review.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


FINAL_ANSWER_OPEN_TAG = "<|FINAL_ANSWER|>"
FINAL_ANSWER_CLOSE_TAG = "<|/FINAL_ANSWER|>"
GO_ID_PATTERN = re.compile(r"GO:\d{7}")
TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}
ACTIVE_STATES = {"CONFIGURING", "COMPLETING", "PENDING", "RUNNING", "RESIZING", "SUSPENDED"}


@dataclass(frozen=True)
class Recipe:
    name: str
    env: Mapping[str, str]
    rationale: str


RECIPES: List[Recipe] = [
    Recipe(
        name="b_live_reasoning_nohardfilter",
        env={
            "REWARD_EXTRACTION_SOURCE": "reasoning_trace",
            "APPLY_FORMAT_REWARD_GATE": "true",
            "INVALID_FORMAT_PENALTY": "0.05",
            "GO_COUNT_PENALTY": "0.0",
            "FILTER_INVALID_ROLLOUTS_FOR_LOSS": "false",
        },
        rationale="Current B-fixed baseline: reasoning_trace reward, weak format penalty, no GO-count penalty, no hard filtering.",
    ),
    Recipe(
        name="b_gate_off_penalty_002",
        env={
            "REWARD_EXTRACTION_SOURCE": "reasoning_trace",
            "APPLY_FORMAT_REWARD_GATE": "false",
            "INVALID_FORMAT_PENALTY": "0.02",
            "GO_COUNT_PENALTY": "0.0",
            "FILTER_INVALID_ROLLOUTS_FOR_LOSS": "false",
            "REWARD_WEIGHTS": "0.05,0.05,1.0,0.02",
        },
        rationale="If validation is below baseline, reduce format gating pressure and add small positive structure rewards without changing tensor shapes.",
    ),
    Recipe(
        name="b_gate_off_penalty_000",
        env={
            "REWARD_EXTRACTION_SOURCE": "reasoning_trace",
            "APPLY_FORMAT_REWARD_GATE": "false",
            "INVALID_FORMAT_PENALTY": "0.0",
            "GO_COUNT_PENALTY": "0.0",
            "FILTER_INVALID_ROLLOUTS_FOR_LOSS": "false",
            "REWARD_WEIGHTS": "0.10,0.05,1.0,0.05",
        },
        rationale="Remove invalid-format punishment entirely and reward final-answer/format presence softly to avoid prediction-density collapse.",
    ),
    Recipe(
        name="b_longer_generation_soft_format",
        env={
            "REWARD_EXTRACTION_SOURCE": "reasoning_trace",
            "APPLY_FORMAT_REWARD_GATE": "false",
            "INVALID_FORMAT_PENALTY": "0.0",
            "GO_COUNT_PENALTY": "0.0",
            "FILTER_INVALID_ROLLOUTS_FOR_LOSS": "false",
            "REWARD_WEIGHTS": "0.05,0.10,1.0,0.05",
            "MAX_NEW_TOKENS": "12000",
            "ROLLOUT_MAX_NEW_TOKENS": "12000",
        },
        rationale="Give completions more room and softly reward GO-summary/final-answer structure when short or missing final answers dominate.",
    ),
    Recipe(
        name="b_no_precision_pressure",
        env={
            "REWARD_MODE": "per_aspect_ia_f1",
            "REWARD_EXTRACTION_SOURCE": "reasoning_trace",
            "APPLY_FORMAT_REWARD_GATE": "false",
            "INVALID_FORMAT_PENALTY": "0.0",
            "GO_COUNT_PENALTY": "0.0",
            "FILTER_INVALID_ROLLOUTS_FOR_LOSS": "false",
            "REWARD_WEIGHTS": "0.05,0.05,1.0,0.05",
        },
        rationale="Remove precision pressure if the model becomes too conservative and under-predicts annotations.",
    ),
    Recipe(
        name="b_more_rollout_diversity",
        env={
            "REWARD_EXTRACTION_SOURCE": "reasoning_trace",
            "APPLY_FORMAT_REWARD_GATE": "false",
            "INVALID_FORMAT_PENALTY": "0.0",
            "GO_COUNT_PENALTY": "0.0",
            "FILTER_INVALID_ROLLOUTS_FOR_LOSS": "false",
            "REWARD_WEIGHTS": "0.05,0.05,1.0,0.05",
            "ROLLOUTS_PER_QUERY": "16",
            "OPTIMIZER_MICRO_BATCH_SIZE_PER_GPU": "4",
        },
        rationale="Last retry: lower rollout load per query to reduce instability while preserving the same 10-step/eval structure.",
    ),
]


def run_command(command: Sequence[str], *, cwd: Optional[Path] = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=str(cwd) if cwd else None, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def metric_value(metrics: Mapping[str, Any]) -> float:
    for key in ("overall_mean_f1", "overall_mean_fmax", "fmax_mean", "mean_fmax"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return float("nan")


def query_job(job_id: str) -> Tuple[str, str]:
    result = run_command(["sacct", "-j", job_id, "--format=JobIDRaw,State,ExitCode", "-P", "-n"])
    if result.returncode != 0:
        return "", ""
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 3 and fields[0] == job_id:
            return fields[1], fields[2]
    return "", ""


def wait_for_job(job_id: str, *, poll_s: float, timeout_s: float) -> Tuple[str, str]:
    started = time.monotonic()
    while True:
        state, exit_code = query_job(job_id)
        if state in TERMINAL_STATES:
            return state, exit_code
        queue = run_command(["squeue", "-h", "-j", job_id])
        if queue.returncode == 0 and not queue.stdout.strip() and state and state not in ACTIVE_STATES:
            return state, exit_code
        if timeout_s > 0 and time.monotonic() - started > timeout_s:
            run_command(["scancel", job_id])
            return "TIMEOUT", "0:0"
        time.sleep(max(poll_s, 5.0))


def export_arg(env: Mapping[str, Any]) -> str:
    parts = ["ALL"]
    for key, value in env.items():
        text = str(value)
        if not text:
            continue
        if "," in text or "\n" in text or "\r" in text:
            raise ValueError(f"Cannot pass {key} through sbatch --export because it contains comma/newline.")
        parts.append(f"{key}={text}")
    return ",".join(parts)


def submit_job(
    *,
    script: Path,
    job_name: str,
    partition: str,
    nodes: int,
    gpus_per_node: Optional[int],
    gpus: Optional[int],
    cpus_per_task: int,
    mem: str,
    time_limit: str,
    log_path: Path,
    env: Mapping[str, Any],
) -> str:
    command = [
        "sbatch",
        "--parsable",
        "--partition",
        partition,
        "--nodes",
        str(nodes),
        "--cpus-per-task",
        str(cpus_per_task),
        "--mem",
        mem,
        "--time",
        time_limit,
        "--job-name",
        job_name,
        "--output",
        str(log_path),
        "--export",
        export_arg(env),
    ]
    if gpus_per_node is not None:
        command.extend(["--ntasks-per-node", "1", "--gpus-per-node", str(gpus_per_node)])
    elif gpus is not None:
        command.extend(["--ntasks", "1", "--gpus", str(gpus)])
    command.append(str(script))
    result = run_command(command, check=True)
    return result.stdout.strip().split(";", 1)[0].splitlines()[-1]


def ensure_baseline_validation(args: argparse.Namespace, log_dir: Path) -> Path:
    baseline_metrics = Path(args.baseline_metrics_path)
    if baseline_metrics.exists():
        return baseline_metrics
    eval_root = baseline_metrics.parent.parent
    env = {
        "PROJECT_ROOT": args.project_root,
        "PYTHON_BIN": str(Path(args.project_root) / ".venv-gpu/bin/python"),
        "MODEL_PATH": args.baseline_model_path,
        "MODEL_NAME": "bioreason-pro-rl-paper-validation-baseline",
        "EVAL_SPLIT": "validation",
        "MAX_SAMPLES": args.validation_num_proteins,
        "SAMPLE_STRATEGY": args.sample_strategy,
        "EVALS_PATH": eval_root,
        "CAFA5_DATASET": args.cafa5_dataset,
        "DATASET_NAME": args.dataset_name,
        "REASONING_DATASET_NAME": args.reasoning_dataset_name,
        "DATASET_ARTIFACT": args.dataset_artifact,
        "TEMPORAL_SPLIT_ARTIFACT": args.temporal_split_artifact,
        "IA_FILE_PATH": args.ia_file_path,
        "GO_OBO_PATH": args.go_obo_path,
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_RUN_NAME": f"baseline-validation-{args.run_group}",
        "WEAVE_PROJECT": f"{args.wandb_entity}/{args.wandb_project}",
        "WEAVE_EVAL_NAME": f"baseline-validation-{args.run_group}",
        "KEEP_LOCAL_EVAL_OUTPUTS": "1",
    }
    job_id = submit_job(
        script=Path(args.project_root) / "scripts/sh_eval.sh",
        job_name=f"baseline_val_{args.run_group}",
        partition=args.partition,
        nodes=1,
        gpus_per_node=None,
        gpus=1,
        cpus_per_task=8,
        mem="128G",
        time_limit=args.eval_time_limit,
        log_path=log_dir / f"baseline_validation_%j.log",
        env=env,
    )
    state, exit_code = wait_for_job(job_id, poll_s=args.poll_s, timeout_s=args.eval_timeout_s)
    if state != "COMPLETED" or exit_code != "0:0":
        raise RuntimeError(f"Baseline validation job {job_id} failed: state={state} exit_code={exit_code}")
    if not baseline_metrics.exists():
        raise FileNotFoundError(f"Baseline validation completed but metrics were not found at {baseline_metrics}")
    return baseline_metrics


def summarize_rollout_traces(output_dir: Path) -> Dict[str, Any]:
    total = 0
    with_final = 0
    with_close = 0
    with_go = 0
    go_counts: List[int] = []
    rewards: List[float] = []
    for path in output_dir.glob("rollout_traces.rank*.jsonl"):
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for line in lines:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            completion = str(payload.get("completion", ""))
            total += 1
            if FINAL_ANSWER_OPEN_TAG in completion:
                with_final += 1
            if FINAL_ANSWER_CLOSE_TAG in completion:
                with_close += 1
            go_ids = GO_ID_PATTERN.findall(completion)
            if go_ids:
                with_go += 1
                go_counts.append(len(set(go_ids)))
            reward = payload.get("reward")
            if isinstance(reward, (int, float)):
                rewards.append(float(reward))
    return {
        "rollouts": total,
        "final_answer_open_rate": with_final / total if total else 0.0,
        "final_answer_close_rate": with_close / total if total else 0.0,
        "go_id_rate": with_go / total if total else 0.0,
        "mean_unique_go_ids": sum(go_counts) / len(go_counts) if go_counts else 0.0,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
    }


def classify_failure(log_text: str, state: str) -> str:
    text = log_text.lower()
    if "required eval outputs missing" in text or "wandb init failed" in text or "weave eval logging failed" in text:
        return "eval_tracking"
    if "external validation job" in text:
        return "external_eval"
    if "out of memory" in text or "cuda oom" in text or state == "OUT_OF_MEMORY":
        return "oom"
    if "worknccl" in text or "nccl" in text and "timeout" in text:
        return "nccl_timeout"
    if "deepspeed" in text and "checkpoint" in text:
        return "checkpoint"
    if "dependencyneversatisfied" in text:
        return "dependency"
    if "traceback" in text:
        return "python_exception"
    if state == "TIMEOUT":
        return "timeout"
    return "unknown"


def build_attempt_recipe(base_recipe: Recipe, previous_diagnosis: Optional[Mapping[str, Any]], attempt_index: int) -> Recipe:
    env = dict(base_recipe.env)
    rationale_parts = [base_recipe.rationale]
    if previous_diagnosis:
        failure_class = str(previous_diagnosis.get("failure_class") or "")
        recommended_next = str(previous_diagnosis.get("recommended_next") or "")
        if failure_class:
            rationale_parts.append(f"Previous failure class was {failure_class}; no automatic resource or rollout-size changes are applied.")
        if recommended_next:
            rationale_parts.append(f"Previous recommendation: {recommended_next}")
    return Recipe(name=base_recipe.name, env=env, rationale=" ".join(rationale_parts))


def diagnose_attempt(
    *,
    attempt_dir: Path,
    output_dir: Path,
    train_log: Path,
    state: str,
    exit_code: str,
    baseline_metric: float,
    step10_metrics: Mapping[str, Any],
    recipe: Recipe,
) -> Dict[str, Any]:
    log_text = ""
    if train_log.exists():
        log_text = train_log.read_text(encoding="utf-8", errors="ignore")[-20000:]
    rollout_summary = summarize_rollout_traces(output_dir)
    observed = metric_value(step10_metrics)
    status = "ok" if state == "COMPLETED" and exit_code == "0:0" else "error"
    below_baseline = status == "ok" and observed == observed and observed < baseline_metric
    diagnosis = {
        "status": status,
        "state": state,
        "exit_code": exit_code,
        "recipe": recipe.name,
        "recipe_rationale": recipe.rationale,
        "baseline_metric": baseline_metric,
        "step10_metric": observed,
        "below_baseline": below_baseline,
        "failure_class": classify_failure(log_text, state) if status == "error" else "",
        "rollout_summary": rollout_summary,
        "recommended_next": "",
    }
    if status == "error":
        failure_class = diagnosis["failure_class"]
        if failure_class == "oom":
            diagnosis["recommended_next"] = "Lower vLLM memory pressure, max_num_seqs, rollout count, or microbatch; keep external eval separate."
        elif failure_class == "nccl_timeout":
            diagnosis["recommended_next"] = "Avoid hard filtering/removing rollouts; preserve per-rank tensor shapes and retry with no hard filter."
        elif failure_class == "checkpoint":
            diagnosis["recommended_next"] = "Inspect DeepSpeed checkpoint save path and frozen-parameter handling before retry."
        elif failure_class in {"external_eval", "eval_tracking"}:
            diagnosis["recommended_next"] = "Retry with metrics-complete eval accepted even if W&B/Weave tracking is transiently unavailable."
        else:
            diagnosis["recommended_next"] = "Inspect the saved log tail and patch the failing code path before retry."
    elif below_baseline:
        if rollout_summary["final_answer_open_rate"] < 0.5 or rollout_summary["go_id_rate"] < 0.5:
            diagnosis["recommended_next"] = "Prediction density/format coverage is weak; reduce penalties/gating and add soft structure rewards."
        elif rollout_summary["mean_unique_go_ids"] < 2.0:
            diagnosis["recommended_next"] = "Predictions are too sparse; remove precision pressure or GO-count penalties."
        else:
            diagnosis["recommended_next"] = "Metrics are below baseline without obvious collapse; try the next conservative reward recipe."
    else:
        diagnosis["recommended_next"] = "Step10 validation meets or exceeds baseline; stop guarded loop."
    write_json(attempt_dir / "diagnosis.json", diagnosis)
    report = [
        f"# B guarded attempt diagnosis: {recipe.name}",
        "",
        f"- status: {diagnosis['status']}",
        f"- slurm: {state} {exit_code}",
        f"- baseline validation metric: {baseline_metric}",
        f"- step10 validation metric: {observed}",
        f"- below baseline: {below_baseline}",
        f"- failure class: {diagnosis['failure_class'] or 'n/a'}",
        f"- recommendation: {diagnosis['recommended_next']}",
        "",
        "## Rollout summary",
        json.dumps(rollout_summary, indent=2, sort_keys=True),
    ]
    (attempt_dir / "codex_analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return diagnosis


def find_step10_metrics(output_dir: Path) -> Path:
    candidates = [
        output_dir / "external_eval" / "step-000010" / "cafa_metrics" / "metrics_summary.json",
        output_dir / "external_eval" / "step-000010" / "results" / "cafa_metrics" / "metrics_summary.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(output_dir.glob("external_eval/step-000010/**/metrics_summary.json"))
    if matches:
        return matches[0]
    return candidates[0]


def submit_b_attempt(args: argparse.Namespace, attempt_index: int, recipe: Recipe, log_dir: Path, attempt_dir: Path) -> Tuple[str, Path, Path]:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"rlB-guarded-{args.run_group}-a{attempt_index}-{ts}"
    output_dir = Path(args.runtime_root) / "data/artifacts/models" / f"train_rl_output_B_guarded_{args.run_group}_a{attempt_index}_{ts}"
    train_log = log_dir / f"b_guarded_a{attempt_index}_%j.log"
    notes = (
        f"B guarded attempt {attempt_index}. 2-node training stays alive; external 1-GPU validation at steps 5 and 10; "
        f"DeepSpeed/inference checkpoints saved at eval steps; eval logs to same W&B run on train/global_step. "
        f"Recipe {recipe.name}: {recipe.rationale}"
    )
    env: Dict[str, str] = {
        "ABLATION": "B",
        "TS": ts,
        "MAX_STEPS": "10",
        "VALIDATION_EVERY_N_STEPS": "5",
        "SAVE_EVERY_N_STEPS": "5",
        "CHECKPOINT_EXPORT_ONLY": "false",
        "EXTERNAL_VALIDATION_ENABLED": "true",
        "EXTERNAL_VALIDATION_MAX_SAMPLES": str(args.validation_num_proteins),
        "EXTERNAL_VALIDATION_TIMEOUT_S": str(int(args.eval_timeout_s)),
        "VALIDATION_RANK0_TIMEOUT_S": str(int(args.eval_timeout_s)),
        "OUTPUT_DIR": str(output_dir),
        "BIOREASON_WANDB_RUN_ID": run_id,
        "BIOREASON_WANDB_RESUME": "allow",
        "WANDB_NOTES": notes,
    }
    env.update({key: str(value) for key, value in recipe.env.items()})
    write_json(attempt_dir / "submitted_env.json", env)
    env_file = attempt_dir / "guarded_env.sh"
    env_file.write_text(
        "\n".join(f"export {key}={shlex.quote(str(value))}" for key, value in sorted(env.items())) + "\n",
        encoding="utf-8",
    )
    job_id = submit_job(
        script=Path(args.project_root) / "runtime_logs/run_reward_ablation_paper_rollouts.sh",
        job_name=f"brp_B_guard_a{attempt_index}",
        partition=args.partition,
        nodes=2,
        gpus_per_node=8,
        gpus=None,
        cpus_per_task=64,
        mem="0",
        time_limit=args.train_time_limit,
        log_path=train_log,
        env={"GUARDED_ENV_FILE": str(env_file)},
    )
    write_json(attempt_dir / "job.json", {"job_id": job_id, "run_id": run_id, "output_dir": str(output_dir), "log": str(train_log)})
    return job_id, output_dir, Path(str(train_log).replace("%j", job_id))


def main() -> int:
    parser = argparse.ArgumentParser()
    user = os.environ.get("USER", "kkamata+cwb607")
    parser.add_argument("--project-root", default=f"/mnt/home/{user}/BioReason-Pro")
    parser.add_argument("--runtime-root", default=f"/mnt/data/{user}/BioReason-Pro")
    parser.add_argument("--partition", default="h100")
    parser.add_argument("--wandb-entity", default="wandb-healthcare")
    parser.add_argument("--wandb-project", default="bioreason-pro")
    parser.add_argument("--run-group", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    parser.add_argument("--validation-num-proteins", type=int, default=200)
    parser.add_argument("--sample-strategy", default="stratified_aspect_profile")
    parser.add_argument("--max-improvement-runs", type=int, default=5)
    parser.add_argument("--poll-s", type=float, default=60.0)
    parser.add_argument("--train-time-limit", default="1-00:00:00")
    parser.add_argument("--eval-time-limit", default="12:00:00")
    parser.add_argument("--eval-timeout-s", type=float, default=43200.0)
    parser.add_argument("--train-timeout-s", type=float, default=90000.0)
    parser.add_argument("--baseline-model-path", default=f"/mnt/data/{user}/BioReason-Pro/data/artifacts/models/bioreason_pro_rl_paper")
    parser.add_argument(
        "--cafa5-dataset",
        default=f"/mnt/data/{user}/BioReason-Pro/data/artifacts/datasets/disease_temporal_hc_reasoning_v2/213_221_225_228",
    )
    parser.add_argument("--dataset-name", default="disease_temporal_hc_reasoning_v2")
    parser.add_argument("--reasoning-dataset-name", default="disease_temporal_hc_reasoning_v2")
    parser.add_argument("--dataset-artifact", default="wandb-healthcare/bioreason-pro/disease-temporal-reasoning:production")
    parser.add_argument("--temporal-split-artifact", default="wandb-healthcare/bioreason-pro/disease-temporal-split:production")
    parser.add_argument(
        "--ia-file-path",
        default=f"/mnt/data/{user}/BioReason-Pro/data/artifacts/benchmarks/213_221_225_228/temporal_split/IA.txt",
    )
    parser.add_argument("--go-obo-path", default="bioreason2/dataset/go-basic.obo")
    parser.add_argument(
        "--baseline-metrics-path",
        default=f"/mnt/data/{user}/BioReason-Pro/data/artifacts/eval/bioreason-pro-rl-paper/validation/results/cafa_metrics/metrics_summary.json",
    )
    args = parser.parse_args()

    root = Path(args.runtime_root) / "guarded_b_runs" / args.run_group
    log_dir = root / "logs"
    root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json", vars(args))

    baseline_path = ensure_baseline_validation(args, log_dir)
    baseline_metrics = load_json(baseline_path)
    baseline_score = metric_value(baseline_metrics)
    if baseline_score != baseline_score:
        raise RuntimeError(f"Could not read baseline validation metric from {baseline_path}")
    write_json(root / "baseline.json", {"path": str(baseline_path), "metrics": baseline_metrics, "score": baseline_score})

    max_attempts = 1 + max(int(args.max_improvement_runs), 0)
    recipes = RECIPES[:max_attempts]
    final_status: Dict[str, Any] = {"status": "not_started"}
    previous_diagnosis: Optional[Mapping[str, Any]] = None
    for attempt_index, base_recipe in enumerate(recipes):
        recipe = build_attempt_recipe(base_recipe, previous_diagnosis, attempt_index)
        attempt_dir = root / f"attempt_{attempt_index:02d}_{recipe.name}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        job_id, output_dir, train_log = submit_b_attempt(args, attempt_index, recipe, log_dir, attempt_dir)
        state, exit_code = wait_for_job(job_id, poll_s=args.poll_s, timeout_s=args.train_timeout_s)
        step10_path = find_step10_metrics(output_dir)
        step10_metrics = load_json(step10_path)
        diagnosis = diagnose_attempt(
            attempt_dir=attempt_dir,
            output_dir=output_dir,
            train_log=train_log,
            state=state,
            exit_code=exit_code,
            baseline_metric=baseline_score,
            step10_metrics=step10_metrics,
            recipe=recipe,
        )
        final_status = {"attempt": attempt_index, **diagnosis}
        write_json(root / "latest_status.json", final_status)
        if diagnosis["status"] == "ok" and not diagnosis["below_baseline"]:
            break
        previous_diagnosis = diagnosis
    write_json(root / "final_status.json", final_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

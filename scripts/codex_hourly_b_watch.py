#!/usr/bin/env python3
"""Hourly status collector for B training/eval runs.

This monitor is deliberately read-only. It never changes training parameters,
submits retry jobs, or cancels Slurm jobs. It only snapshots Slurm state,
available validation metrics, and recent log signals so Codex can review the
run on a fixed cadence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


METRIC_KEYS = ("overall_mean_f1", "overall_mean_fmax", "fmax_mean", "mean_fmax")
ERROR_PATTERNS = {
    "oom": re.compile(r"(out of memory|cuda oom)", re.IGNORECASE),
    "nccl": re.compile(r"(nccl|worknccl).*timeout|timeout.*(nccl|worknccl)", re.IGNORECASE),
    "external_eval": re.compile(r"external validation job .*failed", re.IGNORECASE),
    "eval_tracking": re.compile(r"(required eval outputs missing|wandb init failed|weave eval logging failed)", re.IGNORECASE),
    "traceback": re.compile(r"traceback \(most recent call last\)", re.IGNORECASE),
}


def run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def metric_value(metrics: Mapping[str, Any]) -> Optional[float]:
    for key in METRIC_KEYS:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def read_tail(path: Path, limit: int = 40000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return text[-limit:]


def discover_train_log(run_root: Path, job_id: str) -> Optional[Path]:
    candidates = sorted((run_root / "logs").glob(f"*_{job_id}.log"))
    if candidates:
        return candidates[-1]
    candidates = sorted((run_root / "logs").glob("*.log"))
    return candidates[-1] if candidates else None


def discover_metrics(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    found: Dict[str, Dict[str, Any]] = {}
    for step in (5, 10):
        step_name = f"step-{step:06d}"
        matches = sorted((output_dir / "external_eval" / step_name).glob("**/metrics_summary.json"))
        if matches:
            metrics = load_json(matches[0])
            found[step_name] = {
                "path": str(matches[0]),
                "metric": metric_value(metrics),
                "metrics": metrics,
            }
    return found


def classify_log_tail(log_tail: str) -> Dict[str, bool]:
    return {name: bool(pattern.search(log_tail)) for name, pattern in ERROR_PATTERNS.items()}


def snapshot(args: argparse.Namespace) -> Dict[str, Any]:
    run_root = Path(args.runtime_root) / "guarded_b_runs" / args.run_group
    output_dir = Path(args.output_dir)
    train_log = Path(args.train_log) if args.train_log else discover_train_log(run_root, args.job_id)
    log_tail = read_tail(train_log) if train_log else ""
    squeue = run_command(["squeue", "-j", args.job_id])
    sacct = run_command(["sacct", "-j", args.job_id, "--format=JobIDRaw,State,ExitCode,Elapsed", "-P", "-n"])
    metrics = discover_metrics(output_dir)
    payload: Dict[str, Any] = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_group": args.run_group,
        "job_id": args.job_id,
        "output_dir": str(output_dir),
        "train_log": str(train_log) if train_log else "",
        "squeue": squeue.stdout.strip(),
        "sacct": sacct.stdout.strip(),
        "metrics": metrics,
        "log_signals": classify_log_tail(log_tail),
        "recent_log_tail": log_tail[-8000:],
    }
    if args.baseline_metric is not None:
        payload["baseline_metric"] = args.baseline_metric
    latest_metric = None
    for step_name in sorted(metrics):
        if metrics[step_name].get("metric") is not None:
            latest_metric = metrics[step_name]["metric"]
    if latest_metric is not None and args.baseline_metric is not None:
        payload["latest_metric"] = latest_metric
        payload["below_baseline"] = latest_metric < args.baseline_metric
    return payload


def write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    metric_lines = []
    for step_name, info in sorted(metrics.items()):
        if isinstance(info, dict):
            metric_lines.append(f"- {step_name}: {info.get('metric')} ({info.get('path')})")
    if not metric_lines:
        metric_lines.append("- no external validation metrics found yet")
    signals = payload.get("log_signals") if isinstance(payload.get("log_signals"), dict) else {}
    signal_lines = [f"- {key}: {value}" for key, value in sorted(signals.items())]
    text = [
        f"# B hourly Codex status: {payload.get('run_group')}",
        "",
        f"- checked_at: {payload.get('checked_at')}",
        f"- job_id: {payload.get('job_id')}",
        f"- output_dir: {payload.get('output_dir')}",
        f"- train_log: {payload.get('train_log')}",
        f"- latest_metric: {payload.get('latest_metric', 'n/a')}",
        f"- baseline_metric: {payload.get('baseline_metric', 'n/a')}",
        f"- below_baseline: {payload.get('below_baseline', 'n/a')}",
        "",
        "## Slurm",
        "```",
        str(payload.get("sacct", "")),
        "```",
        "",
        "## Metrics",
        *metric_lines,
        "",
        "## Log Signals",
        *signal_lines,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    user = os.environ.get("USER", "kkamata+cwb607")
    parser.add_argument("--runtime-root", default=f"/mnt/data/{user}/BioReason-Pro")
    parser.add_argument("--run-group", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-log", default="")
    parser.add_argument("--baseline-metric", type=float, default=None)
    parser.add_argument("--interval-s", type=float, default=3600.0)
    parser.add_argument("--iterations", type=int, default=0, help="0 means run forever.")
    args = parser.parse_args()

    run_root = Path(args.runtime_root) / "guarded_b_runs" / args.run_group
    watch_dir = run_root / "codex_hourly_watch"
    count = 0
    while True:
        payload = snapshot(args)
        write_json(watch_dir / "latest_status.json", payload)
        append_jsonl(watch_dir / "hourly_status.jsonl", payload)
        write_markdown(watch_dir / "latest_status.md", payload)
        count += 1
        if args.iterations and count >= args.iterations:
            return 0
        time.sleep(max(args.interval_s, 60.0))


if __name__ == "__main__":
    raise SystemExit(main())

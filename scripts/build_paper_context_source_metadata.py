#!/usr/bin/env python3
"""
Build a paper-context source metadata dataset from a temporal split artifact.

The output is a DatasetDict on disk that contains the columns expected by
build_disease_benchmark_datasets.py:

- protein_id
- sequence
- organism
- protein_name
- protein_function
- go_pred
- interpro_formatted
- ppi_formatted

It reuses GO-GPT and InterPro inference utilities from this repo and supports
checkpoint-based resume for long-running jobs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gogpt" / "src"))

import pandas as pd
from datasets import Dataset, DatasetDict


def log(message: str) -> None:
    print(message, flush=True)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def load_json(path: Path, *, resume: bool) -> Dict[str, str]:
    if not resume:
        return {}
    if not path.exists():
        return {}
    with path.open() as handle:
        payload = json.load(handle)
    return {str(key): normalize_text(value) for key, value in payload.items()}


def save_json(path: Path, payload: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", required=True, help="Temporal split artifact directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for metadata build artifacts.")
    parser.add_argument(
        "--input-tsv",
        default="uniprot_protein_metadata.tsv",
        help="Metadata TSV inside --split-dir.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional protein limit for smoke tests.")
    parser.add_argument("--resume", action="store_true", help="Resume from JSON checkpoints and existing rows.")
    parser.add_argument("--interpro-workers", type=int, default=8, help="InterPro online concurrency.")
    parser.add_argument(
        "--interpro-email",
        default=os.getenv("INTERPRO_EMAIL") or "anonymous@example.com",
        help="Email used for InterPro online API submissions.",
    )
    parser.add_argument("--skip-interpro", action="store_true")
    parser.add_argument("--skip-gogpt", action="store_true")
    parser.add_argument("--gogpt-model", default="wanglab/gogpt")
    parser.add_argument("--gogpt-cache-dir", default=os.getenv("HF_HOME") or "")
    parser.add_argument("--ppi-default", default="None")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser.parse_args()


def load_input_rows(split_dir: Path, input_tsv: str, limit: int) -> List[MutableMapping[str, str]]:
    input_path = split_dir / input_tsv
    if not input_path.exists():
        raise FileNotFoundError(f"Metadata TSV not found: {input_path}")

    df = pd.read_csv(input_path, sep="\t").fillna("")
    required = ["protein_id", "sequence", "organism", "protein_name", "protein_function"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Metadata TSV missing required columns: {missing}")

    if limit > 0:
        df = df.head(limit)

    rows = df[required].copy()
    rows["protein_id"] = rows["protein_id"].astype(str)
    return rows.to_dict(orient="records")


def interpro_single(protein_id: str, sequence: str, email: str) -> tuple[str, str]:
    from interpro_api import format_interpro_output, run_interproscan_online

    try:
        result_df = run_interproscan_online(sequence, email=email)
        if result_df.empty:
            return protein_id, "None"
        return protein_id, normalize_text(format_interpro_output(result_df, {})) or "None"
    except Exception as exc:
        log(f"InterPro failed for {protein_id}: {exc}")
        return protein_id, "None"


def run_interpro(rows: List[MutableMapping[str, str]], checkpoint_path: Path, workers: int, email: str, checkpoint_every: int, resume: bool) -> Dict[str, str]:
    results = load_json(checkpoint_path, resume=resume)
    todo = [row for row in rows if normalize_text(results.get(row["protein_id"])) == ""]
    if not todo:
        log(f"InterPro: all {len(rows)} proteins already cached.")
        return results

    max_workers = max(1, min(workers, len(todo)))
    log(f"InterPro: processing {len(todo)} proteins with {max_workers} workers.")
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(interpro_single, row["protein_id"], row["sequence"], email): row["protein_id"]
            for row in todo
        }
        for future in as_completed(futures):
            protein_id, formatted = future.result()
            results[protein_id] = normalize_text(formatted) or "None"
            completed += 1
            if completed % checkpoint_every == 0:
                save_json(checkpoint_path, results)
                log(f"InterPro: checkpointed {completed}/{len(todo)} new rows.")

    save_json(checkpoint_path, results)
    return results


def run_gogpt(rows: List[MutableMapping[str, str]], checkpoint_path: Path, model_name: str, cache_dir: str, checkpoint_every: int, resume: bool) -> Dict[str, str]:
    from gogpt_api import format_go_output, load_predictor, predict_go_terms

    results = load_json(checkpoint_path, resume=resume)
    todo = [row for row in rows if normalize_text(results.get(row["protein_id"])) == ""]
    if not todo:
        log(f"GO-GPT: all {len(rows)} proteins already cached.")
        return results

    log(f"GO-GPT: loading predictor '{model_name}'.")
    predictor = load_predictor(model_name=model_name, cache_dir=normalize_text(cache_dir) or None)
    log(f"GO-GPT: processing {len(todo)} proteins.")
    completed = 0
    try:
        for row in todo:
            protein_id = row["protein_id"]
            try:
                predictions = predict_go_terms(
                    predictor,
                    sequence=row["sequence"],
                    organism=normalize_text(row["organism"]) or "Homo sapiens",
                )
                results[protein_id] = normalize_text(format_go_output(predictions))
            except Exception as exc:
                log(f"GO-GPT failed for {protein_id}: {exc}")
                results[protein_id] = "None"
            completed += 1
            if completed % checkpoint_every == 0:
                save_json(checkpoint_path, results)
                log(f"GO-GPT: checkpointed {completed}/{len(todo)} new rows.")
    finally:
        del predictor
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    save_json(checkpoint_path, results)
    return results


@dataclass
class BuildSummary:
    rows: int
    nonempty_go_pred: int
    nonempty_interpro_formatted: int
    nonempty_ppi_formatted: int
    go_pred_coverage: float
    interpro_coverage: float
    ppi_coverage: float


def build_summary(df: pd.DataFrame) -> BuildSummary:
    rows = int(len(df))
    nonempty_go_pred = int(df["go_pred"].astype(str).str.strip().ne("").sum()) if rows else 0
    nonempty_interpro = int(df["interpro_formatted"].astype(str).str.strip().ne("").sum()) if rows else 0
    nonempty_ppi = int(df["ppi_formatted"].astype(str).str.strip().ne("").sum()) if rows else 0
    return BuildSummary(
        rows=rows,
        nonempty_go_pred=nonempty_go_pred,
        nonempty_interpro_formatted=nonempty_interpro,
        nonempty_ppi_formatted=nonempty_ppi,
        go_pred_coverage=(nonempty_go_pred / rows) if rows else 0.0,
        interpro_coverage=(nonempty_interpro / rows) if rows else 0.0,
        ppi_coverage=(nonempty_ppi / rows) if rows else 0.0,
    )


def save_output_dataset(output_dir: Path, rows: List[MutableMapping[str, str]]) -> None:
    dataset_dir = output_dir / "hf_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).fillna("")
    dataset = Dataset.from_pandas(df, preserve_index=False)
    DatasetDict({"metadata": dataset}).save_to_disk(str(dataset_dir))

    tsv_path = output_dir / "source_metadata.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)

    metadata = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "coverage": asdict(build_summary(df)),
    }
    (output_dir / "build_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


def main() -> int:
    args = parse_args()
    split_dir = Path(args.split_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    rows = load_input_rows(split_dir, args.input_tsv, args.limit)
    log(f"Loaded {len(rows)} proteins from {split_dir / args.input_tsv}.")

    if args.skip_interpro:
        interpro_results = {row["protein_id"]: "None" for row in rows}
    else:
        interpro_results = run_interpro(
            rows,
            checkpoint_path=checkpoints_dir / "interpro_results.json",
            workers=args.interpro_workers,
            email=args.interpro_email,
            checkpoint_every=max(1, args.checkpoint_every),
            resume=args.resume,
        )

    if args.skip_gogpt:
        gogpt_results = {row["protein_id"]: "None" for row in rows}
    else:
        gogpt_results = run_gogpt(
            rows,
            checkpoint_path=checkpoints_dir / "gogpt_results.json",
            model_name=args.gogpt_model,
            cache_dir=args.gogpt_cache_dir,
            checkpoint_every=max(1, args.checkpoint_every),
            resume=args.resume,
        )

    final_rows: List[MutableMapping[str, str]] = []
    for row in rows:
        protein_id = row["protein_id"]
        final_rows.append(
            {
                "protein_id": protein_id,
                "sequence": normalize_text(row["sequence"]),
                "organism": normalize_text(row["organism"]),
                "protein_name": normalize_text(row["protein_name"]),
                "protein_function": normalize_text(row["protein_function"]),
                "go_pred": normalize_text(gogpt_results.get(protein_id)) or "None",
                "interpro_formatted": normalize_text(interpro_results.get(protein_id)) or "None",
                "ppi_formatted": normalize_text(args.ppi_default) or "None",
            }
        )

    save_output_dataset(output_dir, final_rows)
    summary = build_summary(pd.DataFrame(final_rows))
    log(f"Saved source metadata to {output_dir / 'hf_dataset'}")
    log(
        "Coverage: "
        f"go_pred={summary.nonempty_go_pred}/{summary.rows}, "
        f"interpro={summary.nonempty_interpro_formatted}/{summary.rows}, "
        f"ppi={summary.nonempty_ppi_formatted}/{summary.rows}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

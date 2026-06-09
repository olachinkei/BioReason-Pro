#!/usr/bin/env python3
"""Build the Hugging Face DatasetDict used by Senpai training/eval.

This script converts a temporal split artifact into a reasoning dataset with
train, validation, and test splits. Optional source metadata can provide the
paper-context columns used by the default training prompts:

    go_pred, interpro_formatted, ppi_formatted
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import pandas as pd
import requests
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SPLIT_TO_FILE_PREFIX = {
    "train": "train",
    "validation": "dev",
    "test": "test",
}
ASPECT_TO_COLUMN = {
    "P": "go_bp",
    "F": "go_mf",
    "C": "go_cc",
}
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FIELDS = "accession,organism_name,protein_name,cc_function,sequence,xref_mim,xref_orphanet"
REQUIRED_METADATA_COLUMNS = ("sequence", "organism", "protein_name", "protein_function")
PAPER_CONTEXT_COLUMNS = ("go_pred", "interpro_formatted", "ppi_formatted")
GO_ID_PATTERN = re.compile(r"GO:\d{7}")
METADATA_COLUMN_ALIASES = {
    "protein_id": ("protein_id", "Entry", "accession", "primaryAccession", "DB_ID"),
    "protein_name": ("protein_name", "protein_names", "Entry Name"),
    "protein_function": ("protein_function", "cc_function"),
    "interpro_formatted": ("interpro_formatted",),
    "ppi_formatted": ("ppi_formatted",),
    "go_pred": ("go_pred",),
    "interpro_ids": ("interpro_ids",),
    "interpro_location": ("interpro_location",),
    "is_disease_priority": ("is_disease_priority",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-split-dir", required=True)
    parser.add_argument("--reasoning-output-dir", required=True)
    parser.add_argument(
        "--metadata-cache-path",
        default=None,
        help="Cached UniProt metadata TSV. Defaults under the temporal split directory.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument(
        "--source-metadata-dataset",
        default=os.environ.get("BIOREASON_SOURCE_METADATA_DATASET", ""),
        help="Optional HF dataset id or local DatasetDict with paper-context columns.",
    )
    parser.add_argument("--source-metadata-name", default=os.environ.get("BIOREASON_SOURCE_METADATA_NAME", ""))
    parser.add_argument("--source-metadata-local-dir", default=os.environ.get("BIOREASON_SOURCE_METADATA_LOCAL_DIR", ""))
    parser.add_argument("--source-metadata-cache-dir", default=os.environ.get("BIOREASON_SOURCE_METADATA_CACHE_DIR", ""))
    parser.add_argument(
        "--interpro-metadata-dataset",
        default=os.environ.get("BIOREASON_INTERPRO_METADATA_DATASET", ""),
        help="Optional HF dataset id or local DatasetDict with interpro_id, entry_name, type.",
    )
    parser.add_argument("--interpro-metadata-name", default=os.environ.get("BIOREASON_INTERPRO_METADATA_NAME", "interpro_metadata"))
    parser.add_argument("--interpro-metadata-local-dir", default=os.environ.get("BIOREASON_INTERPRO_METADATA_LOCAL_DIR", ""))
    parser.add_argument(
        "--allow-missing-paper-context",
        action="store_true",
        help="Allow empty go_pred/interpro/ppi context. Use only for debug or ablations.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def normalize_optional_bool(value: Any) -> Optional[bool]:
    text = normalize_text(value).lower()
    if not text:
        return None
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def stringify_bool(value: bool) -> str:
    return "true" if value else "false"


def parse_structured_value(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, dict):
        return value
    text = normalize_text(value)
    if not text:
        return []
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        if "," in text:
            return [item.strip() for item in text.split(",") if item.strip()]
        return text


def normalize_multiline_slot(value: Any) -> str:
    parsed = parse_structured_value(value)
    if isinstance(parsed, list):
        return "\n".join(normalize_text(item) for item in parsed if normalize_text(item))
    if isinstance(parsed, dict):
        return "\n".join(f"{key}: {parsed[key]}" for key in sorted(parsed))
    return normalize_text(parsed)


def normalize_go_predictions(value: Any) -> str:
    text = normalize_multiline_slot(value)
    if not text:
        return ""
    go_ids = GO_ID_PATTERN.findall(text)
    if not go_ids:
        return text
    seen: List[str] = []
    for go_id in go_ids:
        if go_id not in seen:
            seen.append(go_id)
    return ", ".join(seen)


def strip_interpro_tokens(value: str) -> str:
    text = normalize_text(value)
    text = text.replace("InterPro terms:", "")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _load_dataset_source(dataset: str, dataset_name: Optional[str] = None, cache_dir: Optional[str] = None):
    dataset_path = Path(os.path.expanduser(dataset))
    if dataset_path.exists():
        if dataset_path.is_dir() and (dataset_path / "dataset_dict.json").exists():
            return load_from_disk(str(dataset_path))
        if dataset_name:
            named_path = dataset_path / dataset_name
            if named_path.is_dir() and (named_path / "dataset_dict.json").exists():
                return load_from_disk(str(named_path))
    if dataset_name:
        return load_dataset(dataset, name=dataset_name, cache_dir=cache_dir or None)
    return load_dataset(dataset, cache_dir=cache_dir or None)


def load_optional_dataset(dataset: str, dataset_name: str = "", local_dir: str = "", cache_dir: str = ""):
    if local_dir:
        return _load_dataset_source(local_dir, dataset_name=dataset_name or None)
    if dataset:
        return _load_dataset_source(dataset, dataset_name=dataset_name or None, cache_dir=cache_dir or None)
    return None


def dataset_to_frames(dataset_obj: Any) -> List[pd.DataFrame]:
    if dataset_obj is None:
        return []
    if isinstance(dataset_obj, DatasetDict):
        return [dataset_obj[split].to_pandas() for split in dataset_obj.keys()]
    if isinstance(dataset_obj, Dataset):
        return [dataset_obj.to_pandas()]
    if isinstance(dataset_obj, Mapping):
        return [dataset_obj[key].to_pandas() for key in dataset_obj.keys()]
    raise TypeError(f"Unsupported dataset object: {type(dataset_obj)!r}")


def first_present_column(frame: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    for alias in aliases:
        if alias in frame.columns:
            return alias
    return None


def collapse_metadata_rows(frame: pd.DataFrame) -> pd.DataFrame:
    def choose_value(values: Iterable[Any]) -> Any:
        for value in values:
            normalized = normalize_text(value)
            if normalized:
                return value
        return ""

    return frame.groupby("protein_id", dropna=False).agg(choose_value).reset_index()


def load_source_metadata_table(args: argparse.Namespace) -> pd.DataFrame:
    dataset_obj = load_optional_dataset(
        args.source_metadata_dataset,
        dataset_name=args.source_metadata_name,
        local_dir=args.source_metadata_local_dir,
        cache_dir=args.source_metadata_cache_dir,
    )
    frames = dataset_to_frames(dataset_obj)
    if not frames:
        return pd.DataFrame(columns=["protein_id", *REQUIRED_METADATA_COLUMNS, *PAPER_CONTEXT_COLUMNS])

    normalized_frames: List[pd.DataFrame] = []
    for frame in frames:
        protein_id_column = first_present_column(frame, METADATA_COLUMN_ALIASES["protein_id"])
        if protein_id_column is None:
            continue
        normalized = pd.DataFrame({"protein_id": frame[protein_id_column].astype(str)})
        for canonical_name, aliases in METADATA_COLUMN_ALIASES.items():
            if canonical_name == "protein_id":
                continue
            column = first_present_column(frame, aliases)
            normalized[canonical_name] = frame[column] if column is not None else ""
        normalized_frames.append(normalized)
    if not normalized_frames:
        raise ValueError("Unable to resolve protein_id from source metadata dataset.")

    merged = pd.concat(normalized_frames, ignore_index=True).fillna("")
    return collapse_metadata_rows(merged)


def load_interpro_metadata_table(args: argparse.Namespace) -> Optional[pd.DataFrame]:
    dataset_obj = load_optional_dataset(
        args.interpro_metadata_dataset,
        dataset_name=args.interpro_metadata_name,
        local_dir=args.interpro_metadata_local_dir,
    )
    frames = dataset_to_frames(dataset_obj)
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True).fillna("")
    required = {"interpro_id", "entry_name", "type"}
    if not required.issubset(set(merged.columns)):
        raise ValueError(f"InterPro metadata is missing columns: {sorted(required - set(merged.columns))}")
    return merged[list(required)].drop_duplicates(subset=["interpro_id"])


def load_split_labels(temporal_split_dir: Path, split: str) -> pd.DataFrame:
    prefix = SPLIT_TO_FILE_PREFIX[split]
    path = temporal_split_dir / f"{prefix}_assigned_labels.tsv"
    frame = pd.read_csv(path, sep="\t")
    expected = {"DB_ID", "GO_ID", "Aspect"}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame[list(expected)].copy()


def aggregate_labels_by_protein(label_df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for protein_id, group in label_df.groupby("DB_ID"):
        row: Dict[str, Any] = {
            "protein_id": str(protein_id),
            "split": split,
            "go_bp": [],
            "go_mf": [],
            "go_cc": [],
        }
        for aspect, column in ASPECT_TO_COLUMN.items():
            row[column] = sorted(group.loc[group["Aspect"] == aspect, "GO_ID"].astype(str).drop_duplicates())
        rows.append(row)
    rows.sort(key=lambda item: item["protein_id"])
    return pd.DataFrame(rows)


def extract_protein_name(entry: Mapping[str, Any]) -> str:
    description = entry.get("proteinDescription") or {}
    recommended = description.get("recommendedName") or {}
    full_name = (recommended.get("fullName") or {}).get("value")
    if full_name:
        return str(full_name)
    for alternative in description.get("alternativeNames") or []:
        value = (alternative.get("fullName") or {}).get("value")
        if value:
            return str(value)
    return ""


def extract_function_text(entry: Mapping[str, Any]) -> str:
    comments = entry.get("comments") or []
    texts: List[str] = []
    for comment in comments:
        for text in comment.get("texts") or []:
            value = text.get("value")
            if value:
                normalized = " ".join(str(value).split())
                if normalized and normalized not in texts:
                    texts.append(normalized)
    return "\n".join(texts)


def extract_disease_xref_flags(entry: Mapping[str, Any]) -> Dict[str, str]:
    refs = entry.get("uniProtKBCrossReferences") or []
    databases = {normalize_text((ref or {}).get("database")).upper() for ref in refs}
    return {
        "has_omim_xref": stringify_bool("MIM" in databases),
        "has_orphanet_xref": stringify_bool("ORPHANET" in databases),
    }


def fetch_uniprot_metadata(
    accessions: Sequence[str],
    cache_path: Path,
    *,
    batch_size: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    columns = [
        "protein_id",
        "sequence",
        "organism",
        "protein_name",
        "protein_function",
        "has_omim_xref",
        "has_orphanet_xref",
    ]
    existing = pd.DataFrame(columns=columns)
    if cache_path.exists():
        existing = pd.read_csv(cache_path, sep="\t").fillna("")
    cached_ids = set(existing["protein_id"].astype(str)) if not existing.empty else set()
    missing_ids = [protein_id for protein_id in accessions if protein_id not in cached_ids]
    if not missing_ids:
        return existing

    session = requests.Session()
    fetched_rows: List[Dict[str, str]] = []
    total_batches = math.ceil(len(missing_ids) / max(batch_size, 1))
    for index in range(total_batches):
        batch = missing_ids[index * batch_size : (index + 1) * batch_size]
        query = " OR ".join(f"accession:{protein_id}" for protein_id in batch)
        response = session.get(
            UNIPROT_SEARCH_URL,
            params={"query": query, "format": "json", "fields": UNIPROT_FIELDS, "size": len(batch)},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        seen: Set[str] = set()
        for entry in payload.get("results") or []:
            accession = normalize_text(entry.get("primaryAccession"))
            if not accession:
                continue
            seen.add(accession)
            flags = extract_disease_xref_flags(entry)
            fetched_rows.append(
                {
                    "protein_id": accession,
                    "sequence": normalize_text((entry.get("sequence") or {}).get("value")),
                    "organism": normalize_text((entry.get("organism") or {}).get("scientificName")),
                    "protein_name": extract_protein_name(entry),
                    "protein_function": extract_function_text(entry),
                    "has_omim_xref": flags["has_omim_xref"],
                    "has_orphanet_xref": flags["has_orphanet_xref"],
                }
            )
        for protein_id in batch:
            if protein_id not in seen:
                fetched_rows.append({column: "" for column in columns} | {"protein_id": protein_id})
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    fetched = pd.DataFrame(fetched_rows).drop_duplicates(subset=["protein_id"], keep="last")
    combined = pd.concat([existing, fetched], ignore_index=True).drop_duplicates(subset=["protein_id"], keep="last")
    ensure_dir(cache_path.parent)
    combined.fillna("").to_csv(cache_path, sep="\t", index=False)
    return combined.fillna("")


def build_split_tables(temporal_split_dir: Path) -> Dict[str, pd.DataFrame]:
    return {
        split: aggregate_labels_by_protein(load_split_labels(temporal_split_dir, split), split)
        for split in ("train", "validation", "test")
    }


def hydrate_context_columns(frame: pd.DataFrame, interpro_metadata: Optional[pd.DataFrame]) -> pd.DataFrame:
    hydrated = frame.copy()
    for column in ("go_pred", "interpro_formatted", "ppi_formatted"):
        if column not in hydrated.columns:
            hydrated[column] = ""
    hydrated["go_pred"] = hydrated["go_pred"].apply(normalize_go_predictions)
    hydrated["ppi_formatted"] = hydrated["ppi_formatted"].apply(normalize_multiline_slot)

    if interpro_metadata is not None:
        from bioreason2.dataset.cafa5.processor import _process_interpro_data

        rebuilt: List[str] = []
        for _, row in hydrated.iterrows():
            interpro_text = strip_interpro_tokens(normalize_text(row.get("interpro_formatted")))
            if not interpro_text:
                working_row = pd.Series(row.to_dict())
                working_row["interpro_ids"] = parse_structured_value(row.get("interpro_ids"))
                working_row["interpro_location"] = normalize_text(row.get("interpro_location"))
                generated, _ = _process_interpro_data(working_row, interpro_metadata)
                interpro_text = strip_interpro_tokens(generated)
            rebuilt.append(interpro_text)
        hydrated["interpro_formatted"] = rebuilt
    else:
        hydrated["interpro_formatted"] = hydrated["interpro_formatted"].apply(strip_interpro_tokens)

    hydrated["interpro_formatted"] = hydrated["interpro_formatted"].apply(lambda value: normalize_text(value) or "None")
    hydrated["ppi_formatted"] = hydrated["ppi_formatted"].apply(lambda value: normalize_text(value) or "None")
    return hydrated


def attach_metadata(
    split_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    interpro_metadata: Optional[pd.DataFrame],
) -> pd.DataFrame:
    merged = split_df.merge(metadata_df.copy(), on="protein_id", how="left").fillna("")
    for column in (*REQUIRED_METADATA_COLUMNS, *PAPER_CONTEXT_COLUMNS, "interpro_ids", "interpro_location"):
        if column not in merged.columns:
            merged[column] = ""
    priorities: List[str] = []
    for _, row in merged.iterrows():
        explicit = normalize_optional_bool(row.get("is_disease_priority"))
        if explicit is not None:
            priorities.append(stringify_bool(explicit))
            continue
        has_omim = normalize_optional_bool(row.get("has_omim_xref")) is True
        has_orphanet = normalize_optional_bool(row.get("has_orphanet_xref")) is True
        priorities.append(stringify_bool(has_omim or has_orphanet))
    merged["is_disease_priority"] = priorities
    return hydrate_context_columns(merged, interpro_metadata)


def summarize_context_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    total = int(len(df))
    coverage: Dict[str, Any] = {"rows": total}
    for column in ("go_pred", "interpro_formatted", "ppi_formatted"):
        coverage[f"nonempty_{column}"] = int(df[column].astype(str).str.strip().ne("").sum()) if column in df else 0
        coverage[f"{column}_coverage"] = coverage[f"nonempty_{column}"] / total if total else 0.0
    coverage["paper_context_ready"] = bool(
        total
        and coverage["nonempty_go_pred"] == total
        and coverage["nonempty_interpro_formatted"] == total
    )
    return coverage


def validate_source_context(metadata_df: pd.DataFrame, allow_missing: bool) -> None:
    if allow_missing:
        return
    if metadata_df.empty:
        raise ValueError(
            "Source metadata is required for paper-native prompts. "
            "Pass --source-metadata-dataset or --source-metadata-local-dir, "
            "or use --allow-missing-paper-context for debug builds."
        )
    missing = [column for column in PAPER_CONTEXT_COLUMNS[:2] if column not in metadata_df.columns]
    if missing:
        raise ValueError(f"Source metadata is missing paper-context columns: {missing}")


def ordered_go_terms_from_record(record: Mapping[str, Any]) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()
    for column in ("go_mf", "go_bp", "go_cc"):
        values = record.get(column, [])
        if isinstance(values, str):
            parsed = parse_structured_value(values)
            values = parsed if isinstance(parsed, list) else GO_ID_PATTERN.findall(values)
        for go_id in values or []:
            normalized = normalize_text(go_id)
            if GO_ID_PATTERN.fullmatch(normalized) and normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)
    return ordered


def build_paper_reasoning_and_final_answer(record: MutableMapping[str, Any]) -> Tuple[str, str]:
    from bioreason2.dataset.cafa5.processor import _build_response

    raw_reasoning, raw_final_answer = _build_response(
        pd.Series(record),
        interpro_metadata=None,
        include_go_defs=True,
        interpro_in_prompt=False,
        predict_interpro=False,
    )
    reasoning_body = normalize_text(raw_reasoning)
    reasoning = "<|REASONING|>\n"
    reasoning += f"{reasoning_body}\n" if reasoning_body else ""
    reasoning += "<|/REASONING|>"

    final_lines = ["<|FINAL_ANSWER|>"]
    final_lines.extend(ordered_go_terms_from_record(record))
    protein_function = normalize_text(record.get("protein_function"))
    if protein_function:
        final_lines.extend(["", "Function summary:", protein_function])

    legacy_text = normalize_text(raw_final_answer)
    for token in ("<|GO_SUMMARY_START|>", "<|GO_SUMMARY_END|>", "<|FUNCTION_SUMMARY_START|>", "<|FUNCTION_SUMMARY_END|>"):
        legacy_text = legacy_text.replace(token, "")
    retained: List[str] = []
    for line in (line.strip() for line in legacy_text.splitlines() if line.strip()):
        if GO_ID_PATTERN.search(line) or line.lower().startswith(("bp:", "mf:", "cc:")):
            continue
        if protein_function and line == protein_function:
            continue
        if line not in retained:
            retained.append(line)
    if retained:
        final_lines.extend(["", *retained])
    final_lines.append("<|/FINAL_ANSWER|>")
    return reasoning, "\n".join(final_lines)


def build_reasoning_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[MutableMapping[str, Any]] = []
    for record in df.to_dict(orient="records"):
        reasoning, final_answer = build_paper_reasoning_and_final_answer(record)
        record["reasoning"] = reasoning
        record["final_answer"] = final_answer
        rows.append(record)
    return pd.DataFrame(rows)


def dataframe_to_dataset(df: pd.DataFrame) -> Dataset:
    normalized = df.copy()
    for list_column in ("go_bp", "go_mf", "go_cc"):
        normalized[list_column] = normalized[list_column].apply(lambda value: value if isinstance(value, list) else [])
    for column in normalized.columns:
        if column not in ("go_bp", "go_mf", "go_cc"):
            normalized[column] = normalized[column].fillna("").astype(str)
    return Dataset.from_pandas(normalized, preserve_index=False)


def write_dataset_dict(split_tables: Mapping[str, pd.DataFrame], output_dir: Path) -> Dict[str, int]:
    dataset_dict = DatasetDict({split: dataframe_to_dataset(df) for split, df in split_tables.items()})
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dataset_dict.save_to_disk(str(output_dir))
    summary = {split: len(dataset_dict[split]) for split in dataset_dict.keys()}
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    temporal_split_dir = Path(args.temporal_split_dir).expanduser().resolve()
    reasoning_output_dir = Path(args.reasoning_output_dir).expanduser().resolve()
    metadata_cache_path = (
        Path(args.metadata_cache_path).expanduser().resolve()
        if args.metadata_cache_path
        else temporal_split_dir / "uniprot_protein_metadata.tsv"
    )

    split_tables = build_split_tables(temporal_split_dir)
    accessions = sorted(
        {
            protein_id
            for split_df in split_tables.values()
            for protein_id in split_df["protein_id"].astype(str).tolist()
        }
    )
    if args.force_refresh and metadata_cache_path.exists():
        metadata_cache_path.unlink()
    metadata_df = fetch_uniprot_metadata(
        accessions,
        metadata_cache_path,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
    )

    source_metadata_df = load_source_metadata_table(args)
    validate_source_context(source_metadata_df, allow_missing=args.allow_missing_paper_context)
    if not source_metadata_df.empty:
        source_metadata_df = source_metadata_df.fillna("")
        source_metadata_df["protein_id"] = source_metadata_df["protein_id"].astype(str)
        drop_columns = [column for column in REQUIRED_METADATA_COLUMNS if column in source_metadata_df.columns]
        metadata_df = metadata_df.merge(source_metadata_df.drop(columns=drop_columns), on="protein_id", how="left")

    interpro_metadata = load_interpro_metadata_table(args)
    reasoning_tables: Dict[str, pd.DataFrame] = {}
    coverage_by_split: Dict[str, Dict[str, Any]] = {}
    for split, split_df in split_tables.items():
        merged = attach_metadata(split_df, metadata_df, interpro_metadata)
        coverage_by_split[split] = summarize_context_coverage(merged)
        reasoning_tables[split] = build_reasoning_columns(merged)

    if not args.allow_missing_paper_context:
        broken = {split: coverage for split, coverage in coverage_by_split.items() if not coverage["paper_context_ready"]}
        if broken:
            raise ValueError(f"Paper context coverage failed: {json.dumps(broken, sort_keys=True)}")

    counts = write_dataset_dict(reasoning_tables, reasoning_output_dir)
    build_metadata = {
        "temporal_split_dir": str(temporal_split_dir),
        "metadata_cache_path": str(metadata_cache_path),
        "reasoning_output_dir": str(reasoning_output_dir),
        "reasoning_counts": counts,
        "context_coverage_by_split": coverage_by_split,
        "proteins": len(accessions),
        "source_metadata_dataset": normalize_text(args.source_metadata_local_dir) or normalize_text(args.source_metadata_dataset),
        "interpro_metadata_dataset": normalize_text(args.interpro_metadata_local_dir) or normalize_text(args.interpro_metadata_dataset),
    }
    (reasoning_output_dir / "build_metadata.json").write_text(
        json.dumps(build_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(build_metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

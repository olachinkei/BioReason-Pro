#!/usr/bin/env python3
"""Build the reasoning dataset from a temporal split artifact."""

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
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import pandas as pd
import requests
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioreason2.dataset.cafa5.processor import _build_response, _process_interpro_data


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
UNIPROT_FIELDS = "accession,organism_name,protein_name,cc_function,sequence"
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
    "interaction_partners": ("interaction_partners",),
    "string_id": ("string_id",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-split-dir", required=True, help="Path to the temporal split artifact directory.")
    parser.add_argument("--reasoning-output-dir", required=True, help="Directory for the reasoning DatasetDict.")
    parser.add_argument(
        "--metadata-cache-path",
        default=None,
        help="Optional path for cached UniProt metadata TSV. Defaults under the temporal split artifact.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument(
        "--source-metadata-dataset",
        default=os.environ.get("BIOREASON_SOURCE_METADATA_DATASET", ""),
        help="Optional HF dataset id or local dataset dir containing paper-context columns such as go_pred.",
    )
    parser.add_argument(
        "--source-metadata-name",
        default=os.environ.get("BIOREASON_SOURCE_METADATA_NAME", ""),
        help="Optional dataset config name for --source-metadata-dataset.",
    )
    parser.add_argument(
        "--source-metadata-local-dir",
        default=os.environ.get("BIOREASON_SOURCE_METADATA_LOCAL_DIR", ""),
        help="Optional local DatasetDict directory containing metadata/context columns.",
    )
    parser.add_argument(
        "--source-metadata-cache-dir",
        default=os.environ.get("BIOREASON_SOURCE_METADATA_CACHE_DIR", ""),
        help="Optional cache dir when loading --source-metadata-dataset from the hub.",
    )
    parser.add_argument(
        "--interpro-metadata-dataset",
        default=os.environ.get("BIOREASON_INTERPRO_METADATA_DATASET", ""),
        help="Optional dataset id or local dir providing InterPro entry_name/type metadata.",
    )
    parser.add_argument(
        "--interpro-metadata-name",
        default=os.environ.get("BIOREASON_INTERPRO_METADATA_NAME", "interpro_metadata"),
        help="Optional dataset config name for --interpro-metadata-dataset.",
    )
    parser.add_argument(
        "--interpro-metadata-local-dir",
        default=os.environ.get("BIOREASON_INTERPRO_METADATA_LOCAL_DIR", ""),
        help="Optional local DatasetDict directory for InterPro metadata.",
    )
    parser.add_argument(
        "--allow-missing-paper-context",
        action="store_true",
        help="Allow building a dataset even when GO-GPT / InterPro / PPI prompt context is missing.",
    )
    return parser.parse_args()


def ensure_dir(path_value: Path) -> Path:
    path_value.mkdir(parents=True, exist_ok=True)
    return path_value


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def parse_structured_value(value: Any) -> Any:
    if isinstance(value, float) and pd.isna(value):
        return None
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped[0] not in "[{(":
        return value
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(stripped)
        except Exception:
            continue
    return value


def normalize_multiline_slot(value: Any) -> str:
    parsed = parse_structured_value(value)
    if parsed is None:
        return ""
    if isinstance(parsed, dict):
        lines = []
        for key, item in parsed.items():
            text = normalize_text(item)
            if text:
                lines.append(f"- {key}: {text}")
        return "\n".join(lines)
    if isinstance(parsed, (list, tuple, set)):
        lines = [normalize_text(item) for item in parsed]
        lines = [line for line in lines if line]
        return "\n".join(f"- {line}" for line in lines)
    return normalize_text(parsed)


def normalize_go_predictions(value: Any) -> str:
    parsed = parse_structured_value(value)
    if parsed is None:
        return ""
    if isinstance(parsed, dict):
        lines: List[str] = []
        aspect_aliases = {
            "mf": "MF",
            "molecular function": "MF",
            "bp": "BP",
            "biological process": "BP",
            "cc": "CC",
            "cellular component": "CC",
        }
        for raw_key, raw_value in parsed.items():
            aspect = aspect_aliases.get(normalize_text(raw_key).lower(), normalize_text(raw_key))
            if isinstance(raw_value, (list, tuple, set)):
                normalized_items = [normalize_text(item) for item in raw_value]
                normalized_items = [item for item in normalized_items if item]
                normalized_slot = ", ".join(normalized_items)
            else:
                normalized_slot = normalize_multiline_slot(raw_value)
            if normalized_slot:
                normalized_slot = normalized_slot.replace("\n", "; ")
                lines.append(f"{aspect}: {normalized_slot}")
        return "\n".join(lines)
    if isinstance(parsed, (list, tuple, set)):
        entries = [normalize_text(item) for item in parsed]
        entries = [entry for entry in entries if entry]
        return ", ".join(entries)
    return normalize_text(parsed)


def strip_interpro_tokens(value: str) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    for token in ("<|INTERPRO_SUMMARY_START|>", "<|INTERPRO_SUMMARY_END|>"):
        text = text.replace(token, "")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _load_dataset_source(dataset: str, dataset_name: Optional[str] = None, cache_dir: Optional[str] = None):
    dataset_path = Path(os.path.expanduser(str(dataset)))
    if dataset_path.exists():
        if dataset_path.is_dir() and (dataset_path / "dataset_dict.json").exists():
            return load_from_disk(str(dataset_path))
        if dataset_name:
            named_path = dataset_path / dataset_name
            if named_path.is_dir() and (named_path / "dataset_dict.json").exists():
                return load_from_disk(str(named_path))
    return load_dataset(dataset, name=dataset_name or None, cache_dir=cache_dir or None)


def load_optional_dataset(
    *,
    dataset: str,
    dataset_name: str = "",
    local_dir: str = "",
    cache_dir: str = "",
):
    source = normalize_text(local_dir) or normalize_text(dataset)
    if not source:
        return None
    return _load_dataset_source(source, dataset_name=normalize_text(dataset_name) or None, cache_dir=normalize_text(cache_dir) or None)


def dataset_to_frames(dataset_obj: Any) -> List[pd.DataFrame]:
    if dataset_obj is None:
        return []
    if isinstance(dataset_obj, dict):
        return [split.to_pandas() for split in dataset_obj.values()]
    return [dataset_obj.to_pandas()]


def first_present_column(frame: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    for alias in aliases:
        if alias in frame.columns:
            return alias
    return None


def collapse_metadata_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    def choose_value(series: pd.Series):
        for item in series.tolist():
            if isinstance(item, list) and item:
                return item
            if isinstance(item, dict) and item:
                return item
            text = normalize_text(item)
            if text:
                return item
        return ""

    grouped = frame.groupby("protein_id", dropna=False).agg(choose_value).reset_index()
    return grouped


def load_source_metadata_table(args: argparse.Namespace) -> pd.DataFrame:
    dataset_obj = load_optional_dataset(
        dataset=args.source_metadata_dataset,
        dataset_name=args.source_metadata_name,
        local_dir=args.source_metadata_local_dir,
        cache_dir=args.source_metadata_cache_dir,
    )
    if dataset_obj is None:
        return pd.DataFrame(columns=["protein_id", *REQUIRED_METADATA_COLUMNS, *PAPER_CONTEXT_COLUMNS])

    frames: List[pd.DataFrame] = []
    for frame in dataset_to_frames(dataset_obj):
        protein_id_column = first_present_column(frame, METADATA_COLUMN_ALIASES["protein_id"])
        if protein_id_column is None:
            continue
        normalized = pd.DataFrame({"protein_id": frame[protein_id_column].astype(str)})
        for canonical_name, aliases in METADATA_COLUMN_ALIASES.items():
            if canonical_name == "protein_id":
                continue
            source_column = first_present_column(frame, aliases)
            if source_column is not None:
                normalized[canonical_name] = frame[source_column]
        frames.append(normalized)

    if not frames:
        raise ValueError(
            "Unable to resolve protein_id from source metadata dataset. "
            "Provide a local metadata DatasetDict with a protein_id/Entry/accession column."
        )

    merged = collapse_metadata_rows(pd.concat(frames, ignore_index=True))
    for column in ("protein_id", *REQUIRED_METADATA_COLUMNS, *PAPER_CONTEXT_COLUMNS, "interpro_ids", "interpro_location"):
        if column not in merged.columns:
            merged[column] = ""
    return merged


def load_interpro_metadata_table(args: argparse.Namespace) -> Optional[pd.DataFrame]:
    dataset_obj = load_optional_dataset(
        dataset=args.interpro_metadata_dataset,
        dataset_name=args.interpro_metadata_name,
        local_dir=args.interpro_metadata_local_dir,
    )
    if dataset_obj is None:
        return None
    frames = dataset_to_frames(dataset_obj)
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    required_columns = {"interpro_id", "entry_name", "type"}
    if not required_columns.issubset(set(merged.columns)):
        return None
    return merged[list(required_columns)].drop_duplicates(subset=["interpro_id"])


def load_split_labels(temporal_split_dir: Path, split: str) -> pd.DataFrame:
    prefix = SPLIT_TO_FILE_PREFIX[split]
    path = temporal_split_dir / f"{prefix}_assigned_labels.tsv"
    df = pd.read_csv(path, sep="\t")
    expected_columns = {"DB_ID", "GO_ID", "Aspect"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df[list(expected_columns)].copy()


def aggregate_labels_by_protein(label_df: pd.DataFrame, split: str) -> pd.DataFrame:
    protein_rows: List[Dict[str, Any]] = []
    for protein_id, group in label_df.groupby("DB_ID"):
        row = {
            "protein_id": str(protein_id),
            "split": split,
            "go_bp": [],
            "go_mf": [],
            "go_cc": [],
        }
        for aspect, column_name in ASPECT_TO_COLUMN.items():
            terms = sorted(group.loc[group["Aspect"] == aspect, "GO_ID"].astype(str).drop_duplicates().tolist())
            row[column_name] = terms
        protein_rows.append(row)

    protein_rows.sort(key=lambda item: item["protein_id"])
    return pd.DataFrame(protein_rows)


def extract_protein_name(entry: Mapping[str, Any]) -> str:
    description = entry.get("proteinDescription") or {}
    recommended = description.get("recommendedName") or {}
    full_name = (recommended.get("fullName") or {}).get("value")
    if full_name:
        return str(full_name)

    for alt in description.get("alternativeNames") or []:
        value = (alt.get("fullName") or {}).get("value")
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


def fetch_uniprot_metadata(accessions: Sequence[str], cache_path: Path, *, batch_size: int, sleep_seconds: float) -> pd.DataFrame:
    existing = pd.DataFrame(columns=["protein_id", "sequence", "organism", "protein_name", "protein_function"])
    if cache_path.exists():
        existing = pd.read_csv(cache_path, sep="\t")
        existing = existing.fillna("")

    cached_ids = set(existing["protein_id"].astype(str).tolist()) if not existing.empty else set()
    missing_ids = [protein_id for protein_id in accessions if protein_id not in cached_ids]
    if not missing_ids:
        return existing

    session = requests.Session()
    fetched_rows: List[Dict[str, str]] = []
    total_batches = math.ceil(len(missing_ids) / batch_size)

    for index in range(total_batches):
        batch = missing_ids[index * batch_size : (index + 1) * batch_size]
        query = " OR ".join(f"accession:{protein_id}" for protein_id in batch)
        params = {
            "query": query,
            "format": "json",
            "fields": UNIPROT_FIELDS,
            "size": len(batch),
        }
        response = session.get(UNIPROT_SEARCH_URL, params=params, timeout=120)
        response.raise_for_status()
        payload = response.json()
        seen: set[str] = set()
        for entry in payload.get("results") or []:
            accession = str(entry.get("primaryAccession") or "").strip()
            if not accession:
                continue
            seen.add(accession)
            fetched_rows.append(
                {
                    "protein_id": accession,
                    "sequence": str((entry.get("sequence") or {}).get("value") or ""),
                    "organism": str((entry.get("organism") or {}).get("scientificName") or ""),
                    "protein_name": extract_protein_name(entry),
                    "protein_function": extract_function_text(entry),
                }
            )

        missing_after_batch = [protein_id for protein_id in batch if protein_id not in seen]
        for protein_id in missing_after_batch:
            fetched_rows.append(
                {
                    "protein_id": protein_id,
                    "sequence": "",
                    "organism": "",
                    "protein_name": "",
                    "protein_function": "",
                }
            )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    fetched_df = pd.DataFrame(fetched_rows).drop_duplicates(subset=["protein_id"], keep="last")
    combined = pd.concat([existing, fetched_df], ignore_index=True).drop_duplicates(subset=["protein_id"], keep="last")
    combined = combined.fillna("")
    ensure_dir(cache_path.parent)
    combined.to_csv(cache_path, sep="\t", index=False)
    return combined


def build_split_tables(temporal_split_dir: Path) -> Dict[str, pd.DataFrame]:
    return {
        split: aggregate_labels_by_protein(load_split_labels(temporal_split_dir, split), split)
        for split in ("train", "validation", "test")
    }


def hydrate_context_columns(merged: pd.DataFrame, interpro_metadata: Optional[pd.DataFrame]) -> pd.DataFrame:
    hydrated = merged.copy()
    for column in ("go_pred", "interpro_formatted", "ppi_formatted"):
        if column not in hydrated.columns:
            hydrated[column] = ""

    hydrated["go_pred"] = hydrated["go_pred"].apply(normalize_go_predictions)
    hydrated["ppi_formatted"] = hydrated["ppi_formatted"].apply(normalize_multiline_slot)

    if interpro_metadata is not None:
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

    hydrated["interpro_formatted"] = hydrated["interpro_formatted"].apply(
        lambda value: normalize_text(value) or "None"
    )
    hydrated["ppi_formatted"] = hydrated["ppi_formatted"].apply(
        lambda value: normalize_text(value) or "None"
    )

    return hydrated


def attach_metadata(split_df: pd.DataFrame, metadata_df: pd.DataFrame, interpro_metadata: Optional[pd.DataFrame]) -> pd.DataFrame:
    metadata = metadata_df.copy()
    metadata["protein_id"] = metadata["protein_id"].astype(str)
    merged = split_df.merge(metadata, on="protein_id", how="left")
    for column in ("sequence", "organism", "protein_name", "protein_function", "go_pred", "interpro_formatted", "ppi_formatted", "interpro_ids", "interpro_location"):
        if column not in merged.columns:
            merged[column] = ""
    merged = merged.fillna("")
    return hydrate_context_columns(merged, interpro_metadata)


def summarize_context_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    total = int(len(df))
    coverage: Dict[str, Any] = {
        "rows": total,
        "nonempty_go_pred": int(df["go_pred"].astype(str).str.strip().ne("").sum()) if "go_pred" in df.columns else 0,
        "nonempty_interpro_formatted": int(df["interpro_formatted"].astype(str).str.strip().ne("").sum()) if "interpro_formatted" in df.columns else 0,
        "nonempty_ppi_formatted": int(df["ppi_formatted"].astype(str).str.strip().ne("").sum()) if "ppi_formatted" in df.columns else 0,
    }
    coverage["go_pred_coverage"] = (coverage["nonempty_go_pred"] / total) if total else 0.0
    coverage["interpro_coverage"] = (coverage["nonempty_interpro_formatted"] / total) if total else 0.0
    coverage["ppi_coverage"] = (coverage["nonempty_ppi_formatted"] / total) if total else 0.0
    coverage["paper_context_ready"] = bool(
        total
        and coverage["nonempty_go_pred"] == total
        and coverage["nonempty_interpro_formatted"] == total
    )
    return coverage


def validate_source_context(metadata_df: pd.DataFrame, allow_missing_paper_context: bool) -> None:
    if metadata_df.empty:
        raise ValueError(
            "Source metadata dataset is required to populate paper-native prompt context. "
            "Pass --source-metadata-dataset or --source-metadata-local-dir."
        )
    missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in metadata_df.columns]
    if missing:
        raise ValueError(f"Source metadata is missing required columns: {missing}")
    if allow_missing_paper_context:
        return
    go_pred_nonempty = metadata_df["go_pred"].apply(lambda value: normalize_text(value) != "").all()
    interpro_nonempty = metadata_df["interpro_formatted"].apply(lambda value: normalize_text(value) != "").all()
    if not bool(go_pred_nonempty):
        raise ValueError(
            "Source metadata must provide non-empty go_pred for every protein. "
            "The current source does not satisfy the paper initial-hypotheses contract."
        )
    if not bool(interpro_nonempty):
        raise ValueError(
            "Source metadata must provide non-empty interpro_formatted for every protein. "
            "Use an explicit 'None' marker when no InterPro annotation is available."
        )


def ordered_go_terms_from_record(record: Mapping[str, Any]) -> List[str]:
    ordered: List[str] = []
    seen = set()
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


def build_paper_reasoning_and_final_answer(record: MutableMapping[str, Any]) -> tuple[str, str]:
    response_row = pd.Series(record)
    raw_reasoning, raw_final_answer = _build_response(
        response_row,
        interpro_metadata=None,
        include_go_defs=True,
        interpro_in_prompt=False,
        predict_interpro=False,
    )

    reasoning_body = normalize_text(raw_reasoning)
    if reasoning_body.startswith("<|REASONING|>"):
        reasoning = reasoning_body
    else:
        reasoning = "<|REASONING|>\n"
        reasoning += f"{reasoning_body}\n" if reasoning_body else ""
        reasoning += "<|/REASONING|>"

    go_terms = ordered_go_terms_from_record(record)
    final_lines = ["<|FINAL_ANSWER|>"]
    final_lines.extend(go_terms)

    protein_function = normalize_text(record.get("protein_function"))
    if protein_function:
        final_lines.extend(["", "Function summary:", protein_function])

    ppi_formatted = normalize_text(record.get("ppi_formatted"))
    if ppi_formatted:
        final_lines.extend(["", "Hypothesized interaction partners:", ppi_formatted])

    # Keep a small amount of legacy structured information only if it adds
    # non-GO text that is not already covered by the protein_function summary.
    legacy_text = normalize_text(raw_final_answer)
    legacy_text = legacy_text.replace("<|GO_SUMMARY_START|>", "")
    legacy_text = legacy_text.replace("<|GO_SUMMARY_END|>", "")
    legacy_text = legacy_text.replace("<|FUNCTION_SUMMARY_START|>", "")
    legacy_text = legacy_text.replace("<|FUNCTION_SUMMARY_END|>", "")
    legacy_lines = [line.strip() for line in legacy_text.splitlines() if line.strip()]
    retained_legacy = []
    for line in legacy_lines:
        if GO_ID_PATTERN.search(line):
            continue
        if protein_function and line == protein_function:
            continue
        if line.lower().startswith(("bp:", "mf:", "cc:")):
            continue
        if line not in retained_legacy:
            retained_legacy.append(line)
    if retained_legacy:
        final_lines.extend(["", *retained_legacy])

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
    dataset_dict = DatasetDict(
        {
            split: dataframe_to_dataset(df)
            for split, df in split_tables.items()
        }
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dataset_dict.save_to_disk(str(output_dir))
    summary = {split: len(dataset_dict[split]) for split in dataset_dict.keys()}
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    temporal_split_dir = Path(args.temporal_split_dir).resolve()
    reasoning_output_dir = Path(args.reasoning_output_dir).resolve()
    metadata_cache_path = (
        Path(args.metadata_cache_path).resolve()
        if args.metadata_cache_path
        else (temporal_split_dir / "uniprot_protein_metadata.tsv")
    )

    split_tables = build_split_tables(temporal_split_dir)
    all_accessions = sorted(
        {
            protein_id
            for split_df in split_tables.values()
            for protein_id in split_df["protein_id"].astype(str).tolist()
        }
    )

    if args.force_refresh and metadata_cache_path.exists():
        metadata_cache_path.unlink()

    metadata_df = fetch_uniprot_metadata(
        all_accessions,
        metadata_cache_path,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
    )
    source_metadata_df = load_source_metadata_table(args)
    validate_source_context(source_metadata_df, allow_missing_paper_context=args.allow_missing_paper_context)
    source_metadata_df = source_metadata_df.fillna("")
    source_metadata_df["protein_id"] = source_metadata_df["protein_id"].astype(str)
    metadata_df = metadata_df.merge(
        source_metadata_df.drop(columns=[column for column in REQUIRED_METADATA_COLUMNS if column in source_metadata_df.columns], errors="ignore"),
        on="protein_id",
        how="left",
    )
    interpro_metadata = load_interpro_metadata_table(args)

    reasoning_tables: Dict[str, pd.DataFrame] = {}
    context_coverage_by_split: Dict[str, Dict[str, Any]] = {}
    for split, split_df in split_tables.items():
        merged = attach_metadata(split_df, metadata_df, interpro_metadata)
        context_coverage_by_split[split] = summarize_context_coverage(merged)
        reasoning_tables[split] = build_reasoning_columns(merged)

    if not args.allow_missing_paper_context:
        not_ready = {
            split: coverage
            for split, coverage in context_coverage_by_split.items()
            if not coverage["paper_context_ready"]
        }
        if not_ready:
            raise ValueError(
                "Refusing to build/upload a broken reasoning dataset. "
                f"Paper context coverage failed for splits: {json.dumps(not_ready, sort_keys=True)}"
            )

    reasoning_summary = write_dataset_dict(reasoning_tables, reasoning_output_dir)

    build_summary = {
        "temporal_split_dir": str(temporal_split_dir),
        "metadata_cache_path": str(metadata_cache_path),
        "reasoning_output_dir": str(reasoning_output_dir),
        "reasoning_counts": reasoning_summary,
        "context_coverage_by_split": context_coverage_by_split,
        "source_metadata_dataset": normalize_text(args.source_metadata_local_dir) or normalize_text(args.source_metadata_dataset),
        "interpro_metadata_dataset": normalize_text(args.interpro_metadata_local_dir) or normalize_text(args.interpro_metadata_dataset),
        "proteins": len(all_accessions),
    }
    (reasoning_output_dir / "build_metadata.json").write_text(
        json.dumps(build_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(build_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

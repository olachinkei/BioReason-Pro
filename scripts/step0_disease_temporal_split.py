#!/usr/bin/env python3
"""
Construct a disease-focused temporal split for BioReason-Pro Step 0.

This script:
1. Builds a reviewed human disease-protein shortlist from UniProt.
2. Downloads and filters UniProt GOA GAF releases.
3. Computes temporal deltas across release windows.
4. Makes protein-disjoint train/dev/test splits by earliest appearance.
5. Computes NK/LK statistics against the CAFA5 training split.
6. Saves per-window TSVs plus a machine-readable summary JSON.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import requests
from cafaeval.graph import propagate
from cafaeval.parser import obo_parser
from datasets import load_dataset


EXPERIMENTAL_EVIDENCE_CODES = ["IDA", "IPI", "EXP", "IGI", "IMP", "IEP", "IC", "TAS"]
GAF_COLUMNS = [
    "DB",
    "DB_ID",
    "DB_Symbol",
    "Qualifier",
    "GO_ID",
    "DB_Reference",
    "Evidence_Code",
    "With_From",
    "Aspect",
    "DB_Name",
    "DB_Synonym",
    "DB_Type",
    "Taxon",
    "Date",
    "Assigned_By",
    "Extension",
    "Gene_Product_Form_ID",
]
RELEASE_TO_DATE = {
    212: "2022-11-17",
    213: "2022-09-16",
    214: "2023-02-02",
    215: "2023-02-07",
    216: "2023-03-15",
    217: "2023-05-18",
    218: "2023-07-12",
    219: "2023-09-21",
    220: "2023-12-04",
    221: "2024-02-09",
    222: "2024-04-16",
    223: "2024-06-14",
    224: "2024-08-01",
    225: "2024-10-20",
    226: "2024-12-21",
    227: "2025-03-07",
    228: "2025-05-03",
}
MAIN_SHORTLIST_QUERY = (
    "reviewed:true AND organism_id:9606 AND cc_disease:* "
    "AND (go_exp:* OR go_ida:* OR go_ipi:* OR go_igi:* "
    "OR go_imp:* OR go_iep:* OR go_ic:* OR go_tas:*)"
)
HIGH_CONFIDENCE_SHORTLIST_QUERY = (
    MAIN_SHORTLIST_QUERY
    + " AND (xref:mim-* OR xref:orphanet-*)"
)
SHORTLIST_FIELDS = "accession,id"
UNIPROT_BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
GOA_OLD_UNIPROT_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/GO/goa/old/UNIPROT"
GOA_OLD_HUMAN_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/GO/goa/old/HUMAN"
GO_OBO_URL = "https://purl.obolibrary.org/obo/go/go-basic.obo"
ASPECT_TO_NAMESPACE = {
    "P": "biological_process",
    "F": "molecular_function",
    "C": "cellular_component",
}
ASPECT_NAMES = {"P": "BP", "F": "MF", "C": "CC"}
RELEASE_TO_URL = {
    212: f"{GOA_OLD_UNIPROT_BASE_URL}/goa_uniprot_all.gaf.212.gz",
    213: f"{GOA_OLD_HUMAN_BASE_URL}/goa_human.gaf.213.gz",
    214: f"{GOA_OLD_HUMAN_BASE_URL}/goa_human.gaf.214.gz",
    215: f"{GOA_OLD_HUMAN_BASE_URL}/goa_human.gaf.215.gz",
    216: f"{GOA_OLD_HUMAN_BASE_URL}/goa_human.gaf.216.gz",
    217: f"{GOA_OLD_HUMAN_BASE_URL}/goa_human.gaf.217.gz",
    218: f"{GOA_OLD_HUMAN_BASE_URL}/goa_human.gaf.218.gz",
    219: f"{GOA_OLD_HUMAN_BASE_URL}/goa_human.gaf.219.gz",
    220: f"{GOA_OLD_HUMAN_BASE_URL}/goa_human.gaf.220.gz",
    221: f"{GOA_OLD_HUMAN_BASE_URL}/goa_human.gaf.221.gz",
    222: f"{GOA_OLD_HUMAN_BASE_URL}/goa_human.gaf.222.gz",
    223: f"{GOA_OLD_HUMAN_BASE_URL}/goa_human.gaf.223.gz",
    224: f"{GOA_OLD_HUMAN_BASE_URL}/goa_human.gaf.224.gz",
    225: f"{GOA_OLD_HUMAN_BASE_URL}/goa_human.gaf.225.gz",
    226: f"{GOA_OLD_HUMAN_BASE_URL}/goa_human.gaf.226.gz",
    227: f"{GOA_OLD_HUMAN_BASE_URL}/goa_human.gaf.227.gz",
    228: f"{GOA_OLD_HUMAN_BASE_URL}/goa_human.gaf.228.gz",
}
RELEASE_TO_ARCHIVE = {
    212: "goa_uniprot_all.gaf.212.gz",
    213: "goa_human.gaf.213.gz",
    214: "goa_human.gaf.214.gz",
    215: "goa_human.gaf.215.gz",
    216: "goa_human.gaf.216.gz",
    217: "goa_human.gaf.217.gz",
    218: "goa_human.gaf.218.gz",
    219: "goa_human.gaf.219.gz",
    220: "goa_human.gaf.220.gz",
    221: "goa_human.gaf.221.gz",
    222: "goa_human.gaf.222.gz",
    223: "goa_human.gaf.223.gz",
    224: "goa_human.gaf.224.gz",
    225: "goa_human.gaf.225.gz",
    226: "goa_human.gaf.226.gz",
    227: "goa_human.gaf.227.gz",
    228: "goa_human.gaf.228.gz",
}


@dataclass
class SplitSummary:
    split: str
    start_release: int
    end_release: int
    start_date: str
    end_date: str
    disease_raw_records: int
    disease_unique_labels: int
    disease_proteins_before_assignment: int
    disease_proteins_after_assignment: int
    unique_labels_after_assignment: int
    propagated_labels_after_assignment: int
    nk_proteins: int
    lk_proteins: int
    nk_lk_proteins: int
    nk_raw_records: int
    lk_raw_records: int
    nk_lk_raw_records: int
    nk_lk_propagated_labels: int
    avg_unique_labels_per_protein: float
    avg_propagated_labels_per_protein: float
    aspect_counts_after_assignment: Dict[str, int]
    propagated_aspect_counts_after_assignment: Dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="domain/specification/busiless-rules/artifacts/step0_human_ub_20260406",
        help="Directory where intermediate files and summaries will be written.",
    )
    parser.add_argument(
        "--train-start-release",
        type=int,
        default=213,
        help="GOA release used as the train baseline snapshot. Default 213 (2022-09-16, current implementation baseline).",
    )
    parser.add_argument(
        "--train-end-release",
        type=int,
        default=221,
        help="GOA release used as the train/dev boundary. Default 221 (2024-02-09, HUMAN archive).",
    )
    parser.add_argument(
        "--dev-end-release",
        type=int,
        default=225,
        help="GOA release used as the dev/test boundary. Default 225 (2024-10-20, HUMAN archive).",
    )
    parser.add_argument(
        "--test-end-release",
        type=int,
        default=228,
        help="GOA release used as the test end snapshot. Default 228 (2025-05-03, HUMAN archive).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download remote assets even if local files already exist.",
    )
    parser.add_argument(
        "--use-shell-filter",
        action="store_true",
        help="Prefer gzip/awk filtering over the Python fallback when available.",
    )
    parser.add_argument(
        "--skip-propagation",
        action="store_true",
        help="Skip GO propagation if you only need raw label counts.",
    )
    parser.add_argument(
        "--shortlist-mode",
        choices=["main", "high-confidence"],
        default="high-confidence",
        help=(
            "Protein shortlist definition. "
            "'main' uses disease comment + experimental GO; "
            "'high-confidence' additionally requires MIM/Orphanet cross-references."
        ),
    )
    return parser.parse_args()


def build_windows(args: argparse.Namespace) -> List[Tuple[str, int, int]]:
    releases = [
        args.train_start_release,
        args.train_end_release,
        args.dev_end_release,
        args.test_end_release,
    ]
    missing = [release for release in releases if release not in RELEASE_TO_DATE or release not in RELEASE_TO_URL]
    if missing:
        raise ValueError(f"Release metadata is missing for: {sorted(set(missing))}")
    return [
        ("train", args.train_start_release, args.train_end_release),
        ("dev", args.train_end_release, args.dev_end_release),
        ("test", args.dev_end_release, args.test_end_release),
    ]


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and not force:
        log(f"[download] reuse {dest.name}")
        return
    log(f"[download] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def get_next_link(headers: requests.structures.CaseInsensitiveDict[str]) -> Optional[str]:
    link = headers.get("Link")
    if not link:
        return None
    match = re.search(r'<([^>]+)>;\s*rel="next"', link)
    return match.group(1) if match else None


def shortlist_query_for_mode(mode: str) -> str:
    if mode == "main":
        return MAIN_SHORTLIST_QUERY
    if mode == "high-confidence":
        return HIGH_CONFIDENCE_SHORTLIST_QUERY
    raise ValueError(f"Unsupported shortlist mode: {mode}")


def fetch_shortlist(output_path: Path, shortlist_query: str) -> pd.DataFrame:
    session = requests.Session()
    params = {
        "query": shortlist_query,
        "format": "tsv",
        "fields": SHORTLIST_FIELDS,
        "size": 500,
    }
    try:
        response = session.get(UNIPROT_BASE_URL, params=params, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        if output_path.exists():
            log(f"[shortlist] live refresh failed; reuse cached shortlist at {output_path} ({exc})")
            return pd.read_csv(output_path, sep="\t")
        raise
    expected_total = int(response.headers.get("x-total-results", "0") or 0)
    first_page = pd.read_csv(io.StringIO(response.text), sep="\t")

    if output_path.exists():
        existing = pd.read_csv(output_path, sep="\t")
        if expected_total and len(existing) == expected_total:
            log(f"[shortlist] reuse {output_path} ({len(existing):,} proteins)")
            return existing
        log(
            f"[shortlist] cached file has {len(existing):,} rows but live query reports "
            f"{expected_total:,}; rebuilding"
        )

    rows: List[pd.DataFrame] = [first_page]
    url = get_next_link(response.headers)
    page = 1
    log(f"[shortlist] page {page}: {len(first_page):,} rows")

    while url:
        response = session.get(url, timeout=120)
        response.raise_for_status()
        text = response.text
        page_df = pd.read_csv(io.StringIO(text), sep="\t")
        rows.append(page_df)
        page += 1
        log(f"[shortlist] page {page}: {len(page_df):,} rows")
        url = get_next_link(response.headers)

    shortlist = pd.concat(rows, ignore_index=True).drop_duplicates()
    if expected_total and len(shortlist) != expected_total:
        raise RuntimeError(
            f"UniProt shortlist is incomplete: expected {expected_total:,} rows, got {len(shortlist):,}"
        )
    shortlist.to_csv(output_path, sep="\t", index=False)
    log(f"[shortlist] saved {len(shortlist):,} proteins to {output_path}")
    return shortlist


def filtered_gaf_path(asset_dir: Path, release: int) -> Path:
    return asset_dir / f"filtered_goa_uniprot_disease_exp-protein-only_{release}.gaf"


def compressed_gaf_path(asset_dir: Path, release: int) -> Path:
    return asset_dir / RELEASE_TO_ARCHIVE[release]


def goa_url_for_release(release: int) -> str:
    if release not in RELEASE_TO_URL:
        raise KeyError(f"No GOA archive configured for release anchor {release}")
    return RELEASE_TO_URL[release]


def maybe_filter_remote_with_shell(url: str, shortlist_path: Path, output_path: Path) -> bool:
    curl_cmd = shutil.which("curl")
    gzip_cmd = shutil.which("pigz") or shutil.which("gzip")
    awk_cmd = shutil.which("awk")
    if not curl_cmd or not gzip_cmd or not awk_cmd:
        return False

    awk_program = (
        r'NR==FNR { if (FNR > 1) keep[$1]=1; next } '
        r'!/^!/ && $1=="UniProtKB" && ($2 in keep) && '
        r'$7~/^(IDA|IPI|EXP|IGI|IMP|IEP|IC|TAS)$/ && $12=="protein"'
    )
    cmd = (
        f"{curl_cmd} -Lsf {shlex.quote(url)} | "
        f"{gzip_cmd} -dc | "
        f"{awk_cmd} -F'\\t' {shlex.quote(awk_program)} "
        f"{shlex.quote(str(shortlist_path))} - > {shlex.quote(str(output_path))}"
    )
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        if output_path.exists():
            output_path.unlink()
        return False
    return True


def maybe_filter_local_with_shell(input_path: Path, shortlist_path: Path, output_path: Path) -> bool:
    gzip_cmd = shutil.which("pigz") or shutil.which("gzip")
    awk_cmd = shutil.which("awk")
    if not gzip_cmd or not awk_cmd:
        return False

    awk_program = (
        r'NR==FNR { if (FNR > 1) keep[$1]=1; next } '
        r'!/^!/ && $1=="UniProtKB" && ($2 in keep) && '
        r'$7~/^(IDA|IPI|EXP|IGI|IMP|IEP|IC|TAS)$/ && $12=="protein"'
    )
    cmd = (
        f"{gzip_cmd} -dc {shlex.quote(str(input_path))} | "
        f"{awk_cmd} -F'\\t' {shlex.quote(awk_program)} "
        f"{shlex.quote(str(shortlist_path))} - > {shlex.quote(str(output_path))}"
    )
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        if output_path.exists():
            output_path.unlink()
        return False
    return True


def filter_gaf(
    input_path: Path,
    output_path: Path,
    shortlist_path: Path,
    shortlist_set: Set[str],
    use_shell_filter: bool,
) -> None:
    if output_path.exists():
        log(f"[filter] reuse {output_path.name}")
        return

    log(f"[filter] {input_path.name} -> {output_path.name}")
    if use_shell_filter and maybe_filter_local_with_shell(input_path, shortlist_path, output_path):
        return

    with gzip.open(input_path, "rt") as src, output_path.open("w") as dst:
        for line in src:
            if line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 12:
                continue
            if cols[0] != "UniProtKB":
                continue
            if cols[6] not in EXPERIMENTAL_EVIDENCE_CODES:
                continue
            if cols[11] != "protein":
                continue
            if cols[1] not in shortlist_set:
                continue
            dst.write(line)


def prepare_filtered_gaf(
    release: int,
    asset_dir: Path,
    shortlist_path: Path,
    shortlist_set: Set[str],
    use_shell_filter: bool,
    force_download: bool,
) -> Path:
    output_path = filtered_gaf_path(asset_dir, release)
    if output_path.exists() and not force_download:
        log(f"[filter] reuse {output_path.name}")
        return output_path

    if force_download and output_path.exists():
        output_path.unlink()

    remote_url = goa_url_for_release(release)
    log(f"[filter] prepare release {release} from {remote_url}")
    if use_shell_filter and maybe_filter_remote_with_shell(remote_url, shortlist_path, output_path):
        log(f"[filter] ready {output_path.name} ({output_path.stat().st_size / 1024**2:.2f} MiB)")
        return output_path

    gz_path = compressed_gaf_path(asset_dir, release)
    download_file(remote_url, gz_path, force=force_download)
    filter_gaf(
        gz_path,
        output_path,
        shortlist_path=shortlist_path,
        shortlist_set=shortlist_set,
        use_shell_filter=use_shell_filter,
    )
    log(f"[filter] ready {output_path.name} ({output_path.stat().st_size / 1024**2:.2f} MiB)")
    return output_path


def load_filtered_gaf(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        names=GAF_COLUMNS,
        header=None,
        usecols=["DB_ID", "GO_ID", "Evidence_Code", "Aspect", "Date"],
        dtype=str,
        low_memory=False,
    )


def collapse_labels(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["DB_ID", "GO_ID", "Aspect"]
    return df[cols].drop_duplicates().reset_index(drop=True)


def raw_records_for_labels(raw_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df.merge(label_df, on=["DB_ID", "GO_ID", "Aspect"], how="inner")


def aspect_counts(df: pd.DataFrame) -> Dict[str, int]:
    counts = df["Aspect"].value_counts().to_dict()
    return {ASPECT_NAMES[k]: int(counts.get(k, 0)) for k in ["F", "P", "C"]}


def compute_delta(old_df: pd.DataFrame, new_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_key_cols = ["DB_ID", "GO_ID", "Aspect", "Evidence_Code"]
    label_key_cols = ["DB_ID", "GO_ID", "Aspect"]

    old_raw = old_df.drop_duplicates(raw_key_cols).copy()
    new_raw = new_df.drop_duplicates(raw_key_cols).copy()

    old_labels = old_raw[label_key_cols].drop_duplicates()
    new_labels = new_raw[label_key_cols].drop_duplicates()

    old_keys = set(map(tuple, old_labels[label_key_cols].itertuples(index=False, name=None)))
    mask = [tuple(row) not in old_keys for row in new_labels[label_key_cols].itertuples(index=False, name=None)]
    novel_labels = new_labels.loc[mask].reset_index(drop=True)
    novel_raw = raw_records_for_labels(new_raw, novel_labels)
    return novel_raw.reset_index(drop=True), novel_labels


def assign_earliest_split(
    window_to_labels: Dict[str, pd.DataFrame],
    windows: Sequence[Tuple[str, int, int]],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    earliest_split: Dict[str, str] = {}
    for split, _, _ in windows:
        proteins = window_to_labels[split]["DB_ID"].drop_duplicates().tolist()
        for protein_id in proteins:
            earliest_split.setdefault(protein_id, split)

    assigned: Dict[str, pd.DataFrame] = {}
    for split, _, _ in windows:
        df = window_to_labels[split]
        mask = df["DB_ID"].map(earliest_split.get) == split
        assigned[split] = df.loc[mask].reset_index(drop=True)
    return assigned, earliest_split


def validate_split_integrity(
    windows: Sequence[Tuple[str, int, int]],
    window_to_labels: Dict[str, pd.DataFrame],
    assigned_labels: Dict[str, pd.DataFrame],
    earliest_split: Dict[str, str],
) -> Dict[str, object]:
    split_order = [split for split, _, _ in windows]

    for split, start_release, end_release in windows:
        start_date = pd.Timestamp(RELEASE_TO_DATE[start_release])
        end_date = pd.Timestamp(RELEASE_TO_DATE[end_release])
        if start_date >= end_date:
            raise ValueError(
                f"Temporal order is invalid for {split}: "
                f"{start_release} ({start_date.date()}) !< {end_release} ({end_date.date()})"
            )

    for idx in range(len(windows) - 1):
        left_split, _, left_end = windows[idx]
        right_split, right_start, _ = windows[idx + 1]
        if left_end != right_start:
            raise ValueError(
                f"Window boundary mismatch: {left_split} ends at {left_end}, "
                f"but {right_split} starts at {right_start}"
            )

    protein_sets = {
        split: set(df["DB_ID"].drop_duplicates().tolist())
        for split, df in assigned_labels.items()
    }
    overlap_counts: Dict[str, int] = {}
    for idx, left_split in enumerate(split_order):
        for right_split in split_order[idx + 1 :]:
            overlap = protein_sets[left_split] & protein_sets[right_split]
            overlap_counts[f"{left_split}__{right_split}"] = len(overlap)
            if overlap:
                sample = sorted(overlap)[:5]
                raise ValueError(
                    f"Protein overlap detected between {left_split} and {right_split}: {sample}"
                )

    split_to_index = {split: idx for idx, split in enumerate(split_order)}
    for split, df in window_to_labels.items():
        current_idx = split_to_index[split]
        for protein_id in df["DB_ID"].drop_duplicates().tolist():
            assigned_split = earliest_split.get(protein_id)
            if assigned_split is None:
                raise ValueError(f"Protein {protein_id} has no assigned split")
            if split_to_index[assigned_split] > current_idx:
                raise ValueError(
                    f"Protein {protein_id} appears in {split} before its assigned split {assigned_split}"
                )

    return {
        "time_order_valid": True,
        "protein_disjoint_valid": True,
        "window_boundaries": {
            split: {
                "start_release": start_release,
                "end_release": end_release,
                "start_date": RELEASE_TO_DATE[start_release],
                "end_date": RELEASE_TO_DATE[end_release],
            }
            for split, start_release, end_release in windows
        },
        "protein_overlap_counts": overlap_counts,
    }


def propagate_labels(df: pd.DataFrame, obo_path: Path) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    ontologies = obo_parser(str(obo_path), valid_rel=("is_a", "part_of"))
    propagated_frames: List[pd.DataFrame] = []

    for aspect, namespace in ASPECT_TO_NAMESPACE.items():
        ns_df = df[df["Aspect"] == aspect].drop_duplicates(["DB_ID", "GO_ID", "Aspect"]).copy()
        if ns_df.empty or namespace not in ontologies:
            continue

        ontology = ontologies[namespace]
        proteins = ns_df["DB_ID"].drop_duplicates().tolist()
        protein_to_idx = {protein_id: idx for idx, protein_id in enumerate(proteins)}
        matrix = np.zeros((len(proteins), ontology.idxs), dtype=bool)

        for row in ns_df.itertuples(index=False):
            if row.GO_ID not in ontology.terms_dict:
                continue
            protein_idx = protein_to_idx[row.DB_ID]
            term_idx = ontology.terms_dict[row.GO_ID]["index"]
            matrix[protein_idx, term_idx] = True

        propagate(matrix, ontology, ontology.order, mode="max")
        idx_to_term = {info["index"]: term for term, info in ontology.terms_dict.items()}

        propagated_rows: List[Tuple[str, str, str]] = []
        for protein_id, protein_idx in protein_to_idx.items():
            term_indices = np.where(matrix[protein_idx])[0]
            for term_idx in term_indices:
                term_id = idx_to_term.get(int(term_idx))
                if term_id is not None:
                    propagated_rows.append((protein_id, term_id, aspect))

        propagated_frames.append(pd.DataFrame(propagated_rows, columns=["DB_ID", "GO_ID", "Aspect"]))

    if not propagated_frames:
        return pd.DataFrame(columns=["DB_ID", "GO_ID", "Aspect"])

    return pd.concat(propagated_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)


def parse_aspect_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "None" or text == "[]":
            return False
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def load_cafa5_train_minimal(cache_dir: Path) -> pd.DataFrame:
    ds = load_dataset("wanglab/cafa5", name="cafa5_reasoning", cache_dir=str(cache_dir))
    train_df = ds["train"].to_pandas()
    keep_cols = ["protein_id", "go_bp", "go_mf", "go_cc"]
    return train_df[keep_cols].copy()


def compute_nk_lk(label_df: pd.DataFrame, train_df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if label_df.empty or train_df is None or train_df.empty:
        empty = pd.DataFrame(columns=["DB_ID", "GO_ID", "Aspect", "bucket"])
        return empty, {
            "nk_proteins": 0,
            "lk_proteins": 0,
            "nk_lk_proteins": 0,
            "nk_raw_records": 0,
            "lk_raw_records": 0,
            "nk_lk_raw_records": 0,
        }

    aspect_mapping = {"P": "go_bp", "F": "go_mf", "C": "go_cc"}
    train_lookup = train_df.drop_duplicates("protein_id").set_index("protein_id")

    rows: List[Tuple[str, str, str, str]] = []
    nk_proteins: Set[str] = set()
    lk_proteins: Set[str] = set()

    for row in label_df.itertuples(index=False):
        protein_id = row.DB_ID
        if protein_id not in train_lookup.index:
            nk_proteins.add(protein_id)
            rows.append((row.DB_ID, row.GO_ID, row.Aspect, "NK"))
            continue

        train_row = train_lookup.loc[protein_id]
        column = aspect_mapping[row.Aspect]
        if not parse_aspect_value(train_row[column]):
            lk_proteins.add(protein_id)
            rows.append((row.DB_ID, row.GO_ID, row.Aspect, "LK"))

    nk_lk_df = pd.DataFrame(rows, columns=["DB_ID", "GO_ID", "Aspect", "bucket"])
    stats = {
        "nk_proteins": len(nk_proteins),
        "lk_proteins": len(lk_proteins),
        "nk_lk_proteins": len(nk_proteins | lk_proteins),
        "nk_raw_records": int((nk_lk_df["bucket"] == "NK").sum()) if not nk_lk_df.empty else 0,
        "lk_raw_records": int((nk_lk_df["bucket"] == "LK").sum()) if not nk_lk_df.empty else 0,
        "nk_lk_raw_records": int(len(nk_lk_df)),
    }
    return nk_lk_df, stats


def save_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False)


def mean_labels_per_protein(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return round(len(df) / df["DB_ID"].nunique(), 2)


def build_summary(
    split: str,
    start_release: int,
    end_release: int,
    disease_raw: pd.DataFrame,
    assigned_labels: pd.DataFrame,
    propagated_assigned: pd.DataFrame,
    nk_lk_stats: Dict[str, int],
    nk_lk_propagated_labels: int,
) -> SplitSummary:
    return SplitSummary(
        split=split,
        start_release=start_release,
        end_release=end_release,
        start_date=RELEASE_TO_DATE[start_release],
        end_date=RELEASE_TO_DATE[end_release],
        disease_raw_records=int(len(disease_raw)),
        disease_unique_labels=int(collapse_labels(disease_raw).shape[0]),
        disease_proteins_before_assignment=int(disease_raw["DB_ID"].nunique()),
        disease_proteins_after_assignment=int(assigned_labels["DB_ID"].nunique()),
        unique_labels_after_assignment=int(len(assigned_labels)),
        propagated_labels_after_assignment=int(len(propagated_assigned)),
        nk_proteins=nk_lk_stats["nk_proteins"],
        lk_proteins=nk_lk_stats["lk_proteins"],
        nk_lk_proteins=nk_lk_stats["nk_lk_proteins"],
        nk_raw_records=nk_lk_stats["nk_raw_records"],
        lk_raw_records=nk_lk_stats["lk_raw_records"],
        nk_lk_raw_records=nk_lk_stats["nk_lk_raw_records"],
        nk_lk_propagated_labels=nk_lk_propagated_labels,
        avg_unique_labels_per_protein=mean_labels_per_protein(assigned_labels),
        avg_propagated_labels_per_protein=mean_labels_per_protein(propagated_assigned),
        aspect_counts_after_assignment=aspect_counts(assigned_labels),
        propagated_aspect_counts_after_assignment=aspect_counts(propagated_assigned),
    )


def write_markdown_report(
    output_path: Path,
    shortlist_count: int,
    summaries: Sequence[SplitSummary],
    shortlist_query: str,
) -> None:
    lines = [
        "# Step 0 Disease Temporal Split Report",
        "",
        "## Shortlist",
        "",
        f"- Query: `{shortlist_query}`",
        f"- Proteins: **{shortlist_count:,}**",
        "",
        "## Splits",
        "",
        "| Split | Window | Proteins | Unique labels | Propagated labels | NK proteins | LK proteins | NK/LK proteins | Avg labels/protein |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        window = f"{summary.start_release}->{summary.end_release}"
        lines.append(
            f"| {summary.split} | {window} | {summary.disease_proteins_after_assignment:,} | "
            f"{summary.unique_labels_after_assignment:,} | {summary.propagated_labels_after_assignment:,} | "
            f"{summary.nk_proteins:,} | {summary.lk_proteins:,} | {summary.nk_lk_proteins:,} | "
            f"{summary.avg_unique_labels_per_protein:.2f} |"
        )

    lines.extend(["", "## Notes", ""])
    lines.append("- Counts are based on protein-disjoint assignment by earliest temporal appearance.")
    lines.append("- Unique labels are counted as `(DB_ID, GO_ID, Aspect)` after collapsing evidence codes.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    windows = build_windows(args)
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / args.output_dir
    assets_dir = output_dir / "assets"
    ensure_dir(output_dir)
    ensure_dir(assets_dir)

    shortlist_query = shortlist_query_for_mode(args.shortlist_mode)
    shortlist_path = output_dir / "uniprot_disease_shortlist.tsv"
    shortlist_df = fetch_shortlist(shortlist_path, shortlist_query)
    accession_col = "Entry" if "Entry" in shortlist_df.columns else shortlist_df.columns[0]
    shortlist_set = set(shortlist_df[accession_col].astype(str).tolist())
    log(f"[shortlist] mode={args.shortlist_mode} total proteins: {len(shortlist_set):,}")

    releases = sorted({release for _, start, end in windows for release in (start, end)})
    for release in releases:
        prepare_filtered_gaf(
            release=release,
            asset_dir=assets_dir,
            shortlist_path=shortlist_path,
            shortlist_set=shortlist_set,
            use_shell_filter=args.use_shell_filter,
            force_download=args.force_download,
        )

    obo_path = output_dir / "go-basic.obo"
    repo_obo_path = repo_root / "bioreason2/dataset/go-basic.obo"
    if repo_obo_path.exists():
        if args.force_download or not obo_path.exists():
            shutil.copy2(repo_obo_path, obo_path)
            log(f"[download] copied local GO OBO from {repo_obo_path}")
    else:
        download_file(GO_OBO_URL, obo_path, force=args.force_download)

    train_df: Optional[pd.DataFrame] = None
    nk_lk_status = "available"
    nk_lk_error: Optional[str] = None
    try:
        log("[train] loading CAFA5 training split for NK/LK")
        train_df = load_cafa5_train_minimal(output_dir / "hf_cache")
        log(f"[train] loaded {len(train_df):,} rows")
    except Exception as exc:  # pragma: no cover - network/auth dependent
        nk_lk_status = "skipped"
        nk_lk_error = str(exc)
        log(f"[train] NK/LK skipped because CAFA5 train split is unavailable: {exc}")

    raw_deltas: Dict[str, pd.DataFrame] = {}
    label_deltas: Dict[str, pd.DataFrame] = {}

    for split, start_release, end_release in windows:
        log(f"[window] {split} {start_release}->{end_release}")
        old_df = load_filtered_gaf(filtered_gaf_path(assets_dir, start_release))
        new_df = load_filtered_gaf(filtered_gaf_path(assets_dir, end_release))
        raw_delta, label_delta = compute_delta(old_df, new_df)
        disease_labels = label_delta[label_delta["DB_ID"].isin(shortlist_set)].reset_index(drop=True)
        disease_raw = raw_records_for_labels(raw_delta, disease_labels)
        raw_deltas[split] = disease_raw
        label_deltas[split] = disease_labels
        save_tsv(disease_raw, output_dir / f"{split}_disease_delta_raw.tsv")
        save_tsv(disease_labels, output_dir / f"{split}_disease_delta_labels.tsv")
        log(
            f"[window] {split}: raw={len(disease_raw):,} "
            f"labels={len(disease_labels):,} proteins={disease_labels['DB_ID'].nunique():,}"
        )

    assigned_labels, earliest_split = assign_earliest_split(label_deltas, windows=windows)
    split_validation = validate_split_integrity(
        windows=windows,
        window_to_labels=label_deltas,
        assigned_labels=assigned_labels,
        earliest_split=earliest_split,
    )
    with (output_dir / "earliest_split_by_protein.json").open("w", encoding="utf-8") as f:
        json.dump(earliest_split, f, indent=2, sort_keys=True)

    summaries: List[SplitSummary] = []
    for split, start_release, end_release in windows:
        labels = assigned_labels[split]
        raw = raw_records_for_labels(raw_deltas[split], labels)
        propagated = (
            propagate_labels(labels, obo_path) if not args.skip_propagation else labels.copy()
        )
        nk_lk_df, nk_lk_stats = compute_nk_lk(labels, train_df)
        nk_lk_propagated = (
            propagate_labels(nk_lk_df[["DB_ID", "GO_ID", "Aspect"]], obo_path)
            if (not args.skip_propagation and not nk_lk_df.empty)
            else nk_lk_df[["DB_ID", "GO_ID", "Aspect"]].copy()
        )

        save_tsv(labels, output_dir / f"{split}_assigned_labels.tsv")
        save_tsv(raw, output_dir / f"{split}_assigned_raw.tsv")
        save_tsv(propagated, output_dir / f"{split}_assigned_propagated.tsv")
        save_tsv(nk_lk_df, output_dir / f"{split}_assigned_nk_lk.tsv")
        save_tsv(nk_lk_propagated, output_dir / f"{split}_assigned_nk_lk_propagated.tsv")

        summary = build_summary(
            split=split,
            start_release=start_release,
            end_release=end_release,
            disease_raw=raw_deltas[split],
            assigned_labels=labels,
            propagated_assigned=propagated,
            nk_lk_stats=nk_lk_stats,
            nk_lk_propagated_labels=int(len(nk_lk_propagated)),
        )
        summaries.append(summary)
        log(
            f"[summary] {split}: proteins={summary.disease_proteins_after_assignment:,}, "
            f"labels={summary.unique_labels_after_assignment:,}, "
            f"propagated={summary.propagated_labels_after_assignment:,}, "
            f"nk_lk={summary.nk_lk_proteins:,}"
        )

    nk_lk_eda = pd.DataFrame(
        [
            {
                "split": summary.split,
                "start_release": summary.start_release,
                "end_release": summary.end_release,
                "proteins": summary.disease_proteins_after_assignment,
                "unique_labels": summary.unique_labels_after_assignment,
                "nk_proteins": summary.nk_proteins,
                "lk_proteins": summary.lk_proteins,
                "nk_lk_proteins": summary.nk_lk_proteins,
                "nk_raw_records": summary.nk_raw_records,
                "lk_raw_records": summary.lk_raw_records,
                "nk_lk_raw_records": summary.nk_lk_raw_records,
                "nk_lk_propagated_labels": summary.nk_lk_propagated_labels,
            }
            for summary in summaries
        ]
    )
    save_tsv(nk_lk_eda, output_dir / "nk_lk_eda.tsv")

    summary_json = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "shortlist_mode": args.shortlist_mode,
        "shortlist_query": shortlist_query,
        "shortlist_proteins": len(shortlist_set),
        "nk_lk_status": nk_lk_status,
        "nk_lk_error": nk_lk_error,
        "split_validation": split_validation,
        "release_archives": {str(release): RELEASE_TO_ARCHIVE[release] for release in releases},
        "windows": [asdict(summary) for summary in summaries],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, sort_keys=True)

    write_markdown_report(output_dir / "report.md", len(shortlist_set), summaries, shortlist_query)
    log(f"[done] outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

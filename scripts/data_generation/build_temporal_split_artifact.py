#!/usr/bin/env python3
"""Build the disease temporal split artifact used by BioReason-Pro.

The artifact layout is intentionally aligned with train.py and eval.py:

    train_assigned_labels.tsv
    dev_assigned_labels.tsv
    test_assigned_labels.tsv
    *_assigned_propagated.tsv
    IA.txt is built by build_ia_weights.py from train_assigned_propagated.tsv

The default benchmark is the 213/221/225/228 disease split. This script keeps
network-facing steps explicit so future releases can document every external
source involved in the data build.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import re
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd
import requests


EXPERIMENTAL_EVIDENCE_CODES = {"IDA", "IPI", "EXP", "IGI", "IMP", "IEP", "IC", "TAS"}
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
GOA_OLD_HUMAN_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/GO/goa/old/HUMAN"
GO_OBO_URL = "https://purl.obolibrary.org/obo/go/go-basic.obo"
UNIPROT_BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
MAIN_SHORTLIST_QUERY = (
    "reviewed:true AND organism_id:9606 AND cc_disease:* "
    "AND (go_exp:* OR go_ida:* OR go_ipi:* OR go_igi:* OR go_imp:* OR go_iep:* OR go_ic:* OR go_tas:*)"
)
HIGH_CONFIDENCE_SHORTLIST_QUERY = MAIN_SHORTLIST_QUERY + " AND (xref:mim-* OR xref:orphanet-*)"
SHORTLIST_FIELDS = "accession,id"
ASPECT_NAMES = {"P": "BP", "F": "MF", "C": "CC"}
RELEASE_TO_ARCHIVE = {release: f"goa_human.gaf.{release}.gz" for release in RELEASE_TO_DATE}
RELEASE_TO_URL = {
    release: f"{GOA_OLD_HUMAN_BASE_URL}/{archive}"
    for release, archive in RELEASE_TO_ARCHIVE.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/artifacts/benchmarks/213_221_225_228/temporal_split",
        help="Output artifact directory.",
    )
    parser.add_argument("--train-start-release", type=int, default=213)
    parser.add_argument("--train-end-release", type=int, default=221)
    parser.add_argument("--dev-end-release", type=int, default=225)
    parser.add_argument("--test-end-release", type=int, default=228)
    parser.add_argument("--validation-proteins", type=int, default=200)
    parser.add_argument("--holdout-proteins", type=int, default=400)
    parser.add_argument("--partition-seed", type=int, default=23)
    parser.add_argument(
        "--shortlist-mode",
        choices=["main", "high-confidence"],
        default="high-confidence",
        help="UniProt disease shortlist definition.",
    )
    parser.add_argument(
        "--shortlist-tsv",
        default="",
        help="Optional local shortlist TSV. Must include Entry, accession, protein_id, or DB_ID.",
    )
    parser.add_argument(
        "--gaf-dir",
        default="",
        help="Optional directory containing pre-downloaded goa_human.gaf.<release>.gz files.",
    )
    parser.add_argument(
        "--obo",
        default="",
        help="Optional GO OBO path. Defaults to bioreason2/dataset/go-basic.obo or live download.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--skip-propagation",
        action="store_true",
        help="Write propagated files as raw labels. Useful for quick debug builds only.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_windows(args: argparse.Namespace) -> List[Tuple[str, int, int]]:
    releases = [
        args.train_start_release,
        args.train_end_release,
        args.dev_end_release,
        args.test_end_release,
    ]
    missing = [release for release in releases if release not in RELEASE_TO_DATE]
    if missing:
        raise ValueError(f"Release metadata is missing for {sorted(set(missing))}")
    return [
        ("train", args.train_start_release, args.train_end_release),
        ("dev", args.train_end_release, args.dev_end_release),
        ("test", args.dev_end_release, args.test_end_release),
    ]


def get_next_link(headers: Mapping[str, str]) -> Optional[str]:
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


def first_present_column(frame: pd.DataFrame, names: Sequence[str]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"None of these columns are present: {', '.join(names)}")


def load_shortlist_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    accession_column = first_present_column(frame, ("Entry", "accession", "protein_id", "DB_ID"))
    return pd.DataFrame({"Entry": frame[accession_column].astype(str).str.strip()}).drop_duplicates()


def fetch_shortlist(output_path: Path, query: str, *, force_download: bool) -> pd.DataFrame:
    if output_path.exists() and not force_download:
        log(f"[shortlist] reuse {output_path}")
        return load_shortlist_file(output_path)

    session = requests.Session()
    params = {"query": query, "format": "tsv", "fields": SHORTLIST_FIELDS, "size": 500}
    response = session.get(UNIPROT_BASE_URL, params=params, timeout=120)
    response.raise_for_status()
    frames = [pd.read_csv(io.StringIO(response.text), sep="\t")]
    url = get_next_link(response.headers)
    while url:
        response = session.get(url, timeout=120)
        response.raise_for_status()
        frames.append(pd.read_csv(io.StringIO(response.text), sep="\t"))
        url = get_next_link(response.headers)

    shortlist = pd.concat(frames, ignore_index=True).drop_duplicates()
    ensure_dir(output_path.parent)
    shortlist.to_csv(output_path, sep="\t", index=False)
    log(f"[shortlist] wrote {len(shortlist):,} rows to {output_path}")
    return load_shortlist_file(output_path)


def download_file(url: str, dest: Path, *, force_download: bool) -> None:
    if dest.exists() and not force_download:
        log(f"[download] reuse {dest}")
        return
    ensure_dir(dest.parent)
    log(f"[download] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def resolve_gaf_source(
    release: int,
    gaf_dir: Optional[Path],
    assets_dir: Path,
    *,
    force_download: bool,
) -> Path:
    archive_name = RELEASE_TO_ARCHIVE[release]
    if gaf_dir is not None:
        for candidate in (gaf_dir / archive_name, gaf_dir / archive_name.removesuffix(".gz")):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not find release {release} GAF in {gaf_dir}")

    dest = assets_dir / archive_name
    download_file(RELEASE_TO_URL[release], dest, force_download=force_download)
    return dest


def open_text_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def filtered_gaf_path(assets_dir: Path, release: int) -> Path:
    return assets_dir / f"filtered_goa_human_disease_exp_{release}.gaf"


def filter_gaf(
    source_path: Path,
    output_path: Path,
    shortlist_set: Set[str],
    *,
    force_download: bool,
) -> Path:
    if output_path.exists() and not force_download:
        log(f"[filter] reuse {output_path.name}")
        return output_path

    ensure_dir(output_path.parent)
    kept = 0
    with open_text_maybe_gzip(source_path) as src, output_path.open("w", encoding="utf-8") as dst:
        for raw_line in src:
            if not raw_line or raw_line.startswith("!"):
                continue
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) < len(GAF_COLUMNS):
                continue
            row = dict(zip(GAF_COLUMNS, parts))
            qualifier = row["Qualifier"]
            if "NOT" in qualifier.split("|"):
                continue
            if row["DB_ID"] not in shortlist_set:
                continue
            if row["Evidence_Code"] not in EXPERIMENTAL_EVIDENCE_CODES:
                continue
            if row["Aspect"] not in ASPECT_NAMES:
                continue
            if row["DB_Type"].lower() != "protein":
                continue
            dst.write(raw_line)
            kept += 1
    log(f"[filter] {source_path.name}: kept {kept:,} rows")
    return output_path


def load_filtered_gaf(path: Path) -> pd.DataFrame:
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) >= len(GAF_COLUMNS):
                rows.append(parts[: len(GAF_COLUMNS)])
    if not rows:
        return pd.DataFrame(columns=GAF_COLUMNS)
    return pd.DataFrame(rows, columns=GAF_COLUMNS)


def collapse_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["DB_ID", "GO_ID", "Aspect"])
    return df[["DB_ID", "GO_ID", "Aspect"]].drop_duplicates().reset_index(drop=True)


def raw_records_for_labels(raw_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty or label_df.empty:
        return pd.DataFrame(columns=GAF_COLUMNS)
    keys = label_df[["DB_ID", "GO_ID", "Aspect"]].drop_duplicates()
    merged = raw_df.merge(keys, on=["DB_ID", "GO_ID", "Aspect"], how="inner")
    return merged.drop_duplicates().reset_index(drop=True)


def compute_delta(old_df: pd.DataFrame, new_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    label_columns = ["DB_ID", "GO_ID", "Aspect"]
    raw_columns = ["DB_ID", "GO_ID", "Aspect", "Evidence_Code", "DB_Reference"]
    old_labels = old_df[label_columns].drop_duplicates() if not old_df.empty else pd.DataFrame(columns=label_columns)
    new_labels = new_df[label_columns].drop_duplicates() if not new_df.empty else pd.DataFrame(columns=label_columns)
    old_key_set = set(map(tuple, old_labels.itertuples(index=False, name=None)))
    novel_mask = [tuple(row) not in old_key_set for row in new_labels.itertuples(index=False, name=None)]
    novel_labels = new_labels.loc[novel_mask].reset_index(drop=True)

    if new_df.empty or novel_labels.empty:
        return pd.DataFrame(columns=GAF_COLUMNS), novel_labels
    novel_raw = new_df.drop_duplicates(raw_columns).merge(novel_labels, on=label_columns, how="inner")
    return novel_raw.reset_index(drop=True), novel_labels


def assign_earliest_split(
    window_to_labels: Mapping[str, pd.DataFrame],
    windows: Sequence[Tuple[str, int, int]],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    earliest_split: Dict[str, str] = {}
    for split, _, _ in windows:
        for protein_id in window_to_labels[split]["DB_ID"].drop_duplicates().astype(str):
            earliest_split.setdefault(protein_id, split)

    assigned: Dict[str, pd.DataFrame] = {}
    for split, _, _ in windows:
        df = window_to_labels[split].copy()
        assigned[split] = df[df["DB_ID"].map(earliest_split.get) == split].reset_index(drop=True)
    return assigned, earliest_split


def validate_disjoint(label_by_split: Mapping[str, pd.DataFrame], split_order: Sequence[str]) -> Dict[str, int]:
    protein_sets = {
        split: set(label_by_split[split]["DB_ID"].astype(str).drop_duplicates())
        for split in split_order
    }
    overlaps: Dict[str, int] = {}
    for index, left in enumerate(split_order):
        for right in split_order[index + 1 :]:
            overlap = protein_sets[left] & protein_sets[right]
            overlaps[f"{left}__{right}"] = len(overlap)
            if overlap:
                raise ValueError(f"Protein overlap detected between {left} and {right}: {sorted(overlap)[:5]}")
    return overlaps


def aspect_profile(label_df: pd.DataFrame, protein_id: str) -> str:
    values = sorted(ASPECT_NAMES.get(value, value) for value in label_df[label_df["DB_ID"] == protein_id]["Aspect"])
    return "+".join(dict.fromkeys(values)) or "none"


def partition_future_pool(
    future_labels: pd.DataFrame,
    *,
    validation_proteins: int,
    holdout_proteins: int,
    seed: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    proteins = sorted(future_labels["DB_ID"].astype(str).drop_duplicates().tolist())
    rng = random.Random(seed)
    rng.shuffle(proteins)
    dev_ids = set(proteins[: max(validation_proteins, 0)])
    test_ids = set(proteins[len(dev_ids) : len(dev_ids) + max(holdout_proteins, 0)])
    reserve_ids = set(proteins) - dev_ids - test_ids

    partitions = {
        "dev": future_labels[future_labels["DB_ID"].isin(dev_ids)].reset_index(drop=True),
        "test": future_labels[future_labels["DB_ID"].isin(test_ids)].reset_index(drop=True),
        "reserve": future_labels[future_labels["DB_ID"].isin(reserve_ids)].reset_index(drop=True),
    }
    profile_counts: Dict[str, Dict[str, int]] = {}
    for split, frame in partitions.items():
        counts: Dict[str, int] = defaultdict(int)
        for protein_id in frame["DB_ID"].drop_duplicates().astype(str):
            counts[aspect_profile(frame, protein_id)] += 1
        profile_counts[split] = dict(sorted(counts.items()))
    return partitions, {
        "seed": seed,
        "future_pool_proteins": len(proteins),
        "validation_proteins": len(dev_ids),
        "holdout_proteins": len(test_ids),
        "reserve_proteins": len(reserve_ids),
        "aspect_profile_counts": profile_counts,
    }


def load_go_parent_map(obo_path: Path) -> Dict[str, Tuple[str, ...]]:
    parents_by_term: Dict[str, List[str]] = {}
    current_id = ""
    current_parents: List[str] = []
    current_obsolete = False
    in_term = False

    def finalize() -> None:
        nonlocal current_id, current_parents, current_obsolete
        if current_id and not current_obsolete:
            parents_by_term[current_id] = list(dict.fromkeys(current_parents))
        current_id = ""
        current_parents = []
        current_obsolete = False

    with obo_path.open("r", encoding="utf-8") as handle:
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
            if not in_term:
                continue
            if line.startswith("id: "):
                current_id = line.split(":", 1)[1].strip()
            elif line.startswith("is_obsolete: "):
                current_obsolete = line.split(":", 1)[1].strip().lower() == "true"
            elif line.startswith("is_a: "):
                current_parents.append(line.split("!", 1)[0].split(":", 1)[1].strip())
            elif line.startswith("relationship: part_of "):
                current_parents.append(line.split()[2].strip())
    finalize()
    return {term: tuple(parents) for term, parents in parents_by_term.items()}


def ancestors(go_id: str, parent_map: Mapping[str, Tuple[str, ...]], cache: Dict[str, Set[str]]) -> Set[str]:
    if go_id in cache:
        return cache[go_id]
    seen: Set[str] = set()
    stack = list(parent_map.get(go_id, ()))
    while stack:
        parent = stack.pop()
        if parent in seen:
            continue
        seen.add(parent)
        stack.extend(parent_map.get(parent, ()))
    cache[go_id] = seen
    return seen


def propagate_labels(df: pd.DataFrame, obo_path: Path) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["DB_ID", "GO_ID", "Aspect"])
    parent_map = load_go_parent_map(obo_path)
    cache: Dict[str, Set[str]] = {}
    rows: Set[Tuple[str, str, str]] = set()
    for row in df[["DB_ID", "GO_ID", "Aspect"]].drop_duplicates().itertuples(index=False):
        rows.add((row.DB_ID, row.GO_ID, row.Aspect))
        for parent in ancestors(row.GO_ID, parent_map, cache):
            rows.add((row.DB_ID, parent, row.Aspect))
    return pd.DataFrame(sorted(rows), columns=["DB_ID", "GO_ID", "Aspect"])


def save_tsv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, sep="\t", index=False)


def mean_labels_per_protein(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return round(len(df) / max(df["DB_ID"].nunique(), 1), 3)


def aspect_counts(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty:
        return {"MF": 0, "BP": 0, "CC": 0}
    counts = df["Aspect"].map(ASPECT_NAMES).value_counts().to_dict()
    return {aspect: int(counts.get(aspect, 0)) for aspect in ("MF", "BP", "CC")}


def split_summary(
    split: str,
    start_release: int,
    end_release: int,
    labels: pd.DataFrame,
    propagated: pd.DataFrame,
) -> Dict[str, Any]:
    return {
        "split": split,
        "start_release": start_release,
        "end_release": end_release,
        "start_date": RELEASE_TO_DATE[start_release],
        "end_date": RELEASE_TO_DATE[end_release],
        "proteins": int(labels["DB_ID"].nunique()) if not labels.empty else 0,
        "unique_labels": int(len(labels)),
        "propagated_labels": int(len(propagated)),
        "avg_unique_labels_per_protein": mean_labels_per_protein(labels),
        "avg_propagated_labels_per_protein": mean_labels_per_protein(propagated),
        "aspect_counts": aspect_counts(labels),
        "propagated_aspect_counts": aspect_counts(propagated),
    }


def write_report(output_path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Disease Temporal Split Artifact",
        "",
        f"Shortlist mode: `{summary['shortlist_mode']}`",
        f"Shortlist proteins: {summary['shortlist_proteins']}",
        "",
        "| Split | Window | Proteins | Labels | Propagated labels |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in summary["windows"]:
        lines.append(
            f"| {item['split']} | {item['start_release']}->{item['end_release']} | "
            f"{item['proteins']} | {item['unique_labels']} | {item['propagated_labels']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_obo(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.obo:
        return Path(args.obo).expanduser().resolve()
    artifact_obo = output_dir / "go-basic.obo"
    local_obo = repo_root() / "bioreason2/dataset/go-basic.obo"
    if local_obo.exists():
        if args.force_download or not artifact_obo.exists():
            shutil.copy2(local_obo, artifact_obo)
        return artifact_obo
    download_file(GO_OBO_URL, artifact_obo, force_download=args.force_download)
    return artifact_obo


def main() -> int:
    args = parse_args()
    root = repo_root()
    output_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    assets_dir = ensure_dir(output_dir / "assets")
    windows = build_windows(args)

    if args.shortlist_tsv:
        shortlist_df = load_shortlist_file(Path(args.shortlist_tsv).expanduser().resolve())
        shortlist_query = "local shortlist"
    else:
        shortlist_query = shortlist_query_for_mode(args.shortlist_mode)
        shortlist_df = fetch_shortlist(output_dir / "uniprot_disease_shortlist.tsv", shortlist_query, force_download=args.force_download)
    shortlist_set = set(shortlist_df["Entry"].astype(str).str.strip())
    log(f"[shortlist] proteins={len(shortlist_set):,}")

    gaf_dir = Path(args.gaf_dir).expanduser().resolve() if args.gaf_dir else None
    releases = sorted({release for _, start, end in windows for release in (start, end)})
    for release in releases:
        source_path = resolve_gaf_source(release, gaf_dir, assets_dir, force_download=args.force_download)
        filter_gaf(
            source_path,
            filtered_gaf_path(assets_dir, release),
            shortlist_set,
            force_download=args.force_download,
        )

    raw_deltas: Dict[str, pd.DataFrame] = {}
    label_deltas: Dict[str, pd.DataFrame] = {}
    for split, start_release, end_release in windows:
        old_df = load_filtered_gaf(filtered_gaf_path(assets_dir, start_release))
        new_df = load_filtered_gaf(filtered_gaf_path(assets_dir, end_release))
        raw_delta, label_delta = compute_delta(old_df, new_df)
        disease_labels = label_delta[label_delta["DB_ID"].isin(shortlist_set)].reset_index(drop=True)
        disease_raw = raw_records_for_labels(raw_delta, disease_labels)
        raw_deltas[split] = disease_raw
        label_deltas[split] = disease_labels
        save_tsv(disease_raw, output_dir / f"{split}_disease_delta_raw.tsv")
        save_tsv(disease_labels, output_dir / f"{split}_disease_delta_labels.tsv")
        log(f"[window] {split}: proteins={disease_labels['DB_ID'].nunique():,} labels={len(disease_labels):,}")

    assigned_by_window, earliest_split = assign_earliest_split(label_deltas, windows)
    validate_disjoint(assigned_by_window, ["train", "dev", "test"])
    (output_dir / "earliest_split_by_protein.json").write_text(
        json.dumps(earliest_split, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    future_partitions, future_partition_meta = partition_future_pool(
        assigned_by_window["test"],
        validation_proteins=args.validation_proteins,
        holdout_proteins=args.holdout_proteins,
        seed=args.partition_seed,
    )
    final_labels: Dict[str, pd.DataFrame] = {
        "train": pd.concat([assigned_by_window["train"], assigned_by_window["dev"]], ignore_index=True)
        .drop_duplicates(["DB_ID", "GO_ID", "Aspect"])
        .reset_index(drop=True),
        "dev": future_partitions["dev"],
        "test": future_partitions["test"],
        "reserve": future_partitions["reserve"],
    }
    overlaps = validate_disjoint(final_labels, ["train", "dev", "test", "reserve"])
    obo_path = resolve_obo(args, output_dir)

    final_specs = [
        ("train", args.train_start_release, args.dev_end_release, pd.concat([raw_deltas["train"], raw_deltas["dev"]], ignore_index=True)),
        ("dev", args.dev_end_release, args.test_end_release, raw_deltas["test"]),
        ("test", args.dev_end_release, args.test_end_release, raw_deltas["test"]),
        ("reserve", args.dev_end_release, args.test_end_release, raw_deltas["test"]),
    ]
    windows_summary: List[Dict[str, Any]] = []
    empty_nk_lk = pd.DataFrame(columns=["DB_ID", "GO_ID", "Aspect", "bucket"])
    for split, start_release, end_release, source_raw in final_specs:
        labels = final_labels[split]
        raw = raw_records_for_labels(source_raw, labels)
        propagated = labels.copy() if args.skip_propagation else propagate_labels(labels, obo_path)
        save_tsv(labels, output_dir / f"{split}_assigned_labels.tsv")
        save_tsv(raw, output_dir / f"{split}_assigned_raw.tsv")
        save_tsv(propagated, output_dir / f"{split}_assigned_propagated.tsv")
        save_tsv(empty_nk_lk, output_dir / f"{split}_assigned_nk_lk.tsv")
        save_tsv(empty_nk_lk[["DB_ID", "GO_ID", "Aspect"]], output_dir / f"{split}_assigned_nk_lk_propagated.tsv")
        windows_summary.append(split_summary(split, start_release, end_release, labels, propagated))
        log(f"[split] {split}: proteins={labels['DB_ID'].nunique():,} labels={len(labels):,}")

    nk_lk_eda = pd.DataFrame(
        [
            {
                "split": item["split"],
                "proteins": item["proteins"],
                "unique_labels": item["unique_labels"],
                "nk_proteins": 0,
                "lk_proteins": 0,
                "nk_lk_proteins": 0,
            }
            for item in windows_summary
        ]
    )
    save_tsv(nk_lk_eda, output_dir / "nk_lk_eda.tsv")

    summary = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "shortlist_mode": args.shortlist_mode,
        "shortlist_query": shortlist_query,
        "shortlist_proteins": len(shortlist_set),
        "release_archives": {str(release): RELEASE_TO_ARCHIVE[release] for release in releases},
        "split_validation": {
            "protein_disjoint_valid": True,
            "protein_overlap_counts": overlaps,
        },
        "future_partition": future_partition_meta,
        "windows": windows_summary,
        "nk_lk_status": "not_computed",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(output_dir / "report.md", summary)
    log(f"[done] outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

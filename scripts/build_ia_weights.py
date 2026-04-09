#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CAFA-style IA weights from propagated training annotations.")
    parser.add_argument("--annotations", required=True, help="Path to propagated training annotations TSV.")
    parser.add_argument("--obo", required=True, help="Path to GO OBO file.")
    parser.add_argument("--output", required=True, help="Output IA.txt path.")
    parser.add_argument(
        "--log-base",
        type=float,
        default=2.0,
        help="Log base used for information accretion. Defaults to 2.",
    )
    return parser.parse_args()


def load_go_metadata(obo_path: Path) -> Dict[str, Dict[str, object]]:
    metadata: Dict[str, Dict[str, object]] = {}
    current_id = ""
    current_namespace = ""
    current_parents: List[str] = []
    current_obsolete = False
    in_term = False

    def finalize_term() -> None:
        nonlocal current_id, current_namespace, current_parents, current_obsolete
        if current_id and not current_obsolete and current_namespace:
            ordered_parents: List[str] = []
            seen: Set[str] = set()
            for parent in current_parents:
                if parent and parent not in seen:
                    seen.add(parent)
                    ordered_parents.append(parent)
            metadata[current_id] = {
                "namespace": current_namespace,
                "parents": tuple(ordered_parents),
            }
        current_id = ""
        current_namespace = ""
        current_parents = []
        current_obsolete = False

    with obo_path.open("r", encoding="utf-8") as handle:
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
                current_id = line.split(":", 1)[1].strip()
            elif line.startswith("namespace: "):
                current_namespace = line.split(":", 1)[1].strip()
            elif line.startswith("is_obsolete: "):
                current_obsolete = line.split(":", 1)[1].strip().lower() == "true"
            elif line.startswith("is_a: "):
                current_parents.append(line.split("!", 1)[0].split(":", 1)[1].strip())
            elif line.startswith("relationship: part_of "):
                current_parents.append(line.split()[2].strip())
    finalize_term()
    return metadata


def load_propagated_annotations(tsv_path: Path) -> Dict[str, Set[str]]:
    by_protein: Dict[str, Set[str]] = defaultdict(set)
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            protein_id = (row.get("DB_ID") or "").strip()
            go_id = (row.get("GO_ID") or "").strip()
            if protein_id and go_id:
                by_protein[protein_id].add(go_id)
    return by_protein


def count_term_support(protein_to_terms: Mapping[str, Set[str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for go_terms in protein_to_terms.values():
        counts.update(go_terms)
    return counts


def count_parent_context_support(
    protein_to_terms: Mapping[str, Set[str]],
    go_metadata: Mapping[str, Mapping[str, object]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for go_id, metadata in go_metadata.items():
        parents = tuple(metadata.get("parents", ()))
        if not parents:
            continue
        support = 0
        parent_set = set(parents)
        for assigned_terms in protein_to_terms.values():
            if parent_set.issubset(assigned_terms):
                support += 1
        counts[go_id] = support
    return counts


def compute_ia_weights(
    protein_to_terms: Mapping[str, Set[str]],
    go_metadata: Mapping[str, Mapping[str, object]],
    *,
    log_base: float = 2.0,
) -> Dict[str, float]:
    term_support = count_term_support(protein_to_terms)
    parent_support = count_parent_context_support(protein_to_terms, go_metadata)
    total_proteins = max(len(protein_to_terms), 1)
    weights: Dict[str, float] = {}
    log_denominator = math.log(log_base)

    for go_id, metadata in go_metadata.items():
        term_count = float(term_support.get(go_id, 0))
        if term_count <= 0:
            continue
        parents = tuple(metadata.get("parents", ()))
        if parents:
            context_count = float(parent_support.get(go_id, 0))
        else:
            context_count = float(total_proteins)
        if context_count <= 0:
            continue
        conditional_probability = min(max(term_count / context_count, 1e-12), 1.0)
        ia_value = -math.log(conditional_probability) / log_denominator
        if ia_value > 0.0:
            weights[go_id] = ia_value
    return weights


def write_ia_weights(output_path: Path, weights: Mapping[str, float]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for go_id in sorted(weights):
            handle.write(f"{go_id}\t{weights[go_id]:.10f}\n")


def main() -> int:
    args = parse_args()
    annotations_path = Path(args.annotations).expanduser().resolve()
    obo_path = Path(args.obo).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    protein_to_terms = load_propagated_annotations(annotations_path)
    go_metadata = load_go_metadata(obo_path)
    weights = compute_ia_weights(protein_to_terms, go_metadata, log_base=float(args.log_base))
    write_ia_weights(output_path, weights)

    print(f"Wrote {len(weights)} IA weights to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

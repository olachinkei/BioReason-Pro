from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


ASPECT_COLUMNS: Sequence[Tuple[str, str]] = (
    ("BP", "go_bp"),
    ("MF", "go_mf"),
    ("CC", "go_cc"),
)

EXPLICIT_ASPECT_TO_CODE = {
    "biological_process": "BP",
    "molecular_function": "MF",
    "cellular_component": "CC",
    "bp": "BP",
    "mf": "MF",
    "cc": "CC",
}


def _count_terms(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        return len([item for item in text.split(",") if item.strip()])
    if isinstance(value, (list, tuple, set)):
        return len([item for item in value if item not in (None, "")])
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return int(bool(value))


def build_aspect_profile(example: Mapping[str, Any]) -> str:
    explicit_aspect = str(example.get("go_aspect") or "").strip()
    if explicit_aspect:
        normalized_aspect = explicit_aspect.lower()
        if normalized_aspect not in {"all", "mixed", "multi", "any"}:
            return f"aspect:{explicit_aspect}"

    present_aspects = [aspect for aspect, column in ASPECT_COLUMNS if _count_terms(example.get(column)) > 0]
    if not present_aspects:
        present_aspects = ["NONE"]

    total_terms = sum(_count_terms(example.get(column)) for _, column in ASPECT_COLUMNS)
    if total_terms <= 2:
        size_bucket = "labels:1-2"
    elif total_terms <= 5:
        size_bucket = "labels:3-5"
    elif total_terms <= 10:
        size_bucket = "labels:6-10"
    else:
        size_bucket = "labels:11+"

    return f"profile:{'+'.join(present_aspects)}|{size_bucket}"


def _present_aspects(example: Mapping[str, Any]) -> Tuple[str, ...]:
    explicit_aspect = str(example.get("go_aspect") or "").strip().lower()
    if explicit_aspect:
        mapped = EXPLICIT_ASPECT_TO_CODE.get(explicit_aspect)
        if mapped:
            return (mapped,)

    present = [aspect for aspect, column in ASPECT_COLUMNS if _count_terms(example.get(column)) > 0]
    return tuple(present)


def _allocate_group_counts(group_sizes: Mapping[str, int], target_size: int) -> Dict[str, int]:
    allocations = {key: 0 for key in group_sizes}
    if target_size <= 0 or not group_sizes:
        return allocations

    keys = list(group_sizes.keys())
    total_items = sum(group_sizes.values())
    if total_items <= target_size:
        return dict(group_sizes)

    if target_size >= len(keys):
        for key in keys:
            allocations[key] = 1
        remaining = target_size - len(keys)
        capacities = {key: max(group_sizes[key] - 1, 0) for key in keys}
    else:
        remaining = target_size
        capacities = dict(group_sizes)

    if remaining <= 0:
        return allocations

    allocatable_total = sum(capacities.values())
    if allocatable_total <= 0:
        return allocations

    raw_targets = {key: remaining * capacities[key] / allocatable_total for key in keys}
    base_additions = {key: min(int(raw_targets[key]), capacities[key]) for key in keys}
    for key in keys:
        allocations[key] += base_additions[key]

    leftover = remaining - sum(base_additions.values())
    if leftover <= 0:
        return allocations

    ranked_keys = sorted(
        keys,
        key=lambda key: (
            -(raw_targets[key] - base_additions[key]),
            -capacities[key],
            key,
        ),
    )
    capacity_left = {key: capacities[key] - base_additions[key] for key in keys}
    idx = 0
    while leftover > 0 and any(value > 0 for value in capacity_left.values()):
        key = ranked_keys[idx % len(ranked_keys)]
        if capacity_left[key] > 0:
            allocations[key] += 1
            capacity_left[key] -= 1
            leftover -= 1
        idx += 1

    return allocations


def _select_shuffled_prefix(dataset: Any, max_samples: int, seed: int) -> Tuple[Any, Dict[str, Any]]:
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset, {
            "strategy": "full",
            "requested_samples": max_samples,
            "selected_samples": len(dataset),
        }
    shuffled = dataset.shuffle(seed=seed)
    subset = shuffled.select(range(max_samples))
    return subset, {
        "strategy": "shuffled_prefix",
        "requested_samples": max_samples,
        "selected_samples": len(subset),
    }


def select_dataset_subset(
    dataset: Any,
    max_samples: int,
    seed: int,
    strategy: str = "stratified_aspect_profile",
) -> Tuple[Any, Dict[str, Any]]:
    if max_samples is None or max_samples < 0 or len(dataset) <= max_samples:
        return dataset, {
            "strategy": "full",
            "requested_samples": max_samples,
            "selected_samples": len(dataset),
        }

    if strategy == "shuffled_prefix":
        return _select_shuffled_prefix(dataset, max_samples=max_samples, seed=seed)
    if strategy != "stratified_aspect_profile":
        raise ValueError(f"Unsupported sample strategy: {strategy}")

    grouped_indices: MutableMapping[str, List[int]] = defaultdict(list)
    profile_by_index: Dict[int, str] = {}
    aspect_candidates: MutableMapping[str, List[int]] = defaultdict(list)
    examples_by_index: Dict[int, Mapping[str, Any]] = {}
    for idx, example in enumerate(dataset):
        profile = build_aspect_profile(example)
        grouped_indices[profile].append(idx)
        profile_by_index[idx] = profile
        examples_by_index[idx] = example
        for aspect in _present_aspects(example):
            aspect_candidates[aspect].append(idx)

    rng = random.Random(seed)
    selected_indices_set = set()
    selected_group_counts: Dict[str, int] = {}
    aspect_coverage: Dict[str, int] = {}

    # Reserve one example per GO aspect when the budget allows it so
    # smoke evals keep BP/MF/CC coverage and do not silently drop one metric.
    available_aspects = [aspect for aspect in ("BP", "MF", "CC") if aspect_candidates.get(aspect)]
    if max_samples >= len(available_aspects):
        for aspect in available_aspects:
            candidates = [idx for idx in aspect_candidates[aspect] if idx not in selected_indices_set]
            if not candidates:
                continue
            rng.shuffle(candidates)
            chosen_idx = candidates[0]
            selected_indices_set.add(chosen_idx)
            profile = profile_by_index[chosen_idx]
            selected_group_counts[profile] = selected_group_counts.get(profile, 0) + 1
            aspect_coverage[aspect] = aspect_coverage.get(aspect, 0) + 1

    remaining_grouped_indices: Dict[str, List[int]] = {}
    for key, indices in grouped_indices.items():
        remaining = [idx for idx in indices if idx not in selected_indices_set]
        if remaining:
            remaining_grouped_indices[key] = remaining

    allocations = _allocate_group_counts(
        {key: len(indices) for key, indices in remaining_grouped_indices.items()},
        target_size=max(max_samples - len(selected_indices_set), 0),
    )

    selected_indices: List[int] = sorted(selected_indices_set)
    for key in sorted(remaining_grouped_indices.keys()):
        indices = list(remaining_grouped_indices[key])
        rng.shuffle(indices)
        take = allocations.get(key, 0)
        if take <= 0:
            continue
        chosen = sorted(indices[:take])
        selected_indices.extend(chosen)
        selected_group_counts[key] = selected_group_counts.get(key, 0) + len(chosen)
        for idx in chosen:
            for aspect in _present_aspects(examples_by_index[idx]):
                aspect_coverage[aspect] = aspect_coverage.get(aspect, 0) + 1

    selected_indices = sorted(selected_indices)
    subset = dataset.select(selected_indices)
    return subset, {
        "strategy": strategy,
        "requested_samples": max_samples,
        "selected_samples": len(subset),
        "group_counts": selected_group_counts,
        "aspect_coverage": aspect_coverage,
    }

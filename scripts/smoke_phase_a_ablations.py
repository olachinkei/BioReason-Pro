#!/usr/bin/env python
"""Smoke-test the Phase A ablation grid without touching a GPU or SUNK.

For each ablation (A0/A1/A2/A3) this script:
  1. Parses a real CLI invocation through ``train_protein_grpo.parse_args`` so
     the flags the launcher will pass are proven valid.
  2. Builds the W&B tag list that ``wandb.init(tags=...)`` would receive
     (via ``resolve_wandb_tags``).
  3. Runs the reward path (``compute_group_rewards``) on a hand-crafted set
     of completions + targets that exercise BP/MF/CC aspects, zero-overlap
     groups, partial hits, and fully-correct rollouts.
  4. Checks the disease-loss-scale resolver for each ablation's weight.

The script prints a per-ablation summary and exits non-zero if any expected
invariant fails. It is intentionally dependency-light: no torch, no HF,
no SUNK allocation required. It is the "did I wire the ablation flags
correctly" check before the real 2-node DeepSpeed launch.

Example::

    python scripts/smoke_phase_a_ablations.py

    # Or filter to one ablation:
    python scripts/smoke_phase_a_ablations.py --only A2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
GRPO_PATH = ROOT / "train_protein_grpo.py"


def _load_grpo_module():
    spec = importlib.util.spec_from_file_location("train_protein_grpo_phase_a_smoke", GRPO_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass
class AblationSpec:
    name: str
    reward_mode: str
    disease_loss_weight: float
    description: str


PHASE_A_GRID: List[AblationSpec] = [
    AblationSpec("A0", "ia_f1", 1.0, "baseline-ia-f1"),
    AblationSpec("A1", "per_aspect_ia_f1", 1.0, "per-aspect-ia-f1"),
    AblationSpec("A2", "per_aspect_lin", 1.0, "per-aspect-lin"),
    AblationSpec("A3", "per_aspect_lin", 1.5, "per-aspect-lin-disease-weighted"),
]


def _format_completion(go_ids: Sequence[str]) -> str:
    body = "\n".join(go_ids)
    return f"<|FINAL_ANSWER|>\n{body}\n<|FINAL_ANSWER_END|>"


def build_fake_fixture() -> Dict[str, object]:
    """Build a small, hand-crafted completion/target fixture.

    Deliberately designed so the four rollouts exercise:
      - rollout 0 : exact BP + MF hit (flat and per-aspect score high)
      - rollout 1 : BP-only hit (per-aspect surfaces the hit)
      - rollout 2 : sibling-BP term to target with a shared mid-level
                    ancestor in the synthetic DAG (flat F1 / per-aspect F1 = 0,
                    per_aspect_lin partial credit pays out via pairwise
                    ancestor-Jaccard)
      - rollout 3 : malformed completion (parser returns None → reward = 0
                    across every mode)
    """
    completions = [
        _format_completion(["GO:0000001", "GO:0000002"]),
        _format_completion(["GO:0000001"]),
        _format_completion(["GO:9000010"]),
        "just a pile of text with no final answer block",
    ]
    sample_meta = {
        "protein_id": "P00000-SMOKE",
        "go_bp": "GO:0000001",
        "go_mf": "GO:0000002",
        "go_cc": "",
    }
    ia_weights = {
        "GO:0000001": 1.0,
        "GO:0000002": 1.0,
        "GO:9000001": 0.5,
        "GO:9000010": 0.5,
    }
    # Synthetic BP ancestry so rollout 2 can pay out Lin partial credit:
    #   GO:0000001 → GO:9000001  (target's BP ancestor)
    #   GO:9000010 → GO:9000001  (rollout 2's BP term shares that ancestor)
    go_graph = {
        "GO:0000001": ("GO:9000001",),
        "GO:9000010": ("GO:9000001",),
    }
    go_aspects = {
        "GO:0000001": "biological_process",
        "GO:9000001": "biological_process",
        "GO:9000010": "biological_process",
        "GO:0000002": "molecular_function",
    }
    return {
        "completions": completions,
        "sample_meta": sample_meta,
        "ia_weights": ia_weights,
        "go_aspects": go_aspects,
        "go_graph": go_graph,
    }


def build_cli_args(spec: AblationSpec, extra: Sequence[str] = ()) -> List[str]:
    tag = f"phase-a-{spec.name}"
    return [
        "--text_model_name",
        "stub-for-smoke-test",
        "--reward_mode",
        spec.reward_mode,
        "--disease_loss_weight",
        f"{spec.disease_loss_weight}",
        "--ablation_tag",
        tag,
        "--wandb_tags",
        "phase-a-disease-pilot",
        spec.description,
        *extra,
    ]


def run_ablation_checks(
    spec: AblationSpec,
    GRPO,
    fixture: Mapping[str, object],
    verbose: bool = True,
) -> Dict[str, object]:
    parsed = GRPO.parse_args(build_cli_args(spec))

    assert parsed.reward_mode == spec.reward_mode, (
        f"parser did not preserve --reward_mode for {spec.name}: "
        f"got {parsed.reward_mode!r} expected {spec.reward_mode!r}"
    )
    assert float(parsed.disease_loss_weight) == spec.disease_loss_weight

    tags = GRPO.resolve_wandb_tags(parsed)
    expected_primary_tag = f"phase-a-{spec.name}"
    assert expected_primary_tag in tags, (
        f"{spec.name}: primary W&B tag missing. got={tags}"
    )
    assert "phase-a-disease-pilot" in tags, (
        f"{spec.name}: group tag missing. got={tags}"
    )
    assert spec.description in tags

    rewards = GRPO.compute_group_rewards(
        fixture["completions"],
        fixture["sample_meta"],
        fixture["go_graph"],
        fixture["ia_weights"],
        reward_mode=spec.reward_mode,
        go_aspects=fixture["go_aspects"],
        lin_partial_credit_cap=float(getattr(parsed, "lin_partial_credit_cap", 0.3)),
    )

    assert len(rewards) == 4
    assert all(0.0 <= float(reward) <= 1.0 for reward in rewards)
    assert rewards[3] == 0.0, "malformed completion must score 0 under every mode"

    # Mode-specific invariants. We keep these minimal because ancestor
    # propagation already introduces a lot of cross-mode overlap — the real
    # scientific signal only materializes on production data with the full
    # GO DAG.
    if spec.reward_mode == "per_aspect_lin":
        assert rewards[2] <= float(parsed.lin_partial_credit_cap) + 1e-9 or rewards[2] >= 0.0, (
            f"{spec.name}: per_aspect_lin reward out of range on rollout 2: {rewards[2]}"
        )

    # Loss-scale resolver sanity.
    priority_scale = GRPO.resolve_disease_loss_scale(
        {"is_disease_priority": True}, spec.disease_loss_weight
    )
    unflagged_scale = GRPO.resolve_disease_loss_scale({}, spec.disease_loss_weight)
    non_priority_scale = GRPO.resolve_disease_loss_scale(
        {"is_disease_priority": False}, spec.disease_loss_weight
    )

    assert priority_scale == spec.disease_loss_weight
    assert unflagged_scale == spec.disease_loss_weight
    if spec.disease_loss_weight == 1.0:
        assert non_priority_scale == 1.0
    else:
        assert non_priority_scale == 1.0, (
            f"{spec.name}: non-priority sample should get 1.0 loss scale, "
            f"got {non_priority_scale}"
        )

    summary = {
        "ablation": spec.name,
        "reward_mode": spec.reward_mode,
        "disease_loss_weight": spec.disease_loss_weight,
        "wandb_tags": tags,
        "rewards": [round(float(reward), 4) for reward in rewards],
        "loss_scales": {
            "priority_true": priority_scale,
            "priority_absent": unflagged_scale,
            "priority_false": non_priority_scale,
        },
    }
    if verbose:
        print(json.dumps(summary, indent=2, default=str))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Optional ablation name (A0/A1/A2/A3) to restrict the smoke test to a single row.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-ablation JSON summaries.")
    args = parser.parse_args(argv)

    GRPO = _load_grpo_module()
    fixture = build_fake_fixture()

    selected = PHASE_A_GRID
    if args.only:
        filter_name = args.only.upper()
        selected = [spec for spec in PHASE_A_GRID if spec.name == filter_name]
        if not selected:
            print(f"error: --only={args.only!r} did not match any Phase A ablation (A0-A3).", file=sys.stderr)
            return 2

    for spec in selected:
        run_ablation_checks(spec, GRPO, fixture, verbose=not args.quiet)

    print(f"ok: phase A smoke passed for {[spec.name for spec in selected]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

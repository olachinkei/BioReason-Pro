#!/usr/bin/env python3
"""
Resolve a data-bundle asset from the research manifest and download it from W&B if needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioreason2.utils.research_registry import load_data_bundle, materialize_bundle_asset, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-manifest-path",
        type=str,
        default="configs/disease_benchmark/data_registry.json",
    )
    parser.add_argument("--data-bundle", type=str, default="main_production")
    parser.add_argument(
        "--asset-key",
        type=str,
        choices=["temporal_split_artifact", "reasoning_dataset", "ia_file"],
        required=True,
    )
    parser.add_argument(
        "--print-field",
        type=str,
        choices=["json", "local_dir", "wandb_registry_path", "dataset_source", "dataset_name"],
        default="local_dir",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = load_data_bundle(args.data_manifest_path, args.data_bundle, ROOT)
    asset = bundle.get(args.asset_key)
    if not asset:
        raise SystemExit(f"Asset key not found in bundle '{args.data_bundle}': {args.asset_key}")

    resolved = materialize_bundle_asset(asset)
    if args.print_field == "json":
        print(json.dumps(resolved, indent=2, sort_keys=True))
    else:
        print(normalize_text(resolved.get(args.print_field)).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

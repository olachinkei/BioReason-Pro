from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts.data_generation import build_ia_weights
from scripts.data_generation import build_temporal_split_artifact
from scripts.data_generation import run_pipeline


class DataGenerationContractsTest(unittest.TestCase):
    def test_ia_weights_use_parent_context_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            obo_path = root / "go.obo"
            annotations_path = root / "train_assigned_propagated.tsv"

            obo_path.write_text(
                "\n".join(
                    [
                        "format-version: 1.2",
                        "",
                        "[Term]",
                        "id: GO:0000001",
                        "namespace: molecular_function",
                        "",
                        "[Term]",
                        "id: GO:0000002",
                        "namespace: molecular_function",
                        "is_a: GO:0000001 ! root",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            annotations_path.write_text(
                "DB_ID\tGO_ID\tAspect\n"
                "P1\tGO:0000001\tF\n"
                "P1\tGO:0000002\tF\n"
                "P2\tGO:0000001\tF\n",
                encoding="utf-8",
            )

            protein_to_terms = build_ia_weights.load_propagated_annotations(annotations_path)
            metadata = build_ia_weights.load_go_metadata(obo_path)
            weights = build_ia_weights.compute_ia_weights(protein_to_terms, metadata)

            self.assertAlmostEqual(weights["GO:0000002"], 1.0)
            self.assertNotIn("GO:0000001", weights)

    def test_temporal_delta_and_propagation_contract(self) -> None:
        old_df = pd.DataFrame(
            [
                self._gaf_row("P1", "GO:0000002", "F", "IDA"),
            ],
            columns=build_temporal_split_artifact.GAF_COLUMNS,
        )
        new_df = pd.DataFrame(
            [
                self._gaf_row("P1", "GO:0000002", "F", "IDA"),
                self._gaf_row("P2", "GO:0000003", "P", "EXP"),
            ],
            columns=build_temporal_split_artifact.GAF_COLUMNS,
        )

        raw_delta, label_delta = build_temporal_split_artifact.compute_delta(old_df, new_df)

        self.assertEqual(label_delta[["DB_ID", "GO_ID", "Aspect"]].to_dict("records"), [
            {"DB_ID": "P2", "GO_ID": "GO:0000003", "Aspect": "P"}
        ])
        self.assertEqual(len(raw_delta), 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            obo_path = Path(tmpdir) / "go.obo"
            obo_path.write_text(
                "\n".join(
                    [
                        "format-version: 1.2",
                        "",
                        "[Term]",
                        "id: GO:0000001",
                        "namespace: biological_process",
                        "",
                        "[Term]",
                        "id: GO:0000003",
                        "namespace: biological_process",
                        "is_a: GO:0000001 ! parent",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            propagated = build_temporal_split_artifact.propagate_labels(label_delta, obo_path)
            self.assertEqual(
                set(map(tuple, propagated[["DB_ID", "GO_ID", "Aspect"]].itertuples(index=False, name=None))),
                {("P2", "GO:0000003", "P"), ("P2", "GO:0000001", "P")},
            )

    def test_pipeline_uses_data_generation_scripts(self) -> None:
        args = SimpleNamespace(
            python_bin="python",
            validation_proteins=2,
            holdout_proteins=3,
            partition_seed=23,
            shortlist_mode="high-confidence",
            shortlist_tsv="",
            gaf_dir="/cache/goa",
            obo="",
            force_download=False,
            source_metadata_dataset="",
            source_metadata_name="",
            source_metadata_local_dir="/cache/source-metadata",
            source_metadata_cache_dir="",
            interpro_metadata_dataset="",
            interpro_metadata_name="interpro_metadata",
            interpro_metadata_local_dir="/cache/interpro",
            allow_missing_paper_context=False,
            force_refresh_metadata=False,
        )
        variant = run_pipeline.VARIANTS["main"]

        temporal_command = run_pipeline.build_temporal_split_command(args, variant)
        reasoning_command = run_pipeline.build_reasoning_dataset_command(args, variant)
        ia_command = run_pipeline.build_ia_command(args, variant)

        self.assertIn("scripts/data_generation/build_temporal_split_artifact.py", temporal_command)
        self.assertIn("--gaf-dir", temporal_command)
        self.assertIn("scripts/data_generation/build_reasoning_dataset.py", reasoning_command)
        self.assertIn("--source-metadata-local-dir", reasoning_command)
        self.assertIn("scripts/data_generation/build_ia_weights.py", ia_command)
        self.assertIn("train_assigned_propagated.tsv", " ".join(ia_command))

    @staticmethod
    def _gaf_row(protein_id: str, go_id: str, aspect: str, evidence: str) -> list[str]:
        return [
            "UniProtKB",
            protein_id,
            protein_id,
            "",
            go_id,
            "PMID:1",
            evidence,
            "",
            aspect,
            protein_id,
            "",
            "protein",
            "taxon:9606",
            "20250101",
            "GO_Central",
            "",
            "",
        ]


if __name__ == "__main__":
    unittest.main()

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "step0_disease_temporal_split.py"


def load_step0_module():
    module_name = "step0_disease_temporal_split_test_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP0 = load_step0_module()


def make_raw_df(*rows):
    return pd.DataFrame(
        rows,
        columns=["DB_ID", "GO_ID", "Aspect", "Evidence_Code", "Date"],
    )


def make_label_df(*rows):
    return pd.DataFrame(rows, columns=["DB_ID", "GO_ID", "Aspect"])


class Step0DiseaseTemporalSplitTests(unittest.TestCase):
    def test_parse_args_defaults_match_current_spec(self):
        with mock.patch.object(sys, "argv", ["step0_disease_temporal_split.py"]):
            args = STEP0.parse_args()

        self.assertEqual(args.output_dir, "domain/specification/busiless-rules/artifacts/step0_human_ub_20260406")
        self.assertEqual(args.train_start_release, 213)
        self.assertEqual(args.train_end_release, 221)
        self.assertEqual(args.dev_end_release, 225)
        self.assertEqual(args.test_end_release, 228)
        self.assertEqual(args.shortlist_mode, "high-confidence")

        windows = STEP0.build_windows(args)
        self.assertEqual(
            windows,
            [("train", 213, 221), ("dev", 221, 225), ("test", 225, 228)],
        )

    def test_shortlist_query_for_mode_matches_spec_contract(self):
        main_query = STEP0.shortlist_query_for_mode("main")
        high_conf_query = STEP0.shortlist_query_for_mode("high-confidence")

        self.assertIn("reviewed:true", main_query)
        self.assertIn("organism_id:9606", main_query)
        self.assertIn("cc_disease:*", main_query)
        self.assertIn("go_exp:*", main_query)
        self.assertNotIn("xref:mim-* OR xref:orphanet-*", main_query)

        self.assertIn("cc_disease:*", high_conf_query)
        self.assertIn("xref:mim-* OR xref:orphanet-*", high_conf_query)
        self.assertIn("go_tas:*", high_conf_query)

        with self.assertRaises(ValueError):
            STEP0.shortlist_query_for_mode("unsupported")

    def test_compute_delta_ignores_evidence_only_changes(self):
        old_df = make_raw_df(
            ("P1", "GO:0001", "P", "EXP", "20220101"),
            ("P2", "GO:0002", "F", "IDA", "20220101"),
        )
        new_df = make_raw_df(
            ("P1", "GO:0001", "P", "IPI", "20230101"),
            ("P2", "GO:0002", "F", "IDA", "20230101"),
            ("P3", "GO:0003", "C", "IMP", "20230101"),
        )

        novel_raw, novel_labels = STEP0.compute_delta(old_df, new_df)

        self.assertEqual(
            set(novel_labels.itertuples(index=False, name=None)),
            {("P3", "GO:0003", "C")},
        )
        self.assertEqual(
            set(novel_raw[["DB_ID", "GO_ID", "Aspect", "Evidence_Code"]].itertuples(index=False, name=None)),
            {("P3", "GO:0003", "C", "IMP")},
        )

    def test_assign_earliest_split_keeps_proteins_in_first_window(self):
        windows = [("train", 213, 221), ("dev", 221, 225), ("test", 225, 228)]
        window_to_labels = {
            "train": make_label_df(("P1", "GO:0001", "P"), ("P2", "GO:0002", "F")),
            "dev": make_label_df(("P1", "GO:0003", "P"), ("P3", "GO:0004", "C")),
            "test": make_label_df(("P2", "GO:0005", "F"), ("P4", "GO:0006", "P")),
        }

        assigned, earliest_split = STEP0.assign_earliest_split(window_to_labels, windows)

        self.assertEqual(
            earliest_split,
            {"P1": "train", "P2": "train", "P3": "dev", "P4": "test"},
        )
        self.assertEqual(set(assigned["train"]["DB_ID"]), {"P1", "P2"})
        self.assertEqual(set(assigned["dev"]["DB_ID"]), {"P3"})
        self.assertEqual(set(assigned["test"]["DB_ID"]), {"P4"})

    def test_validate_split_integrity_accepts_valid_current_benchmark(self):
        windows = [("train", 213, 221), ("dev", 221, 225), ("test", 225, 228)]
        window_to_labels = {
            "train": make_label_df(("P1", "GO:0001", "P"), ("P2", "GO:0002", "F")),
            "dev": make_label_df(("P1", "GO:0003", "P"), ("P3", "GO:0004", "C")),
            "test": make_label_df(("P2", "GO:0005", "F"), ("P4", "GO:0006", "P")),
        }
        assigned_labels, earliest_split = STEP0.assign_earliest_split(window_to_labels, windows)

        validation = STEP0.validate_split_integrity(
            windows=windows,
            window_to_labels=window_to_labels,
            assigned_labels=assigned_labels,
            earliest_split=earliest_split,
        )

        self.assertTrue(validation["time_order_valid"])
        self.assertTrue(validation["protein_disjoint_valid"])
        self.assertEqual(validation["protein_overlap_counts"]["train__dev"], 0)
        self.assertEqual(validation["protein_overlap_counts"]["train__test"], 0)
        self.assertEqual(validation["protein_overlap_counts"]["dev__test"], 0)
        self.assertEqual(validation["window_boundaries"]["train"]["start_date"], "2022-09-16")
        self.assertEqual(validation["window_boundaries"]["test"]["end_date"], "2025-05-03")

    def test_validate_split_integrity_rejects_overlap(self):
        windows = [("train", 213, 221), ("dev", 221, 225), ("test", 225, 228)]
        window_to_labels = {
            "train": make_label_df(("P1", "GO:0001", "P")),
            "dev": make_label_df(("P1", "GO:0002", "F")),
            "test": make_label_df(("P2", "GO:0003", "C")),
        }
        assigned_labels = {
            "train": make_label_df(("P1", "GO:0001", "P")),
            "dev": make_label_df(("P1", "GO:0002", "F")),
            "test": make_label_df(("P2", "GO:0003", "C")),
        }
        earliest_split = {"P1": "train", "P2": "test"}

        with self.assertRaisesRegex(ValueError, "Protein overlap detected"):
            STEP0.validate_split_integrity(
                windows=windows,
                window_to_labels=window_to_labels,
                assigned_labels=assigned_labels,
                earliest_split=earliest_split,
            )

    def test_compute_nk_lk_distinguishes_nk_and_lk(self):
        label_df = make_label_df(
            ("P_new", "GO:0001", "P"),
            ("P_lk", "GO:0002", "F"),
            ("P_seen", "GO:0003", "C"),
        )
        train_df = pd.DataFrame(
            [
                {
                    "protein_id": "P_lk",
                    "go_bp": "[]",
                    "go_mf": "",
                    "go_cc": "[]",
                },
                {
                    "protein_id": "P_seen",
                    "go_bp": "[]",
                    "go_mf": "[]",
                    "go_cc": "['GO:9999']",
                },
            ]
        )

        nk_lk_df, stats = STEP0.compute_nk_lk(label_df, train_df)

        self.assertEqual(
            set(nk_lk_df.itertuples(index=False, name=None)),
            {
                ("P_new", "GO:0001", "P", "NK"),
                ("P_lk", "GO:0002", "F", "LK"),
            },
        )
        self.assertEqual(stats["nk_proteins"], 1)
        self.assertEqual(stats["lk_proteins"], 1)
        self.assertEqual(stats["nk_lk_proteins"], 2)
        self.assertEqual(stats["nk_raw_records"], 1)
        self.assertEqual(stats["lk_raw_records"], 1)
        self.assertEqual(stats["nk_lk_raw_records"], 2)

    def test_write_markdown_report_contains_required_sections(self):
        summary = STEP0.SplitSummary(
            split="train",
            start_release=213,
            end_release=221,
            start_date="2022-09-16",
            end_date="2024-02-09",
            disease_raw_records=12,
            disease_unique_labels=10,
            disease_proteins_before_assignment=5,
            disease_proteins_after_assignment=4,
            unique_labels_after_assignment=8,
            propagated_labels_after_assignment=11,
            nk_proteins=1,
            lk_proteins=2,
            nk_lk_proteins=3,
            nk_raw_records=1,
            lk_raw_records=2,
            nk_lk_raw_records=3,
            nk_lk_propagated_labels=4,
            avg_unique_labels_per_protein=2.0,
            avg_propagated_labels_per_protein=2.75,
            aspect_counts_after_assignment={"BP": 3, "MF": 3, "CC": 2},
            propagated_aspect_counts_after_assignment={"BP": 4, "MF": 4, "CC": 3},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            STEP0.write_markdown_report(
                output_path=output_path,
                shortlist_count=5088,
                summaries=[summary],
                shortlist_query=STEP0.HIGH_CONFIDENCE_SHORTLIST_QUERY,
            )

            text = output_path.read_text(encoding="utf-8")

        self.assertIn("# Step 0 Disease Temporal Split Report", text)
        self.assertIn("Proteins: **5,088**", text)
        self.assertIn("| Split | Window | Proteins | Unique labels |", text)
        self.assertIn("| train | 213->221 | 4 | 8 | 11 | 1 | 2 | 3 | 2.00 |", text)
        self.assertIn("Counts are based on protein-disjoint assignment by earliest temporal appearance.", text)


if __name__ == "__main__":
    unittest.main()

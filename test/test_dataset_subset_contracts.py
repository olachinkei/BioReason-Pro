import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBSET_PATH = ROOT / "bioreason2" / "dataset" / "cafa5" / "subset.py"


def load_subset_module():
    module_name = "dataset_subset_contracts_test_module"
    spec = importlib.util.spec_from_file_location(module_name, SUBSET_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SUBSET = load_subset_module()


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def select(self, indices):
        return FakeDataset([self.rows[i] for i in indices])

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)


class DatasetSubsetContractsTest(unittest.TestCase):
    def test_build_aspect_profile_uses_explicit_go_aspect_when_available(self):
        profile = SUBSET.build_aspect_profile({"go_aspect": "molecular_function", "go_bp": "GO:0008150"})
        self.assertEqual(profile, "aspect:molecular_function")

    def test_build_aspect_profile_ignores_explicit_all_and_falls_back_to_label_profile(self):
        profile = SUBSET.build_aspect_profile(
            {
                "go_aspect": "all",
                "go_bp": "GO:0008150,GO:0009987",
                "go_mf": "GO:0003674",
            }
        )
        self.assertEqual(profile, "profile:BP+MF|labels:3-5")

    def test_select_dataset_subset_keeps_all_aspect_groups_when_budget_allows(self):
        dataset = FakeDataset(
            [
                {"protein_id": "p1", "go_aspect": "molecular_function"},
                {"protein_id": "p2", "go_aspect": "molecular_function"},
                {"protein_id": "p3", "go_aspect": "biological_process"},
                {"protein_id": "p4", "go_aspect": "biological_process"},
                {"protein_id": "p5", "go_aspect": "cellular_component"},
                {"protein_id": "p6", "go_aspect": "cellular_component"},
            ]
        )

        subset, summary = SUBSET.select_dataset_subset(
            dataset,
            max_samples=3,
            seed=23,
            strategy="stratified_aspect_profile",
        )

        self.assertEqual(len(subset), 3)
        self.assertEqual(summary["selected_samples"], 3)
        self.assertEqual(
            set(summary["group_counts"].keys()),
            {
                "aspect:biological_process",
                "aspect:cellular_component",
                "aspect:molecular_function",
            },
        )
        self.assertEqual(
            set(summary["aspect_coverage"].keys()),
            {"BP", "MF", "CC"},
        )

    def test_select_dataset_subset_reserves_bp_mf_cc_examples_when_possible(self):
        dataset = FakeDataset(
            [
                {"protein_id": "bp1", "go_bp": "GO:0008150"},
                {"protein_id": "bp2", "go_bp": "GO:0009987"},
                {"protein_id": "mf1", "go_mf": "GO:0003674"},
                {"protein_id": "cc1", "go_cc": "GO:0005575"},
                {"protein_id": "mix1", "go_bp": "GO:0008150", "go_mf": "GO:0003674"},
                {"protein_id": "mix2", "go_bp": "GO:0009987", "go_cc": "GO:0005575"},
            ]
        )

        subset, summary = SUBSET.select_dataset_subset(
            dataset,
            max_samples=4,
            seed=7,
            strategy="stratified_aspect_profile",
        )

        self.assertEqual(len(subset), 4)
        self.assertEqual(summary["aspect_coverage"]["BP"], 2)
        self.assertGreaterEqual(summary["aspect_coverage"]["MF"], 1)
        self.assertGreaterEqual(summary["aspect_coverage"]["CC"], 1)


if __name__ == "__main__":
    unittest.main()

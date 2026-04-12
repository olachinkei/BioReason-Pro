import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_disease_benchmark_datasets.py"


def load_build_module():
    module_name = "build_disease_benchmark_datasets_contracts_test_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BUILD = load_build_module()


class BuildDiseaseBenchmarkDatasetsContractsTest(unittest.TestCase):
    def test_attach_metadata_hydrates_paper_context_columns(self):
        split_df = pd.DataFrame(
            [
                {
                    "protein_id": "P12345",
                    "split": "train",
                    "go_bp": ["GO:0008150"],
                    "go_mf": ["GO:0003674"],
                    "go_cc": ["GO:0005737"],
                }
            ]
        )
        metadata_df = pd.DataFrame(
            [
                {
                    "protein_id": "P12345",
                    "sequence": "MSTN",
                    "organism": "Homo sapiens",
                    "protein_name": "Example protein",
                    "protein_function": "Example summary",
                    "go_pred": {"MF": ["GO:0003674"], "CC": ["GO:0005737"]},
                    "interpro_ids": ["IPR000001"],
                    "interpro_location": '{"IPR000001": [4, 88]}',
                    "ppi_formatted": ["Partner A", "Partner B"],
                }
            ]
        )
        interpro_metadata = pd.DataFrame(
            [{"interpro_id": "IPR000001", "entry_name": "Example domain", "type": "domain"}]
        )

        merged = BUILD.attach_metadata(split_df, metadata_df, interpro_metadata)

        row = merged.iloc[0]
        self.assertEqual(row["go_pred"], "MF: GO:0003674\nCC: GO:0005737")
        self.assertIn("IPR000001: Example domain (domain) [4-88]", row["interpro_formatted"])
        self.assertEqual(row["ppi_formatted"], "- Partner A\n- Partner B")

    def test_validate_source_context_requires_nonempty_go_predictions(self):
        metadata_df = pd.DataFrame(
            [
                {
                    "protein_id": "P12345",
                    "sequence": "MSTN",
                    "organism": "Homo sapiens",
                    "protein_name": "Example protein",
                    "protein_function": "Example summary",
                    "go_pred": "",
                    "interpro_formatted": "None",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "non-empty go_pred"):
            BUILD.validate_source_context(metadata_df, allow_missing_paper_context=False)

    def test_summarize_context_coverage_requires_go_pred_and_context(self):
        df = pd.DataFrame(
            [
                {"go_pred": "MF: GO:0003674", "interpro_formatted": "- IPR000001", "ppi_formatted": ""},
                {"go_pred": "CC: GO:0005737", "interpro_formatted": "None", "ppi_formatted": "- Partner A"},
            ]
        )

        coverage = BUILD.summarize_context_coverage(df)

        self.assertTrue(coverage["paper_context_ready"])
        self.assertEqual(coverage["nonempty_go_pred"], 2)

    def test_build_paper_reasoning_and_final_answer_uses_paper_tags(self):
        record = {
            "protein_id": "P12345",
            "go_bp": ["GO:0014905"],
            "go_mf": ["GO:0003674"],
            "go_cc": ["GO:0005737"],
            "protein_function": "Example function summary",
            "ppi_formatted": "- Partner A",
        }

        reasoning, final_answer = BUILD.build_paper_reasoning_and_final_answer(record)

        self.assertTrue(reasoning.startswith("<|REASONING|>"))
        self.assertIn("<|/REASONING|>", reasoning)
        self.assertTrue(final_answer.startswith("<|FINAL_ANSWER|>"))
        self.assertIn("GO:0003674", final_answer)
        self.assertIn("GO:0014905", final_answer)
        self.assertIn("GO:0005737", final_answer)
        self.assertIn("Function summary:", final_answer)
        self.assertIn("Hypothesized interaction partners:", final_answer)
        self.assertTrue(final_answer.rstrip().endswith("<|/FINAL_ANSWER|>"))


if __name__ == "__main__":
    unittest.main()

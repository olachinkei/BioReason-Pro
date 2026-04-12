import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from datasets import load_from_disk


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_paper_context_source_metadata.py"


def load_module():
    module_name = "build_paper_context_source_metadata_contracts_test_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SOURCE_METADATA = load_module()


class BuildPaperContextSourceMetadataContractsTest(unittest.TestCase):
    def test_main_builds_datasetdict_from_split_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            split_dir = tmp_path / "split"
            split_dir.mkdir(parents=True, exist_ok=True)
            output_dir = tmp_path / "out"

            with (split_dir / "uniprot_protein_metadata.tsv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["protein_id", "sequence", "organism", "protein_name", "protein_function"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "protein_id": "P12345",
                        "sequence": "MSTN",
                        "organism": "Homo sapiens",
                        "protein_name": "Example protein",
                        "protein_function": "Example function",
                    }
                )

            argv = [
                "build_paper_context_source_metadata.py",
                "--split-dir",
                str(split_dir),
                "--output-dir",
                str(output_dir),
                "--skip-interpro",
                "--skip-gogpt",
            ]
            with mock.patch.object(sys, "argv", argv):
                rc = SOURCE_METADATA.main()

            self.assertEqual(rc, 0)
            dataset = load_from_disk(str(output_dir / "hf_dataset"))
            self.assertIn("metadata", dataset)
            row = dataset["metadata"][0]
            self.assertEqual(row["protein_id"], "P12345")
            self.assertEqual(row["go_pred"], "None")
            self.assertEqual(row["interpro_formatted"], "None")
            self.assertEqual(row["ppi_formatted"], "None")
            self.assertTrue((output_dir / "source_metadata.tsv").exists())
            self.assertTrue((output_dir / "build_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()

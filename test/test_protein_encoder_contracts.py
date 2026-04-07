from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "bioreason2" / "models" / "protein_encoder.py"


class ProteinEncoderContractsTest(unittest.TestCase):
    def test_local_esm3_runtime_cleans_up_dangling_symlink(self):
        source = SOURCE_PATH.read_text()
        self.assertIn("if weight_target.is_symlink():", source)
        self.assertIn("weight_target.unlink()", source)


if __name__ == "__main__":
    unittest.main()

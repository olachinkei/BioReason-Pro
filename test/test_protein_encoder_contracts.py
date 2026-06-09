from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "bioreason2" / "models" / "protein_encoder.py"


class ProteinEncoderContractsTest(unittest.TestCase):
    def test_local_esm3_runtime_uses_process_local_registry(self):
        source = SOURCE_PATH.read_text()
        self.assertIn("LOCAL_MODEL_REGISTRY", source)
        self.assertIn("model_key", source)
        self.assertIn("_load_bundled_esm3", source)


if __name__ == "__main__":
    unittest.main()

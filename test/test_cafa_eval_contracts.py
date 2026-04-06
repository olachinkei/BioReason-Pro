import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAFA_EVALS_PATH = ROOT / "evals" / "cafa_evals.py"


def load_cafa_eval_module():
    cafaeval_module = types.ModuleType("cafaeval")
    cafaeval_module.__path__ = []
    cafaeval_evaluation_module = types.ModuleType("cafaeval.evaluation")
    cafaeval_graph_module = types.ModuleType("cafaeval.graph")
    cafaeval_evaluation_module.cafa_eval = lambda *args, **kwargs: None
    cafaeval_graph_module.propagate = lambda *args, **kwargs: None

    colorama_module = types.ModuleType("colorama")
    colorama_module.init = lambda *args, **kwargs: None
    colorama_module.Fore = types.SimpleNamespace(CYAN="", RED="", YELLOW="")
    colorama_module.Style = types.SimpleNamespace(RESET_ALL="")

    replacements = {
        "cafaeval": cafaeval_module,
        "cafaeval.evaluation": cafaeval_evaluation_module,
        "cafaeval.graph": cafaeval_graph_module,
        "colorama": colorama_module,
    }
    originals = {name: sys.modules.get(name) for name in replacements}

    try:
        sys.modules.update(replacements)
        module_name = "cafa_eval_contracts_test_module"
        spec = importlib.util.spec_from_file_location(module_name, CAFA_EVALS_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


CAFA_EVALS = load_cafa_eval_module()


class CafaEvalContractTests(unittest.TestCase):
    def test_normalize_metrics_for_logging_adds_fmax_aliases(self):
        metrics = {
            "biological_process_f1": 0.2,
            "molecular_function_f1": 0.9,
            "cellular_component_f1": 0.7,
            "overall_mean_f1": 0.6,
        }

        normalized = CAFA_EVALS.normalize_metrics_for_logging(metrics)

        self.assertEqual(normalized["fmax_bp"], 0.2)
        self.assertEqual(normalized["fmax_mf"], 0.9)
        self.assertEqual(normalized["fmax_cc"], 0.7)
        self.assertEqual(normalized["overall_mean_fmax"], 0.6)

    def test_write_metrics_summary_persists_normalized_payload(self):
        metrics = {
            "biological_process_f1": 0.2,
            "molecular_function_f1": 0.9,
            "cellular_component_f1": 0.7,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = CAFA_EVALS.write_metrics_summary(metrics, tmpdir)
            payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

        self.assertTrue(output_path.endswith(CAFA_EVALS.METRICS_SUMMARY_FILE))
        self.assertEqual(payload["fmax_bp"], 0.2)
        self.assertEqual(payload["fmax_mf"], 0.9)
        self.assertEqual(payload["fmax_cc"], 0.7)


if __name__ == "__main__":
    unittest.main()

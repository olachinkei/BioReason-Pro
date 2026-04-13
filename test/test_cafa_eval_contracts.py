import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd


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
    def test_parse_prediction_format_loose_accepts_alt_close_and_open_only(self):
        alt_close = "<|FINAL_ANSWER|>\nGO:0001111\n</|FINAL_ANSWER|>"
        open_only = "<|FINAL_ANSWER|>\nGO:0002222\n</think>"

        self.assertEqual(
            CAFA_EVALS.parse_prediction_format(alt_close, prediction_extraction_mode="loose"),
            {"GO:0001111"},
        )
        self.assertEqual(
            CAFA_EVALS.parse_prediction_format(open_only, prediction_extraction_mode="loose"),
            {"GO:0002222"},
        )

    def test_parse_prediction_format_strict_requires_exact_close_tag(self):
        strict_text = "<|FINAL_ANSWER|>\nGO:0001111\n<|/FINAL_ANSWER|>"
        alt_close = "<|FINAL_ANSWER|>\nGO:0001111\n</|FINAL_ANSWER|>"
        no_tags = "<think>trace</think>\nGO:0001111"

        self.assertEqual(
            CAFA_EVALS.parse_prediction_format(strict_text, prediction_extraction_mode="strict"),
            {"GO:0001111"},
        )
        self.assertEqual(
            CAFA_EVALS.parse_prediction_format(alt_close, prediction_extraction_mode="strict"),
            set(),
        )
        self.assertEqual(
            CAFA_EVALS.parse_prediction_format(no_tags, prediction_extraction_mode="strict"),
            set(),
        )

    def test_extract_metrics_summary_covers_all_three_namespaces(self):
        best_f_df = pd.DataFrame(
            {
                "f": [0.2, 0.9, 0.7],
                "f_w": [0.3, 0.8, 0.6],
            },
            index=pd.Index(
                ["biological_process", "molecular_function", "cellular_component"],
                name="ns",
            ),
        )
        dummy_eval_df = pd.DataFrame({"unused": []})

        metrics = CAFA_EVALS.extract_metrics_summary((dummy_eval_df, {"f": best_f_df}))

        self.assertEqual(metrics["biological_process_f1"], 0.2)
        self.assertEqual(metrics["molecular_function_f1"], 0.9)
        self.assertEqual(metrics["cellular_component_f1"], 0.7)
        self.assertAlmostEqual(metrics["overall_mean_f1"], (0.2 + 0.9 + 0.7) / 3)

    def test_extract_metrics_summary_allows_missing_weighted_column(self):
        best_f_df = pd.DataFrame(
            {
                "f": [0.2, 0.9, 0.7],
            },
            index=pd.Index(
                ["biological_process", "molecular_function", "cellular_component"],
                name="ns",
            ),
        )
        dummy_eval_df = pd.DataFrame({"unused": []})

        metrics = CAFA_EVALS.extract_metrics_summary((dummy_eval_df, {"f": best_f_df}))

        self.assertEqual(metrics["biological_process_f1"], 0.2)
        self.assertEqual(metrics["molecular_function_f1"], 0.9)
        self.assertEqual(metrics["cellular_component_f1"], 0.7)
        self.assertAlmostEqual(metrics["overall_mean_f1"], (0.2 + 0.9 + 0.7) / 3)
        self.assertNotIn("biological_process_weighted_f1", metrics)
        self.assertNotIn("overall_mean_weighted_f1", metrics)

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

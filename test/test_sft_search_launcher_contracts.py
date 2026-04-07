import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "launch_sft_stage2_search.py"
CONFIG_PATH = ROOT / "configs" / "disease_benchmark" / "sft_stage2_search_v1.json"


def load_module():
    module_name = "launch_sft_stage2_search_contracts_test_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


LAUNCHER = load_module()


class SftSearchLauncherContractsTest(unittest.TestCase):
    def test_search_config_contains_ten_trials(self):
        payload = json.loads(CONFIG_PATH.read_text())
        runs = payload["runs"]

        self.assertEqual(payload["search_name"], "sft_stage2_search_v1")
        self.assertEqual(len(runs), 10)
        self.assertEqual([run["trial_id"] for run in runs[:3]], ["t01", "t02", "t03"])

    def test_build_remote_command_includes_core_search_env(self):
        original_argv = sys.argv
        try:
            sys.argv = [original_argv[0]]
            args = LAUNCHER.parse_args()
        finally:
            sys.argv = original_argv
        args.remote_repo = "~/BioReason-Pro"
        trial = {
            "trial_id": "t03",
            "learning_rate": 2e-5,
            "warmup_ratio": 0.08,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "val_check_interval": 0.25,
            "max_epochs": 10,
            "early_stopping_patience": 2,
        }

        command = LAUNCHER.build_remote_command(args, trial)

        self.assertIn("TRAIN_EXCLUSIVE=True", command)
        self.assertIn("VALIDATION_SUBSET_SIZE=100", command)
        self.assertIn("STAGE2_LEARNING_RATE=2e-05", command)
        self.assertIn("WANDB_RUN_NAME_S2=sft-s2-t03", command)
        self.assertIn("bash scripts/sh_train_protein_qwen_staged.sh", command)


if __name__ == "__main__":
    unittest.main()

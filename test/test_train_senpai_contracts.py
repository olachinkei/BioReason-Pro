import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "train.py"


def load_train_module():
    module_name = "train_senpai_contracts_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TRAIN = load_train_module()


def parse_result_marker(output: str) -> dict:
    for line in output.splitlines():
        if line.startswith("SENPAI-RESULT: "):
            return json.loads(line.split(": ", 1)[1])
    raise AssertionError(f"SENPAI-RESULT marker not found in output: {output}")


class TrainSenpaiContractsTest(unittest.TestCase):
    def assets(self, tmpdir: str) -> dict:
        root = Path(tmpdir)
        return {
            "base_checkpoint": "demo/entity/bioreason-pro-rl-paper:production",
            "model_dir": str(root / "baseline-model"),
            "dataset_dir": str(root / "dataset"),
            "dataset_name": "disease_temporal_hc_reasoning_v2",
            "dataset_artifact": "demo/entity/dataset:production",
            "temporal_split_artifact": "demo/entity/split:production",
            "ia_file": str(root / "IA.txt"),
        }

    def run_senpai(self, argv, *, eval_payloads, train_payloads=None):
        train_payloads = list(train_payloads or [])
        eval_payloads = list(eval_payloads)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "run"
            fake_assets = self.assets(tmpdir)

            def fake_eval(**kwargs):
                payload = eval_payloads.pop(0)
                metrics_path = Path(kwargs["output_dir"]) / "cafa_metrics" / "metrics_summary.json"
                if payload == "raise_eval_error":
                    raise TRAIN.SenpaiEvalPhaseError("validation failed before metrics", metrics_path)
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_path.write_text(json.dumps(payload), encoding="utf-8")
                return payload, metrics_path

            def fake_train(**kwargs):
                payload = train_payloads.pop(0) if train_payloads else {}
                checkpoint = Path(kwargs["output_dir"]) / "checkpoints" / f"step-{int(kwargs['save_every_n_steps']):06d}"
                (checkpoint / "inference_export").mkdir(parents=True, exist_ok=True)
                return checkpoint, payload.get("run_id", f"{kwargs['phase']}-run")

            stdout = io.StringIO()
            with (
                mock.patch.object(TRAIN, "resolve_senpai_assets", return_value=fake_assets),
                mock.patch.object(TRAIN, "run_eval_phase", side_effect=fake_eval) as eval_mock,
                mock.patch.object(TRAIN, "run_train_phase", side_effect=fake_train) as train_mock,
                redirect_stdout(stdout),
            ):
                TRAIN.senpai_main(["--output_dir", str(out), *argv])

            return {
                "stdout": stdout.getvalue(),
                "result": parse_result_marker(stdout.getvalue()),
                "summary": json.loads((out / "senpai_summary.json").read_text(encoding="utf-8")),
                "eval_calls": eval_mock.call_count,
                "train_calls": train_mock.call_count,
            }

    def test_parse_senpai_args_exposes_expected_defaults(self):
        args = TRAIN.parse_senpai_args([])
        self.assertEqual(args.gate_steps, 5)
        self.assertEqual(args.continue_steps, 15)
        self.assertEqual(args.primary_metric, "overall_mean_fmax")
        self.assertEqual(args.max_val_samples, 100)
        self.assertEqual(args.eval_max_new_tokens, 8192)
        self.assertEqual(args.nnodes, 1)
        self.assertEqual(args.gpus_per_node, 8)

    def test_backend_train_command_defaults_are_paper_rollout_shape_with_8k_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = TRAIN.parse_senpai_args(["--output_dir", tmpdir, "--wandb_name", "unit/seed"])
            command = TRAIN.build_backend_train_command(
                args=args,
                assets=self.assets(tmpdir),
                phase="gate",
                max_steps=5,
                save_every_n_steps=5,
                output_dir=Path(tmpdir) / "train-gate",
            )

        def value_after(flag: str) -> str:
            return command[command.index(flag) + 1]

        self.assertEqual(value_after("--queries_per_step"), "8")
        self.assertEqual(value_after("--rollouts_per_query"), "24")
        self.assertEqual(value_after("--optimizer_micro_batch_size_per_gpu"), "6")
        self.assertEqual(value_after("--gradient_accumulation_steps"), "4")
        self.assertEqual(value_after("--max_new_tokens"), "8192")
        self.assertEqual(value_after("--vllm_max_model_len"), "12288")
        self.assertEqual(value_after("--vllm_max_num_seqs"), "8")

    def test_vllm_max_num_seqs_is_not_upscaled_to_rollout_count(self):
        args = TRAIN.parse_args(
            [
                "--backend_train",
                "--text_model_name",
                "/tmp/model",
                "--target_num_nodes",
                "1",
                "--target_gpus_per_node",
                "8",
                "--queries_per_step",
                "8",
                "--rollouts_per_query",
                "24",
                "--vllm_max_num_seqs",
                "8",
            ]
        )

        self.assertEqual(TRAIN.resolve_effective_vllm_max_num_seqs(args), 8)
        self.assertEqual(TRAIN.resolve_rollout_generation_batch_size(args, 24), 8)

    def test_baseline_mode_emits_baseline_result(self):
        result = self.run_senpai(
            ["--mode", "baseline"],
            eval_payloads=[{"overall_mean_fmax": 0.42}],
        )
        self.assertEqual(result["eval_calls"], 1)
        self.assertEqual(result["train_calls"], 0)
        self.assertEqual(result["result"]["primary_metric"]["value"], 0.42)

    def test_gate_failure_stops_without_continuation(self):
        result = self.run_senpai(
            [],
            eval_payloads=[
                {"overall_mean_fmax": 0.70},
                {"overall_mean_fmax": 0.69},
            ],
            train_payloads=[{"run_id": "gate123"}],
        )
        self.assertEqual(result["train_calls"], 1)
        self.assertEqual(result["eval_calls"], 2)
        self.assertFalse(result["summary"]["improved_after_gate"])
        self.assertEqual(result["result"]["wandb_run_ids"], ["gate123"])
        self.assertEqual(result["result"]["primary_metric"]["value"], 0.69)

    def test_gate_success_continues_to_20_steps_and_reports_best(self):
        result = self.run_senpai(
            [],
            eval_payloads=[
                {"overall_mean_fmax": 0.50},
                {"overall_mean_fmax": 0.60},
                {"overall_mean_fmax": 0.65},
            ],
            train_payloads=[{"run_id": "gate123"}, {"run_id": "continue456"}],
        )
        self.assertEqual(result["train_calls"], 2)
        self.assertEqual(result["eval_calls"], 3)
        self.assertTrue(result["summary"]["improved_after_gate"])
        self.assertEqual(result["summary"]["best_step"], 20)
        self.assertEqual(result["result"]["wandb_run_ids"], ["gate123", "continue456"])
        self.assertEqual(result["result"]["primary_metric"]["value"], 0.65)

    def test_missing_gate_metric_is_terminal_negative_result(self):
        result = self.run_senpai(
            [],
            eval_payloads=[
                {"overall_mean_fmax": 0.50},
                {},
            ],
            train_payloads=[{"run_id": "gate123"}],
        )
        self.assertEqual(result["train_calls"], 1)
        self.assertFalse(result["summary"]["improved_after_gate"])
        self.assertIsNone(result["summary"]["gate_value"])
        self.assertEqual(result["result"]["primary_metric"]["value"], 0.0)

    def test_gate_eval_failure_without_metrics_is_terminal_negative_result(self):
        result = self.run_senpai(
            [],
            eval_payloads=[
                {"overall_mean_fmax": 0.50},
                "raise_eval_error",
            ],
            train_payloads=[{"run_id": "gate123"}],
        )
        self.assertEqual(result["train_calls"], 1)
        self.assertFalse(result["summary"]["improved_after_gate"])
        self.assertIsNone(result["summary"]["gate_value"])
        self.assertIn("gate_eval_error", result["summary"])
        self.assertEqual(result["result"]["primary_metric"]["value"], 0.0)

    def test_checkpoint_export_mode_is_available_to_backend(self):
        args = TRAIN.parse_args(
            [
                "--backend_train",
                "--text_model_name",
                "/tmp/model",
                "--checkpoint_export_mode",
                "inference_export",
            ]
        )
        self.assertEqual(args.checkpoint_export_mode, "inference_export")


if __name__ == "__main__":
    unittest.main()

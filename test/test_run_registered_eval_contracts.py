import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_registered_eval.py"
REGISTRY_PATH = ROOT / "bioreason2" / "utils" / "research_registry.py"
EVAL_TARGET_REGISTRY_JSON = ROOT / "configs" / "disease_benchmark" / "eval_target_registry.json"


def load_research_registry_module():
    module_name = "run_registered_eval_research_registry_test_module"
    spec = importlib.util.spec_from_file_location(module_name, REGISTRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_registered_eval_module():
    module_name = "run_registered_eval_contracts_test_module"
    registry_module = load_research_registry_module()

    bioreason2_module = sys.modules.get("bioreason2", types.ModuleType("bioreason2"))
    utils_module = types.ModuleType("bioreason2.utils")
    utils_module.research_registry = registry_module

    previous_bioreason2 = sys.modules.get("bioreason2")
    previous_utils = sys.modules.get("bioreason2.utils")
    previous_registry = sys.modules.get("bioreason2.utils.research_registry")

    sys.modules["bioreason2"] = bioreason2_module
    sys.modules["bioreason2.utils"] = utils_module
    sys.modules["bioreason2.utils.research_registry"] = registry_module

    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_bioreason2 is None:
            sys.modules.pop("bioreason2", None)
        else:
            sys.modules["bioreason2"] = previous_bioreason2

        if previous_utils is None:
            sys.modules.pop("bioreason2.utils", None)
        else:
            sys.modules["bioreason2.utils"] = previous_utils

        if previous_registry is None:
            sys.modules.pop("bioreason2.utils.research_registry", None)
        else:
            sys.modules["bioreason2.utils.research_registry"] = previous_registry
    return module


REGISTERED_EVAL = load_registered_eval_module()


class RunRegisteredEvalContractsTest(unittest.TestCase):
    def test_protein_llm_targets_require_bundled_protein_assets(self):
        registry = json.loads(EVAL_TARGET_REGISTRY_JSON.read_text(encoding="utf-8"))
        required_paths = {
            "config.json",
            "go_embedding.pt",
            "go_projection.pt",
            "protein_projection.pt",
            "protein_model/pytorch_model.bin",
        }

        for target_name in ("bioreason-pro-rl-paper", "train-sft-output", "train-rl-output"):
            target = registry["targets"][target_name]
            source = target["model_sources"][0]
            self.assertEqual(set(source["required_paths"]), required_paths)

    def test_with_bundle_context_formats_target_sources(self):
        bundle = {
            "benchmark_alias": "213.221.225.228",
            "benchmark_alias_dir": "213_221_225_228",
            "benchmark_version": "213 -> 221 -> 225 -> 228",
        }
        target = {
            "target_name": "blast-diamond-baseline",
            "prediction_sources": [
                {
                    "wandb_registry_path": "demo/project/disease-temporal-blast-diamond-predictions:{benchmark_alias}",
                    "local_dir": "data/artifacts/predictions/blast_diamond/{benchmark_alias_dir}",
                }
            ],
        }

        rendered = REGISTERED_EVAL.with_bundle_context(target, bundle)

        self.assertEqual(
            rendered["prediction_sources"][0]["wandb_registry_path"],
            "demo/project/disease-temporal-blast-diamond-predictions:213.221.225.228",
        )
        self.assertEqual(
            rendered["prediction_sources"][0]["local_dir"],
            "data/artifacts/predictions/blast_diamond/213_221_225_228",
        )

    def test_run_protein_llm_target_passes_registry_driven_env(self):
        bundle = {
            "benchmark_version": "213 -> 221 -> 225 -> 228",
            "benchmark_alias": "213.221.225.228",
            "shortlist_mode": "high-confidence",
            "shortlist_query": "demo query",
            "train_start_release": 213,
            "train_end_release": 221,
            "dev_end_release": 225,
            "test_end_release": 228,
            "temporal_split_artifact": {"wandb_registry_path": "demo/project/disease-temporal-split:production"},
            "reasoning_dataset": {
                "wandb_registry_path": "demo/project/disease-temporal-reasoning:production",
                "dataset_source": "wanglab/cafa5",
                "dataset_name": "disease_temporal_hc_reasoning_v1",
            },
        }
        target = {
            "target_name": "bioreason-pro-base",
            "display_name": "bioreason-pro-base",
            "runner": "protein_llm",
            "model_sources": [{"type": "wandb_artifact", "wandb_registry_path": "demo/project/base:production"}],
        }
        args = types.SimpleNamespace(
            split="validation",
            wandb_project="demo-project",
            wandb_entity="demo-entity",
            wandb_mode="offline",
            weave_project="demo-entity/demo-project",
            metric_threads=4,
            metric_threshold_step=0.95,
            max_samples=25,
            sample_strategy="stratified_aspect_profile",
            num_chunks=1,
            chunk_id=0,
            keep_local_eval_outputs=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "bioreason-base"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            runtime_paths = {
                "output_root": tempfile.mkdtemp(),
                "go_obo_path": "/tmp/go-basic.obo",
                "ia_file_path": "/tmp/IA.txt",
                "go_embeddings_path": "/tmp/go-embeddings",
                "dataset_cache_dir": "/tmp/hf-cache",
                "structure_dir": "/tmp/structures",
            }

            captured = {}

            def fake_run_shell_command(command, env):
                captured["command"] = list(command)
                captured["env"] = dict(env)

            with mock.patch.object(
                REGISTERED_EVAL,
                "materialize_first_available_source",
                return_value={
                    "local_path": str(model_dir),
                    "source_ref": "demo/project/bioreason-pro-base:production",
                },
            ), mock.patch.object(REGISTERED_EVAL, "run_shell_command", side_effect=fake_run_shell_command), mock.patch.object(
                REGISTERED_EVAL.shutil,
                "which",
                return_value=None,
            ):
                status = REGISTERED_EVAL.run_protein_llm_target(args, bundle, target, runtime_paths)

        self.assertEqual(status["status"], "completed")
        self.assertEqual(captured["command"], ["bash", "scripts/sh_eval.sh"])
        self.assertEqual(captured["env"]["MODEL_PATH"], str(model_dir))
        self.assertEqual(captured["env"]["CAFA5_DATASET"], "wanglab/cafa5")
        self.assertEqual(captured["env"]["DATASET_NAME"], "disease_temporal_hc_reasoning_v1")
        self.assertEqual(captured["env"]["TEMPORAL_SPLIT_ARTIFACT"], "demo/project/disease-temporal-split:production")
        self.assertEqual(captured["env"]["DATASET_ARTIFACT"], "demo/project/disease-temporal-reasoning:production")
        self.assertEqual(captured["env"]["MODEL_ARTIFACT"], "demo/project/bioreason-pro-base:production")
        self.assertEqual(captured["env"]["EVAL_SPLIT"], "validation")
        self.assertEqual(captured["env"]["WANDB_RUN_NAME"], "eval-bioreason-pro-base-validation-213.221.225.228")
        self.assertEqual(captured["env"]["KEEP_LOCAL_EVAL_OUTPUTS"], "0")

    def test_run_protein_llm_target_wraps_eval_in_srun_when_requested(self):
        bundle = {
            "benchmark_version": "213 -> 221 -> 225 -> 228",
            "benchmark_alias": "213.221.225.228",
            "shortlist_mode": "high-confidence",
            "shortlist_query": "demo query",
            "train_start_release": 213,
            "train_end_release": 221,
            "dev_end_release": 225,
            "test_end_release": 228,
            "temporal_split_artifact": {"wandb_registry_path": "demo/project/disease-temporal-split:production"},
            "reasoning_dataset": {
                "wandb_registry_path": "demo/project/disease-temporal-reasoning:production",
                "dataset_source": "wanglab/cafa5",
                "dataset_name": "disease_temporal_hc_reasoning_v1",
            },
        }
        target = {
            "target_name": "train-sft-output",
            "display_name": "train-sft-output",
            "runner": "protein_llm",
            "model_sources": [{"type": "wandb_artifact", "wandb_registry_path": "demo/project/train-sft-output:latest"}],
        }
        args = types.SimpleNamespace(
            split="test",
            wandb_project="demo-project",
            wandb_entity="demo-entity",
            wandb_mode="online",
            weave_project="demo-entity/demo-project",
            metric_threads=4,
            metric_threshold_step=0.95,
            max_samples=8,
            sample_strategy="stratified_aspect_profile",
            num_chunks=1,
            chunk_id=0,
            keep_local_eval_outputs=False,
            use_srun=True,
            srun_partition="h100",
            srun_gpus=1,
            srun_cpus_per_task=8,
            srun_mem="128G",
            srun_time_limit="02:00:00",
            srun_account="",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "train-sft-output"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            runtime_paths = {
                "output_root": tempfile.mkdtemp(),
                "go_obo_path": "/tmp/go-basic.obo",
                "ia_file_path": "/tmp/IA.txt",
                "go_embeddings_path": "/tmp/go-embeddings",
                "dataset_cache_dir": "/tmp/hf-cache",
                "structure_dir": "/tmp/structures",
            }

            captured = {}

            def fake_run_shell_command(command, env):
                captured["command"] = list(command)
                captured["env"] = dict(env)

            with mock.patch.object(
                REGISTERED_EVAL,
                "materialize_first_available_source",
                return_value={
                    "local_path": str(model_dir),
                    "source_ref": "demo/project/train-sft-output:latest",
                },
            ), mock.patch.object(REGISTERED_EVAL, "run_shell_command", side_effect=fake_run_shell_command), mock.patch.object(
                REGISTERED_EVAL.shutil,
                "which",
                return_value="/usr/bin/srun",
            ):
                REGISTERED_EVAL.run_protein_llm_target(args, bundle, target, runtime_paths)

        self.assertEqual(
            captured["command"][:14],
            [
                "srun",
                "--job-name",
                "eval-train-sft-output-test",
                "--partition",
                "h100",
                "--nodes",
                "1",
                "--ntasks-per-node",
                "1",
                "--gpus",
                "1",
                "--cpus-per-task",
                "8",
                "--mem",
            ],
        )
        self.assertEqual(captured["command"][-2:], ["bash", "scripts/sh_eval.sh"])
        self.assertEqual(captured["env"]["MODEL_ARTIFACT"], "demo/project/train-sft-output:latest")
        self.assertEqual(captured["env"]["WANDB_RUN_NAME"], "eval-train-sft-output-test-213.221.225.228")

    def test_run_protein_llm_target_auto_wraps_eval_in_srun_on_login_node(self):
        bundle = {
            "benchmark_version": "213 -> 221 -> 225 -> 228",
            "benchmark_alias": "213.221.225.228",
            "shortlist_mode": "high-confidence",
            "shortlist_query": "demo query",
            "train_start_release": 213,
            "train_end_release": 221,
            "dev_end_release": 225,
            "test_end_release": 228,
            "temporal_split_artifact": {"wandb_registry_path": "demo/project/disease-temporal-split:production"},
            "reasoning_dataset": {
                "wandb_registry_path": "demo/project/disease-temporal-reasoning:production",
                "dataset_source": "wanglab/cafa5",
                "dataset_name": "disease_temporal_hc_reasoning_v1",
            },
        }
        target = {
            "target_name": "train-sft-output",
            "display_name": "train-sft-output",
            "runner": "protein_llm",
            "model_sources": [{"type": "wandb_artifact", "wandb_registry_path": "demo/project/train-sft-output:latest"}],
        }
        args = types.SimpleNamespace(
            split="test",
            wandb_project="demo-project",
            wandb_entity="demo-entity",
            wandb_mode="online",
            weave_project="demo-entity/demo-project",
            metric_threads=4,
            metric_threshold_step=0.95,
            max_samples=8,
            sample_strategy="stratified_aspect_profile",
            num_chunks=1,
            chunk_id=0,
            keep_local_eval_outputs=False,
            use_srun=False,
            srun_partition="h100",
            srun_gpus=1,
            srun_cpus_per_task=8,
            srun_mem="128G",
            srun_time_limit="02:00:00",
            srun_account="",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "train-sft-output"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            runtime_paths = {
                "output_root": tempfile.mkdtemp(),
                "go_obo_path": "/tmp/go-basic.obo",
                "ia_file_path": "/tmp/IA.txt",
                "go_embeddings_path": "/tmp/go-embeddings",
                "dataset_cache_dir": "/tmp/hf-cache",
                "structure_dir": "/tmp/structures",
            }

            captured = {}

            def fake_run_shell_command(command, env):
                captured["command"] = list(command)
                captured["env"] = dict(env)

            with mock.patch.object(
                REGISTERED_EVAL,
                "materialize_first_available_source",
                return_value={
                    "local_path": str(model_dir),
                    "source_ref": "demo/project/train-sft-output:latest",
                },
            ), mock.patch.object(REGISTERED_EVAL, "run_shell_command", side_effect=fake_run_shell_command), mock.patch.object(
                REGISTERED_EVAL.shutil,
                "which",
                return_value="/usr/bin/srun",
            ), mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SLURM_JOB_ID", None)
                REGISTERED_EVAL.run_protein_llm_target(args, bundle, target, runtime_paths)

        self.assertEqual(captured["command"][:2], ["srun", "--job-name"])
        self.assertEqual(captured["command"][-2:], ["bash", "scripts/sh_eval.sh"])

    def test_build_run_names_uses_specific_artifact_name_for_tuned_targets(self):
        names = REGISTERED_EVAL.build_run_names(
            "train-sft-output",
            "test",
            "213.221.225.228",
            model_source_ref="wandb-healthcare/bioreason-pro-custom/train-sft-output-verify40:v0",
        )

        self.assertEqual(names["run_name"], "eval-train-sft-output-verify40-test-213.221.225.228")
        self.assertEqual(names["artifact_name"], "eval-train-sft-output-verify40-test-213.221.225.228")
        self.assertEqual(names["weave_eval_name"], "eval-train-sft-output-verify40-test-213.221.225.228")

    def test_run_protein_llm_target_cleans_stale_scratch_before_execution(self):
        bundle = {
            "benchmark_version": "213 -> 221 -> 225 -> 228",
            "benchmark_alias": "213.221.225.228",
            "shortlist_mode": "high-confidence",
            "shortlist_query": "demo query",
            "train_start_release": 213,
            "train_end_release": 221,
            "dev_end_release": 225,
            "test_end_release": 228,
            "temporal_split_artifact": {"wandb_registry_path": "demo/project/disease-temporal-split:production"},
            "reasoning_dataset": {
                "wandb_registry_path": "demo/project/disease-temporal-reasoning:production",
                "dataset_source": "wanglab/cafa5",
                "dataset_name": "disease_temporal_hc_reasoning_v1",
            },
        }
        target = {
            "target_name": "bioreason-pro-base",
            "display_name": "bioreason-pro-base",
            "runner": "protein_llm",
            "model_sources": [{"type": "wandb_artifact", "wandb_registry_path": "demo/project/base:production"}],
        }
        args = types.SimpleNamespace(
            split="test",
            wandb_project="demo-project",
            wandb_entity="demo-entity",
            wandb_mode="online",
            weave_project="demo-entity/demo-project",
            metric_threads=4,
            metric_threshold_step=0.95,
            max_samples=3,
            sample_strategy="shuffled_prefix",
            num_chunks=1,
            chunk_id=0,
            keep_local_eval_outputs=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            stale_file = output_root / "bioreason-pro-base" / "test" / "results" / "stale.json"
            stale_file.parent.mkdir(parents=True, exist_ok=True)
            stale_file.write_text("{}", encoding="utf-8")
            runtime_paths = {
                "output_root": str(output_root),
                "go_obo_path": "/tmp/go-basic.obo",
                "ia_file_path": "/tmp/IA.txt",
                "go_embeddings_path": "/tmp/go-embeddings",
                "dataset_cache_dir": "/tmp/hf-cache",
                "structure_dir": "/tmp/structures",
            }

            model_dir = output_root / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")

            with mock.patch.object(
                REGISTERED_EVAL,
                "materialize_first_available_source",
                return_value={
                    "local_path": str(model_dir),
                    "source_ref": "demo/project/bioreason-pro-base:production",
                },
            ), mock.patch.object(REGISTERED_EVAL, "run_shell_command", return_value=None):
                REGISTERED_EVAL.run_protein_llm_target(args, bundle, target, runtime_paths)

            self.assertFalse(stale_file.exists())

    def test_run_protein_llm_target_converts_raw_sft_checkpoint_before_eval(self):
        bundle = {
            "benchmark_version": "213 -> 221 -> 225 -> 228",
            "benchmark_alias": "213.221.225.228",
            "shortlist_mode": "high-confidence",
            "shortlist_query": "demo query",
            "train_start_release": 213,
            "train_end_release": 221,
            "dev_end_release": 225,
            "test_end_release": 228,
            "temporal_split_artifact": {"wandb_registry_path": "demo/project/disease-temporal-split:production"},
            "reasoning_dataset": {
                "wandb_registry_path": "demo/project/disease-temporal-reasoning:production",
                "dataset_source": "wanglab/cafa5",
                "dataset_name": "disease_temporal_hc_reasoning_v1",
            },
        }
        target = {
            "target_name": "train-sft-output",
            "display_name": "train-sft-output",
            "runner": "protein_llm",
            "model_sources": [
                {"type": "wandb_artifact", "wandb_registry_path": "demo/project/train-sft-output:latest"}
            ],
        }
        args = types.SimpleNamespace(
            split="test",
            wandb_project="demo-project",
            wandb_entity="demo-entity",
            wandb_mode="online",
            weave_project="demo-entity/demo-project",
            metric_threads=4,
            metric_threshold_step=0.95,
            max_samples=3,
            sample_strategy="shuffled_prefix",
            num_chunks=1,
            chunk_id=0,
            keep_local_eval_outputs=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_model_dir = tmp_path / "train_sft_output"
            raw_model_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = raw_model_dir / "demo-best.ckpt"
            ckpt_path.write_text("checkpoint", encoding="utf-8")

            paper_model_dir = tmp_path / "paper_model"
            paper_model_dir.mkdir(parents=True, exist_ok=True)
            (paper_model_dir / "config.json").write_text("{}", encoding="utf-8")
            (paper_model_dir / "go_embedding.pt").write_text("stub", encoding="utf-8")

            runtime_paths = {
                "output_root": str(tmp_path / "eval"),
                "go_obo_path": "/tmp/go-basic.obo",
                "ia_file_path": "/tmp/IA.txt",
                "go_embeddings_path": "/tmp/go-embeddings",
                "dataset_cache_dir": "/tmp/hf-cache",
                "structure_dir": "/tmp/structures",
            }

            commands = []

            def fake_materialize(sources, allow_missing=False):
                source = sources[0]
                if source.get("wandb_registry_path") == "demo/project/train-sft-output:latest":
                    return {
                        "local_path": str(raw_model_dir),
                        "source_ref": "demo/project/train-sft-output:latest",
                    }
                if source.get("wandb_registry_path") == "demo/project/bioreason-pro-rl:production":
                    return {
                        "local_path": str(paper_model_dir),
                        "source_ref": "demo/project/bioreason-pro-rl:production",
                    }
                raise AssertionError(f"Unexpected source: {source}")

            def fake_run_shell(command, env):
                commands.append(list(command))
                if command[1].endswith("save_unsloth_ckpt.py"):
                    save_dir = Path(command[command.index("--save_dir") + 1])
                    save_dir.mkdir(parents=True, exist_ok=True)
                    (save_dir / "config.json").write_text("{}", encoding="utf-8")

            with mock.patch.dict(os.environ, {"BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH": "demo/project/bioreason-pro-rl:production"}, clear=False), mock.patch.object(
                REGISTERED_EVAL,
                "materialize_first_available_source",
                side_effect=fake_materialize,
            ), mock.patch.object(REGISTERED_EVAL, "run_shell_command", side_effect=fake_run_shell):
                status = REGISTERED_EVAL.run_protein_llm_target(args, bundle, target, runtime_paths)

            self.assertEqual(status["status"], "completed")
            self.assertEqual(commands[0][0], sys.executable)
            self.assertTrue(commands[0][1].endswith("save_unsloth_ckpt.py"))
            self.assertIn(str(ckpt_path), commands[0])
            self.assertEqual(commands[1], ["bash", "scripts/sh_eval.sh"])

    def test_resolve_effective_max_samples_defaults_validation_to_100(self):
        args = types.SimpleNamespace(split="validation", max_samples=None)
        self.assertEqual(REGISTERED_EVAL.resolve_effective_max_samples(args), 100)

    def test_resolve_effective_max_samples_defaults_test_to_full_split(self):
        args = types.SimpleNamespace(split="test", max_samples=None)
        self.assertEqual(REGISTERED_EVAL.resolve_effective_max_samples(args), -1)

    def test_run_prediction_artifact_target_writes_metrics_and_samples(self):
        bundle = {
            "benchmark_version": "213 -> 221 -> 225 -> 228",
            "benchmark_alias": "213.221.225.228",
            "reasoning_dataset": {
                "wandb_registry_path": "demo/project/disease-temporal-reasoning:production",
                "dataset_source": "wanglab/cafa5",
                "dataset_name": "disease_temporal_hc_reasoning_v1",
            },
            "temporal_split_artifact": {"wandb_registry_path": "demo/project/disease-temporal-split:production"},
        }
        target = {
            "target_name": "blast-diamond-baseline",
            "display_name": "blast-diamond-baseline",
            "runner": "prediction_artifact",
            "prediction_glob": "*.tsv",
            "prediction_sources": [{"type": "wandb_artifact", "wandb_registry_path": "demo/project/preds:latest"}],
        }
        args = types.SimpleNamespace(
            split="test",
            metric_threads=2,
            metric_threshold_step=0.95,
            wandb_project=None,
            wandb_entity=None,
            wandb_mode=None,
            keep_local_eval_outputs=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "eval"
            prediction_dir = Path(tmpdir) / "predictions"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            (prediction_dir / "predictions.tsv").write_text(
                "P1\tGO:0001111\t1.0\nP1\tGO:0002222\t0.8\n",
                encoding="utf-8",
            )

            runtime_paths = {
                "output_root": str(output_root),
                "go_obo_path": "/tmp/go-basic.obo",
                "ia_file_path": "/tmp/IA.txt",
                "dataset_cache_dir": "/tmp/hf-cache",
                "go_embeddings_path": "/tmp/go-embeddings",
                "structure_dir": "/tmp/structures",
            }

            fake_evals_module = types.ModuleType("evals")
            fake_cafa_module = types.SimpleNamespace(
                run_cafa_evaluation=lambda *args, **kwargs: ("evaluation_df", {"f": object()}),
                extract_metrics_summary=lambda result: {
                    "molecular_function_f1": 0.5,
                    "biological_process_f1": 0.6,
                    "cellular_component_f1": 0.7,
                    "overall_mean_f1": 0.6,
                },
                normalize_metrics_for_logging=lambda metrics: {
                    **metrics,
                    "fmax_mf": metrics["molecular_function_f1"],
                    "fmax_bp": metrics["biological_process_f1"],
                    "fmax_cc": metrics["cellular_component_f1"],
                    "overall_mean_fmax": metrics["overall_mean_f1"],
                },
                write_metrics_summary=lambda metrics, output_dir: str(Path(output_dir) / "metrics_summary.json"),
            )
            fake_evals_module.cafa_evals = fake_cafa_module

            with mock.patch.object(
                REGISTERED_EVAL,
                "materialize_first_available_source",
                return_value={"local_path": str(prediction_dir), "source_ref": "demo/project/preds:latest"},
            ), mock.patch.object(
                REGISTERED_EVAL,
                "load_ground_truth_split",
                return_value=[{"protein_id": "P1", "ground_truth_terms": {"GO:0001111"}}],
            ), mock.patch.object(
                REGISTERED_EVAL,
                "maybe_log_prediction_eval_to_wandb",
                return_value=None,
            ), mock.patch.dict(
                sys.modules,
                {"evals": fake_evals_module},
            ):
                status = REGISTERED_EVAL.run_prediction_artifact_target(args, bundle, target, runtime_paths)

            self.assertEqual(status["status"], "completed")
            results_dir = output_root / "blast-diamond-baseline" / "test" / "results"
            self.assertTrue((results_dir / "ground_truth.tsv").exists())
            self.assertTrue((results_dir / "sample_results.tsv").exists())
            self.assertTrue((results_dir / "run_summary.json").exists())
            summary = json.loads((results_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["prediction_source"], "demo/project/preds:latest")

    def test_run_prediction_artifact_target_cleans_local_scratch_after_wandb_logging(self):
        bundle = {
            "benchmark_version": "213 -> 221 -> 225 -> 228",
            "benchmark_alias": "213.221.225.228",
            "reasoning_dataset": {
                "wandb_registry_path": "demo/project/disease-temporal-reasoning:production",
                "dataset_source": "wanglab/cafa5",
                "dataset_name": "disease_temporal_hc_reasoning_v1",
            },
            "temporal_split_artifact": {"wandb_registry_path": "demo/project/disease-temporal-split:production"},
        }
        target = {
            "target_name": "blast-diamond-baseline",
            "display_name": "blast-diamond-baseline",
            "runner": "prediction_artifact",
            "prediction_glob": "*.tsv",
            "prediction_sources": [{"type": "wandb_artifact", "wandb_registry_path": "demo/project/preds:latest"}],
        }
        args = types.SimpleNamespace(
            split="test",
            metric_threads=2,
            metric_threshold_step=0.95,
            wandb_project="demo-project",
            wandb_entity="demo-entity",
            wandb_mode="online",
            keep_local_eval_outputs=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "eval"
            prediction_dir = Path(tmpdir) / "predictions"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            (prediction_dir / "predictions.tsv").write_text("P1\tGO:0001111\t1.0\n", encoding="utf-8")

            runtime_paths = {
                "output_root": str(output_root),
                "go_obo_path": "/tmp/go-basic.obo",
                "ia_file_path": "/tmp/IA.txt",
                "dataset_cache_dir": "/tmp/hf-cache",
                "go_embeddings_path": "/tmp/go-embeddings",
                "structure_dir": "/tmp/structures",
            }

            fake_evals_module = types.ModuleType("evals")
            fake_cafa_module = types.SimpleNamespace(
                run_cafa_evaluation=lambda *args, **kwargs: ("evaluation_df", {"f": object()}),
                extract_metrics_summary=lambda result: {
                    "molecular_function_f1": 0.5,
                    "biological_process_f1": 0.6,
                    "cellular_component_f1": 0.7,
                    "overall_mean_f1": 0.6,
                },
                normalize_metrics_for_logging=lambda metrics: {
                    **metrics,
                    "fmax_mf": metrics["molecular_function_f1"],
                    "fmax_bp": metrics["biological_process_f1"],
                    "fmax_cc": metrics["cellular_component_f1"],
                    "overall_mean_fmax": metrics["overall_mean_f1"],
                },
                write_metrics_summary=lambda metrics, output_dir: str(Path(output_dir) / "metrics_summary.json"),
            )
            fake_evals_module.cafa_evals = fake_cafa_module

            with mock.patch.object(
                REGISTERED_EVAL,
                "materialize_first_available_source",
                return_value={"local_path": str(prediction_dir), "source_ref": "demo/project/preds:latest"},
            ), mock.patch.object(
                REGISTERED_EVAL,
                "load_ground_truth_split",
                return_value=[{"protein_id": "P1", "ground_truth_terms": {"GO:0001111"}}],
            ), mock.patch.object(
                REGISTERED_EVAL,
                "maybe_log_prediction_eval_to_wandb",
                return_value="eval-artifact:v0",
            ), mock.patch.dict(
                sys.modules,
                {"evals": fake_evals_module},
            ):
                status = REGISTERED_EVAL.run_prediction_artifact_target(args, bundle, target, runtime_paths)

            self.assertEqual(status["status"], "completed")
            self.assertFalse((output_root / "blast-diamond-baseline" / "test").exists())


if __name__ == "__main__":
    unittest.main()

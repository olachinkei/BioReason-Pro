import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACKING_PATH = ROOT / "bioreason2" / "utils" / "tracking.py"


def load_tracking_module():
    module_name = "training_tracking_contracts_test_module"
    spec = importlib.util.spec_from_file_location(module_name, TRACKING_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TRACKING = load_tracking_module()


class FakeConfig(dict):
    def update(self, values, allow_val_change=False):
        self["allow_val_change"] = allow_val_change
        super().update(values)


class FakeRun:
    def __init__(self):
        self.config = FakeConfig()
        self.artifacts = []
        self.used_artifacts = []

    def log_artifact(self, artifact, aliases=None):
        artifact.logged_aliases = aliases
        self.artifacts.append(artifact)

    def use_artifact(self, artifact_ref, type=None):
        self.used_artifacts.append((artifact_ref, type))


class FakeArtifact:
    def __init__(self, name, type, metadata=None):
        self.name = name
        self.type = type
        self.metadata = metadata or {}
        self.added_dirs = []
        self.logged_aliases = None

    def add_dir(self, path):
        self.added_dirs.append(path)


class TrainingTrackingContractsTest(unittest.TestCase):
    def test_build_training_tracking_config_matches_spec_shape(self):
        args = types.SimpleNamespace(
            wandb_job_type="train_sft",
            benchmark_version="213 -> 221 -> 225 -> 228",
            temporal_split_artifact="data/artifacts/benchmarks/213_221_225_228/temporal_split",
            dataset_config=None,
            reasoning_dataset_config=None,
            dataset_artifact="disease_temporal_hc_reasoning_v1:latest",
            shortlist_query="reviewed:true AND organism_id:9606",
            shortlist_mode="high-confidence",
            train_start_release=213,
            train_end_release=221,
            dev_end_release=225,
            test_end_release=228,
            base_checkpoint="bowang-lab/bioreason-pro-base",
            model_artifact=None,
            checkpoint_artifact_name="disease-sft-checkpoints",
            checkpoint_dir="checkpoints/disease-sft",
            output_dir=None,
            seed=23,
            learning_rate=1e-4,
            batch_size=4,
            gradient_accumulation_steps=2,
            max_epochs=10,
            validation_subset_size=100,
            validation_subset_strategy="stratified_aspect_profile",
            max_eval_samples=None,
            eval_sample_strategy=None,
            job_time_limit="12:00:00",
            training_stage=2,
            cafa5_dataset_name="disease_temporal_hc_reasoning_v1",
            reasoning_dataset_name="disease_temporal_hc_reasoning_v1",
            ckpt_path=None,
            projector_checkpoint_path=None,
        )

        config = TRACKING.build_training_tracking_config(args, run_name="demo-run")

        self.assertEqual(config["job_type"], "train_sft")
        self.assertEqual(config["benchmark_version"], "213 -> 221 -> 225 -> 228")
        self.assertEqual(
            config["temporal_split_artifact"],
            "data/artifacts/benchmarks/213_221_225_228/temporal_split",
        )
        self.assertEqual(config["dataset_config"], "disease_temporal_hc_reasoning_v1")
        self.assertEqual(config["reasoning_dataset_config"], "disease_temporal_hc_reasoning_v1")
        self.assertEqual(config["dataset_artifact"], "disease_temporal_hc_reasoning_v1:latest")
        self.assertEqual(config["model_artifact"], "disease-sft-checkpoints")
        self.assertEqual(config["job_time_limit"], "12:00:00")
        self.assertEqual(config["num_train_epochs"], 10)
        self.assertEqual(config["validation_subset_size"], 100)
        self.assertEqual(config["validation_subset_strategy"], "stratified_aspect_profile")
        self.assertEqual(config["weave_project"], "")
        self.assertIsNone(config["weave_trace_budget"])

    def test_build_training_tracking_config_uses_output_dir_when_checkpoint_dir_missing(self):
        args = types.SimpleNamespace(
            wandb_job_type="train_rl",
            benchmark_version="213 -> 221 -> 225 -> 228",
            temporal_split_artifact="wandb-healthcare/project/disease-temporal-split:production",
            dataset_config="disease_temporal_hc_reasoning_v1",
            reasoning_dataset_config="disease_temporal_hc_reasoning_v1",
            dataset_artifact="wandb-healthcare/project/disease-temporal-reasoning:production",
            shortlist_query="reviewed:true",
            shortlist_mode="high-confidence",
            train_start_release=213,
            train_end_release=221,
            dev_end_release=225,
            test_end_release=228,
            base_checkpoint="wandb-healthcare/project/train-sft-output:latest",
            model_artifact="train-rl-output",
            checkpoint_artifact_name="train-rl-output",
            checkpoint_dir=None,
            output_dir="data/artifacts/models/train_rl_output/demo",
            seed=23,
            learning_rate=5e-6,
            batch_size=1,
            train_batch_size=1,
            eval_batch_size=4,
            gradient_accumulation_steps=1,
            max_epochs=1,
            validation_subset_size=None,
            validation_subset_strategy=None,
            max_eval_samples=100,
            eval_sample_strategy="stratified_aspect_profile",
            max_eval_batches=0,
            eval_every_n_steps=50,
            save_every_n_steps=50,
            rotating_eval_every_n_steps=100,
            rotating_eval_max_samples=256,
            rotating_eval_sample_strategy="stratified_aspect_profile",
            rotating_eval_seed_stride=9973,
            num_generations=8,
            max_new_tokens=512,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            reward_funcs="strict_format,summary_schema,go_overlap,structural_noise",
            reward_weights="0.5,0.75,2.0,1.0",
            kl_beta=0.02,
            job_time_limit="12:00:00",
            training_stage=None,
            cafa5_dataset_name="disease_temporal_hc_reasoning_v1",
            reasoning_dataset_name="disease_temporal_hc_reasoning_v1",
            ckpt_path=None,
            projector_checkpoint_path=None,
        )

        config = TRACKING.build_training_tracking_config(args, run_name="train-rl-demo")
        metadata = TRACKING.build_checkpoint_artifact_metadata(
            args,
            run_name="train-rl-demo",
            tracking_config=config,
        )

        self.assertEqual(config["output_dir"], "data/artifacts/models/train_rl_output/demo")
        self.assertEqual(config["train_batch_size"], 1)
        self.assertEqual(config["eval_batch_size"], 4)
        self.assertEqual(config["max_eval_samples"], 100)
        self.assertEqual(config["eval_sample_strategy"], "stratified_aspect_profile")
        self.assertEqual(config["max_eval_batches"], 0)
        self.assertEqual(config["num_generations"], 8)
        self.assertEqual(config["max_new_tokens"], 512)
        self.assertEqual(config["temperature"], 1.0)
        self.assertEqual(config["top_k"], 20)
        self.assertEqual(config["reward_funcs"], "strict_format,summary_schema,go_overlap,structural_noise")
        self.assertEqual(config["reward_weights"], "0.5,0.75,2.0,1.0")
        self.assertEqual(config["rotating_eval_every_n_steps"], 100)
        self.assertEqual(metadata["checkpoint_dir"], "data/artifacts/models/train_rl_output/demo")

    def test_build_sft_sample_row_is_one_row_per_sample(self):
        batch = {
            "protein_ids": ["P12345"],
            "sample_splits": [""],
            "go_bp_targets": ["GO:0008150, GO:0007165"],
            "go_mf_targets": ["GO:0003674"],
            "go_cc_targets": ["GO:0005575"],
        }
        result = {
            "user_input": "Summarize the disease-relevant GO functions.",
            "generation": "<think>Mutant signaling is impaired.</think>\nGO:0007165",
            "ground_truth": "GO:0007165",
        }

        row = TRACKING.build_sft_sample_row(batch=batch, prefix="val", result=result)

        self.assertEqual(set(row.keys()), set(TRACKING.SFT_SAMPLE_TABLE_COLUMNS))
        self.assertEqual(row["protein_id"], "P12345")
        self.assertEqual(row["split"], "validation")
        self.assertEqual(row["reasoning"], "Mutant signaling is impaired.")
        self.assertEqual(row["final_answer"], "GO:0007165")
        self.assertEqual(row["expected_go_bp"], "GO:0008150, GO:0007165")
        self.assertEqual(row["expected_go_mf"], "GO:0003674")
        self.assertEqual(row["expected_go_cc"], "GO:0005575")

    def test_sync_run_config_updates_existing_run(self):
        run = FakeRun()
        applied = TRACKING.sync_run_config(run, {"job_type": "train_sft", "benchmark_version": "demo"})

        self.assertTrue(applied)
        self.assertEqual(run.config["job_type"], "train_sft")
        self.assertEqual(run.config["benchmark_version"], "demo")
        self.assertTrue(run.config["allow_val_change"])

    def test_maybe_log_directory_artifact_logs_checkpoint_dir(self):
        run = FakeRun()
        fake_wandb = types.SimpleNamespace(Artifact=FakeArtifact)

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = TRACKING.build_checkpoint_artifact_metadata(
                types.SimpleNamespace(checkpoint_dir=tmpdir, training_stage=2),
                run_name="demo-run",
                tracking_config={"benchmark_version": "213 -> 221 -> 225 -> 228"},
            )

            status = TRACKING.maybe_log_directory_artifact(
                run=run,
                wandb_module=fake_wandb,
                artifact_name="demo-checkpoints",
                artifact_type="model",
                directory=tmpdir,
                aliases=["latest", "best"],
                metadata=metadata,
            )

        self.assertTrue(status["logged"])
        self.assertEqual(status["aliases"], ["latest", "best"])
        self.assertEqual(run.artifacts[0].name, "demo-checkpoints")
        self.assertEqual(run.artifacts[0].added_dirs, [status["directory"]])
        self.assertEqual(run.artifacts[0].logged_aliases, ["latest", "best"])
        self.assertEqual(run.artifacts[0].metadata["run_name"], "demo-run")

    def test_prepare_model_artifact_directory_selects_single_checkpoint(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as export:
            source_path = Path(source)
            (source_path / "last.ckpt").write_text("last", encoding="utf-8")
            (source_path / "demo-best.ckpt").write_text("best", encoding="utf-8")
            (source_path / "demo-recent.ckpt").write_text("recent", encoding="utf-8")
            (source_path / "training_metadata.json").write_text("{}", encoding="utf-8")

            manifest = TRACKING.prepare_model_artifact_directory(source, export)

            self.assertTrue(manifest["prepared"])
            self.assertEqual(manifest["mode"], "raw_checkpoint")
            self.assertEqual(manifest["selected_checkpoint"], "demo-best.ckpt")
            exported = Path(export)
            self.assertTrue((exported / "demo-best.ckpt").is_file())
            self.assertFalse((exported / "last.ckpt").exists())
            self.assertTrue((exported / "training_metadata.json").is_file())
            self.assertTrue((exported / "artifact_manifest.json").is_file())

    def test_prepare_model_artifact_directory_slims_torch_checkpoints(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as export:
            source_path = Path(source)
            checkpoint_path = source_path / "demo-best.ckpt"
            torch.save(
                {
                    "state_dict": {"weight": torch.ones(2)},
                    "optimizer_states": [{"big": torch.zeros(128)}],
                    "hyper_parameters": {"foo": "bar"},
                    "epoch": 3,
                    "global_step": 9,
                },
                checkpoint_path,
            )

            manifest = TRACKING.prepare_model_artifact_directory(source, export)

            self.assertTrue(manifest["prepared"])
            self.assertTrue(manifest["checkpoint_slimmed"])
            exported_checkpoint = Path(export) / "demo-best.ckpt"
            slim = torch.load(exported_checkpoint, map_location="cpu", weights_only=False)
            self.assertIn("state_dict", slim)
            self.assertNotIn("optimizer_states", slim)
            self.assertEqual(slim["epoch"], 3)

    def test_prepare_model_artifact_directory_excludes_raw_checkpoints_for_hf_exports(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as export:
            source_path = Path(source)
            (source_path / "config.json").write_text("{}", encoding="utf-8")
            (source_path / "tokenizer.json").write_text("{}", encoding="utf-8")
            raw_dir = source_path / "raw_checkpoints"
            raw_dir.mkdir()
            (raw_dir / "checkpoint-1.bin").write_text("ignore", encoding="utf-8")

            manifest = TRACKING.prepare_model_artifact_directory(source, export)

            self.assertTrue(manifest["prepared"])
            self.assertEqual(manifest["mode"], "hf_export")
            exported = Path(export)
            self.assertTrue((exported / "config.json").is_file())
            self.assertTrue((exported / "tokenizer.json").is_file())
            self.assertFalse((exported / "raw_checkpoints").exists())
            self.assertTrue((exported / "artifact_manifest.json").is_file())

    def test_parse_artifact_aliases_deduplicates(self):
        aliases = TRACKING.parse_artifact_aliases("latest, best,latest")
        self.assertEqual(aliases, ["latest", "best"])

    def test_maybe_use_artifact_refs_registers_input_lineage(self):
        run = FakeRun()

        statuses = TRACKING.maybe_use_artifact_refs(
            run,
            {
                "temporal_split_artifact": "wandb-healthcare/bioreasoning-pro/disease-temporal-split:production",
                "dataset_artifact": "wandb-healthcare/bioreasoning-pro/disease-temporal-reasoning:production",
                "base_checkpoint": "/tmp/local-checkpoint",
            },
        )

        self.assertTrue(statuses["temporal_split_artifact"]["used"])
        self.assertTrue(statuses["dataset_artifact"]["used"])
        self.assertFalse(statuses["base_checkpoint"]["used"])
        self.assertEqual(statuses["base_checkpoint"]["reason"], "not_wandb_artifact_ref")
        self.assertEqual(
            run.used_artifacts,
            [
                ("wandb-healthcare/bioreasoning-pro/disease-temporal-split:production", None),
                ("wandb-healthcare/bioreasoning-pro/disease-temporal-reasoning:production", None),
            ],
        )


if __name__ == "__main__":
    unittest.main()

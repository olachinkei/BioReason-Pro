import tempfile
import types
import unittest

from bioreason2.utils.tracking import (
    SFT_SAMPLE_TABLE_COLUMNS,
    build_checkpoint_artifact_metadata,
    build_sft_sample_row,
    build_training_tracking_config,
    maybe_log_directory_artifact,
    parse_artifact_aliases,
    sync_run_config,
)


class FakeConfig(dict):
    def update(self, values, allow_val_change=False):
        self["allow_val_change"] = allow_val_change
        super().update(values)


class FakeRun:
    def __init__(self):
        self.config = FakeConfig()
        self.artifacts = []

    def log_artifact(self, artifact, aliases=None):
        artifact.logged_aliases = aliases
        self.artifacts.append(artifact)


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
            step0_artifact="domain/specification/busiless-rules/artifacts/step0_human_ub_20260406",
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
            seed=23,
            learning_rate=1e-4,
            batch_size=4,
            gradient_accumulation_steps=2,
            max_epochs=10,
            job_time_limit="12:00:00",
            training_stage=2,
            cafa5_dataset_name="disease_temporal_hc_v1",
            reasoning_dataset_name="disease_temporal_hc_reasoning_v1",
            ckpt_path=None,
            projector_checkpoint_path=None,
        )

        config = build_training_tracking_config(args, run_name="demo-run")

        self.assertEqual(config["job_type"], "train_sft")
        self.assertEqual(config["benchmark_version"], "213 -> 221 -> 225 -> 228")
        self.assertEqual(config["dataset_config"], "disease_temporal_hc_v1")
        self.assertEqual(config["reasoning_dataset_config"], "disease_temporal_hc_reasoning_v1")
        self.assertEqual(config["dataset_artifact"], "disease_temporal_hc_reasoning_v1:latest")
        self.assertEqual(config["model_artifact"], "disease-sft-checkpoints")
        self.assertEqual(config["job_time_limit"], "12:00:00")
        self.assertEqual(config["num_train_epochs"], 10)

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

        row = build_sft_sample_row(batch=batch, prefix="val", result=result)

        self.assertEqual(set(row.keys()), set(SFT_SAMPLE_TABLE_COLUMNS))
        self.assertEqual(row["protein_id"], "P12345")
        self.assertEqual(row["split"], "validation")
        self.assertEqual(row["reasoning"], "Mutant signaling is impaired.")
        self.assertEqual(row["final_answer"], "GO:0007165")
        self.assertEqual(row["expected_go_bp"], "GO:0008150, GO:0007165")
        self.assertEqual(row["expected_go_mf"], "GO:0003674")
        self.assertEqual(row["expected_go_cc"], "GO:0005575")

    def test_sync_run_config_updates_existing_run(self):
        run = FakeRun()
        applied = sync_run_config(run, {"job_type": "train_sft", "benchmark_version": "demo"})

        self.assertTrue(applied)
        self.assertEqual(run.config["job_type"], "train_sft")
        self.assertEqual(run.config["benchmark_version"], "demo")
        self.assertTrue(run.config["allow_val_change"])

    def test_maybe_log_directory_artifact_logs_checkpoint_dir(self):
        run = FakeRun()
        fake_wandb = types.SimpleNamespace(Artifact=FakeArtifact)

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = build_checkpoint_artifact_metadata(
                types.SimpleNamespace(checkpoint_dir=tmpdir, training_stage=2),
                run_name="demo-run",
                tracking_config={"benchmark_version": "213 -> 221 -> 225 -> 228"},
            )

            status = maybe_log_directory_artifact(
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

    def test_parse_artifact_aliases_deduplicates(self):
        aliases = parse_artifact_aliases("latest, best,latest")
        self.assertEqual(aliases, ["latest", "best"])


if __name__ == "__main__":
    unittest.main()

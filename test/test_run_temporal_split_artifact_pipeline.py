import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_temporal_split_artifact_pipeline.py"


def load_pipeline_module():
    module_name = "run_temporal_split_artifact_pipeline_test_module"

    fake_wandb = types.ModuleType("wandb")

    class FakeArtifact:
        def __init__(self, name, type, metadata=None):
            self.name = name
            self.type = type
            self.metadata = metadata or {}
            self.added_dirs = []

        def add_dir(self, path):
            self.added_dirs.append(path)

    class FakeRun:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.logged_artifacts = []
            self.finished = False

        def log_artifact(self, artifact, aliases=None):
            self.logged_artifacts.append({"artifact": artifact, "aliases": aliases})

        def finish(self):
            self.finished = True

    fake_wandb.Artifact = FakeArtifact
    fake_wandb.init_calls = []
    fake_wandb.runs = []

    def init(**kwargs):
        fake_wandb.init_calls.append(kwargs)
        run = FakeRun(**kwargs)
        fake_wandb.runs.append(run)
        return run

    fake_wandb.init = init

    previous_wandb = sys.modules.get("wandb")
    sys.modules["wandb"] = fake_wandb
    try:
        spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if previous_wandb is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = previous_wandb
    return module


PIPELINE = load_pipeline_module()


def write_valid_temporal_split_outputs(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "split_validation": {
            "time_order_valid": True,
            "protein_disjoint_valid": True,
        },
        "windows": [
            {"split": "train", "disease_proteins_after_assignment": 10},
            {"split": "dev", "disease_proteins_after_assignment": 4},
            {"split": "test", "disease_proteins_after_assignment": 5},
        ],
    }
    (base_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (base_dir / "report.md").write_text(
        "# Temporal Split Artifact\n\n| Split | Window | Proteins | Unique labels |\n|---|---|---|---|\n",
        encoding="utf-8",
    )
    for filename in PIPELINE.REQUIRED_TEMPORAL_SPLIT_FILES:
        path = base_dir / filename
        if path.exists():
            continue
        path.write_text("ok\n", encoding="utf-8")


class RunTemporalSplitArtifactPipelineContractsTest(unittest.TestCase):
    def test_prepare_local_storage_creates_required_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            variant = PIPELINE.VARIANT_CONFIGS["main"]

            status = PIPELINE.prepare_local_storage(repo_root, variant)

        self.assertTrue(status["ok"])
        created_dirs = set(status["created_dirs"])
        self.assertIn(str((repo_root / "data" / "artifacts").resolve()), created_dirs)
        self.assertIn(str((repo_root / "data" / "artifacts" / "eval").resolve()), created_dirs)
        self.assertIn(str((repo_root / variant.temporal_split_output_dir).resolve()), created_dirs)
        self.assertIn(str((repo_root / variant.default_reasoning_dir).resolve()), created_dirs)

    def test_validate_temporal_split_outputs_matches_readme_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_valid_temporal_split_outputs(output_dir)

            status = PIPELINE.validate_temporal_split_outputs(output_dir)

        self.assertTrue(status["ok"])
        self.assertTrue(status["time_order_valid"])
        self.assertTrue(status["protein_disjoint_valid"])
        self.assertTrue(status["split_counts_present"])
        self.assertEqual(status["proteins_by_split"]["train"], 10)

    def test_run_variant_pipeline_skips_upload_when_sanity_fails(self):
        args = types.SimpleNamespace(
            variant="main",
            upload_to_wandb=True,
            wandb_entity="demo",
            wandb_project="project",
            temporal_split_artifact_family="disease-temporal-split",
            reasoning_artifact_family="disease-temporal-reasoning",
            reasoning_dir=None,
            shortlist_mode="high-confidence",
            use_shell_filter=False,
            force_download=False,
            skip_propagation=False,
            temporal_split_script="scripts/build_disease_temporal_split_artifact.py",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            output_dir = repo_root / PIPELINE.VARIANT_CONFIGS["main"].temporal_split_output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "summary.json").write_text("{}", encoding="utf-8")

            with mock.patch.object(
                PIPELINE,
                "run_temporal_split_command",
                return_value={"command": ["python"], "returncode": 0, "completed": True},
            ), mock.patch.object(PIPELINE, "upload_variant_artifacts") as upload_mock:
                status = PIPELINE.run_variant_pipeline(repo_root, args, PIPELINE.VARIANT_CONFIGS["main"])

            self.assertFalse(status["ok"])
            self.assertFalse(status["sanity_check"]["ok"])
            upload_mock.assert_not_called()
            self.assertIn("Sanity check failed", status["error"])
            self.assertTrue((output_dir / "pipeline_status.json").exists())

    def test_run_variant_pipeline_uploads_after_passing_sanity(self):
        args = types.SimpleNamespace(
            variant="main",
            upload_to_wandb=True,
            wandb_entity="demo",
            wandb_project="project",
            temporal_split_artifact_family="disease-temporal-split",
            reasoning_artifact_family="disease-temporal-reasoning",
            reasoning_dir=None,
            shortlist_mode="high-confidence",
            use_shell_filter=False,
            force_download=False,
            skip_propagation=False,
            temporal_split_script="scripts/build_disease_temporal_split_artifact.py",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            variant = PIPELINE.VARIANT_CONFIGS["main"]
            output_dir = repo_root / variant.temporal_split_output_dir
            write_valid_temporal_split_outputs(output_dir)

            uploads = [{"artifact_name": "disease-temporal-split", "uploaded": True}]
            with mock.patch.object(
                PIPELINE,
                "run_temporal_split_command",
                return_value={"command": ["python"], "returncode": 0, "completed": True},
            ), mock.patch.object(PIPELINE, "upload_variant_artifacts", return_value=uploads) as upload_mock:
                status = PIPELINE.run_variant_pipeline(repo_root, args, variant)

            self.assertTrue(status["ok"])
            self.assertTrue(status["local_storage"]["ok"])
            self.assertEqual(status["uploads"], uploads)
            upload_mock.assert_called_once_with(repo_root, args, variant)
            self.assertTrue((output_dir / "pipeline_status.json").exists())

    def test_upload_variant_artifacts_uses_variant_aliases_and_skips_missing_datasets(self):
        args = types.SimpleNamespace(
            wandb_entity="demo",
            wandb_project="project",
            temporal_split_artifact_family="disease-temporal-split",
            reasoning_artifact_family="disease-temporal-reasoning",
            reasoning_dir=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            variant = PIPELINE.VARIANT_CONFIGS["comparison"]
            temporal_split_dir = repo_root / variant.temporal_split_output_dir
            temporal_split_dir.mkdir(parents=True, exist_ok=True)

            uploads = PIPELINE.upload_variant_artifacts(repo_root, args, variant)

        self.assertEqual(uploads[0]["artifact_name"], "disease-temporal-split")
        self.assertEqual(uploads[0]["aliases"], ["214.221.225.228"])
        self.assertTrue(uploads[0]["uploaded"])
        self.assertFalse(uploads[1]["uploaded"])
        self.assertEqual(uploads[1]["skip_reason"], "directory_missing_or_empty")

    def test_validate_reasoning_dataset_outputs_requires_paper_context_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            payload = {
                "context_coverage_by_split": {
                    "train": {"paper_context_ready": True},
                    "validation": {"paper_context_ready": False},
                    "test": {"paper_context_ready": True},
                }
            }
            (dataset_dir / "build_metadata.json").write_text(json.dumps(payload), encoding="utf-8")

            status = PIPELINE.validate_reasoning_dataset_outputs(dataset_dir)

        self.assertFalse(status["ok"])
        self.assertFalse(status["paper_context_ready"])

    def test_upload_variant_artifacts_skips_dataset_when_paper_context_validation_fails(self):
        args = types.SimpleNamespace(
            wandb_entity="demo",
            wandb_project="project",
            temporal_split_artifact_family="disease-temporal-split",
            reasoning_artifact_family="disease-temporal-reasoning",
            reasoning_dir=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            variant = PIPELINE.VARIANT_CONFIGS["main"]
            temporal_split_dir = repo_root / variant.temporal_split_output_dir
            temporal_split_dir.mkdir(parents=True, exist_ok=True)
            reasoning_dir = repo_root / variant.default_reasoning_dir
            reasoning_dir.mkdir(parents=True, exist_ok=True)
            (reasoning_dir / "build_metadata.json").write_text(
                json.dumps(
                    {
                        "context_coverage_by_split": {
                            "train": {"paper_context_ready": False},
                            "validation": {"paper_context_ready": False},
                            "test": {"paper_context_ready": False},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (reasoning_dir / "dataset_dict.json").write_text("{}", encoding="utf-8")

            uploads = PIPELINE.upload_variant_artifacts(repo_root, args, variant)

        self.assertFalse(uploads[1]["uploaded"])
        self.assertEqual(uploads[1]["skip_reason"], "paper_context_validation_failed")


if __name__ == "__main__":
    unittest.main()

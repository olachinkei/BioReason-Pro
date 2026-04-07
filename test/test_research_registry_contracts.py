import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "bioreason2" / "utils" / "research_registry.py"


def load_registry_module():
    module_name = "research_registry_contracts_test_module"
    spec = importlib.util.spec_from_file_location(module_name, REGISTRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


REGISTRY = load_registry_module()


class ResearchRegistryContractsTest(unittest.TestCase):
    def test_load_exported_env_file_reads_export_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {}, clear=True):
            env_path = Path(tmpdir) / "demo.env"
            env_path.write_text(
                'export FOO="bar baz"\nBAR=qux\n# comment\n',
                encoding="utf-8",
            )

            loaded = REGISTRY.load_exported_env_file(str(env_path))
            self.assertEqual(os.environ["FOO"], "bar baz")
            self.assertEqual(os.environ["BAR"], "qux")

        self.assertEqual(loaded["FOO"], "bar baz")
        self.assertEqual(loaded["BAR"], "qux")

    def test_load_data_bundle_resolves_relative_local_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            registry_path = repo_root / "data_registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "default_bundle": "main",
                        "bundles": {
                            "main": {
                                "benchmark_version": "213 -> 221 -> 225 -> 228",
                                "benchmark_alias": "213.221.225.228",
                                "temporal_split_artifact": {
                                    "wandb_registry_path": "demo/project/disease-temporal-split:production",
                                    "local_dir": "data/artifacts/benchmarks/main/temporal_split",
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            bundle = REGISTRY.load_data_bundle(str(registry_path), None, repo_root)

        self.assertEqual(bundle["bundle_name"], "main")
        self.assertTrue(bundle["temporal_split_artifact"]["local_dir"].endswith("data/artifacts/benchmarks/main/temporal_split"))

    def test_materialize_source_uses_existing_directory_when_required_paths_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / "model"
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "config.json").write_text("{}", encoding="utf-8")

            result = REGISTRY.materialize_source(
                {
                    "type": "huggingface",
                    "repo_id": "wanglab/bioreason-pro-sft",
                    "local_dir": str(local_dir),
                    "required_paths": ["config.json"],
                }
            )

        self.assertEqual(result["local_path"], str(local_dir))
        self.assertFalse(result["downloaded"])
        self.assertEqual(result["source_ref"], "hf://wanglab/bioreason-pro-sft")

    def test_materialize_source_downloads_huggingface_snapshot_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / "model"

            with mock.patch.object(REGISTRY, "download_huggingface_snapshot") as download_mock:
                result = REGISTRY.materialize_source(
                    {
                        "type": "huggingface",
                        "repo_id": "wanglab/bioreason-pro-sft",
                        "local_dir": str(local_dir),
                        "required_paths": ["config.json"],
                    }
                )

        download_mock.assert_called_once()
        self.assertEqual(result["source_ref"], "hf://wanglab/bioreason-pro-sft")
        self.assertTrue(result["downloaded"])

    def test_materialize_source_redownloads_when_only_partial_required_paths_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / "model"
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "config.json").write_text("{}", encoding="utf-8")

            with mock.patch.object(REGISTRY, "download_wandb_artifact") as artifact_mock:
                result = REGISTRY.materialize_source(
                    {
                        "type": "wandb_artifact",
                        "wandb_registry_path": "demo/project/train-sft-output:v2",
                        "local_dir": str(local_dir),
                        "required_paths": [
                            "config.json",
                            "go_embedding.pt",
                            "protein_model/pytorch_model.bin",
                        ],
                    }
                )

        artifact_mock.assert_called_once_with("demo/project/train-sft-output:v2", str(local_dir))
        self.assertEqual(result["source_ref"], "demo/project/train-sft-output:v2")
        self.assertTrue(result["downloaded"])

    def test_materialize_source_local_dir_requires_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / "predictions"
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "predictions.tsv").write_text("P1\tGO:1\t1.0\n", encoding="utf-8")

            result = REGISTRY.materialize_source(
                {
                    "type": "local_dir",
                    "local_dir": str(local_dir),
                    "required_paths": ["predictions.tsv"],
                }
            )

        self.assertEqual(result["local_path"], str(local_dir))
        self.assertFalse(result["downloaded"])

    def test_materialize_first_available_source_falls_back_to_wandb_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir) / "model"
            sources = [
                {
                    "type": "huggingface",
                    "repo_id_env": "BIOREASON_PRO_BASE_HF_REPO",
                    "local_dir": str(local_dir),
                    "required_paths": ["config.json"],
                },
                {
                    "type": "wandb_artifact",
                    "wandb_registry_path": "demo/project/bioreason-pro-base:production",
                    "local_dir": str(local_dir),
                    "required_paths": ["config.json"],
                },
            ]

            with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(
                REGISTRY, "download_wandb_artifact"
            ) as artifact_mock:
                result = REGISTRY.materialize_first_available_source(sources)

        artifact_mock.assert_called_once()
        self.assertEqual(result["source_ref"], "demo/project/bioreason-pro-base:production")


if __name__ == "__main__":
    unittest.main()

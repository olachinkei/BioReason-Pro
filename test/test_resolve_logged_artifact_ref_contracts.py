import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "resolve_logged_artifact_ref.py"


def load_module():
    module_name = "resolve_logged_artifact_ref_contracts_test_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RESOLVER = load_module()


class ResolveLoggedArtifactRefContractsTest(unittest.TestCase):
    def test_resolves_artifact_after_retry(self):
        api_calls = {"runs": 0}

        class FakeArtifact:
            def __init__(self, name):
                self.name = name

        class FakeRun:
            created_at = "2026-04-08T00:00:00Z"

            def __init__(self, artifacts):
                self._artifacts = artifacts

            def logged_artifacts(self):
                return self._artifacts

        class FakeApi:
            def __init__(self, timeout):
                self.timeout = timeout

            def runs(self, path, filters=None, per_page=100):
                api_calls["runs"] += 1
                if api_calls["runs"] < 2:
                    return [FakeRun([])]
                return [FakeRun([FakeArtifact("train-sft-output:v3")])]

        fake_wandb = types.SimpleNamespace(Api=FakeApi)
        stdout = io.StringIO()

        with mock.patch.object(
            RESOLVER,
            "parse_args",
            return_value=types.SimpleNamespace(
                wandb_entity="wandb-healthcare",
                wandb_project="bioreason-pro-custom",
                run_name="sft-verify",
                artifact_name="train-sft-output",
                attempts=3,
                sleep_seconds=0.0,
                api_timeout=90,
            ),
        ), mock.patch.dict(sys.modules, {"wandb": fake_wandb}), mock.patch.object(
            RESOLVER.time,
            "sleep",
            return_value=None,
        ), contextlib.redirect_stdout(stdout):
            exit_code = RESOLVER.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "wandb-healthcare/bioreason-pro-custom/train-sft-output:v3")
        self.assertGreaterEqual(api_calls["runs"], 2)


if __name__ == "__main__":
    unittest.main()

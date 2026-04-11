import importlib.util
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_registered_train_rl.py"
REGISTRY_PATH = ROOT / "bioreason2" / "utils" / "research_registry.py"


def load_research_registry_module():
    module_name = "run_registered_train_rl_research_registry_test_module"
    spec = importlib.util.spec_from_file_location(module_name, REGISTRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_registered_train_rl_module():
    module_name = "run_registered_train_rl_contracts_test_module"
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


REGISTERED_TRAIN_RL = load_registered_train_rl_module()


class FakeArtifact:
    def __init__(self, name: str, artifact_type: str = "model") -> None:
        self.name = name
        self.type = artifact_type


class FakeRun:
    def __init__(
        self,
        name: str,
        *,
        state: str = "finished",
        created_at: str | None = None,
        artifacts: list[FakeArtifact] | None = None,
        config: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.state = state
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self._artifacts = artifacts or []
        self.config = config or {}

    def logged_artifacts(self):
        return list(self._artifacts)


class FakeApi:
    def __init__(self, runs):
        self._runs = list(runs)
        self.project_ref = None

    def runs(self, project_ref: str):
        self.project_ref = project_ref
        return list(self._runs)


class RunRegisteredTrainRlContractsTest(unittest.TestCase):
    def make_args(self, **overrides):
        defaults = {
            "registry_env_file": ROOT / "configs" / "disease_benchmark" / "wandb_registry_paths.env",
            "wandb_entity": "demo-entity",
            "wandb_project": "demo-project",
            "run_name": "demo-run",
            "checkpoint_artifact_name": "train-rl-output",
            "checkpoint_artifact_aliases": "latest,production",
            "weave_project": "",
            "nnodes": 2,
            "gpus_per_node": 8,
            "hostfile": None,
            "master_addr": "127.0.0.1",
            "master_port": "29500",
            "node_rank": "0",
        }
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    def test_resolve_registry_env_file_resolves_relative_paths_from_repo_root(self):
        resolved = REGISTERED_TRAIN_RL.resolve_registry_env_file(Path("configs/disease_benchmark/wandb_registry_paths.env"))
        self.assertEqual(resolved, (ROOT / "configs" / "disease_benchmark" / "wandb_registry_paths.env").resolve())

    def test_build_launch_env_requires_exact_2x8_shape(self):
        with self.assertRaisesRegex(ValueError, "--nnodes 2"):
            REGISTERED_TRAIN_RL.build_launch_env(self.make_args(nnodes=1))
        with self.assertRaisesRegex(ValueError, "--gpus-per-node 8"):
            REGISTERED_TRAIN_RL.build_launch_env(self.make_args(gpus_per_node=4))

    def test_build_launch_env_supports_hostfile_without_master_env(self):
        hostfile = Path("/tmp/hosts.txt")
        env = REGISTERED_TRAIN_RL.build_launch_env(
            self.make_args(
                hostfile=hostfile,
                master_addr="",
                master_port="",
                node_rank="",
            )
        )

        self.assertEqual(env["HOSTFILE"], str(hostfile))
        self.assertNotIn("MASTER_ADDR", env)
        self.assertNotIn("MASTER_PORT", env)
        self.assertNotIn("NODE_RANK", env)

    def test_main_updates_registry_env_on_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / "wandb_registry_paths.env"
            env_file.write_text(
                'export BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH="demo/project/bioreason-pro-rl-paper:production"\n',
                encoding="utf-8",
            )
            fake_api = FakeApi([FakeRun("demo-run", artifacts=[FakeArtifact("train-rl-output:v7")])])
            fake_wandb = types.SimpleNamespace(Api=lambda: fake_api)
            argv = [
                "run_registered_train_rl.py",
                "--registry-env-file",
                str(env_file),
                "--wandb-entity",
                "demo-entity",
                "--wandb-project",
                "demo-project",
                "--run-name",
                "demo-run",
                "--master-addr",
                "10.0.0.1",
                "--master-port",
                "29500",
                "--node-rank",
                "0",
            ]

            with mock.patch.object(REGISTERED_TRAIN_RL, "load_exported_env_file") as load_env, mock.patch.object(
                REGISTERED_TRAIN_RL,
                "run_local_command",
            ) as run_local, mock.patch.dict(sys.modules, {"wandb": fake_wandb}), mock.patch.object(sys, "argv", argv):
                REGISTERED_TRAIN_RL.main()

            load_env.assert_called_once()
            self.assertTrue(Path(load_env.call_args.args[0]).samefile(env_file))
            run_local.assert_called_once()
            command, env_updates = run_local.call_args.args
            self.assertEqual(command, ["bash", "scripts/sh_train_protein_grpo.sh"])
            self.assertTrue(Path(env_updates["REGISTRY_ENV_FILE"]).samefile(env_file))
            self.assertEqual(env_updates["NNODES"], "2")
            self.assertEqual(env_updates["GPUS_PER_NODE"], "8")
            self.assertEqual(env_updates["QUERIES_PER_STEP"], "8")
            self.assertEqual(env_updates["ROLLOUTS_PER_QUERY"], "24")
            self.assertEqual(env_updates["GRADIENT_ACCUMULATION_STEPS"], "2")
            self.assertEqual(env_updates["MASTER_ADDR"], "10.0.0.1")
            self.assertEqual(env_updates["MASTER_PORT"], "29500")
            self.assertEqual(env_updates["NODE_RANK"], "0")
            self.assertEqual(fake_api.project_ref, "demo-entity/demo-project")
            self.assertIn(
                'export BIOREASON_TRAIN_RL_MODEL_REGISTRY_PATH="demo-entity/demo-project/train-rl-output:v7"',
                env_file.read_text(encoding="utf-8"),
            )

    def test_main_does_not_update_registry_env_when_run_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / "wandb_registry_paths.env"
            env_file.write_text(
                'export BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH="demo/project/bioreason-pro-rl-paper:production"\n',
                encoding="utf-8",
            )
            fake_api = FakeApi([FakeRun("demo-run", state="failed", artifacts=[FakeArtifact("train-rl-output:v7")])])
            fake_wandb = types.SimpleNamespace(Api=lambda: fake_api)
            argv = [
                "run_registered_train_rl.py",
                "--registry-env-file",
                str(env_file),
                "--wandb-entity",
                "demo-entity",
                "--wandb-project",
                "demo-project",
                "--run-name",
                "demo-run",
                "--master-addr",
                "10.0.0.1",
                "--master-port",
                "29500",
                "--node-rank",
                "0",
            ]

            with mock.patch.object(REGISTERED_TRAIN_RL, "load_exported_env_file"), mock.patch.object(
                REGISTERED_TRAIN_RL,
                "run_local_command",
            ), mock.patch.object(REGISTERED_TRAIN_RL, "update_env_export") as update_env, mock.patch.dict(
                sys.modules,
                {"wandb": fake_wandb},
            ), mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(RuntimeError, "state=failed"):
                    REGISTERED_TRAIN_RL.main()

            update_env.assert_not_called()


if __name__ == "__main__":
    unittest.main()

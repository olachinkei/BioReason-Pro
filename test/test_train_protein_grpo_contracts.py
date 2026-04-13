import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "train_protein_grpo.py"
WRAPPER_PATH = ROOT / "scripts" / "sh_train_protein_grpo.sh"


def load_grpo_module():
    module_name = "train_protein_grpo_spec_test_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GRPO = load_grpo_module()


class FakeWeaveClient:
    def __init__(self) -> None:
        self.flush_called = False

    def flush(self) -> None:
        self.flush_called = True


class FakeWeaveAttributes:
    def __init__(self, owner: "FakeWeaveModule", payload: dict[str, object]) -> None:
        self.owner = owner
        self.payload = dict(payload)

    def __enter__(self) -> "FakeWeaveAttributes":
        self.owner.attribute_calls.append(self.payload)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeWeaveModule:
    def __init__(self) -> None:
        self.client = FakeWeaveClient()
        self.init_calls: list[dict[str, object]] = []
        self.attribute_calls: list[dict[str, object]] = []
        self.trace_payloads: list[dict[str, object]] = []
        self.trace_results: list[dict[str, object]] = []

    def init(self, project: str, global_attributes: dict[str, object] | None = None) -> FakeWeaveClient:
        self.init_calls.append(
            {
                "project": project,
                "global_attributes": dict(global_attributes or {}),
            }
        )
        return self.client

    def op(self, name: str):
        def decorator(fn):
            def wrapped(payload):
                payload_dict = dict(payload)
                self.trace_payloads.append(payload_dict)
                result = fn(payload_dict)
                self.trace_results.append(dict(result))
                return result

            wrapped._weave_name = name
            return wrapped

        return decorator

    def attributes(self, payload: dict[str, object]) -> FakeWeaveAttributes:
        return FakeWeaveAttributes(self, payload)


class FakeWandbConfig(dict):
    def update(self, values, allow_val_change=False):
        self["allow_val_change"] = allow_val_change
        super().update(values)


class FakeWandbRun:
    def __init__(self, *, entity: str, project: str, run_id: str = "demo123") -> None:
        self.entity = entity
        self.project = project
        self.id = run_id
        self.path = f"{entity}/{project}/{run_id}" if entity and project else ""
        self.url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}" if entity and project else ""
        self.logged: list[dict[str, object]] = []
        self.define_metric_calls: list[dict[str, object]] = []
        self.finished = False
        self.used_artifacts: list[tuple[str, object]] = []
        self.config = FakeWandbConfig()

    def log(self, payload, step=None):
        self.logged.append({"payload": dict(payload), "step": step})

    def define_metric(self, name, **kwargs):
        self.define_metric_calls.append({"name": name, "kwargs": dict(kwargs)})

    def finish(self):
        self.finished = True

    def use_artifact(self, artifact_ref, type=None):
        self.used_artifacts.append((artifact_ref, type))


class FakeWandbModule:
    def __init__(self) -> None:
        self.init_calls: list[dict[str, object]] = []
        self.define_metric_calls: list[dict[str, object]] = []
        self.runs: list[FakeWandbRun] = []
        self.run = None

    def init(self, **kwargs):
        self.init_calls.append(dict(kwargs))
        run = FakeWandbRun(
            entity=str(kwargs.get("entity") or ""),
            project=str(kwargs.get("project") or ""),
        )
        self.runs.append(run)
        self.run = run
        return run

    def define_metric(self, name, **kwargs):
        self.define_metric_calls.append({"name": name, "kwargs": dict(kwargs)})


class TrainProteinGrpoContractsTest(unittest.TestCase):
    def write_executable(self, path: Path, body: str) -> Path:
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
        return path

    def create_model_bundle(self, root: Path) -> Path:
        model_dir = root / "model_bundle"
        (model_dir / "protein_model").mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (model_dir / "protein_projection.pt").write_text("stub", encoding="utf-8")
        (model_dir / "protein_model" / "pytorch_model.bin").write_text("stub", encoding="utf-8")
        return model_dir

    def create_dataset_dir(self, root: Path) -> Path:
        dataset_dir = root / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "dataset_dict.json").write_text("{}", encoding="utf-8")
        return dataset_dir

    def create_wrapper_harness(self, root: Path) -> dict[str, str]:
        model_dir = self.create_model_bundle(root)
        dataset_dir = self.create_dataset_dir(root)
        ia_dir = root / "ia_bundle"
        ia_dir.mkdir(parents=True, exist_ok=True)
        ia_path = ia_dir / "IA.txt"
        ia_path.write_text("GO:0000001 1.0\n", encoding="utf-8")
        obo_path = root / "mini.obo"
        obo_path.write_text("[Term]\nid: GO:0000001\nname: root\n", encoding="utf-8")
        hostfile = root / "hosts.txt"
        hostfile.write_text("worker-0 slots=4\nworker-1 slots=4\n", encoding="utf-8")
        registry_env = root / "wandb_registry_paths.env"
        registry_env.write_text(
            "\n".join(
                [
                    'export BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH="env/project/bioreason-pro-rl-paper:production"',
                    'export WANDB_PROJECT="env-project"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        log_path = root / "tool_log.jsonl"

        fake_python = self.write_executable(
            root / "fake_python.py",
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json
                import os
                import sys
                from pathlib import Path

                log_path = Path(os.environ["FAKE_LOG_PATH"])
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"tool": "python", "argv": sys.argv[1:]}) + "\\n")

                script_name = Path(sys.argv[1]).name if len(sys.argv) > 1 else ""
                argv = sys.argv[2:]

                def value_after(flag: str) -> str:
                    if flag not in argv:
                        return ""
                    index = argv.index(flag)
                    if index + 1 >= len(argv):
                        return ""
                    return argv[index + 1]

                if script_name == "materialize_model_source.py":
                    sys.stdout.write(os.environ["FAKE_MODEL_DIR"])
                elif script_name == "materialize_data_bundle.py":
                    asset_key = value_after("--asset-key")
                    field = value_after("--print-field")
                    if asset_key == "reasoning_dataset" and field == "local_dir":
                        sys.stdout.write(os.environ["FAKE_DATASET_DIR"])
                    elif asset_key == "reasoning_dataset" and field == "dataset_name":
                        sys.stdout.write("demo_reasoning_dataset")
                    elif asset_key == "reasoning_dataset" and field == "wandb_registry_path":
                        sys.stdout.write(os.environ["FAKE_DATASET_ARTIFACT"])
                    elif asset_key == "temporal_split_artifact" and field == "wandb_registry_path":
                        sys.stdout.write(os.environ["FAKE_TEMPORAL_ARTIFACT"])
                    elif asset_key == "ia_file" and field == "local_dir":
                        sys.stdout.write(os.environ["FAKE_IA_DIR"])
                    else:
                        raise SystemExit(f"unexpected data bundle resolver call: {argv}")
                elif script_name == "train_protein_grpo.py":
                    pass
                """
            ),
        )
        fake_deepspeed = self.write_executable(
            root / "fake_deepspeed.py",
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json
                import os
                import sys
                from pathlib import Path

                log_path = Path(os.environ["FAKE_LOG_PATH"])
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"tool": "deepspeed", "argv": sys.argv[1:]}) + "\\n")
                """
            ),
        )
        fake_scontrol = self.write_executable(
            root / "scontrol",
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import os
                import sys

                if sys.argv[1:] != ["show", "hostnames", os.environ.get("SLURM_JOB_NODELIST", "")]:
                    raise SystemExit(f"unexpected scontrol argv: {sys.argv[1:]}")
                for host in os.environ.get("FAKE_SCONTROL_HOSTS", "worker-0\\nworker-1").splitlines():
                    host = host.strip()
                    if host:
                        print(host)
                """
            ),
        )
        model_resolver = root / "materialize_model_source.py"
        model_resolver.write_text("# resolver placeholder\n", encoding="utf-8")
        data_resolver = root / "materialize_data_bundle.py"
        data_resolver.write_text("# resolver placeholder\n", encoding="utf-8")

        return {
            "FAKE_LOG_PATH": str(log_path),
            "PYTHON_BIN": str(fake_python),
            "DEEPSPEED_BIN": str(fake_deepspeed),
            "MODEL_SOURCE_RESOLVER": str(model_resolver),
            "DATA_BUNDLE_RESOLVER": str(data_resolver),
            "REGISTRY_ENV_FILE": str(registry_env),
            "GO_OBO_PATH": str(obo_path),
            "FAKE_MODEL_DIR": str(model_dir),
            "FAKE_DATASET_DIR": str(dataset_dir),
            "FAKE_IA_DIR": str(ia_dir),
            "FAKE_TEMPORAL_ARTIFACT": "env/project/disease-temporal-split:production",
            "FAKE_DATASET_ARTIFACT": "env/project/disease-temporal-reasoning:production",
            "HOSTFILE_PATH": str(hostfile),
            "PATH": f"{root}:{os.environ['PATH']}",
            "FAKE_SCONTROL_HOSTS": "worker-0\nworker-1\n",
        }

    def read_tool_log(self, log_path: Path) -> list[dict[str, object]]:
        if not log_path.exists():
            return []
        return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def value_after(self, argv: list[str], flag: str) -> str:
        index = argv.index(flag)
        return argv[index + 1]

    def test_defaults_match_specification(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            args = GRPO.parse_args(["--text_model_name", "/tmp/demo-model"])
            algorithm = GRPO.build_algorithm_spec(args)
            runtime_spec = GRPO.build_runtime_spec(args)

        self.assertEqual(args.max_steps, 1200)
        self.assertEqual(algorithm.queries_per_step, 8)
        self.assertEqual(algorithm.rollouts_per_query, 24)
        self.assertEqual(algorithm.total_trajectories, 192)
        self.assertEqual(algorithm.steps_per_generation, 2)
        self.assertEqual(algorithm.num_iterations, 1)
        self.assertEqual(algorithm.max_new_tokens, 10000)
        self.assertEqual(runtime_spec.optimizer_micro_batch_size_per_gpu, 6)
        self.assertEqual(runtime_spec.gradient_accumulation_steps, 2)
        self.assertEqual(runtime_spec.target_num_nodes, 2)
        self.assertEqual(runtime_spec.target_gpus_per_node, 8)
        self.assertEqual(runtime_spec.target_world_size, 16)
        self.assertEqual(runtime_spec.local_trajectories_per_rank, 12)
        self.assertEqual(runtime_spec.runtime_stack, "deepspeed_vllm_colocate")
        self.assertEqual(args.attn_implementation, "auto")
        self.assertEqual(args.dataset_num_proc, 4)
        self.assertEqual(args.reasoning_prompt_style, "paper_native_tight")
        self.assertEqual(args.vllm_attention_backend, "XFORMERS")
        self.assertEqual(args.vllm_worker_multiproc_method, "spawn")
        self.assertFalse(args.vllm_enable_sleep_mode)
        self.assertFalse(args.vllm_use_v1)

    def test_parse_args_accepts_deepspeed_local_rank_flag(self):
        args = GRPO.parse_args(
            [
                "--text_model_name",
                "/tmp/demo-model",
                "--local_rank=3",
            ]
        )

        self.assertEqual(args.local_rank, 3)

    def test_resolve_dataset_num_proc_treats_non_positive_as_auto(self):
        self.assertIsNone(GRPO.resolve_dataset_num_proc(0))
        self.assertIsNone(GRPO.resolve_dataset_num_proc(-3))
        self.assertEqual(GRPO.resolve_dataset_num_proc(4), 4)

    def test_resolve_effective_dataset_num_proc_clamps_distributed_workers(self):
        self.assertEqual(GRPO.resolve_effective_dataset_num_proc(4, distributed=True), 1)
        self.assertEqual(GRPO.resolve_effective_dataset_num_proc(4, distributed=False), 4)
        self.assertIsNone(GRPO.resolve_effective_dataset_num_proc(0, distributed=True))

    def test_extract_go_terms_requires_final_answer_block(self):
        self.assertIsNone(GRPO.extract_go_terms_from_final_answer("GO:0007165"))
        self.assertEqual(
            GRPO.extract_go_terms_from_final_answer(
                "<|FINAL_ANSWER|>\nGO:0007165\nGO:0005515\n<|/FINAL_ANSWER|>"
            ),
            ["GO:0007165", "GO:0005515"],
        )

    def test_extract_go_terms_from_completion_requires_final_answer_block(self):
        self.assertIsNone(
            GRPO.extract_go_terms_from_completion(
                "<|GO_SUMMARY_START|>\nBP: GO:0007165 (signal transduction)\nMF: GO:0005515 (protein binding)\n<|GO_SUMMARY_END|>"
            )
        )

    def test_extract_go_terms_from_completion_accepts_alternate_final_answer_close_tag(self):
        self.assertEqual(
            GRPO.extract_go_terms_from_completion(
                "<|FINAL_ANSWER|>\nGO:0007165\nGO:0005515\n</|FINAL_ANSWER|>"
            ),
            ["GO:0007165", "GO:0005515"],
        )

    def test_extract_go_terms_from_completion_accepts_open_final_answer_until_end_marker(self):
        self.assertEqual(
            GRPO.extract_go_terms_from_completion(
                "<|FINAL_ANSWER|>\nGO:0007165\nGO:0005515\n<|endoftext|>"
            ),
            ["GO:0007165", "GO:0005515"],
        )

    def test_build_completion_format_summary_tracks_broken_final_answer_shapes(self):
        summary = GRPO.build_completion_format_summary(
            "<|FINAL_ANSWER|>\nGO:0007165\n</|FINAL_ANSWER|>\n</think>\n<tool_call>{}</tool_call>\n<|FINAL_ANSWER|>"
        )

        self.assertTrue(summary["has_final_answer_tag"])
        self.assertTrue(summary["uses_alt_final_answer_close_tag"])
        self.assertTrue(summary["has_repeated_final_answer_open_tag"])
        self.assertTrue(summary["has_tool_call_residue"])
        self.assertTrue(summary["has_think_residue"])
        self.assertEqual(summary["parsed_go_ids"], ["GO:0007165"])

    def test_build_completion_format_summary_tracks_unclosed_final_answer(self):
        summary = GRPO.build_completion_format_summary(
            "<|FINAL_ANSWER|>\nGO:0007165\n<|endoftext|>"
        )

        self.assertTrue(summary["has_final_answer_tag"])
        self.assertTrue(summary["has_unclosed_final_answer_tag"])
        self.assertFalse(summary["uses_alt_final_answer_close_tag"])
        self.assertEqual(summary["parsed_go_ids"], ["GO:0007165"])

    def test_compute_group_rewards_requires_final_answer_block(self):
        rewards = GRPO.compute_group_rewards(
            completions=[
                "<|GO_SUMMARY_START|>\nBP: GO:0000001 (root)\n<|GO_SUMMARY_END|>",
                "No structured output",
            ],
            sample_meta={"go_bp": "GO:0000001", "go_mf": "", "go_cc": ""},
            go_graph={},
            ia_weights={"GO:0000001": 1.0},
        )

        self.assertEqual(rewards, [0.0, 0.0])

    def test_build_query_sample_meta_omits_reasoning_and_final_answer(self):
        sample_meta = GRPO.build_query_sample_meta(
            {
                "protein_ids": ["P12345"],
                "sample_splits": ["train"],
                "go_bp_targets": ["GO:0000001"],
                "go_mf_targets": [""],
                "go_cc_targets": ["GO:0000002"],
                "reasoning_targets": ["teacher reasoning"],
                "final_answers": ["teacher final"],
            }
        )

        self.assertEqual(sample_meta["protein_id"], "P12345")
        self.assertEqual(sample_meta["split"], "train")
        self.assertEqual(sample_meta["go_bp"], "GO:0000001")
        self.assertEqual(sample_meta["go_cc"], "GO:0000002")
        self.assertNotIn("reasoning", sample_meta)
        self.assertNotIn("final_answer", sample_meta)

    def test_compute_group_advantages_uses_batch_global_std(self):
        runtime = GRPO.DistributedRuntime(enabled=False, rank=0, world_size=1, local_rank=0, device="cpu")
        global_std = GRPO.compute_global_reward_std([[1.0, 1.0, 3.0], [0.0, 0.0, 2.0]], runtime, epsilon=1e-6)
        self.assertGreater(global_std, 0.0)

        first_group = GRPO.compute_group_advantages([1.0, 1.0, 3.0], global_std)
        second_group = GRPO.compute_group_advantages([0.0, 0.0, 2.0], global_std)
        self.assertAlmostEqual(sum(first_group) + sum(second_group), 0.0, places=5)

    def test_partition_queries_for_rank_matches_one_query_per_rank_layout(self):
        global_indices = list(range(8))
        self.assertEqual(GRPO.partition_queries_for_rank(global_indices, rank=0, world_size=8, queries_per_step=8), [0])
        self.assertEqual(GRPO.partition_queries_for_rank(global_indices, rank=3, world_size=8, queries_per_step=8), [3])
        self.assertEqual(GRPO.partition_queries_for_rank(global_indices, rank=7, world_size=8, queries_per_step=8), [7])
        self.assertEqual(GRPO.partition_queries_for_rank(global_indices, rank=0, world_size=16, queries_per_step=8), [0])
        self.assertEqual(GRPO.partition_queries_for_rank(global_indices, rank=1, world_size=16, queries_per_step=8), [0])
        self.assertEqual(GRPO.partition_queries_for_rank(global_indices, rank=14, world_size=16, queries_per_step=8), [7])

    def test_resolve_local_cuda_visible_device_prefers_single_visible_gpu(self):
        self.assertEqual(GRPO.resolve_local_cuda_visible_device(local_rank=3, cuda_visible_devices="5"), "5")
        self.assertEqual(GRPO.resolve_local_cuda_visible_device(local_rank=2, cuda_visible_devices="0,2,4,6"), "4")

    def test_select_rollout_indices_for_loss_filters_long_completions(self):
        if GRPO.torch is None:
            self.skipTest("torch is required for completion-id tensors")
        completion_ids = [
            GRPO.torch.tensor([1, 2, 3], dtype=GRPO.torch.long),
            GRPO.torch.tensor([1, 2, 3, 4, 5], dtype=GRPO.torch.long),
        ]
        self.assertEqual(
            GRPO.select_rollout_indices_for_loss(completion_ids, max_loss_completion_tokens=4),
            [0],
        )

    def test_resolve_effective_vllm_sleep_mode_disables_subprocess_backend(self):
        subprocess_args = GRPO.parse_args(
            [
                "--text_model_name",
                "/tmp/demo-model",
                "--rollout_backend",
                "subprocess",
                "--vllm_enable_sleep_mode",
                "true",
            ]
        )
        inprocess_args = GRPO.parse_args(
            [
                "--text_model_name",
                "/tmp/demo-model",
                "--rollout_backend",
                "inprocess",
                "--vllm_enable_sleep_mode",
                "true",
            ]
        )

        self.assertFalse(GRPO.resolve_effective_vllm_sleep_mode(subprocess_args))
        self.assertTrue(GRPO.resolve_effective_vllm_sleep_mode(inprocess_args))

    def test_rollout_worker_refresh_defers_subprocess_restart_until_next_generate(self):
        worker = object.__new__(GRPO.VLLMRolloutWorker)
        worker.backend = "subprocess"
        worker.checkpoint_dir = Path("/tmp/original")
        worker.args = mock.Mock()
        worker.runtime = mock.Mock()

        with mock.patch.object(worker, "_stop_subprocess") as stop_mock, mock.patch.object(
            worker, "_start_subprocess"
        ) as start_mock:
            worker.refresh(Path("/tmp/next"))

        self.assertEqual(worker.checkpoint_dir, Path("/tmp/next"))
        stop_mock.assert_called_once_with()
        start_mock.assert_not_called()

    def test_rollout_worker_unload_stops_subprocess_without_sleep_mode_and_generate_restarts_it(self):
        worker = object.__new__(GRPO.VLLMRolloutWorker)
        worker.backend = "subprocess"
        worker.checkpoint_dir = Path("/tmp/current")
        worker.args = GRPO.parse_args(
            [
                "--text_model_name",
                "/tmp/demo-model",
                "--rollout_backend",
                "subprocess",
                "--vllm_enable_sleep_mode",
                "false",
            ]
        )
        worker.runtime = types.SimpleNamespace(rank=0)
        worker._connection = None
        worker._process = None
        worker._generation_counter = 0

        with mock.patch.object(worker, "_stop_subprocess") as stop_mock:
            worker.unload()
        stop_mock.assert_called_once_with()

        connection = mock.Mock()

        def fake_start(checkpoint_dir: Path) -> None:
            self.assertEqual(checkpoint_dir, Path("/tmp/current"))
            worker._connection = connection

        with mock.patch.object(worker, "_start_subprocess", side_effect=fake_start) as start_mock, mock.patch.object(
            worker, "_recv_response", return_value={"status": "ok", "outputs": ["demo"]}
        ) as recv_mock, mock.patch.object(GRPO, "build_rollout_query_payload", return_value={"query": "payload"}):
            worker._connection = None
            worker.generate_group(
                query=GRPO.PreparedQuery(
                    input_ids=None,
                    attention_mask=None,
                    protein_sequences=[],
                    batch_idx_map=[],
                    structure_coords=None,
                    go_aspects=[],
                    sample_meta={},
                    prompt_text="",
                    multimodal_cache=None,
                ),
                repeat_count=1,
                sampling=GRPO.SamplingSpec(),
            )
        start_mock.assert_called_once_with(Path("/tmp/current"))
        recv_mock.assert_called_once()

    def test_build_deepspeed_config_matches_spec_shape(self):
        args = GRPO.parse_args(["--text_model_name", "/tmp/demo-model"])
        runtime_spec = GRPO.build_runtime_spec(args)
        ds_config = GRPO.build_deepspeed_config(args, runtime_spec)

        self.assertEqual(ds_config["train_micro_batch_size_per_gpu"], 6)
        self.assertEqual(ds_config["gradient_accumulation_steps"], 2)
        self.assertEqual(ds_config["gradient_clipping"], 1.0)
        self.assertEqual(ds_config["zero_optimization"]["stage"], 2)
        self.assertTrue(ds_config["bf16"]["enabled"])

    def test_build_tracking_config_carries_required_spec_fields(self):
        args = GRPO.parse_args(["--text_model_name", "/tmp/demo-model"])
        args.temporal_split_artifact = "wandb-healthcare/bioreasoning-pro/disease-temporal-split:production"
        args.dataset_artifact = "wandb-healthcare/bioreasoning-pro/disease-temporal-reasoning:production"
        args.base_checkpoint = "wandb-healthcare/bioreasoning-pro/bioreason-pro-rl-paper:production"
        algorithm = GRPO.build_algorithm_spec(args)
        runtime_spec = GRPO.build_runtime_spec(args)
        runtime = GRPO.DistributedRuntime(enabled=False, rank=0, world_size=1, local_rank=0, device="cpu")

        config = GRPO.build_tracking_config(args, algorithm, runtime_spec, runtime, run_name="demo-run")

        self.assertEqual(config["job_type"], "train_rl")
        self.assertEqual(config["benchmark_version"], "213 -> 221 -> 225 -> 228")
        self.assertEqual(
            config["temporal_split_artifact"],
            "wandb-healthcare/bioreasoning-pro/disease-temporal-split:production",
        )
        self.assertEqual(
            config["dataset_artifact"],
            "wandb-healthcare/bioreasoning-pro/disease-temporal-reasoning:production",
        )
        self.assertEqual(
            config["base_checkpoint"],
            "wandb-healthcare/bioreasoning-pro/bioreason-pro-rl-paper:production",
        )
        self.assertEqual(config["model_artifact"], "train-rl-output")
        self.assertEqual(config["reward_extraction"], "final_answer_only")
        self.assertEqual(config["paper_target_rollouts_per_query"], 24.0)
        self.assertEqual(config["paper_target_max_new_tokens"], 10000.0)
        self.assertEqual(config["paper_deviation_max_new_tokens"], 0.0)
        self.assertEqual(config["query_parallel_degree"], 2)
        self.assertEqual(config["local_rollouts_per_rank"], 12)
        self.assertEqual(config["wandb_project"], "bioreasoning-pro")
        self.assertEqual(config["reward_prediction_source"], "final_answer_block")

    def test_load_reasoning_datasets_uses_configured_reasoning_prompt_style(self):
        args = GRPO.parse_args(["--text_model_name", "/tmp/demo-model", "--reasoning_prompt_style", "paper_compact"])
        runtime = GRPO.DistributedRuntime(enabled=False, rank=0, world_size=1, local_rank=0, device="cpu")

        with mock.patch(
            "bioreason2.dataset.cafa5.load.load_cafa5_dataset",
            return_value=(["train"], ["validation"], []),
        ) as load_mock:
            train_dataset, validation_dataset = GRPO.load_reasoning_datasets(args, runtime)

        self.assertEqual(train_dataset, ["train"])
        self.assertEqual(validation_dataset, ["validation"])
        self.assertEqual(load_mock.call_args.kwargs["reasoning_prompt_style"], "paper_compact")

    def test_parse_args_accepts_paper_native_tight_reasoning_prompt_style(self):
        args = GRPO.parse_args(["--text_model_name", "/tmp/demo-model", "--reasoning_prompt_style", "paper_native_tight"])
        self.assertEqual(args.reasoning_prompt_style, "paper_native_tight")

    def test_build_trace_path_is_rank_scoped_when_distributed(self):
        runtime = GRPO.DistributedRuntime(enabled=True, rank=3, world_size=8, local_rank=3, device="cpu")
        trace_path = GRPO.build_trace_path(Path("/tmp/out"), "rollout_traces.jsonl", runtime)
        self.assertEqual(trace_path, Path("/tmp/out/rollout_traces.rank03.jsonl"))

    def test_validate_runtime_shape_requires_target_world_size_match(self):
        args = GRPO.parse_args(
            [
                "--text_model_name",
                "/tmp/demo-model",
                "--target_num_nodes",
                "1",
                "--target_gpus_per_node",
                "4",
                "--debug_single_process",
                "true",
            ]
        )
        algorithm = GRPO.build_algorithm_spec(args)
        runtime_spec = GRPO.build_runtime_spec(args)
        runtime = GRPO.DistributedRuntime(enabled=False, rank=0, world_size=1, local_rank=0, device="cpu")

        with self.assertRaisesRegex(ValueError, "integer multiple of queries_per_step"):
            GRPO.validate_runtime_shape(runtime, algorithm, runtime_spec, args)

    def test_validate_spec_inputs_requires_existing_ia_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_dir = self.create_model_bundle(tmp_path)
            dataset_dir = self.create_dataset_dir(tmp_path)
            obo_path = tmp_path / "mini.obo"
            obo_path.write_text("[Term]\nid: GO:0000001\nname: root\n", encoding="utf-8")
            args = GRPO.parse_args(
                [
                    "--text_model_name",
                    str(model_dir),
                    "--cafa5_dataset",
                    str(dataset_dir),
                    "--go_obo_path",
                    str(obo_path),
                    "--ia_file_path",
                    str(tmp_path / "missing-ia.txt"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "requires a valid IA file"):
                GRPO.validate_spec_inputs(args)

    def test_run_preflight_reports_missing_dependencies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_dir = self.create_model_bundle(tmp_path)
            dataset_dir = self.create_dataset_dir(tmp_path)
            ia_path = Path(tmpdir) / "ia.txt"
            obo_path = Path(tmpdir) / "mini.obo"
            ia_path.write_text("GO:0000001 1.0\n", encoding="utf-8")
            obo_path.write_text("[Term]\nid: GO:0000001\nname: root\n", encoding="utf-8")
            args = GRPO.parse_args(
                [
                    "--text_model_name",
                    str(model_dir),
                    "--cafa5_dataset",
                    str(dataset_dir),
                    "--ia_file_path",
                    str(ia_path),
                    "--go_obo_path",
                    str(obo_path),
                    "--debug_single_process",
                    "true",
                ]
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                ok = GRPO.run_preflight(args)
            self.assertFalse(ok)
            self.assertIn("Missing runtime dependencies", stdout.getvalue())

    def test_run_preflight_reports_resolved_paths_and_artifact_refs_on_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_dir = self.create_model_bundle(tmp_path)
            dataset_dir = self.create_dataset_dir(tmp_path)
            ia_path = tmp_path / "IA.txt"
            obo_path = tmp_path / "mini.obo"
            ia_path.write_text("GO:0000001 1.0\n", encoding="utf-8")
            obo_path.write_text("[Term]\nid: GO:0000001\nname: root\n", encoding="utf-8")
            args = GRPO.parse_args(
                [
                    "--text_model_name",
                    str(model_dir),
                    "--base_checkpoint",
                    "wandb-healthcare/bioreasoning-pro/bioreason-pro-rl-paper:production",
                    "--cafa5_dataset",
                    str(dataset_dir),
                    "--ia_file_path",
                    str(ia_path),
                    "--go_obo_path",
                    str(obo_path),
                    "--temporal_split_artifact",
                    "wandb-healthcare/bioreasoning-pro/disease-temporal-split:production",
                    "--dataset_artifact",
                    "wandb-healthcare/bioreasoning-pro/disease-temporal-reasoning:production",
                    "--debug_single_process",
                    "true",
                ]
            )
            stdout = io.StringIO()
            with mock.patch.object(
                GRPO,
                "collect_runtime_dependency_statuses",
                return_value={
                    "torch": True,
                    "deepspeed": True,
                    "peft": True,
                    "transformers": True,
                    "vllm": True,
                },
            ), redirect_stdout(stdout):
                ok = GRPO.run_preflight(args)
            self.assertTrue(ok)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(
                payload["artifact_refs"]["base_checkpoint"],
                "wandb-healthcare/bioreasoning-pro/bioreason-pro-rl-paper:production",
            )
            self.assertEqual(payload["artifact_refs"]["dataset_artifact"], "wandb-healthcare/bioreasoning-pro/disease-temporal-reasoning:production")
            self.assertEqual(payload["resolved_paths"]["text_model_name"], str(model_dir))
            self.assertEqual(payload["resolved_paths"]["cafa5_dataset"], str(dataset_dir))
            self.assertEqual(payload["launch_contract"]["target_world_size"], 16)
            self.assertEqual(payload["launch_contract"]["queries_per_step"], 8)
            self.assertEqual(payload["launch_contract"]["query_parallel_degree"], 2)
            self.assertEqual(payload["launch_contract"]["local_rollouts_per_rank"], 12)
            self.assertEqual(payload["launch_contract"]["attn_implementation"], "auto")
            self.assertEqual(payload["launch_contract"]["dataset_num_proc"], 1)
            self.assertEqual(payload["launch_contract"]["vllm_attention_backend"], "XFORMERS")
            self.assertEqual(payload["launch_contract"]["vllm_worker_multiproc_method"], "spawn")
            self.assertFalse(payload["launch_contract"]["vllm_enable_sleep_mode"])
            self.assertFalse(payload["launch_contract"]["vllm_use_v1"])
            self.assertEqual(payload["failures"], [])
            self.assertTrue(any("dataset_num_proc is automatically reduced to 1" in item for item in payload["warnings"]))

    def test_go_graph_loading_and_propagation_supports_part_of_and_is_a(self):
        obo_text = """
[Term]
id: GO:0000001
name: root

[Term]
id: GO:0000002
name: child
is_a: GO:0000001 ! root

[Term]
id: GO:0000003
name: part child
relationship: part_of GO:0000002 ! child
""".strip()
        with tempfile.TemporaryDirectory() as tmpdir:
            obo_path = Path(tmpdir) / "mini.obo"
            obo_path.write_text(obo_text, encoding="utf-8")
            graph = GRPO.load_go_term_graph(str(obo_path))
            propagated = GRPO.propagate_go_ids(["GO:0000003"], graph)
            self.assertEqual(propagated, ["GO:0000003", "GO:0000002", "GO:0000001"])

    def test_wrapper_is_minimal_deepspeed_launcher(self):
        wrapper_text = WRAPPER_PATH.read_text(encoding="utf-8")
        self.assertIn("source_env_file_without_overrides()", wrapper_text)
        self.assertIn('REGISTRY_ENV_FILE=${REGISTRY_ENV_FILE:-"configs/disease_benchmark/wandb_registry_paths.env"}', wrapper_text)
        self.assertIn('MODEL_SOURCE_RESOLVER=${MODEL_SOURCE_RESOLVER:-"scripts/materialize_model_source.py"}', wrapper_text)
        self.assertIn('DATA_BUNDLE_RESOLVER=${DATA_BUNDLE_RESOLVER:-"scripts/materialize_data_bundle.py"}', wrapper_text)
        self.assertIn('BASE_CHECKPOINT=${BASE_CHECKPOINT:-"${BIOREASON_RL_PAPER_MODEL_REGISTRY_PATH:-}"}', wrapper_text)
        self.assertIn('--hostfile "$HOSTFILE"', wrapper_text)
        self.assertIn("--no_ssh", wrapper_text)
        self.assertIn('--num_nodes "$NNODES"', wrapper_text)
        self.assertIn('--num_gpus "$GPUS_PER_NODE"', wrapper_text)
        self.assertIn('QUERIES_PER_STEP=${QUERIES_PER_STEP:-8}', wrapper_text)
        self.assertIn('GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}', wrapper_text)
        self.assertIn('--preflight_only', wrapper_text)
        self.assertNotIn("srun", wrapper_text)
        self.assertNotIn("torch.distributed.run", wrapper_text)

    def test_wrapper_preflight_uses_python_and_prefers_explicit_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = self.create_wrapper_harness(Path(tmpdir))
            env = os.environ.copy()
            env.update(harness)
            env.update(
                {
                    "BASE_CHECKPOINT": "explicit/project/custom-paper:production",
                    "WANDB_PROJECT": "explicit-project",
                }
            )

            result = subprocess.run(
                ["bash", str(WRAPPER_PATH), "--preflight_only", "true"],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            entries = self.read_tool_log(Path(harness["FAKE_LOG_PATH"]))
            model_entry = next(
                entry
                for entry in entries
                if entry["tool"] == "python" and Path(entry["argv"][0]).name == "materialize_model_source.py"
            )
            self.assertEqual(self.value_after(model_entry["argv"], "--wandb-registry-path"), "explicit/project/custom-paper:production")
            train_entry = next(
                entry for entry in entries if entry["tool"] == "python" and Path(entry["argv"][0]).name == "train_protein_grpo.py"
            )
            self.assertEqual(self.value_after(train_entry["argv"], "--wandb_project"), "explicit-project")
            self.assertEqual(self.value_after(train_entry["argv"], "--queries_per_step"), "8")
            self.assertEqual(self.value_after(train_entry["argv"], "--rollouts_per_query"), "24")
            self.assertEqual(self.value_after(train_entry["argv"], "--gradient_accumulation_steps"), "2")
            self.assertEqual(self.value_after(train_entry["argv"], "--max_new_tokens"), "10000")
            self.assertEqual(self.value_after(train_entry["argv"], "--vllm_attention_backend"), "XFORMERS")
            self.assertEqual(self.value_after(train_entry["argv"], "--vllm_worker_multiproc_method"), "spawn")
            self.assertEqual(self.value_after(train_entry["argv"], "--vllm_enable_sleep_mode"), "false")
            self.assertEqual(self.value_after(train_entry["argv"], "--vllm_use_v1"), "0")
            self.assertFalse(any(entry["tool"] == "deepspeed" for entry in entries))
            asset_keys = {
                self.value_after(entry["argv"], "--asset-key")
                for entry in entries
                if entry["tool"] == "python" and Path(entry["argv"][0]).name == "materialize_data_bundle.py"
            }
            self.assertEqual(asset_keys, {"reasoning_dataset", "temporal_split_artifact", "ia_file"})

    def test_wrapper_requires_master_env_when_hostfile_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = self.create_wrapper_harness(Path(tmpdir))
            env = os.environ.copy()
            env.update(harness)

            result = subprocess.run(
                ["bash", str(WRAPPER_PATH)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("MASTER_ADDR is required", result.stdout)

    def test_wrapper_uses_explicit_multinode_deepspeed_args_without_hostfile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = self.create_wrapper_harness(Path(tmpdir))
            env = os.environ.copy()
            env.update(harness)
            env.update(
                {
                    "MASTER_ADDR": "10.0.0.1",
                    "MASTER_PORT": "29500",
                    "NODE_RANK": "0",
                }
            )

            result = subprocess.run(
                ["bash", str(WRAPPER_PATH)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            entries = self.read_tool_log(Path(harness["FAKE_LOG_PATH"]))
            deepspeed_entry = next(entry for entry in entries if entry["tool"] == "deepspeed")
            self.assertIn("--num_nodes", deepspeed_entry["argv"])
            self.assertIn("--num_gpus", deepspeed_entry["argv"])
            self.assertIn("--no_ssh", deepspeed_entry["argv"])
            self.assertEqual(self.value_after(deepspeed_entry["argv"], "--master_addr"), "10.0.0.1")
            self.assertEqual(self.value_after(deepspeed_entry["argv"], "--master_port"), "29500")
            self.assertEqual(self.value_after(deepspeed_entry["argv"], "--node_rank"), "0")
            self.assertEqual(self.value_after(deepspeed_entry["argv"], "--queries_per_step"), "8")
            self.assertEqual(self.value_after(deepspeed_entry["argv"], "--rollouts_per_query"), "24")
            self.assertEqual(self.value_after(deepspeed_entry["argv"], "--gradient_accumulation_steps"), "2")
            self.assertEqual(self.value_after(deepspeed_entry["argv"], "--max_new_tokens"), "10000")
            self.assertEqual(self.value_after(deepspeed_entry["argv"], "--vllm_attention_backend"), "XFORMERS")
            self.assertEqual(self.value_after(deepspeed_entry["argv"], "--vllm_worker_multiproc_method"), "spawn")
            self.assertEqual(self.value_after(deepspeed_entry["argv"], "--vllm_enable_sleep_mode"), "false")
            self.assertEqual(self.value_after(deepspeed_entry["argv"], "--vllm_use_v1"), "0")
            self.assertNotIn("--hostfile", deepspeed_entry["argv"])

    def test_wrapper_auto_generates_hostfile_from_slurm_nodelist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = self.create_wrapper_harness(Path(tmpdir))
            env = os.environ.copy()
            env.update(harness)
            env.update(
                {
                    "MASTER_ADDR": "10.0.0.1",
                    "MASTER_PORT": "29500",
                    "NODE_RANK": "1",
                    "SLURM_JOB_ID": "4242",
                    "SLURM_JOB_NODELIST": "worker-[0-1]",
                }
            )

            result = subprocess.run(
                ["bash", str(WRAPPER_PATH)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            self.assertIn("auto-generated DeepSpeed hostfile", result.stdout)
            entries = self.read_tool_log(Path(harness["FAKE_LOG_PATH"]))
            deepspeed_entry = next(entry for entry in entries if entry["tool"] == "deepspeed")
            generated_hostfile = self.value_after(deepspeed_entry["argv"], "--hostfile")
            self.assertTrue(generated_hostfile.endswith("deepspeed_hosts.4242.txt"))
            self.assertIn("--no_ssh", deepspeed_entry["argv"])
            self.assertEqual(self.value_after(deepspeed_entry["argv"], "--node_rank"), "1")
            hostfile_text = Path(generated_hostfile).read_text(encoding="utf-8")
            self.assertEqual(hostfile_text, "worker-0 slots=8\nworker-1 slots=8\n")

    def test_wrapper_uses_hostfile_when_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = self.create_wrapper_harness(Path(tmpdir))
            env = os.environ.copy()
            env.update(harness)
            env["HOSTFILE"] = harness["HOSTFILE_PATH"]

            result = subprocess.run(
                ["bash", str(WRAPPER_PATH)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            entries = self.read_tool_log(Path(harness["FAKE_LOG_PATH"]))
            deepspeed_entry = next(entry for entry in entries if entry["tool"] == "deepspeed")
            self.assertEqual(self.value_after(deepspeed_entry["argv"], "--hostfile"), harness["HOSTFILE_PATH"])
            self.assertNotIn("--num_nodes", deepspeed_entry["argv"])
            self.assertNotIn("--no_ssh", deepspeed_entry["argv"])

    def test_wrapper_uses_hostfile_with_no_ssh_when_node_rank_is_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = self.create_wrapper_harness(Path(tmpdir))
            env = os.environ.copy()
            env.update(harness)
            env.update(
                {
                    "HOSTFILE": harness["HOSTFILE_PATH"],
                    "MASTER_ADDR": "10.0.0.1",
                    "MASTER_PORT": "29500",
                    "NODE_RANK": "1",
                }
            )

            result = subprocess.run(
                ["bash", str(WRAPPER_PATH)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            entries = self.read_tool_log(Path(harness["FAKE_LOG_PATH"]))
            deepspeed_entry = next(entry for entry in entries if entry["tool"] == "deepspeed")
            self.assertEqual(self.value_after(deepspeed_entry["argv"], "--hostfile"), harness["HOSTFILE_PATH"])
            self.assertIn("--no_ssh", deepspeed_entry["argv"])
            self.assertEqual(self.value_after(deepspeed_entry["argv"], "--master_addr"), "10.0.0.1")
            self.assertEqual(self.value_after(deepspeed_entry["argv"], "--master_port"), "29500")
            self.assertEqual(self.value_after(deepspeed_entry["argv"], "--node_rank"), "1")
            self.assertNotIn("--num_nodes", deepspeed_entry["argv"])

    def test_run_tracker_initializes_weave_with_cache_dir_and_run_name_attributes(self):
        fake_weave = FakeWeaveModule()
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {}, clear=True):
            output_dir = Path(tmpdir) / "train-output"
            args = GRPO.parse_args(
                [
                    "--text_model_name",
                    "/tmp/demo-model",
                    "--output_dir",
                    str(output_dir),
                ]
            )
            args.trace_rollouts_to_weave = True
            args.weave_project = "demo-entity/demo-project"
            args.run_name = "demo-run"
            runtime = GRPO.DistributedRuntime(enabled=False, rank=0, world_size=1, local_rank=0, device="cpu")
            observed: dict[str, str] = {}

            def fake_wandb_init(_tracker_self, _config):
                observed["cache_dir_at_wandb_init"] = os.environ.get("WEAVE_SERVER_CACHE_DIR", "")
                return None

            with mock.patch.object(GRPO, "weave", fake_weave), mock.patch.object(GRPO.RunTracker, "_maybe_init_wandb", fake_wandb_init):
                tracker = GRPO.RunTracker(args=args, config={}, output_dir=output_dir, runtime=runtime)

            self.assertTrue(callable(tracker.weave_trace_fn))
            self.assertEqual(os.environ["WEAVE_SERVER_CACHE_DIR"], str((output_dir / "weave_server_cache").resolve()))
            self.assertEqual(observed["cache_dir_at_wandb_init"], str((output_dir / "weave_server_cache").resolve()))
            self.assertEqual(fake_weave.init_calls[0]["project"], "demo-entity/demo-project")
            self.assertEqual(fake_weave.init_calls[0]["global_attributes"]["run_name"], "demo-run")
            self.assertEqual(fake_weave.init_calls[0]["global_attributes"]["job_type"], "train_rl")

    def test_run_tracker_infers_wandb_identity_and_logs_startup_metrics(self):
        fake_wandb = FakeWandbModule()
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict(
            sys.modules, {"wandb": fake_wandb}
        ):
            output_dir = Path(tmpdir) / "train-output"
            args = GRPO.parse_args(
                [
                    "--text_model_name",
                    "/tmp/demo-model",
                    "--output_dir",
                    str(output_dir),
                    "--base_checkpoint",
                    "wandb-healthcare/bioreasoning-pro/bioreason-pro-rl-paper:production",
                ]
            )
            args.run_name = "demo-run"
            args.trace_rollouts_to_weave = False
            runtime = GRPO.DistributedRuntime(enabled=False, rank=0, world_size=1, local_rank=0, device="cpu")
            config = {
                "queries_per_step": 8,
                "rollouts_per_query": 24,
                "total_trajectories_per_step": 192,
                "max_new_tokens": 10000,
                "steps_per_generation": 2,
                "num_iterations": 1,
            }

            tracker = GRPO.RunTracker(args=args, config=config, output_dir=output_dir, runtime=runtime)

            self.assertEqual(fake_wandb.init_calls[0]["entity"], "wandb-healthcare")
            self.assertEqual(fake_wandb.init_calls[0]["project"], "bioreasoning-pro")
            self.assertEqual(fake_wandb.init_calls[0]["dir"], str((output_dir / "wandb").resolve()))
            self.assertEqual(args.wandb_entity, "wandb-healthcare")
            self.assertEqual(args.wandb_project, "bioreasoning-pro")
            self.assertEqual(tracker.wandb_run_path, "wandb-healthcare/bioreasoning-pro/demo123")
            self.assertTrue((output_dir / "wandb_run_info.json").exists())
            define_metric_names = [item["name"] for item in fake_wandb.runs[0].define_metric_calls]
            self.assertIn("train/global_step", define_metric_names)
            self.assertIn("train/*", define_metric_names)
            self.assertIn("timing/*", define_metric_names)
            self.assertNotIn("paper/*", define_metric_names)
            self.assertNotIn("runtime/*", define_metric_names)
            startup_payload = fake_wandb.runs[0].logged[0]["payload"]
            self.assertEqual(fake_wandb.runs[0].logged[0]["step"], 0)
            self.assertEqual(startup_payload["train/global_step"], 0)
            self.assertEqual(startup_payload["system/wandb_initialized"], 1.0)
            self.assertEqual(startup_payload["system/weave_enabled"], 0.0)
            self.assertNotIn("runtime/rollouts_per_query", startup_payload)
            self.assertNotIn("paper/runtime_deviation_from_spec", startup_payload)

    def test_log_metrics_adds_namespaced_wandb_series(self):
        fake_wandb = FakeWandbModule()
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict(
            sys.modules, {"wandb": fake_wandb}
        ):
            output_dir = Path(tmpdir) / "train-output"
            args = GRPO.parse_args(
                [
                    "--text_model_name",
                    "/tmp/demo-model",
                    "--output_dir",
                    str(output_dir),
                    "--base_checkpoint",
                    "wandb-healthcare/bioreasoning-pro/bioreason-pro-rl-paper:production",
                ]
            )
            args.run_name = "demo-run"
            args.trace_rollouts_to_weave = False
            runtime = GRPO.DistributedRuntime(enabled=False, rank=0, world_size=1, local_rank=0, device="cpu")
            tracker = GRPO.RunTracker(args=args, config={}, output_dir=output_dir, runtime=runtime)

            tracker.log_metrics(
                {
                    "reward_mean": 0.25,
                    "loss_mean": 0.125,
                    "validation_reward_mean": 0.5,
                    "timing_rollout_seconds": 12.0,
                },
                step=7,
            )

            payload = fake_wandb.runs[0].logged[-1]["payload"]
            self.assertEqual(fake_wandb.runs[0].logged[-1]["step"], 7)
            self.assertEqual(payload["train/global_step"], 7)
            self.assertNotIn("reward_mean", payload)
            self.assertEqual(payload["train/reward_mean"], 0.25)
            self.assertEqual(payload["train/loss_mean"], 0.125)
            self.assertEqual(payload["validation/reward_mean"], 0.5)
            self.assertEqual(payload["timing/rollout_seconds"], 12.0)

    def test_log_rollout_trace_includes_run_name_in_jsonl(self):
        fake_weave = FakeWeaveModule()
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {}, clear=True):
            output_dir = Path(tmpdir) / "train-output"
            args = GRPO.parse_args(
                [
                    "--text_model_name",
                    "/tmp/demo-model",
                    "--output_dir",
                    str(output_dir),
                ]
            )
            args.trace_rollouts_to_weave = True
            args.weave_project = "demo-entity/demo-project"
            args.run_name = "demo-run"
            runtime = GRPO.DistributedRuntime(enabled=False, rank=0, world_size=1, local_rank=0, device="cpu")

            with mock.patch.object(GRPO, "weave", fake_weave), mock.patch.object(
                GRPO.RunTracker, "_maybe_init_wandb", return_value=None
            ):
                tracker = GRPO.RunTracker(args=args, config={}, output_dir=output_dir, runtime=runtime)
                tracker.log_rollout_trace(
                    {
                        "step": 3,
                        "rank": 0,
                        "split": "train",
                        "protein_id": "P12345",
                        "rollout_idx": 2,
                        "reward": 1.0,
                    },
                    trace_to_weave=True,
                )

            lines = tracker.trace_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["run_name"], "demo-run")
            self.assertEqual(payload["job_type"], "train_rl")
            self.assertEqual(fake_weave.trace_payloads, [])
            self.assertEqual(fake_weave.attribute_calls, [])

    def test_trace_rollout_call_uses_weave_op_with_input_and_output(self):
        fake_weave = FakeWeaveModule()
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {}, clear=True):
            output_dir = Path(tmpdir) / "train-output"
            args = GRPO.parse_args(
                [
                    "--text_model_name",
                    "/tmp/demo-model",
                    "--output_dir",
                    str(output_dir),
                ]
            )
            args.trace_rollouts_to_weave = True
            args.weave_project = "demo-entity/demo-project"
            args.run_name = "demo-run"
            runtime = GRPO.DistributedRuntime(enabled=False, rank=0, world_size=1, local_rank=0, device="cpu")
            query = GRPO.PreparedQuery(
                input_ids=None,
                attention_mask=None,
                protein_sequences=["MSEQ"],
                batch_idx_map=[0],
                structure_coords=None,
                go_aspects=["bp"],
                sample_meta={
                    "protein_id": "P12345",
                    "split": "train",
                    "go_bp": "GO:0000001",
                    "reasoning": "teacher reasoning",
                    "final_answer": "teacher answer",
                },
                prompt_text="Predict GO terms.",
                multimodal_cache=None,
            )
            sampling = GRPO.SamplingSpec(max_new_tokens=32)
            tokenizer = types.SimpleNamespace(
                encode=lambda text, add_special_tokens=False: [ord(ch) % 17 for ch in str(text)],
            )

            with mock.patch.object(GRPO, "weave", fake_weave), mock.patch.object(
                GRPO.RunTracker, "_maybe_init_wandb", return_value=None
            ):
                tracker = GRPO.RunTracker(args=args, config={}, output_dir=output_dir, runtime=runtime)
                outputs = tracker.trace_rollout_call(
                    step=4,
                    split="train",
                    query=query,
                    repeat_count=3,
                    sampling=sampling,
                    generator=lambda: ["<|FINAL_ANSWER|>GO:0000001<|/FINAL_ANSWER|>"],
                    tokenizer=tokenizer,
                )

            self.assertEqual(outputs, ["<|FINAL_ANSWER|>GO:0000001<|/FINAL_ANSWER|>"])
            self.assertEqual(fake_weave.trace_payloads[0]["run_name"], "demo-run")
            self.assertEqual(fake_weave.trace_payloads[0]["query"]["prompt_text"], "Predict GO terms.")
            self.assertEqual(fake_weave.trace_payloads[0]["repeat_count"], 3)
            self.assertEqual(fake_weave.trace_payloads[0]["stage"], "rollout")
            self.assertEqual(fake_weave.trace_payloads[0]["query"]["sample_meta"], {"protein_id": "P12345", "split": "train"})
            self.assertNotIn("target_go_ids", fake_weave.trace_payloads[0]["query"])
            self.assertNotIn("reasoning", fake_weave.trace_payloads[0]["query"]["sample_meta"])
            self.assertNotIn("final_answer", fake_weave.trace_payloads[0]["query"]["sample_meta"])
            self.assertEqual(fake_weave.attribute_calls[0]["run_name"], "demo-run")
            self.assertEqual(fake_weave.attribute_calls[0]["step"], 4)
            self.assertEqual(fake_weave.trace_results[0]["output_count"], 1)
            self.assertEqual(
                fake_weave.trace_results[0]["output_char_lengths"],
                [len("<|FINAL_ANSWER|>GO:0000001<|/FINAL_ANSWER|>")],
            )
            self.assertEqual(fake_weave.trace_results[0]["output_word_lengths"], [1])
            self.assertEqual(
                fake_weave.trace_results[0]["output_token_lengths"],
                [len("<|FINAL_ANSWER|>GO:0000001<|/FINAL_ANSWER|>")],
            )
            self.assertEqual(fake_weave.trace_results[0]["output_length_summary"]["chars"]["count"], 1)
            self.assertEqual(fake_weave.trace_results[0]["output_length_summary"]["tokens"]["count"], 1)
            self.assertEqual(fake_weave.trace_results[0]["output_format_valid"], [True])
            self.assertEqual(fake_weave.trace_results[0]["output_has_final_answer_tag"], [True])
            self.assertEqual(fake_weave.trace_results[0]["output_has_go_summary_block"], [False])
            self.assertEqual(fake_weave.trace_results[0]["output_uses_alt_final_answer_close_tag"], [False])
            self.assertEqual(fake_weave.trace_results[0]["output_has_unclosed_final_answer_tag"], [False])
            self.assertEqual(fake_weave.trace_results[0]["output_has_repeated_final_answer_open_tag"], [False])
            self.assertEqual(fake_weave.trace_results[0]["output_has_tool_call_residue"], [False])
            self.assertEqual(fake_weave.trace_results[0]["output_has_think_residue"], [False])
            self.assertEqual(fake_weave.trace_results[0]["output_parsed_go_counts"], [1])
            self.assertEqual(fake_weave.trace_results[0]["output_format_summary"]["format_valid"]["true_count"], 1)
            self.assertEqual(fake_weave.trace_results[0]["output_format_summary"]["alt_final_answer_close_tag"]["true_count"], 0)
            self.assertEqual(tracker.weave_remaining_budget, args.weave_trace_budget - 1)

    def test_trace_reward_call_uses_stage_specific_targets(self):
        fake_weave = FakeWeaveModule()
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {}, clear=True):
            output_dir = Path(tmpdir) / "train-output"
            args = GRPO.parse_args(
                [
                    "--text_model_name",
                    "/tmp/demo-model",
                    "--output_dir",
                    str(output_dir),
                ]
            )
            args.trace_rollouts_to_weave = True
            args.weave_project = "demo-entity/demo-project"
            args.run_name = "demo-run"
            runtime = GRPO.DistributedRuntime(enabled=False, rank=0, world_size=1, local_rank=0, device="cpu")
            query = GRPO.PreparedQuery(
                input_ids=None,
                attention_mask=None,
                protein_sequences=["MSEQ"],
                batch_idx_map=[0],
                structure_coords=None,
                go_aspects=["bp"],
                sample_meta={"protein_id": "P12345", "go_bp": "GO:0000001", "reasoning": "teacher"},
                prompt_text="Predict GO terms.",
                multimodal_cache=None,
            )

            with mock.patch.object(GRPO, "weave", fake_weave), mock.patch.object(
                GRPO.RunTracker, "_maybe_init_wandb", return_value=None
            ):
                tracker = GRPO.RunTracker(args=args, config={}, output_dir=output_dir, runtime=runtime)
                rewards = tracker.trace_reward_call(
                    step=5,
                    split="train",
                    query=query,
                    completions=["<|FINAL_ANSWER|>GO:0000001<|/FINAL_ANSWER|>"],
                    callback=lambda: [1.0],
                )

            self.assertEqual(rewards, [1.0])
            self.assertEqual(fake_weave.trace_payloads[0]["stage"], "reward")
            self.assertEqual(fake_weave.trace_payloads[0]["target_go_ids"], ["GO:0000001"])
            self.assertEqual(fake_weave.trace_payloads[0]["completions"], ["<|FINAL_ANSWER|>GO:0000001<|/FINAL_ANSWER|>"])
            self.assertEqual(fake_weave.attribute_calls[0]["stage"], "reward")

    def test_trace_policy_update_call_uses_stage_specific_op(self):
        fake_weave = FakeWeaveModule()
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {}, clear=True):
            output_dir = Path(tmpdir) / "train-output"
            args = GRPO.parse_args(
                [
                    "--text_model_name",
                    "/tmp/demo-model",
                    "--output_dir",
                    str(output_dir),
                ]
            )
            args.trace_rollouts_to_weave = True
            args.weave_project = "demo-entity/demo-project"
            args.run_name = "demo-run"
            runtime = GRPO.DistributedRuntime(enabled=False, rank=0, world_size=1, local_rank=0, device="cpu")

            with mock.patch.object(GRPO, "weave", fake_weave), mock.patch.object(
                GRPO.RunTracker, "_maybe_init_wandb", return_value=None
            ):
                tracker = GRPO.RunTracker(args=args, config={}, output_dir=output_dir, runtime=runtime)
                summary = tracker.trace_policy_update_call(
                    step=6,
                    split="train",
                    payload={"steps_per_generation": 2, "valid_rollout_counts": [24]},
                    callback=lambda: {"update_chunk_count": 8, "policy_loss_mean": 0.12},
                )

            self.assertEqual(summary["stage"], "policy_update")
            self.assertEqual(summary["update_chunk_count"], 8)
            self.assertEqual(fake_weave.trace_payloads[0]["stage"], "policy_update")
            self.assertEqual(fake_weave.attribute_calls[0]["stage"], "policy_update")

    def test_finish_flushes_weave_client(self):
        fake_weave = FakeWeaveModule()
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {}, clear=True):
            output_dir = Path(tmpdir) / "train-output"
            args = GRPO.parse_args(
                [
                    "--text_model_name",
                    "/tmp/demo-model",
                    "--output_dir",
                    str(output_dir),
                ]
            )
            args.trace_rollouts_to_weave = True
            args.weave_project = "demo-entity/demo-project"
            args.run_name = "demo-run"
            runtime = GRPO.DistributedRuntime(enabled=False, rank=0, world_size=1, local_rank=0, device="cpu")

            with mock.patch.object(GRPO, "weave", fake_weave), mock.patch.object(
                GRPO.RunTracker, "_maybe_init_wandb", return_value=None
            ):
                tracker = GRPO.RunTracker(args=args, config={}, output_dir=output_dir, runtime=runtime)
                tracker.finish()

            self.assertTrue(fake_weave.client.flush_called)


if __name__ == "__main__":
    unittest.main()

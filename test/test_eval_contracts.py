import argparse
import asyncio
import contextlib
import csv
import importlib.util
import json
import importlib
import importlib.machinery
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "eval.py"


def install_eval_test_stubs():
    torch_module = types.ModuleType("torch")
    torch_module.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)

    class _CudaModule:
        OutOfMemoryError = RuntimeError

        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

    @contextlib.contextmanager
    def inference_mode():
        yield

    torch_module.cuda = _CudaModule
    torch_module.inference_mode = inference_mode

    bioreason2_module = types.ModuleType("bioreason2")
    models_module = types.ModuleType("bioreason2.models")
    protein_vllm_module = types.ModuleType("bioreason2.models.protein_vllm")
    dataset_module = types.ModuleType("bioreason2.dataset")
    cafa5_module = types.ModuleType("bioreason2.dataset.cafa5")
    cafa5_module.__path__ = []
    cafa5_load_module = types.ModuleType("bioreason2.dataset.cafa5.load")
    cafa5_subset_module = types.ModuleType("bioreason2.dataset.cafa5.subset")
    utils_module = types.ModuleType("bioreason2.utils")
    wandb_module = types.ModuleType("wandb")
    weave_module = types.ModuleType("weave")

    class ProteinLLMModel:
        pass

    def load_cafa5_dataset(*args, **kwargs):
        raise AssertionError("Tests should patch load_cafa5_dataset explicitly")

    def str2bool(value):
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"1", "true", "yes", "y"}

    def maybe_use_artifact_refs(run, artifact_refs, artifact_types=None):
        if run is None:
            return {}
        statuses = {}
        for key, value in artifact_refs.items():
            if value:
                artifact_type = None if artifact_types is None else artifact_types.get(key)
                run.use_artifact(value, type=artifact_type)
                statuses[key] = {"used": True, "artifact_ref": value, "artifact_type": artifact_type}
            else:
                statuses[key] = {"used": False, "artifact_ref": value, "artifact_type": None}
        return statuses

    def select_dataset_subset(dataset, max_samples, seed, strategy="stratified_aspect_profile"):
        if max_samples is None or max_samples < 0 or len(dataset) <= max_samples:
            return dataset, {
                "strategy": "full",
                "requested_samples": max_samples,
                "selected_samples": len(dataset),
            }
        return dataset.select(range(max_samples)), {
            "strategy": strategy,
            "requested_samples": max_samples,
            "selected_samples": max_samples,
        }

    protein_vllm_module.ProteinLLMModel = ProteinLLMModel
    cafa5_load_module.load_cafa5_dataset = load_cafa5_dataset
    cafa5_subset_module.select_dataset_subset = select_dataset_subset
    utils_module.str2bool = str2bool
    utils_module.maybe_use_artifact_refs = maybe_use_artifact_refs

    class FakeTable:
        def __init__(self, columns=None, data=None):
            self.columns = list(columns or [])
            self.data = list(data or [])

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
            self.artifacts = []
            self.used_artifact_refs = []
            self.finished = False
            self.config = dict(kwargs.get("config", {}))

        def log_artifact(self, artifact):
            self.artifacts.append(artifact)

        def use_artifact(self, artifact_ref, type=None):
            self.used_artifact_refs.append({"artifact_ref": artifact_ref, "type": type})

        def finish(self):
            self.finished = True

    wandb_module.Table = FakeTable
    wandb_module.Artifact = FakeArtifact
    wandb_module.init_calls = []
    wandb_module.logged_payloads = []
    wandb_module.last_run = None

    def wandb_init(**kwargs):
        wandb_module.init_calls.append(kwargs)
        wandb_module.last_run = FakeRun(**kwargs)
        return wandb_module.last_run

    def wandb_log(payload):
        wandb_module.logged_payloads.append(payload)

    wandb_module.init = wandb_init
    wandb_module.log = wandb_log

    weave_module.op_calls = []

    def weave_op(func=None, name=None, **kwargs):
        def decorator(actual):
            def wrapped(*args, **inner_kwargs):
                result = actual(*args, **inner_kwargs)
                weave_module.op_calls.append(
                    {
                        "name": name or actual.__name__,
                        "args": args,
                        "kwargs": inner_kwargs,
                        "result": result,
                    }
                )
                return result

            wrapped.__name__ = actual.__name__
            return wrapped

        if func is None:
            return decorator
        return decorator(func)

    class FakeEvaluation:
        instances = []
        async_mode = False

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.dataset = kwargs.get("dataset", [])
            self.scorers = kwargs.get("scorers", [])
            self.preprocess_model_input = kwargs.get("preprocess_model_input", lambda row: row)
            self.last_result = None
            FakeEvaluation.instances.append(self)

        def _run_evaluation(self, model):
            rows = []
            for row in self.dataset:
                model_inputs = self.preprocess_model_input(row)
                model_output = model(**model_inputs)
                scorer_outputs = []
                for scorer in self.scorers:
                    scorer_outputs.append(
                        scorer(
                            expected_output=row.get("expected_output", ""),
                            model_output=model_output,
                        )
                    )
                rows.append({"row": row, "model_output": model_output, "scores": scorer_outputs})
            self.last_result = {"rows": rows}
            return self.last_result

        def evaluate(self, model):
            if self.async_mode:
                async def _evaluate_async():
                    return self._run_evaluation(model)

                return _evaluate_async()
            return self._run_evaluation(model)

    weave_module.init_calls = []
    weave_module.flush_calls = 0
    weave_module.op = weave_op
    weave_module.Evaluation = FakeEvaluation

    def weave_init(project_name):
        weave_module.init_calls.append(project_name)
        return types.SimpleNamespace(
            project=project_name,
            flush=lambda: setattr(weave_module, "flush_calls", weave_module.flush_calls + 1),
        )

    weave_module.init = weave_init

    sys.modules["torch"] = torch_module
    try:
        real_tqdm = importlib.import_module("tqdm")
    except ImportError:
        tqdm_module = types.ModuleType("tqdm")
        tqdm_module.__path__ = []
        tqdm_module.tqdm = lambda iterable, **kwargs: iterable
        tqdm_auto_module = types.ModuleType("tqdm.auto")
        tqdm_auto_module.tqdm = tqdm_module.tqdm
        tqdm_contrib_module = types.ModuleType("tqdm.contrib")
        tqdm_concurrent_module = types.ModuleType("tqdm.contrib.concurrent")
        tqdm_concurrent_module.thread_map = lambda fn, data, **kwargs: [fn(x) for x in data]
        sys.modules["tqdm"] = tqdm_module
        sys.modules["tqdm.auto"] = tqdm_auto_module
        sys.modules["tqdm.contrib"] = tqdm_contrib_module
        sys.modules["tqdm.contrib.concurrent"] = tqdm_concurrent_module
    else:
        sys.modules["tqdm"] = real_tqdm
    sys.modules["bioreason2"] = bioreason2_module
    sys.modules["bioreason2.models"] = models_module
    sys.modules["bioreason2.models.protein_vllm"] = protein_vllm_module
    sys.modules["bioreason2.dataset"] = dataset_module
    sys.modules["bioreason2.dataset.cafa5"] = cafa5_module
    sys.modules["bioreason2.dataset.cafa5.load"] = cafa5_load_module
    sys.modules["bioreason2.dataset.cafa5.subset"] = cafa5_subset_module
    sys.modules["bioreason2.utils"] = utils_module
    sys.modules["wandb"] = wandb_module
    sys.modules["weave"] = weave_module


def load_eval_module():
    install_eval_test_stubs()
    module_name = "eval_contracts_test_module"
    spec = importlib.util.spec_from_file_location(module_name, EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EVAL = load_eval_module()


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def shuffle(self, seed=None):
        return FakeDataset(self.rows)

    def select(self, indices):
        return FakeDataset([self.rows[i] for i in indices])

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)


def make_eval_args(**overrides):
    base = dict(
        ckpt_dir="/tmp/mock-ckpt",
        protein_model_name="esm3_sm_open_v1",
        protein_embedding_layer=37,
        go_obo_path="/tmp/go-basic.obo",
        precomputed_embeddings_path="/tmp/go_embeddings",
        unified_go_encoder=True,
        go_hidden_dim=512,
        go_num_gat_layers=3,
        go_num_heads=8,
        go_num_reduced_embeddings=200,
        go_embedding_dim=2560,
        cafa5_dataset="wanglab/cafa5",
        cafa5_dataset_name="disease_temporal_hc_reasoning_v1",
        cafa5_dataset_subset=None,
        dataset_cache_dir=None,
        structure_dir=None,
        include_go_defs=False,
        interpro_dataset_name="interpro_metadata",
        split_go_aspects=False,
        interpro_in_prompt=True,
        predict_interpro=False,
        ppi_in_prompt=False,
        include_protein_function_summary=True,
        val_split_ratio=0.1,
        seed=23,
        debug=False,
        max_length_protein=2048,
        enable_thinking=True,
        reasoning_dataset_name="disease_temporal_hc_reasoning_v1",
        go_gpt_predictions_column="go_pred",
        min_go_mf_freq=1,
        min_go_bp_freq=1,
        min_go_cc_freq=1,
        apply_go_filtering_to_val_test=False,
        add_uniprot_summary=True,
        eval_split="validation",
        max_samples=-1,
        sample_strategy="stratified_aspect_profile",
        max_model_len=32768,
        max_new_tokens=1024,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.0,
        pass_at_k=1,
        num_chunks=1,
        chunk_id=0,
        evals_path="/tmp/evals",
        benchmark_version=None,
        model_name=None,
        temporal_split_artifact=None,
        dataset_artifact=None,
        model_artifact=None,
        shortlist_query=None,
        shortlist_mode=None,
        train_start_release=None,
        train_end_release=None,
        dev_end_release=None,
        test_end_release=None,
        metrics_summary_path=None,
        ia_file_path=None,
        metric_threads=0,
        metric_threshold_step=0.99,
        metrics_final_answer_only=True,
        reasoning_metrics_mode=None,
        wandb_project=None,
        wandb_entity=None,
        wandb_run_name=None,
        wandb_artifact_name=None,
        wandb_dir=None,
        wandb_mode=None,
        weave_project=None,
        weave_eval_name=None,
        keep_local_eval_outputs=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class EvalContractTests(unittest.TestCase):
    def test_maybe_init_eval_tracking_initializes_wandb_and_weave_early(self):
        EVAL.wandb.init_calls.clear()
        EVAL.weave.init_calls.clear()
        EVAL.weave.op_calls.clear()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = make_eval_args(
                evals_path=tmpdir,
                wandb_project="bioreason-pro",
                wandb_entity="demo-entity",
                weave_project="demo-entity/bioreason-pro",
            )
            tracking_state = EVAL.maybe_init_eval_tracking(args)

            self.assertIsNotNone(tracking_state["wandb_run"])
            self.assertIsNotNone(tracking_state["weave_client"])
            self.assertTrue(callable(tracking_state["trace_sample"]))
            self.assertEqual(EVAL.wandb.init_calls[0]["job_type"], "eval")
            self.assertEqual(EVAL.weave.init_calls[0], "demo-entity/bioreason-pro")

    def test_maybe_trace_eval_sample_emits_weave_payload(self):
        EVAL.weave.op_calls.clear()
        args = make_eval_args(eval_split="test", benchmark_version="213 -> 221 -> 225 -> 228")
        tracking_state = {"trace_sample": lambda payload: EVAL.weave.op_calls.append({"name": "eval_sample_inference", "result": payload})}
        sample = {
            "prompt": [{"role": "assistant", "reasoning_content": "gold", "content": [{"text": "GO:1"}]}],
        }
        result_record = {
            "success": True,
            "ground_truth": "<think>gold</think>\nGO:1",
            "generated_response": "<think>trace</think>\nGO:1",
            "input_prompt": "prompt",
        }

        EVAL.maybe_trace_eval_sample(
            tracking_state,
            args=args,
            sample=sample,
            protein_id="P1",
            go_aspect="molecular_function",
            attempt_idx=0,
            result_record=result_record,
        )

        self.assertEqual(tracking_state["sample_traces_logged"], 1)
        self.assertEqual(EVAL.weave.op_calls[-1]["result"]["protein_id"], "P1")
        self.assertEqual(EVAL.weave.op_calls[-1]["result"]["final_answer"], "GO:1")

    def test_argument_parser_defaults_to_validation_split(self):
        parser = EVAL.setup_argument_parser()
        args = parser.parse_args(
            [
                "--ckpt_dir",
                "/tmp/mock-ckpt",
                "--go_obo_path",
                "/tmp/go-basic.obo",
                "--precomputed_embeddings_path",
                "/tmp/go-emb",
                "--evals_path",
                "/tmp/evals",
            ]
        )

        self.assertEqual(args.eval_split, "validation")

    def test_argument_parser_allows_missing_precomputed_embeddings_path(self):
        parser = EVAL.setup_argument_parser()
        args = parser.parse_args(
            [
                "--ckpt_dir",
                "/tmp/mock-ckpt",
                "--go_obo_path",
                "/tmp/go-basic.obo",
                "--evals_path",
                "/tmp/evals",
            ]
        )

        self.assertIsNone(args.precomputed_embeddings_path)

    def test_initialize_model_treats_empty_precomputed_embeddings_path_as_none(self):
        args = make_eval_args(precomputed_embeddings_path="")

        with mock.patch.object(EVAL, "ProteinLLMModel") as mock_model_cls:
            mock_model = object()
            mock_model_cls.return_value = mock_model
            result = EVAL.initialize_model(args)

        self.assertIs(result, mock_model)
        self.assertEqual(mock_model_cls.call_args.kwargs["precomputed_embeddings_path"], None)

    def test_select_eval_dataset_supports_validation_and_test(self):
        train_ds = FakeDataset([{"protein_id": "train"}])
        val_ds = FakeDataset([{"protein_id": "val"}])
        test_ds = FakeDataset([{"protein_id": "test"}])

        selected_validation = EVAL.select_eval_dataset(train_ds, val_ds, test_ds, "validation")
        selected_test = EVAL.select_eval_dataset(train_ds, val_ds, test_ds, "test")

        self.assertEqual(len(selected_validation), 1)
        self.assertEqual(list(selected_validation)[0]["protein_id"], "val")
        self.assertEqual(list(selected_test)[0]["protein_id"], "test")

        with self.assertRaises(ValueError):
            EVAL.select_eval_dataset(train_ds, val_ds, test_ds, "train")

    def test_load_dataset_uses_requested_eval_split(self):
        train_ds = FakeDataset([{"protein_id": "train"}])
        val_ds = FakeDataset(
            [
                {"protein_id": "val1", "go_aspect": "molecular_function"},
                {"protein_id": "val2", "go_aspect": "biological_process"},
            ]
        )
        test_ds = FakeDataset(
            [
                {"protein_id": "test1", "go_aspect": "molecular_function"},
                {"protein_id": "test2", "go_aspect": "cellular_component"},
                {"protein_id": "test3", "go_aspect": "biological_process"},
            ]
        )

        args = make_eval_args(eval_split="test", max_samples=2, sample_strategy="shuffled_prefix")

        with mock.patch.object(EVAL, "load_cafa5_dataset", return_value=(train_ds, val_ds, test_ds)):
            samples = EVAL.load_dataset(args)

        self.assertEqual(len(samples), 2)
        self.assertEqual([row["protein_id"] for row in samples], ["test1", "test2"])

    def test_collect_and_write_eval_artifacts(self):
        sample_record = {
            "protein_id": "P12345",
            "go_aspect": "molecular_function",
            "ground_truth": "GO:0001111",
            "generated_response": "GO:0002222",
            "success": True,
            "input_prompt": "prompt",
            "sequence_length": 321,
        }
        sample_record_k1 = {
            **sample_record,
            "generated_response": "GO:0003333",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "P12345_MF_k00.json").write_text(json.dumps(sample_record), encoding="utf-8")
            (tmp_path / "P12345_MF_k01.json").write_text(json.dumps(sample_record_k1), encoding="utf-8")
            (tmp_path / EVAL.RUN_SUMMARY_FILE).write_text(json.dumps({"ignore": True}), encoding="utf-8")
            (tmp_path / EVAL.ERROR_LOG_FILE).write_text(json.dumps([{"ignore": True}]), encoding="utf-8")

            result_rows = EVAL.collect_result_rows(str(tmp_path))
            self.assertEqual(len(result_rows), 2)
            self.assertEqual(result_rows[0]["protein_id"], "P12345")

            args = make_eval_args(eval_split="test", max_samples=10)
            sample_rows = EVAL.build_sample_table_rows(args, result_rows)
            self.assertEqual(len(sample_rows), 1)
            self.assertEqual(sample_rows[0]["protein_id"], "P12345")
            self.assertEqual(sample_rows[0]["attempt_count"], 2)
            self.assertIn("attempt_count=2", sample_rows[0]["accuracy_or_match_note"])

            sample_table_path = EVAL.write_sample_results_table(sample_rows, str(tmp_path))
            with open(sample_table_path, "r", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                written_rows = list(reader)

            self.assertEqual(len(written_rows), 1)
            self.assertEqual(written_rows[0]["protein_id"], "P12345")
            self.assertEqual(written_rows[0]["prediction"], "GO:0002222")
            self.assertEqual(written_rows[0]["attempt_count"], "2")
            self.assertIn("GO:0003333", written_rows[0]["attempt_predictions_json"])

            summary = EVAL.build_run_summary(
                args=args,
                loaded_samples=4,
                remaining_samples=3,
                newly_processed=2,
                total_time=1.25,
                result_rows=result_rows,
            )
            self.assertEqual(summary["job_type"], "eval")
            self.assertEqual(summary["eval_split"], "test")
            self.assertEqual(summary["result_files_total"], 2)
            self.assertEqual(summary["unique_sample_keys_total"], 1)

            summary_path = EVAL.write_run_summary(summary, str(tmp_path))
            written_summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
            self.assertEqual(written_summary["job_type"], "eval")
            self.assertEqual(written_summary["eval_split"], "test")

    def test_filter_unprocessed_samples_handles_protein_ids_with_underscores(self):
        samples = [
            {"protein_id": "P_12345", "go_aspect": "molecular_function"},
            {"protein_id": "Q99999", "go_aspect": "biological_process"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "P_12345_MF_k00.json").write_text("{}", encoding="utf-8")

            remaining = EVAL.filter_unprocessed_samples(samples, str(tmp_path))

        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["protein_id"], "Q99999")

    def test_log_error_uses_requested_evals_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            EVAL.log_error(
                tmpdir,
                "other",
                "P12345",
                "molecular_function",
                "",
                "GO:0001111",
                "",
                "",
                "GO:0001111",
                "",
                "boom",
            )

            error_path = Path(tmpdir) / EVAL.ERROR_LOG_FILE
            self.assertTrue(error_path.exists())
            payload = json.loads(error_path.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["protein_id"], "P12345")
            self.assertEqual(payload[0]["error_message"], "boom")

    def test_maybe_compute_metrics_summary_runs_cafa_pipeline(self):
        class FakeFrame:
            def __init__(self, label):
                self.label = label

            def to_csv(self, path, sep="\t"):
                Path(path).write_text(self.label, encoding="utf-8")

        calls = {}

        def process_json_data(base_dir, reasoning_mode=False, final_answer_only=False, go_dag=None):
            calls["process_json_data"] = {
                "base_dir": base_dir,
                "reasoning_mode": reasoning_mode,
                "final_answer_only": final_answer_only,
            }
            return [("P12345", {"GO:0001111"})], [("P12345", {"GO:0001111"})]

        def create_cafa_prediction_file(predictions, output_path):
            calls["prediction_file"] = output_path
            Path(output_path).write_text("predictions", encoding="utf-8")

        def create_cafa_ground_truth_file(ground_truth, output_path):
            calls["ground_truth_file"] = output_path
            Path(output_path).write_text("ground_truth", encoding="utf-8")

        def run_cafa_evaluation(ontology_path, predictions_dir, ground_truth_path, ia_file_path=None, n_cpu=0, th_step=0.99):
            calls["run_cafa_evaluation"] = {
                "ontology_path": ontology_path,
                "predictions_dir": predictions_dir,
                "ground_truth_path": ground_truth_path,
                "ia_file_path": ia_file_path,
                "n_cpu": n_cpu,
                "th_step": th_step,
            }
            return FakeFrame("evaluation"), {"f": FakeFrame("best-f"), "f_w": FakeFrame("best-fw")}

        def extract_metrics_summary(results):
            calls["extract_metrics_summary"] = True
            return {
                "molecular_function_f1": 0.9,
                "biological_process_f1": 0.2,
                "cellular_component_f1": 0.7,
            }

        def write_metrics_summary(metrics, output_dir):
            calls["metrics_output_dir"] = output_dir
            output_path = Path(output_dir) / "metrics_summary.json"
            output_path.write_text(json.dumps(metrics), encoding="utf-8")
            return str(output_path)

        fake_cafa_module = types.SimpleNamespace(
            process_json_data=process_json_data,
            create_cafa_prediction_file=create_cafa_prediction_file,
            create_cafa_ground_truth_file=create_cafa_ground_truth_file,
            run_cafa_evaluation=run_cafa_evaluation,
            extract_metrics_summary=extract_metrics_summary,
            write_metrics_summary=write_metrics_summary,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ia_path = Path(tmpdir) / "IA.txt"
            ia_path.write_text("ia", encoding="utf-8")
            go_obo_path = Path(tmpdir) / "go-basic.obo"
            go_obo_path.write_text("obo", encoding="utf-8")

            args = make_eval_args(
                evals_path=tmpdir,
                go_obo_path=str(go_obo_path),
                ia_file_path=str(ia_path),
                metric_threads=4,
                metric_threshold_step=0.95,
            )

            with mock.patch.object(EVAL.importlib, "import_module", return_value=fake_cafa_module):
                metrics_summary, metrics_summary_path = EVAL.maybe_compute_metrics_summary(args)

            self.assertEqual(metrics_summary["fmax_mf"], 0.9)
            self.assertTrue(metrics_summary_path.endswith("metrics_summary.json"))
            self.assertTrue(Path(metrics_summary_path).exists())
            self.assertEqual(calls["process_json_data"]["base_dir"], tmpdir)
            self.assertTrue(calls["process_json_data"]["reasoning_mode"])
            self.assertTrue(calls["process_json_data"]["final_answer_only"])
            self.assertEqual(calls["run_cafa_evaluation"]["ia_file_path"], str(ia_path))
            self.assertEqual(calls["run_cafa_evaluation"]["n_cpu"], 4)
            self.assertEqual(calls["run_cafa_evaluation"]["th_step"], 0.95)

    def test_maybe_compute_metrics_summary_runs_without_ia_file(self):
        calls = {}

        def process_json_data(base_dir, reasoning_mode=False, final_answer_only=False, go_dag=None):
            return [("P12345", {"GO:0001111"})], [("P12345", {"GO:0001111"})]

        def create_cafa_prediction_file(predictions, output_path):
            Path(output_path).write_text("predictions", encoding="utf-8")

        def create_cafa_ground_truth_file(ground_truth, output_path):
            Path(output_path).write_text("ground_truth", encoding="utf-8")

        def run_cafa_evaluation(ontology_path, predictions_dir, ground_truth_path, ia_file_path=None, n_cpu=0, th_step=0.99):
            calls["run_cafa_evaluation"] = {
                "ontology_path": ontology_path,
                "predictions_dir": predictions_dir,
                "ground_truth_path": ground_truth_path,
                "ia_file_path": ia_file_path,
                "n_cpu": n_cpu,
                "th_step": th_step,
            }
            return types.SimpleNamespace(to_csv=lambda *args, **kwargs: None), {
                "f": types.SimpleNamespace(to_csv=lambda *args, **kwargs: None),
                "f_w": types.SimpleNamespace(to_csv=lambda *args, **kwargs: None),
            }

        def extract_metrics_summary(results):
            return {"molecular_function_f1": 0.7}

        def write_metrics_summary(metrics, output_dir):
            output_path = Path(output_dir) / "metrics_summary.json"
            output_path.write_text(json.dumps(metrics), encoding="utf-8")
            return str(output_path)

        fake_cafa_module = types.SimpleNamespace(
            process_json_data=process_json_data,
            create_cafa_prediction_file=create_cafa_prediction_file,
            create_cafa_ground_truth_file=create_cafa_ground_truth_file,
            run_cafa_evaluation=run_cafa_evaluation,
            extract_metrics_summary=extract_metrics_summary,
            write_metrics_summary=write_metrics_summary,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            go_obo_path = Path(tmpdir) / "go-basic.obo"
            go_obo_path.write_text("obo", encoding="utf-8")

            args = make_eval_args(
                evals_path=tmpdir,
                go_obo_path=str(go_obo_path),
                ia_file_path=None,
            )

            with mock.patch.object(EVAL.importlib, "import_module", return_value=fake_cafa_module):
                metrics_summary, metrics_summary_path = EVAL.maybe_compute_metrics_summary(args)

        self.assertEqual(metrics_summary["fmax_mf"], 0.7)
        self.assertIsNotNone(metrics_summary_path)
        self.assertIsNone(calls["run_cafa_evaluation"]["ia_file_path"])

    def test_maybe_compute_metrics_summary_tolerates_missing_metric_frames(self):
        def process_json_data(base_dir, reasoning_mode=False, final_answer_only=False, go_dag=None):
            return [("P12345", {"GO:0001111"})], [("P12345", {"GO:0001111"})]

        fake_cafa_module = types.SimpleNamespace(
            process_json_data=process_json_data,
            create_cafa_prediction_file=lambda predictions, output_path: Path(output_path).write_text(
                "predictions", encoding="utf-8"
            ),
            create_cafa_ground_truth_file=lambda ground_truth, output_path: Path(output_path).write_text(
                "ground_truth", encoding="utf-8"
            ),
            run_cafa_evaluation=lambda *args, **kwargs: (None, {"f": None, "f_w": None}),
            extract_metrics_summary=lambda results: {},
            write_metrics_summary=lambda metrics, output_dir: str(Path(output_dir) / "metrics_summary.json"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            go_obo_path = Path(tmpdir) / "go-basic.obo"
            go_obo_path.write_text("obo", encoding="utf-8")

            args = make_eval_args(
                evals_path=tmpdir,
                go_obo_path=str(go_obo_path),
                ia_file_path=None,
            )

            with mock.patch.object(EVAL.importlib, "import_module", return_value=fake_cafa_module):
                metrics_summary, metrics_summary_path = EVAL.maybe_compute_metrics_summary(args)

        self.assertEqual(metrics_summary, {})
        self.assertTrue(metrics_summary_path.endswith("metrics_summary.json"))

    def test_maybe_compute_metrics_summary_skips_when_required_files_are_missing(self):
        args = make_eval_args(
            go_obo_path="/tmp/does-not-exist.obo",
            ia_file_path=None,
        )

        with mock.patch.object(EVAL.importlib, "import_module") as import_module:
            metrics_summary, metrics_summary_path = EVAL.maybe_compute_metrics_summary(args)

        self.assertEqual(metrics_summary, {})
        self.assertIsNone(metrics_summary_path)
        import_module.assert_not_called()

    def test_maybe_compute_metrics_summary_recovers_from_cafa_errors(self):
        def process_json_data(*args, **kwargs):
            return [("P12345", {"GO:0001111"})], [("P12345", {"GO:0001111"})]

        fake_cafa_module = types.SimpleNamespace(
            process_json_data=process_json_data,
            create_cafa_prediction_file=lambda predictions, output_path: Path(output_path).write_text(
                "predictions", encoding="utf-8"
            ),
            create_cafa_ground_truth_file=lambda ground_truth, output_path: Path(output_path).write_text(
                "ground_truth", encoding="utf-8"
            ),
            run_cafa_evaluation=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
            extract_metrics_summary=lambda results: {},
            write_metrics_summary=lambda metrics, output_dir: str(Path(output_dir) / "metrics_summary.json"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ia_path = Path(tmpdir) / "IA.txt"
            ia_path.write_text("ia", encoding="utf-8")
            go_obo_path = Path(tmpdir) / "go-basic.obo"
            go_obo_path.write_text("obo", encoding="utf-8")

            args = make_eval_args(
                evals_path=tmpdir,
                go_obo_path=str(go_obo_path),
                ia_file_path=str(ia_path),
            )

            with mock.patch.object(EVAL.importlib, "import_module", return_value=fake_cafa_module):
                metrics_summary, metrics_summary_path = EVAL.maybe_compute_metrics_summary(args)

        self.assertEqual(metrics_summary, {})
        self.assertIsNone(metrics_summary_path)

    def test_log_eval_tracking_uses_optional_wandb_and_weave(self):
        EVAL.wandb.init_calls.clear()
        EVAL.wandb.logged_payloads.clear()
        EVAL.wandb.last_run = None
        EVAL.weave.init_calls.clear()
        EVAL.weave.flush_calls = 0
        EVAL.weave.Evaluation.instances.clear()

        result_rows = [
            {
                "protein_id": "P12345",
                "go_aspect": "molecular_function",
                "success": True,
                "input_prompt": "prompt",
                "ground_truth": "<think>gold trace</think>\nGO:0001111",
                "generated_response": "<think>model trace</think>\nGO:0001111",
                "sequence_length": 321,
            }
        ]
        run_summary = {
            "job_type": "eval",
            "loaded_samples": 1,
            "newly_processed_samples": 1,
            "result_files_total": 1,
            "unique_sample_keys_total": 1,
            "successful_result_files_total": 1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics_summary.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "biological_process_f1": 0.2,
                        "molecular_function_f1": 0.9,
                        "cellular_component_f1": 0.7,
                    }
                ),
                encoding="utf-8",
            )
            args = make_eval_args(
                eval_split="test",
                evals_path=tmpdir,
                metrics_summary_path=str(metrics_path),
                benchmark_version="disease_temporal_hc_reasoning_v1",
                model_name="BioReason-Pro-SFT",
                wandb_project="bioreason-pro",
                wandb_entity="demo-entity",
                weave_project="demo-entity/bioreason-pro",
            )

            tracking_status = EVAL.log_eval_tracking(args, run_summary, result_rows)
            sample_rows = EVAL.build_sample_table_rows(args, result_rows)

        self.assertTrue(tracking_status["metrics_loaded"])
        self.assertTrue(tracking_status["wandb_logged"])
        self.assertTrue(tracking_status["weave_logged"])
        self.assertEqual(EVAL.weave.flush_calls, 1)
        self.assertEqual(len(sample_rows), 1)
        self.assertEqual(sample_rows[0]["reasoning_full"], "model trace")
        self.assertEqual(sample_rows[0]["final_answer"], "GO:0001111")
        self.assertIn("exact_match=True", sample_rows[0]["accuracy_or_match_note"])

    def test_log_eval_tracking_awaits_async_weave_evaluation(self):
        EVAL.wandb.init_calls.clear()
        EVAL.wandb.logged_payloads.clear()
        EVAL.wandb.last_run = None
        EVAL.weave.init_calls.clear()
        EVAL.weave.flush_calls = 0
        EVAL.weave.Evaluation.instances.clear()
        EVAL.weave.Evaluation.async_mode = True
        self.addCleanup(setattr, EVAL.weave.Evaluation, "async_mode", False)

        result_rows = [
            {
                "protein_id": "P12345",
                "go_aspect": "molecular_function",
                "success": True,
                "input_prompt": "prompt",
                "ground_truth": "<think>gold trace</think>\nGO:0001111",
                "generated_response": "<think>model trace</think>\nGO:0001111",
                "sequence_length": 321,
            }
        ]
        run_summary = {
            "job_type": "eval",
            "loaded_samples": 1,
            "newly_processed_samples": 1,
            "result_files_total": 1,
            "unique_sample_keys_total": 1,
            "successful_result_files_total": 1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics_summary.json"
            metrics_path.write_text(json.dumps({"molecular_function_f1": 0.9}), encoding="utf-8")
            args = make_eval_args(
                eval_split="test",
                evals_path=tmpdir,
                metrics_summary_path=str(metrics_path),
                wandb_project="bioreason-pro",
                wandb_entity="demo-entity",
                weave_project="demo-entity/bioreason-pro",
            )

            tracking_status = EVAL.log_eval_tracking(args, run_summary, result_rows)

        self.assertTrue(tracking_status["weave_logged"])
        self.assertEqual(EVAL.weave.flush_calls, 1)
        self.assertEqual(len(EVAL.weave.Evaluation.instances), 1)
        self.assertIsNotNone(EVAL.weave.Evaluation.instances[0].last_result)

    def test_log_eval_tracking_sets_default_weave_cache_dir(self):
        EVAL.wandb.init_calls.clear()
        EVAL.wandb.logged_payloads.clear()
        EVAL.wandb.last_run = None
        EVAL.weave.init_calls.clear()
        EVAL.weave.flush_calls = 0
        EVAL.weave.Evaluation.instances.clear()
        EVAL.weave.op_calls.clear()

        result_rows = [
            {
                "protein_id": "P12345",
                "go_aspect": "molecular_function",
                "success": True,
                "input_prompt": "prompt",
                "ground_truth": "<think>gold trace</think>\nGO:0001111",
                "generated_response": "<think>model trace</think>\nGO:0001111",
                "sequence_length": 321,
            }
        ]
        run_summary = {
            "job_type": "eval",
            "loaded_samples": 1,
            "newly_processed_samples": 1,
            "result_files_total": 1,
            "unique_sample_keys_total": 1,
            "successful_result_files_total": 1,
        }

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(EVAL.os.environ, {}, clear=False):
            EVAL.os.environ.pop("WEAVE_SERVER_CACHE_DIR", None)
            wandb_dir = str(Path(tmpdir) / "wandb")
            metrics_path = Path(tmpdir) / "metrics_summary.json"
            metrics_path.write_text(json.dumps({"molecular_function_f1": 0.9}), encoding="utf-8")
            args = make_eval_args(
                eval_split="test",
                evals_path=tmpdir,
                wandb_dir=wandb_dir,
                metrics_summary_path=str(metrics_path),
                wandb_project="bioreason-pro",
                wandb_entity="demo-entity",
                weave_project="demo-entity/bioreason-pro",
                temporal_split_artifact="demo-entity/bioreason-pro/disease-temporal-split:production",
                dataset_artifact="demo-entity/bioreason-pro/disease-temporal-reasoning:production",
                model_artifact="demo-entity/bioreason-pro/bioreason-pro-rl:production",
            )

            tracking_status = EVAL.log_eval_tracking(args, run_summary, result_rows)

            configured_cache = EVAL.os.environ.get("WEAVE_SERVER_CACHE_DIR")

        self.assertTrue(tracking_status["weave_logged"])
        self.assertEqual(configured_cache, str((Path(wandb_dir) / "weave_server_cache").resolve()))

        self.assertEqual(EVAL.wandb.init_calls[0]["job_type"], "eval")
        self.assertEqual(EVAL.wandb.init_calls[0]["config"]["benchmark_version"], "disease_temporal_hc_reasoning_v1")
        self.assertTrue(EVAL.wandb.last_run.finished)
        self.assertEqual(EVAL.wandb.last_run.artifacts, [])
        self.assertEqual(
            EVAL.wandb.last_run.used_artifact_refs,
            [
                {"artifact_ref": "demo-entity/bioreason-pro/disease-temporal-split:production", "type": "dataset"},
                {"artifact_ref": "demo-entity/bioreason-pro/disease-temporal-reasoning:production", "type": "dataset"},
                {"artifact_ref": "demo-entity/bioreason-pro/bioreason-pro-rl:production", "type": "model"},
            ],
        )
        self.assertTrue(any("fmax_mf" in payload for payload in EVAL.wandb.logged_payloads))
        self.assertTrue(any("eval_summary" in payload for payload in EVAL.wandb.logged_payloads))
        self.assertTrue(any("eval_samples" in payload for payload in EVAL.wandb.logged_payloads))

        self.assertEqual(EVAL.weave.init_calls[0], "demo-entity/bioreason-pro")
        self.assertEqual(len(EVAL.weave.Evaluation.instances), 1)
        self.assertEqual(len(EVAL.weave.Evaluation.instances[0].dataset), 1)

    def test_log_eval_tracking_logs_tables_and_weave_for_validation(self):
        EVAL.wandb.init_calls.clear()
        EVAL.wandb.logged_payloads.clear()
        EVAL.wandb.last_run = None
        EVAL.weave.init_calls.clear()
        EVAL.weave.Evaluation.instances.clear()

        result_rows = [
            {
                "protein_id": "P54321",
                "go_aspect": "biological_process",
                "success": True,
                "input_prompt": "prompt",
                "ground_truth": "<think>gold trace</think>\nGO:0002222",
                "generated_response": "<think>model trace</think>\nGO:0002222",
                "sequence_length": 111,
            }
        ]
        run_summary = {
            "job_type": "eval",
            "loaded_samples": 1,
            "newly_processed_samples": 1,
            "result_files_total": 1,
            "unique_sample_keys_total": 1,
            "successful_result_files_total": 1,
        }

        args = make_eval_args(
            eval_split="validation",
            model_name="BioReason-Pro-RL-Paper",
            wandb_project="bioreason-pro",
            wandb_entity="demo-entity",
            weave_project="demo-entity/bioreason-pro",
        )

        tracking_status = EVAL.log_eval_tracking(
            args,
            run_summary,
            result_rows,
            metrics_summary={"fmax_mf": 0.5, "fmax_bp": 0.4, "fmax_cc": 0.3},
        )

        self.assertTrue(tracking_status["wandb_logged"])
        self.assertTrue(tracking_status["weave_logged"])
        self.assertTrue(any("fmax_mf" in payload for payload in EVAL.wandb.logged_payloads))
        self.assertTrue(any("eval_summary" in payload for payload in EVAL.wandb.logged_payloads))
        self.assertTrue(any("eval_samples" in payload for payload in EVAL.wandb.logged_payloads))
        self.assertEqual(EVAL.wandb.last_run.artifacts, [])
        self.assertEqual(EVAL.weave.init_calls, ["demo-entity/bioreason-pro"])
        self.assertEqual(len(EVAL.weave.Evaluation.instances), 1)

    def test_enforce_required_eval_outputs_requires_metrics_for_validation(self):
        args = make_eval_args(eval_split="validation")

        with self.assertRaisesRegex(RuntimeError, "Fmax metrics"):
            EVAL.enforce_required_eval_outputs(
                args,
                {
                    "wandb_logged": True,
                    "metrics_loaded": False,
                    "weave_logged": False,
                },
            )

    def test_enforce_required_eval_outputs_requires_weave_for_test(self):
        args = make_eval_args(eval_split="test", weave_project="demo-entity/bioreason-pro")

        with self.assertRaisesRegex(RuntimeError, "Weave evaluation"):
            EVAL.enforce_required_eval_outputs(
                args,
                {
                    "wandb_logged": True,
                    "metrics_loaded": True,
                    "weave_logged": False,
                },
            )

    def test_enforce_required_eval_outputs_requires_weave_for_validation_when_project_configured(self):
        args = make_eval_args(eval_split="validation", weave_project="demo-entity/bioreason-pro")

        with self.assertRaisesRegex(RuntimeError, "Weave evaluation"):
            EVAL.enforce_required_eval_outputs(
                args,
                {
                    "wandb_logged": True,
                    "metrics_loaded": True,
                    "weave_logged": False,
                },
            )

    def test_maybe_cleanup_local_eval_outputs_removes_scratch_after_wandb_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scratch_dir = Path(tmpdir) / "results"
            scratch_dir.mkdir()
            (scratch_dir / "sample.json").write_text("{}", encoding="utf-8")

            args = make_eval_args(
                evals_path=str(scratch_dir),
                wandb_project="bioreason-pro",
                wandb_mode="online",
                keep_local_eval_outputs=False,
            )
            status = EVAL.maybe_cleanup_local_eval_outputs(args, {"wandb_logged": True})

            self.assertTrue(status["cleanup_completed"])
            self.assertFalse(scratch_dir.exists())

    def test_maybe_cleanup_local_eval_outputs_keeps_scratch_in_offline_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scratch_dir = Path(tmpdir) / "results"
            scratch_dir.mkdir()
            (scratch_dir / "sample.json").write_text("{}", encoding="utf-8")

            args = make_eval_args(
                evals_path=str(scratch_dir),
                wandb_project="bioreason-pro",
                wandb_mode="offline",
                keep_local_eval_outputs=False,
            )
            status = EVAL.maybe_cleanup_local_eval_outputs(args, {"wandb_logged": True})

            self.assertFalse(status["cleanup_completed"])
            self.assertTrue(scratch_dir.exists())


if __name__ == "__main__":
    unittest.main()

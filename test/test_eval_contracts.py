import argparse
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
    cafa5_load_module = types.ModuleType("bioreason2.dataset.cafa5.load")
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

    protein_vllm_module.ProteinLLMModel = ProteinLLMModel
    cafa5_load_module.load_cafa5_dataset = load_cafa5_dataset
    utils_module.str2bool = str2bool

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
            self.finished = False

        def log_artifact(self, artifact):
            self.artifacts.append(artifact)

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

    def weave_op(func=None, **kwargs):
        if func is None:
            return lambda actual: actual
        return func

    class FakeEvaluation:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.dataset = kwargs.get("dataset", [])
            self.scorers = kwargs.get("scorers", [])
            self.preprocess_model_input = kwargs.get("preprocess_model_input", lambda row: row)
            self.last_result = None
            FakeEvaluation.instances.append(self)

        def evaluate(self, model):
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

    weave_module.init_calls = []
    weave_module.op = weave_op
    weave_module.Evaluation = FakeEvaluation

    def weave_init(project_name):
        weave_module.init_calls.append(project_name)
        return types.SimpleNamespace(project=project_name)

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
        step0_artifact=None,
        dataset_artifact=None,
        model_artifact=None,
        shortlist_query=None,
        shortlist_mode=None,
        train_start_release=None,
        train_end_release=None,
        dev_end_release=None,
        test_end_release=None,
        metrics_summary_path=None,
        wandb_project=None,
        wandb_entity=None,
        wandb_run_name=None,
        wandb_artifact_name=None,
        wandb_mode=None,
        weave_project=None,
        weave_eval_name=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class EvalContractTests(unittest.TestCase):
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

        args = make_eval_args(eval_split="test", max_samples=2)

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

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "P12345_MF_k00.json").write_text(json.dumps(sample_record), encoding="utf-8")
            (tmp_path / EVAL.RUN_SUMMARY_FILE).write_text(json.dumps({"ignore": True}), encoding="utf-8")
            (tmp_path / EVAL.ERROR_LOG_FILE).write_text(json.dumps([{"ignore": True}]), encoding="utf-8")

            rows = EVAL.collect_result_rows(str(tmp_path))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["protein_id"], "P12345")

            sample_table_path = EVAL.write_sample_results_table(rows, str(tmp_path))
            with open(sample_table_path, "r", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                written_rows = list(reader)

            self.assertEqual(len(written_rows), 1)
            self.assertEqual(written_rows[0]["protein_id"], "P12345")
            self.assertEqual(written_rows[0]["generated_response"], "GO:0002222")

            args = make_eval_args(eval_split="test", max_samples=10)
            summary = EVAL.build_run_summary(
                args=args,
                loaded_samples=4,
                remaining_samples=3,
                newly_processed=2,
                total_time=1.25,
                result_rows=rows,
            )
            self.assertEqual(summary["job_type"], "eval")
            self.assertEqual(summary["eval_split"], "test")
            self.assertEqual(summary["result_files_total"], 1)
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

    def test_log_eval_tracking_uses_optional_wandb_and_weave(self):
        EVAL.wandb.init_calls.clear()
        EVAL.wandb.logged_payloads.clear()
        EVAL.wandb.last_run = None
        EVAL.weave.init_calls.clear()
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
                benchmark_version="disease_temporal_hc_v1",
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
        self.assertEqual(len(sample_rows), 1)
        self.assertEqual(sample_rows[0]["reasoning_full"], "model trace")
        self.assertEqual(sample_rows[0]["final_answer"], "GO:0001111")
        self.assertIn("exact_match=True", sample_rows[0]["accuracy_or_match_note"])

        self.assertEqual(EVAL.wandb.init_calls[0]["job_type"], "eval")
        self.assertEqual(EVAL.wandb.init_calls[0]["config"]["benchmark_version"], "disease_temporal_hc_v1")
        self.assertTrue(EVAL.wandb.last_run.finished)
        self.assertEqual(EVAL.wandb.last_run.artifacts[0].added_dirs, [tmpdir])
        self.assertTrue(any("fmax_mf" in payload for payload in EVAL.wandb.logged_payloads))
        self.assertTrue(any("eval_summary" in payload for payload in EVAL.wandb.logged_payloads))
        self.assertTrue(any("eval_samples" in payload for payload in EVAL.wandb.logged_payloads))

        self.assertEqual(EVAL.weave.init_calls[0], "demo-entity/bioreason-pro")
        self.assertEqual(len(EVAL.weave.Evaluation.instances), 1)
        self.assertEqual(len(EVAL.weave.Evaluation.instances[0].dataset), 1)


if __name__ == "__main__":
    unittest.main()

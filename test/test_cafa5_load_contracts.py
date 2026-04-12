import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
LOAD_PATH = ROOT / "bioreason2" / "dataset" / "cafa5" / "load.py"


def load_cafa5_module():
    module_name = "cafa5_load_contracts_test_module"
    spec = importlib.util.spec_from_file_location(module_name, LOAD_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


LOAD = load_cafa5_module()
PROMPTS_SPEC = importlib.util.spec_from_file_location(
    "cafa5_prompts_contracts_test_module",
    ROOT / "bioreason2" / "dataset" / "prompts" / "cafa5.py",
)
PROMPTS = importlib.util.module_from_spec(PROMPTS_SPEC)
sys.modules["cafa5_prompts_contracts_test_module"] = PROMPTS
assert PROMPTS_SPEC.loader is not None
PROMPTS_SPEC.loader.exec_module(PROMPTS)

FORMAT_SPEC = importlib.util.spec_from_file_location(
    "cafa5_format_contracts_test_module",
    ROOT / "bioreason2" / "dataset" / "cafa5" / "format.py",
)
FORMAT = importlib.util.module_from_spec(FORMAT_SPEC)
sys.modules["cafa5_format_contracts_test_module"] = FORMAT
assert FORMAT_SPEC.loader is not None
FORMAT_SPEC.loader.exec_module(FORMAT)


class Cafa5LoadContractsTest(unittest.TestCase):
    def test_resolve_keep_in_memory_defaults_to_false_without_distributed_world(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(LOAD._resolve_keep_in_memory_for_local_dataset())

    def test_resolve_keep_in_memory_defaults_to_true_for_distributed_world(self):
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "8"}, clear=True):
            self.assertTrue(LOAD._resolve_keep_in_memory_for_local_dataset())

    def test_resolve_keep_in_memory_respects_explicit_env_override(self):
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "8", "BIOREASON_DATASET_KEEP_IN_MEMORY": "false"}, clear=True):
            self.assertFalse(LOAD._resolve_keep_in_memory_for_local_dataset())

    def test_load_dataset_source_uses_keep_in_memory_for_local_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "dataset_dict.json").write_text("{}", encoding="utf-8")

            with mock.patch.dict(os.environ, {"WORLD_SIZE": "8"}, clear=True), mock.patch.object(
                LOAD,
                "load_from_disk",
                return_value={"train": []},
            ) as load_from_disk_mock:
                result = LOAD._load_dataset_source(str(dataset_dir))

        self.assertEqual(result, {"train": []})
        load_from_disk_mock.assert_called_once_with(str(dataset_dir), keep_in_memory=True)

    def test_paper_compact_response_format_uses_non_parsable_placeholders(self):
        response_format = LOAD._build_paper_compact_response_format("MF/BP")

        self.assertIn("GO:XXXXXXX", response_format)
        self.assertIn("GO:YYYYYYY", response_format)
        self.assertNotIn("GO:0000000", response_format)
        self.assertNotIn("GO:0000001", response_format)

    def test_paper_compact_prompt_does_not_embed_numeric_dummy_go_ids(self):
        system_prompt = PROMPTS.CAFA5_REASONING_TEMPLATE_PAPER_COMPACT["system_prompt"]

        self.assertIn("GO:XXXXXXX", system_prompt)
        self.assertIn("GO:YYYYYYY", system_prompt)
        self.assertNotIn("GO:0000000", system_prompt)
        self.assertNotIn("GO:0000001", system_prompt)

    def test_paper_native_prompt_mentions_gogpt_and_final_answer_tags(self):
        system_prompt = PROMPTS.CAFA5_REASONING_TEMPLATE_PAPER_NATIVE["system_prompt"]
        user_prompt = PROMPTS.CAFA5_REASONING_TEMPLATE_PAPER_NATIVE["user_prompt"]

        self.assertIn("greedy-decoded GO-GPT predictions", system_prompt)
        self.assertIn("<|REASONING|>", system_prompt)
        self.assertIn("<|FINAL_ANSWER|>", system_prompt)
        self.assertIn("<|/FINAL_ANSWER|>", system_prompt)
        self.assertIn("<|FINAL_ANSWER|>", user_prompt)
        self.assertNotIn("GO:XXXXXXX", user_prompt)
        self.assertNotIn("GO:YYYYYYY", user_prompt)

    def test_limit_multiline_slot_keeps_full_text_when_caps_disabled(self):
        value = "line1\nline2\nline3"

        self.assertEqual(LOAD._limit_multiline_slot(value, max_lines=0), value)

    def test_compact_go_speculations_keeps_all_ids_when_caps_disabled(self):
        text = "MF: GO:0000001, GO:0000002, GO:0000003"

        compact = LOAD._compact_go_speculations(text, max_ids_per_aspect=0)

        self.assertEqual(compact["MF"], "GO:0000001, GO:0000002, GO:0000003")

    def test_format_preserves_paper_contract_without_think_wrapper(self):
        example = {
            "prompt": {
                "system": "system",
                "user": "user",
                "assistant_reasoning": "<|REASONING|>\nreason\n<|/REASONING|>",
                "assistant_answer": "<|FINAL_ANSWER|>\nGO:0000001\n<|/FINAL_ANSWER|>",
            },
            "sequence": "MPEPTIDE",
        }

        formatted = FORMAT.format_cafa5_for_protein_llm(example)
        assistant = formatted["prompt"][1]
        assistant_text = assistant["content"][0]["text"]

        self.assertNotIn("reasoning_content", assistant)
        self.assertTrue(assistant_text.startswith("<|REASONING|>"))
        self.assertIn("<|FINAL_ANSWER|>", assistant_text)


if __name__ == "__main__":
    unittest.main()

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "train_protein_grpo.py"
SFT_SCRIPT_PATH = ROOT / "train_protein_llm.py"
WRAPPER_PATH = ROOT / "scripts" / "sh_train_protein_grpo.sh"
SFT_WRAPPER_PATH = ROOT / "scripts" / "sh_train_protein_qwen_staged.sh"


def load_grpo_module():
    module_name = "train_protein_grpo_contracts_test_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GRPO = load_grpo_module()


class TrainProteinGrpoContractsTest(unittest.TestCase):
    def test_resolve_attn_implementation_falls_back_without_flash_attn(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertEqual(GRPO.resolve_attn_implementation("flash_attention_2"), "sdpa")

    def test_wrapper_uses_sft_conversion_lora_settings(self):
        wrapper_text = WRAPPER_PATH.read_text()

        self.assertIn('BASE_WANDB_PROJECT=${BASE_WANDB_PROJECT:-"${WANDB_PROJECT:-bioreason-pro-custom}"}', wrapper_text)
        self.assertIn('SFT_CONVERSION_LORA_RANK=${SFT_CONVERSION_LORA_RANK:-128}', wrapper_text)
        self.assertIn('SFT_CONVERSION_LORA_ALPHA=${SFT_CONVERSION_LORA_ALPHA:-256}', wrapper_text)
        self.assertIn('SFT_CONVERSION_LORA_DROPOUT=${SFT_CONVERSION_LORA_DROPOUT:-0.05}', wrapper_text)
        self.assertIn('CAFA5_DATASET=${CAFA5_DATASET:-""}', wrapper_text)
        self.assertIn('WEAVE_TRACE_BUDGET=${WEAVE_TRACE_BUDGET:-64}', wrapper_text)
        self.assertIn('MIN_NEW_TOKENS=${MIN_NEW_TOKENS:-1}', wrapper_text)
        self.assertIn('RL_RUN_FAMILY="rl-sft"', wrapper_text)
        self.assertIn('RL_RUN_FAMILY="rl-paper"', wrapper_text)
        self.assertIn('WANDB_RUN_NAME=${WANDB_RUN_NAME:-"${RL_RUN_FAMILY}-${TIMESTAMP}"}', wrapper_text)
        self.assertIn('--asset-key reasoning_dataset', wrapper_text)
        self.assertIn('--lora_rank "$SFT_CONVERSION_LORA_RANK"', wrapper_text)
        self.assertIn('--lora_alpha "$SFT_CONVERSION_LORA_ALPHA"', wrapper_text)
        self.assertIn('--lora_dropout "$SFT_CONVERSION_LORA_DROPOUT"', wrapper_text)
        self.assertIn('--weave_trace_budget "$WEAVE_TRACE_BUDGET"', wrapper_text)
        self.assertIn('--min_new_tokens "$MIN_NEW_TOKENS"', wrapper_text)

    def test_extract_go_ids_preserves_order_and_deduplicates(self):
        text = "GO:0008150 and GO:0003674 and GO:0008150 again"
        self.assertEqual(GRPO.extract_go_ids(text), ["GO:0008150", "GO:0003674"])

    def test_extract_reasoning_and_answer_parses_sections(self):
        text = "<think>first infer signaling loss</think><answer>GO:0007165, GO:0005515</answer>"
        parsed = GRPO.extract_reasoning_and_answer(text)
        self.assertEqual(parsed["reasoning"], "first infer signaling loss")
        self.assertEqual(parsed["final_answer"], "GO:0007165, GO:0005515")

    def test_build_target_go_ids_merges_all_aspects(self):
        sample_meta = {
            "go_bp": "GO:0007165",
            "go_mf": "GO:0005515",
            "go_cc": "GO:0005737",
            "ground_truth_go_terms": "GO:0007165, GO:0009987",
        }
        self.assertEqual(
            GRPO.build_target_go_ids(sample_meta),
            ["GO:0007165", "GO:0005515", "GO:0005737", "GO:0009987"],
        )

    def test_standardize_group_rewards_returns_zeroes_for_constant_group(self):
        self.assertEqual(GRPO.standardize_group_rewards([0.5, 0.5, 0.5]), [0.0, 0.0, 0.0])

    def test_compute_group_rewards_combines_named_components(self):
        completion = "<think>reasoning</think><answer>GO:0007165</answer>"
        sample_meta = {"go_bp": "GO:0007165"}

        totals, components = GRPO.compute_group_rewards(
            [completion],
            sample_meta,
            ["strict_format", "answer_nonempty", "go_overlap"],
            [1.0, 1.0, 2.0],
        )

        self.assertEqual(components["strict_format"], [1.0])
        self.assertEqual(components["answer_nonempty"], [1.0])
        self.assertEqual(components["go_overlap"], [1.0])
        self.assertEqual(totals, [4.0])

    def test_parse_reward_weights_validates_expected_count(self):
        with self.assertRaises(ValueError):
            GRPO.parse_reward_weights("1.0,2.0", 3)

    def test_parse_args_defaults_to_train_rl_contract(self):
        args = GRPO.parse_args(["--text_model_name", "/tmp/demo-model"])

        self.assertEqual(args.wandb_job_type, "train_rl")
        self.assertEqual(args.dataset_config, "disease_temporal_hc_reasoning_v1")
        self.assertEqual(args.reasoning_dataset_config, "disease_temporal_hc_reasoning_v1")
        self.assertEqual(args.checkpoint_artifact_name, "train-rl-output")
        self.assertEqual(args.output_dir, "data/artifacts/models/train_rl_output")
        self.assertEqual(args.max_eval_samples, 100)
        self.assertEqual(args.eval_sample_strategy, "stratified_aspect_profile")
        self.assertEqual(args.min_new_tokens, 1)
        self.assertTrue(args.bnb_4bit_use_double_quant)
        self.assertEqual(args.weave_trace_budget, 64)
        self.assertFalse(args.ablation_from_paper_rl)

    def test_rl_script_uses_canonical_metrics_and_input_artifact_lineage(self):
        source = SCRIPT_PATH.read_text()

        self.assertIn("maybe_use_artifact_refs(", source)
        self.assertIn("maybe_trace_generation(", source)
        self.assertIn('"data_step_num_groups_submitted"', source)
        self.assertIn('"data_step_num_groups_trainable"', source)
        self.assertIn('"data_step_num_trajectories"', source)
        self.assertIn('"data_step_num_datums"', source)
        self.assertIn('"data_step_trainer_tokens"', source)
        self.assertIn('"reward_std_dev"', source)
        self.assertIn('"loss_train"', source)
        self.assertIn('"loss_kl_div"', source)
        self.assertIn('"loss_learning_rate"', source)
        self.assertIn('"loss_grad_norm"', source)
        self.assertIn('"eval_reward"', source)
        self.assertNotIn("train_rl_rollouts", source)
        self.assertNotIn("wandb.Table(", source)
        self.assertNotIn('"dataset/train_size"', source)
        self.assertNotIn('"dataset/validation_size"', source)

    def test_sft_script_registers_input_artifact_lineage(self):
        source = SFT_SCRIPT_PATH.read_text()

        self.assertIn("maybe_use_artifact_refs(", source)
        self.assertIn('"temporal_split_artifact": args.temporal_split_artifact', source)
        self.assertIn('"dataset_artifact": args.dataset_artifact', source)
        self.assertIn('"base_checkpoint": args.base_checkpoint', source)

    def test_sft_wrapper_supports_search_hyperparameters(self):
        wrapper_text = SFT_WRAPPER_PATH.read_text()

        self.assertIn('TRAIN_EXCLUSIVE=${TRAIN_EXCLUSIVE:-"True"}', wrapper_text)
        self.assertIn('STAGE2_LEARNING_RATE=${STAGE2_LEARNING_RATE:-1e-4}', wrapper_text)
        self.assertIn('STAGE2_WARMUP_RATIO=${STAGE2_WARMUP_RATIO:-0.05}', wrapper_text)
        self.assertIn('STAGE2_BATCH_SIZE=${STAGE2_BATCH_SIZE:-4}', wrapper_text)
        self.assertIn('STAGE2_GRADIENT_ACCUMULATION_STEPS=${STAGE2_GRADIENT_ACCUMULATION_STEPS:-1}', wrapper_text)
        self.assertIn('STAGE2_EARLY_STOPPING_PATIENCE=${STAGE2_EARLY_STOPPING_PATIENCE:-2}', wrapper_text)
        self.assertIn('STAGE2_RUN_LABEL=${STAGE2_RUN_LABEL:-""}', wrapper_text)
        self.assertIn('--early_stopping_patience "$STAGE2_EARLY_STOPPING_PATIENCE"', wrapper_text)
        self.assertIn('--learning_rate "$STAGE2_LEARNING_RATE"', wrapper_text)


if __name__ == "__main__":
    unittest.main()

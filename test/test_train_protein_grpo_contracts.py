import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
except ImportError:  # pragma: no cover - contract tests run without torch
    torch = None


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

        self.assertIn("source_env_file_without_overrides()", wrapper_text)
        self.assertIn('source_env_file_without_overrides "$REGISTRY_ENV_FILE"', wrapper_text)
        self.assertIn('BASE_WANDB_PROJECT=${BASE_WANDB_PROJECT:-"${WANDB_PROJECT:-bioreasoning-pro}"}', wrapper_text)
        self.assertIn('SFT_CONVERSION_LORA_RANK=${SFT_CONVERSION_LORA_RANK:-128}', wrapper_text)
        self.assertIn('SFT_CONVERSION_LORA_ALPHA=${SFT_CONVERSION_LORA_ALPHA:-256}', wrapper_text)
        self.assertIn('SFT_CONVERSION_LORA_DROPOUT=${SFT_CONVERSION_LORA_DROPOUT:-0.05}', wrapper_text)
        self.assertIn('CAFA5_DATASET=${CAFA5_DATASET:-""}', wrapper_text)
        self.assertIn('WEAVE_TRACE_BUDGET=${WEAVE_TRACE_BUDGET:-128}', wrapper_text)
        self.assertIn('EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}', wrapper_text)
        self.assertIn('MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-128}', wrapper_text)
        self.assertIn('MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-0}', wrapper_text)
        self.assertIn('ROTATING_EVAL_EVERY_N_STEPS=${ROTATING_EVAL_EVERY_N_STEPS:-100}', wrapper_text)
        self.assertIn('ROTATING_EVAL_MAX_SAMPLES=${ROTATING_EVAL_MAX_SAMPLES:-256}', wrapper_text)
        self.assertIn('ROTATING_EVAL_SAMPLE_STRATEGY=${ROTATING_EVAL_SAMPLE_STRATEGY:-"stratified_aspect_profile"}', wrapper_text)
        self.assertIn('ROTATING_EVAL_SEED_STRIDE=${ROTATING_EVAL_SEED_STRIDE:-9973}', wrapper_text)
        self.assertIn('NUM_GENERATIONS=${NUM_GENERATIONS:-8}', wrapper_text)
        self.assertIn('REWARD_FUNCS=${REWARD_FUNCS:-"strict_format,summary_schema,go_overlap,structural_noise"}', wrapper_text)
        self.assertIn('REWARD_WEIGHTS=${REWARD_WEIGHTS:-"0.5,0.75,2.0,1.0"}', wrapper_text)
        self.assertIn('MIN_NEW_TOKENS=${MIN_NEW_TOKENS:-1}', wrapper_text)
        self.assertIn('MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}', wrapper_text)
        self.assertIn('TEMPERATURE=${TEMPERATURE:-1.0}', wrapper_text)
        self.assertIn('TOP_K=${TOP_K:-20}', wrapper_text)
        self.assertIn('if [ "${CHECKPOINT_ARTIFACT_NAME+x}" = "x" ]; then', wrapper_text)
        self.assertIn('RL_RUN_FAMILY="rl-sft"', wrapper_text)
        self.assertIn('RL_RUN_FAMILY="rl-paper"', wrapper_text)
        self.assertIn('WANDB_RUN_NAME=${WANDB_RUN_NAME:-"${RL_RUN_FAMILY}-${TIMESTAMP}"}', wrapper_text)
        self.assertIn('--asset-key reasoning_dataset', wrapper_text)
        self.assertIn('find "$RESOLVED_TRAIN_SFT_DIR" -maxdepth 2 -type f -name "*best*.ckpt"', wrapper_text)
        self.assertIn('SFT_CKPT_PATH="$RESOLVED_TRAIN_SFT_DIR/last.ckpt"', wrapper_text)
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

    def test_go_overlap_reward_ignores_reasoning_only_go_terms(self):
        completion = "<think>Reasoning mentions GO:0007165.</think><answer>GO:0005515</answer>"
        sample_meta = {"go_bp": "GO:0007165"}

        self.assertEqual(GRPO.go_overlap_reward(completion, sample_meta), 0.0)

    def test_strict_format_reward_rejects_structural_tag_noise(self):
        completion = "<think>reasoning</think></tool_call>"

        self.assertEqual(GRPO.strict_format_reward(completion, {}), 0.0)
        self.assertLess(GRPO.structural_noise_reward(completion, {}), 0.0)

    def test_summary_schema_reward_requires_expected_summary_blocks(self):
        completion = (
            "<think>reasoning</think>"
            "<|GO_SUMMARY_START|>\nBP: GO:0007165\n<|GO_SUMMARY_END|>\n\n"
            "<|FUNCTION_SUMMARY_START|>\nKinase-linked signaling regulator.\n<|FUNCTION_SUMMARY_END|>"
        )

        self.assertEqual(GRPO.summary_schema_reward(completion, {}), 1.0)

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
        self.assertEqual(args.eval_batch_size, 4)
        self.assertEqual(args.max_eval_samples, 128)
        self.assertEqual(args.eval_sample_strategy, "stratified_aspect_profile")
        self.assertEqual(args.max_eval_batches, 0)
        self.assertEqual(args.num_generations, 8)
        self.assertEqual(args.max_new_tokens, 512)
        self.assertEqual(args.temperature, 1.0)
        self.assertEqual(args.top_k, 20)
        self.assertEqual(args.rotating_eval_every_n_steps, 100)
        self.assertEqual(args.rotating_eval_max_samples, 256)
        self.assertEqual(args.reward_funcs, "strict_format,summary_schema,go_overlap,structural_noise")
        self.assertEqual(args.reward_weights, "0.5,0.75,2.0,1.0")
        self.assertEqual(args.min_new_tokens, 1)
        self.assertTrue(args.bnb_4bit_use_double_quant)
        self.assertEqual(args.weave_trace_budget, 64)
        self.assertFalse(args.ablation_from_paper_rl)

    def test_extract_completion_ids_handles_prompt_inclusive_output(self):
        if torch is None:
            self.skipTest("torch is not available in the contract test environment")

        prompt_ids = torch.tensor([[11, 12, 13]])
        generated_ids = torch.tensor([[11, 12, 13, 21, 22]])

        completion_ids = GRPO.extract_completion_ids(generated_ids, prompt_ids)

        self.assertTrue(torch.equal(completion_ids, torch.tensor([21, 22])))

    def test_extract_completion_ids_handles_completion_only_output(self):
        if torch is None:
            self.skipTest("torch is not available in the contract test environment")

        prompt_ids = torch.tensor([[11, 12, 13, 14]])
        generated_ids = torch.tensor([[21, 22]])

        completion_ids = GRPO.extract_completion_ids(generated_ids, prompt_ids)

        self.assertTrue(torch.equal(completion_ids, torch.tensor([21, 22])))

    def test_extract_completion_ids_batch_handles_prompt_inclusive_output(self):
        if torch is None:
            self.skipTest("torch is not available in the contract test environment")

        prompt_ids = torch.tensor([[11, 12, 13], [21, 22, 23]])
        generated_ids = torch.tensor([[11, 12, 13, 31, 32], [21, 22, 23, 41, 42]])

        completion_ids_list = GRPO.extract_completion_ids_batch(generated_ids, prompt_ids)

        self.assertEqual(len(completion_ids_list), 2)
        self.assertTrue(torch.equal(completion_ids_list[0], torch.tensor([31, 32])))
        self.assertTrue(torch.equal(completion_ids_list[1], torch.tensor([41, 42])))

    def test_evaluate_policy_uses_batched_generation_for_eval_batches(self):
        if torch is None:
            self.skipTest("torch is not available in the contract test environment")

        class FakeModel:
            def __init__(self):
                self.generate_calls = []
                self.eval_calls = 0
                self.train_calls = 0

            def eval(self):
                self.eval_calls += 1

            def train(self):
                self.train_calls += 1

            def generate(self, **kwargs):
                self.generate_calls.append(kwargs)
                return torch.tensor([[11, 12, 31], [21, 22, 32]])

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def decode(self, completion_ids, skip_special_tokens=False):
                values = completion_ids.tolist()
                return " ".join(str(value) for value in values)

        batch = {
            "input_ids": torch.tensor([[11, 12], [21, 22]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
            "protein_sequences": ["AAA", "BBB"],
            "batch_idx_map": [0, 1],
            "batch_go_aspects": ["bp", "mf"],
            "protein_ids": ["P1", "P2"],
            "sample_splits": ["validation", "validation"],
            "go_bp_targets": ["", ""],
            "go_mf_targets": ["", ""],
            "go_cc_targets": ["", ""],
            "reasoning_targets": ["", ""],
            "final_answers": ["", ""],
            "prompt": ["prompt-1", "prompt-2"],
        }
        args = mock.Mock(
            max_eval_batches=8,
            do_sample=False,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            min_new_tokens=1,
            max_new_tokens=32,
        )
        model = FakeModel()
        metrics = GRPO.evaluate_policy(
            model=model,
            ref_model=None,
            dataloader=[batch],
            tokenizer=FakeTokenizer(),
            args=args,
            reward_names=[],
            reward_weights=[],
            device=torch.device("cpu"),
            trace_state=None,
            global_step=25,
        )

        self.assertEqual(len(model.generate_calls), 1)
        self.assertEqual(tuple(model.generate_calls[0]["input_ids"].shape), (2, 2))
        self.assertEqual(metrics["eval_completion_length"], 1.0)
        self.assertEqual(metrics["eval_data_step_num_datums"], 2.0)
        self.assertEqual(model.eval_calls, 1)
        self.assertEqual(model.train_calls, 1)

    def test_rl_script_uses_canonical_metrics_and_input_artifact_lineage(self):
        source = SCRIPT_PATH.read_text()

        self.assertIn("maybe_use_artifact_refs(", source)
        self.assertIn("maybe_trace_generation(", source)
        self.assertIn("extract_completion_ids_batch(", source)
        self.assertIn("evaluate_policy_batch(", source)
        self.assertIn("build_rollout_group_inputs(", source)
        self.assertIn("generate_rollouts_for_example(", source)
        self.assertIn('reward_component/{reward_name}', source)
        self.assertIn('"validation_rotating"', source)
        self.assertIn("candidate.is_file()", source)
        self.assertIn("precomputed_go_embedding_cache_path", source)
        self.assertIn("model.load_precomputed_go_embedding_cache(", source)
        self.assertIn("prepare_model_artifact_directory(", source)
        self.assertIn('"artifact_export_mode"', source)
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
        self.assertIn('"train_skipped_update"', source)
        self.assertIn('protein_model_dir = export_dir / "protein_model"', source)
        self.assertIn('torch.save(model.protein_model.state_dict(), protein_model_dir / "pytorch_model.bin")', source)
        self.assertNotIn('"loss_train": 0.0', source)
        self.assertNotIn('"loss_kl_div": 0.0', source)
        self.assertNotIn("train_rl_rollouts", source)
        self.assertNotIn("wandb.Table(", source)
        self.assertNotIn('"dataset/train_size"', source)
        self.assertNotIn('"dataset/validation_size"', source)

    def test_grpo_converter_saves_protein_model_weights(self):
        source = (ROOT / "bioreason2" / "utils" / "save_grpo_ckpt.py").read_text()

        self.assertIn('protein_model_dir = os.path.join(args.save_dir, "protein_model")', source)
        self.assertIn('torch.save(model.protein_model.state_dict(), os.path.join(protein_model_dir, "pytorch_model.bin"))', source)

    def test_sft_script_registers_input_artifact_lineage(self):
        source = SFT_SCRIPT_PATH.read_text()

        self.assertIn("maybe_use_artifact_refs(", source)
        self.assertIn('"temporal_split_artifact": args.temporal_split_artifact', source)
        self.assertIn('"dataset_artifact": args.dataset_artifact', source)
        self.assertIn('"base_checkpoint": args.base_checkpoint', source)

    def test_sft_wrapper_supports_search_hyperparameters(self):
        wrapper_text = SFT_WRAPPER_PATH.read_text()

        self.assertIn("source_env_file_without_overrides()", wrapper_text)
        self.assertIn('source_env_file_without_overrides "$REGISTRY_ENV_FILE"', wrapper_text)
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

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
        self.assertIn('BASE_CHECKPOINT_LOCAL_DIR=${BASE_CHECKPOINT_LOCAL_DIR:-""}', wrapper_text)
        self.assertIn('SFT_CONVERSION_LORA_RANK=${SFT_CONVERSION_LORA_RANK:-128}', wrapper_text)
        self.assertIn('SFT_CONVERSION_LORA_ALPHA=${SFT_CONVERSION_LORA_ALPHA:-256}', wrapper_text)
        self.assertIn('SFT_CONVERSION_LORA_DROPOUT=${SFT_CONVERSION_LORA_DROPOUT:-0.05}', wrapper_text)
        self.assertIn('CAFA5_DATASET=${CAFA5_DATASET:-""}', wrapper_text)
        self.assertIn('WEAVE_TRACE_BUDGET=${WEAVE_TRACE_BUDGET:-128}', wrapper_text)
        self.assertIn('PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-1}}', wrapper_text)
        self.assertIn('PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE:-4}}', wrapper_text)
        self.assertIn('MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-200}', wrapper_text)
        self.assertIn('MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-0}', wrapper_text)
        self.assertIn('ROTATING_EVAL_EVERY_N_STEPS=${ROTATING_EVAL_EVERY_N_STEPS:-100}', wrapper_text)
        self.assertIn('ROTATING_EVAL_MAX_SAMPLES=${ROTATING_EVAL_MAX_SAMPLES:-256}', wrapper_text)
        self.assertIn('ROTATING_EVAL_SAMPLE_STRATEGY=${ROTATING_EVAL_SAMPLE_STRATEGY:-"stratified_aspect_profile"}', wrapper_text)
        self.assertIn('ROTATING_EVAL_SEED_STRIDE=${ROTATING_EVAL_SEED_STRIDE:-9973}', wrapper_text)
        self.assertIn('SEED=${SEED:-42}', wrapper_text)
        self.assertIn('LOSS_TYPE=${LOSS_TYPE:-"dr_grpo"}', wrapper_text)
        self.assertIn('STEPS_PER_GENERATION=${STEPS_PER_GENERATION:-2}', wrapper_text)
        self.assertIn('NUM_ITERATIONS=${NUM_ITERATIONS:-1}', wrapper_text)
        self.assertIn('NUM_GENERATIONS=${NUM_GENERATIONS:-24}', wrapper_text)
        self.assertIn('EVAL_DO_SAMPLE=${EVAL_DO_SAMPLE:-"False"}', wrapper_text)
        self.assertIn('EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.1}', wrapper_text)
        self.assertIn('EVAL_TOP_P=${EVAL_TOP_P:-0.9}', wrapper_text)
        self.assertIn('EVAL_TOP_K=${EVAL_TOP_K:-20}', wrapper_text)
        self.assertIn('REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}', wrapper_text)
        self.assertIn('CLIP_EPSILON_LOW=${CLIP_EPSILON_LOW:-7e-4}', wrapper_text)
        self.assertIn('CLIP_EPSILON_HIGH=${CLIP_EPSILON_HIGH:-9e-4}', wrapper_text)
        self.assertIn('REWARD_SCALING=${REWARD_SCALING:-"batch"}', wrapper_text)
        self.assertIn('IMPORTANCE_SAMPLING_CAP=${IMPORTANCE_SAMPLING_CAP:-2.0}', wrapper_text)
        self.assertIn('REWARD_FINAL_ANSWER_ONLY=${REWARD_FINAL_ANSWER_ONLY:-"False"}', wrapper_text)
        self.assertIn('REWARD_PREDICTION_SOURCE=${REWARD_PREDICTION_SOURCE:-"auto"}', wrapper_text)
        self.assertIn('REWARD_FUNCS=${REWARD_FUNCS:-"ia_weighted_f1"}', wrapper_text)
        self.assertIn('REWARD_WEIGHTS=${REWARD_WEIGHTS:-"1.0"}', wrapper_text)
        self.assertIn('PYTHON_BIN=${PYTHON_BIN:-""}', wrapper_text)
        self.assertIn('if [ -x "$(pwd)/.venv-gpu/bin/python" ]; then', wrapper_text)
        self.assertIn('if [ "$TRAIN_NUM_GPUS" -gt 1 ]; then', wrapper_text)
        self.assertIn('torch.distributed.run', wrapper_text)
        self.assertIn('--nproc_per_node "$TRAIN_NUM_GPUS"', wrapper_text)
        self.assertIn('ADD_UNIPROT_SUMMARY=${ADD_UNIPROT_SUMMARY:-"False"}', wrapper_text)
        self.assertIn('CONTINUATION_MODE=${CONTINUATION_MODE:-"paper_native"}', wrapper_text)
        self.assertIn('if [ "$CONTINUATION_MODE" = "paper_native" ]; then', wrapper_text)
        self.assertIn('INCLUDE_PROTEIN_FUNCTION_SUMMARY="True"', wrapper_text)
        self.assertIn('REASONING_PROMPT_STYLE=${REASONING_PROMPT_STYLE:-"auto"}', wrapper_text)
        self.assertIn('COMPACT_INTERPRO_LIMIT=${COMPACT_INTERPRO_LIMIT:-12}', wrapper_text)
        self.assertIn('COMPACT_PPI_LIMIT=${COMPACT_PPI_LIMIT:-10}', wrapper_text)
        self.assertIn('COMPACT_GO_SPECULATION_LIMIT=${COMPACT_GO_SPECULATION_LIMIT:-8}', wrapper_text)
        self.assertIn('MIN_NEW_TOKENS=${MIN_NEW_TOKENS:-1}', wrapper_text)
        self.assertIn('MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-10000}', wrapper_text)
        self.assertIn('if [ -z "${ROLLOUT_LOGPROB_MICROBATCH_SIZE+x}" ]; then', wrapper_text)
        self.assertIn('ROLLOUT_LOGPROB_MICROBATCH_SIZE=1', wrapper_text)
        self.assertIn('ROLLOUT_LOGPROB_MICROBATCH_SIZE=4', wrapper_text)
        self.assertIn('SAMPLING_CONTRACT=${SAMPLING_CONTRACT:-"auto"}', wrapper_text)
        self.assertIn('TEMPERATURE=${TEMPERATURE:-1.0}', wrapper_text)
        self.assertIn('TOP_P=${TOP_P:-0.95}', wrapper_text)
        self.assertIn('TOP_K=${TOP_K:-20}', wrapper_text)
        self.assertIn('if [ "${CHECKPOINT_ARTIFACT_NAME+x}" = "x" ]; then', wrapper_text)
        self.assertIn('RL_RUN_FAMILY="rl-sft"', wrapper_text)
        self.assertIn('RL_RUN_FAMILY="rl-paper"', wrapper_text)
        self.assertIn('WANDB_RUN_NAME=${WANDB_RUN_NAME:-"${RL_RUN_FAMILY}-${TIMESTAMP}"}', wrapper_text)
        self.assertIn('--asset-key reasoning_dataset', wrapper_text)
        self.assertIn('resolve_existing_dir()', wrapper_text)
        self.assertIn('is_valid_hf_model_dir()', wrapper_text)
        self.assertIn('"$PYTHON_BIN" "$MODEL_SOURCE_RESOLVER"', wrapper_text)
        self.assertIn('"$PYTHON_BIN" - "$config_path"', wrapper_text)
        self.assertIn('find "$RESOLVED_TRAIN_SFT_DIR" -maxdepth 2 -type f -name "*best*.ckpt"', wrapper_text)
        self.assertIn('SFT_CKPT_PATH="$RESOLVED_TRAIN_SFT_DIR/last.ckpt"', wrapper_text)
        self.assertIn('Error: BASE_CHECKPOINT_LOCAL_DIR is not a valid HF model directory', wrapper_text)
        self.assertIn('--- Removing invalid converted HF train-sft-output at $TRAIN_SFT_HF_DIR', wrapper_text)
        self.assertIn('Error: RL init model directory is not a valid HF model directory', wrapper_text)
        self.assertIn('--lora_rank "$SFT_CONVERSION_LORA_RANK"', wrapper_text)
        self.assertIn('--lora_alpha "$SFT_CONVERSION_LORA_ALPHA"', wrapper_text)
        self.assertIn('--lora_dropout "$SFT_CONVERSION_LORA_DROPOUT"', wrapper_text)
        self.assertIn('--seed "$SEED"', wrapper_text)
        self.assertIn('--ia_file_path "$IA_FILE_PATH"', wrapper_text)
        self.assertIn('--require_ia_file "$REQUIRE_IA_FILE"', wrapper_text)
        self.assertIn('--weave_trace_budget "$WEAVE_TRACE_BUDGET"', wrapper_text)
        self.assertIn('--weave_trace_full_group_count "$WEAVE_TRACE_FULL_GROUP_COUNT"', wrapper_text)
        self.assertIn('--weave_trace_full_rollouts_per_group "$WEAVE_TRACE_FULL_ROLLOUTS_PER_GROUP"', wrapper_text)
        self.assertIn('--loss_type "$LOSS_TYPE"', wrapper_text)
        self.assertIn('--steps_per_generation "$STEPS_PER_GENERATION"', wrapper_text)
        self.assertIn('--num_iterations "$NUM_ITERATIONS"', wrapper_text)
        self.assertIn('--min_new_tokens "$MIN_NEW_TOKENS"', wrapper_text)
        self.assertIn('--rollout_logprob_microbatch_size "$ROLLOUT_LOGPROB_MICROBATCH_SIZE"', wrapper_text)
        self.assertIn('--min_p "$MIN_P"', wrapper_text)
        self.assertIn('--repetition_penalty "$REPETITION_PENALTY"', wrapper_text)
        self.assertIn('--clip_epsilon_low "$CLIP_EPSILON_LOW"', wrapper_text)
        self.assertIn('--clip_epsilon_high "$CLIP_EPSILON_HIGH"', wrapper_text)
        self.assertIn('--reward_scaling "$REWARD_SCALING"', wrapper_text)
        self.assertIn('--importance_sampling_cap "$IMPORTANCE_SAMPLING_CAP"', wrapper_text)
        self.assertIn('--reward_prediction_source "$REWARD_PREDICTION_SOURCE"', wrapper_text)
        self.assertIn('--include_protein_function_summary "$INCLUDE_PROTEIN_FUNCTION_SUMMARY"', wrapper_text)
        self.assertIn('--continuation_mode "$CONTINUATION_MODE"', wrapper_text)
        self.assertIn('--reasoning_prompt_style "$REASONING_PROMPT_STYLE"', wrapper_text)
        self.assertIn('--compact_interpro_limit "$COMPACT_INTERPRO_LIMIT"', wrapper_text)
        self.assertIn('--compact_ppi_limit "$COMPACT_PPI_LIMIT"', wrapper_text)
        self.assertIn('--compact_go_speculation_limit "$COMPACT_GO_SPECULATION_LIMIT"', wrapper_text)
        self.assertIn('--sampling_contract "$SAMPLING_CONTRACT"', wrapper_text)
        self.assertIn('--per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"', wrapper_text)
        self.assertIn('--per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"', wrapper_text)
        self.assertIn('TRAIN_LAUNCH_PREFIX=()', wrapper_text)
        self.assertIn('stdbuf -oL -eL "${TRAIN_COMMAND[@]}" "${TRAIN_LAUNCH_PREFIX[@]}" train_protein_grpo.py', wrapper_text)

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

    def test_go_overlap_reward_uses_reasoning_trace_when_configured(self):
        completion = "<think>Reasoning mentions GO:0007165.</think><answer>GO:0005515</answer>"
        sample_meta = {"go_bp": "GO:0007165"}

        with mock.patch.dict(GRPO.REWARD_CONTEXT, {"reward_prediction_source": "reasoning_trace"}):
            self.assertGreater(GRPO.go_overlap_reward(completion, sample_meta), 0.0)

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

        self.assertEqual(GRPO.summary_schema_reward(completion, {"go_aspect": "bp"}), 1.0)

    def test_summary_schema_reward_accepts_go_summary_without_function_summary(self):
        completion = (
            "<think>reasoning</think>"
            "<|GO_SUMMARY_START|>\nBP: GO:0007165\n<|GO_SUMMARY_END|>"
        )

        self.assertEqual(GRPO.summary_schema_reward(completion, {"go_aspect": "bp"}), 1.0)

    def test_go_presence_reward_penalizes_freeform_answers_without_go_ids(self):
        completion = "<think>reasoning</think><answer>This protein likely regulates signaling.</answer>"

        self.assertLess(GRPO.go_presence_reward(completion, {"go_bp": "GO:0007165"}), 0.0)

    def test_go_presence_reward_rewards_tagged_go_summary(self):
        completion = (
            "<think>reasoning</think>"
            "<|GO_SUMMARY_START|>\nBP: GO:0007165\n<|GO_SUMMARY_END|>\n\n"
            "<|FUNCTION_SUMMARY_START|>\nKinase-linked signaling regulator.\n<|FUNCTION_SUMMARY_END|>"
        )

        self.assertEqual(GRPO.go_presence_reward(completion, {"go_bp": "GO:0007165"}), 1.0)

    def test_go_presence_reward_accepts_unstructured_answer_tag_go_ids_in_trace_mode(self):
        completion = "<think>reasoning</think><answer>GO:0007165</answer>"

        self.assertGreater(GRPO.go_presence_reward(completion, {"go_bp": "GO:0007165"}), 0.0)

    def test_build_generation_kwargs_omits_sampling_controls_for_greedy_eval(self):
        args = GRPO.parse_args(
            [
                "--text_model_name",
                "hf-model",
                "--eval_do_sample",
                "false",
            ]
        )
        tokenizer = type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 1, "encode": lambda self, text, add_special_tokens=False: [1]})()

        kwargs = GRPO.build_generation_kwargs(args, tokenizer, for_eval=True)

        self.assertFalse(kwargs["do_sample"])
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)
        self.assertNotIn("top_k", kwargs)

    def test_build_generation_kwargs_uses_checkpoint_native_sampling_for_paper_native_train(self):
        args = GRPO.parse_args(
            [
                "--text_model_name",
                "hf-model",
            ]
        )
        tokenizer = type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 1, "encode": lambda self, text, add_special_tokens=False: [1]})()

        kwargs = GRPO.build_generation_kwargs(args, tokenizer, for_eval=False)

        self.assertNotIn("do_sample", kwargs)
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)
        self.assertNotIn("top_k", kwargs)
        self.assertNotIn("stopping_criteria", kwargs)

    def test_go_aspect_coverage_reward_tracks_requested_aspects(self):
        completion = (
            "<think>reasoning</think>"
            "<|GO_SUMMARY_START|>\nMF: GO:0005515\nBP: GO:0007165\n<|GO_SUMMARY_END|>\n\n"
            "<|FUNCTION_SUMMARY_START|>\nSignal adaptor.\n<|FUNCTION_SUMMARY_END|>"
        )

        self.assertEqual(
            GRPO.go_aspect_coverage_reward(
                completion,
                {"go_bp": "GO:0007165", "go_mf": "GO:0005515", "go_cc": "GO:0005737"},
            ),
            2.0 / 3.0,
        )

    def test_inspect_completion_uses_reasoning_trace_go_ids_by_default(self):
        completion = "<think>reasoning</think>GO:0007165</tool_call>"
        meta = GRPO.inspect_completion(completion)

        self.assertEqual(meta["predicted_go_ids"], ["GO:0007165"])
        self.assertEqual(meta["prediction_source"], "reasoning_trace")

    def test_inspect_completion_in_reasoning_trace_mode_keeps_trace_go_ids(self):
        completion = (
            "<think>Reasoning mentions GO:0001111 speculatively.</think>"
            "<answer>\n"
            "Speculative note GO:0001111 should not count.\n"
            "<|GO_SUMMARY_START|>\nBP: GO:0007165\n<|GO_SUMMARY_END|>\n\n"
            "<|FUNCTION_SUMMARY_START|>\nSignal adaptor.\n<|FUNCTION_SUMMARY_END|>\n"
            "</answer>"
        )
        GRPO.inspect_completion_text.cache_clear()
        with mock.patch.dict(GRPO.REWARD_CONTEXT, {"reward_prediction_source": "reasoning_trace"}):
            meta = GRPO.inspect_completion(completion)

        self.assertIn("GO:0001111", meta["predicted_go_ids"])
        self.assertEqual(meta["prediction_source"], "reasoning_trace")

    def test_go_overlap_reward_uses_reasoning_trace_go_ids(self):
        completion = "<think>reasoning</think><answer>GO:0007165</answer>"

        with mock.patch.dict(GRPO.REWARD_CONTEXT, {"reward_prediction_source": "reasoning_trace"}):
            self.assertGreater(GRPO.go_overlap_reward(completion, {"go_bp": "GO:0007165"}), 0.0)

    def test_go_overlap_reward_uses_full_completion_when_think_never_closes(self):
        completion = "<think>Reasoning cites GO:0007165 and GO:0005515"

        self.assertGreater(GRPO.go_overlap_reward(completion, {"go_bp": "GO:0007165"}), 0.0)

    def test_terminal_summary_markers_include_go_summary_end(self):
        self.assertIn(GRPO.GO_SUMMARY_END, GRPO.TERMINAL_SUMMARY_MARKERS)
        self.assertNotIn(GRPO.FUNCTION_SUMMARY_END, GRPO.TERMINAL_SUMMARY_MARKERS)

    def test_ia_weighted_f1_reward_propagates_go_terms(self):
        completion = (
            "<think>reasoning</think>"
            "<|GO_SUMMARY_START|>\nBP: GO:0009966\n<|GO_SUMMARY_END|>\n\n"
            "<|FUNCTION_SUMMARY_START|>\nSignal response regulator.\n<|FUNCTION_SUMMARY_END|>"
        )

        self.assertGreater(
            GRPO.ia_weighted_f1_reward(completion, {"go_bp": "GO:0009967"}),
            0.0,
        )

    def test_compute_batch_relative_advantages_uses_global_std(self):
        grouped_advantages, global_std = GRPO.compute_batch_relative_advantages(
            [[1.0, 1.0, 3.0], [0.0, 0.0, 2.0]],
            epsilon_std=1e-6,
            reward_scaling="batch",
        )

        self.assertGreater(global_std, 0.0)
        self.assertEqual(len(grouped_advantages), 2)
        self.assertAlmostEqual(sum(grouped_advantages[0]) + sum(grouped_advantages[1]), 0.0, places=5)

    def test_build_batch_semantics_matches_single_node_8gpu_target(self):
        args = mock.Mock(train_batch_size=1, eval_batch_size=4, num_generations=24)

        semantics = GRPO.build_batch_semantics(args, world_size=8)

        self.assertEqual(semantics["per_device_train_batch_size"], 1)
        self.assertEqual(semantics["per_device_eval_batch_size"], 4)
        self.assertEqual(semantics["world_size"], 8)
        self.assertEqual(semantics["global_unique_proteins_per_step"], 8)
        self.assertEqual(semantics["global_num_trajectories_per_step"], 192)

    def test_truncation_penalty_reward_penalizes_long_non_terminal_output(self):
        completion = "<think>reasoning</think>" + " word" * 330

        self.assertLess(GRPO.truncation_penalty_reward(completion, {}), 0.0)

    def test_truncation_penalty_reward_rewards_terminal_summary(self):
        completion = (
            "<think>reasoning</think>"
            "<|GO_SUMMARY_START|>\nBP: GO:0007165\n<|GO_SUMMARY_END|>\n"
            "<|FUNCTION_SUMMARY_START|>\nSignal adaptor.\n<|FUNCTION_SUMMARY_END|>"
        )

        self.assertGreater(GRPO.truncation_penalty_reward(completion, {}), 0.0)

    def test_standardize_group_rewards_returns_zeroes_for_constant_group(self):
        self.assertEqual(GRPO.standardize_group_rewards([0.5, 0.5, 0.5]), [0.0, 0.0, 0.0])

    def test_compute_group_rewards_combines_named_components(self):
        completion = (
            "<think>reasoning</think>"
            "<|GO_SUMMARY_START|>\nBP: GO:0007165\n<|GO_SUMMARY_END|>\n\n"
            "<|FUNCTION_SUMMARY_START|>\nSignal adaptor.\n<|FUNCTION_SUMMARY_END|>"
        )
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
        self.assertEqual(args.train_batch_size, 1)
        self.assertEqual(args.eval_batch_size, 4)
        self.assertEqual(args.max_eval_samples, 200)
        self.assertEqual(args.eval_sample_strategy, "stratified_aspect_profile")
        self.assertEqual(args.max_eval_batches, 0)
        self.assertEqual(args.loss_type, "dr_grpo")
        self.assertEqual(args.steps_per_generation, 2)
        self.assertEqual(args.num_iterations, 1)
        self.assertEqual(args.num_generations, 24)
        self.assertEqual(args.max_new_tokens, 10000)
        self.assertEqual(args.rollout_logprob_microbatch_size, 4)
        self.assertEqual(args.temperature, 1.0)
        self.assertEqual(args.top_p, 0.95)
        self.assertEqual(args.top_k, 20)
        self.assertEqual(args.repetition_penalty, 1.0)
        self.assertEqual(args.clip_epsilon_low, 7e-4)
        self.assertEqual(args.clip_epsilon_high, 9e-4)
        self.assertEqual(args.importance_sampling_cap, 2.0)
        self.assertFalse(args.eval_do_sample)
        self.assertEqual(args.eval_temperature, 0.1)
        self.assertEqual(args.eval_top_p, 0.9)
        self.assertEqual(args.eval_top_k, 20)
        self.assertEqual(args.rotating_eval_every_n_steps, 100)
        self.assertEqual(args.rotating_eval_max_samples, 256)
        self.assertEqual(args.reward_funcs, "ia_weighted_f1")
        self.assertEqual(args.reward_weights, "1.0")
        self.assertEqual(args.continuation_mode, "paper_native")
        self.assertEqual(args.sampling_contract, "checkpoint_native")
        self.assertEqual(args.reward_prediction_source, "reasoning_trace")
        self.assertEqual(args.min_new_tokens, 1)
        self.assertEqual(args.reasoning_prompt_style, "paper_native")
        self.assertEqual(args.compact_interpro_limit, 12)
        self.assertEqual(args.compact_ppi_limit, 10)
        self.assertEqual(args.compact_go_speculation_limit, 8)
        self.assertEqual(args.max_length_text, 512)
        self.assertFalse(args.add_uniprot_summary)
        self.assertTrue(args.bnb_4bit_use_double_quant)
        self.assertEqual(args.weave_trace_budget, 64)
        self.assertEqual(args.weave_trace_full_group_count, 4)
        self.assertEqual(args.weave_trace_full_rollouts_per_group, 24)
        self.assertTrue(args.gradient_checkpointing)
        self.assertTrue(args.disable_model_dropout)
        self.assertFalse(args.reward_final_answer_only)
        self.assertTrue(args.require_ia_file)
        self.assertFalse(args.ablation_from_paper_rl)

    def test_require_training_ia_file_raises_when_missing(self):
        args = mock.Mock(ia_file_path="", require_ia_file=True)

        with self.assertRaises(FileNotFoundError):
            GRPO.require_training_ia_file(args, ["ia_weighted_f1"])

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

    def test_build_rollout_observability_detects_summary_end_and_marker_index(self):
        if torch is None:
            self.skipTest("torch is not available in the contract test environment")

        class FakeTokenizer:
            eos_token_id = 99

            def encode(self, text, add_special_tokens=False):
                mapping = {
                    GRPO.GO_SUMMARY_START: [7, 8],
                }
                return mapping.get(text, [])

        completion_ids = torch.tensor([1, 7, 8, 3, 4])
        completion_text = f"{GRPO.GO_SUMMARY_START}\nBP: GO:0007165\n{GRPO.GO_SUMMARY_END}"

        observability = GRPO.build_rollout_observability(
            FakeTokenizer(),
            completion_ids,
            completion_text,
            total_reward=1.0,
            max_new_tokens=10000,
        )

        self.assertEqual(observability["stop_reason"], "summary_end")
        self.assertEqual(observability["first_go_summary_token_idx"], 1)
        self.assertTrue(observability["has_go_summary_end"])
        self.assertFalse(observability["max_new_tokens_hit"])
        self.assertTrue(observability["reward_nonzero"])

    def test_build_rollout_observability_detects_max_token_stop(self):
        if torch is None:
            self.skipTest("torch is not available in the contract test environment")

        class FakeTokenizer:
            eos_token_id = 99

            def encode(self, text, add_special_tokens=False):
                return []

        completion_ids = torch.tensor([10, 11, 12, 13])

        observability = GRPO.build_rollout_observability(
            FakeTokenizer(),
            completion_ids,
            "plain reasoning without summary",
            total_reward=0.0,
            max_new_tokens=4,
        )

        self.assertEqual(observability["stop_reason"], "max_tokens")
        self.assertEqual(observability["first_go_summary_token_idx"], -1)
        self.assertFalse(observability["has_go_summary_end"])
        self.assertTrue(observability["max_new_tokens_hit"])
        self.assertFalse(observability["reward_nonzero"])

    def test_slice_rollout_group_reindexes_batch_idx_map_for_microbatches(self):
        if torch is None:
            self.skipTest("torch is not available in the contract test environment")

        rollout_group = {
            "combined_input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            "combined_attention_mask": torch.ones((4, 3), dtype=torch.long),
            "completion_attention": torch.tensor([[1, 1], [1, 0], [1, 1], [0, 0]], dtype=torch.long),
            "prompt_token_len": 1,
            "protein_sequences": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "batch_idx_map": [0, 0, 1, 1, 2, 2, 3, 3],
            "structure_coords": torch.arange(4, dtype=torch.float32).unsqueeze(-1),
            "go_aspects": ["bp", "bp", "mf", "mf"],
        }

        sliced = GRPO.slice_rollout_group(rollout_group, 1, 3)

        self.assertEqual(tuple(sliced["combined_input_ids"].shape), (2, 3))
        self.assertEqual(sliced["protein_sequences"], ["A", "B", "A", "B"])
        self.assertEqual(sliced["batch_idx_map"], [0, 0, 1, 1])
        self.assertEqual(sliced["go_aspects"], ["bp", "mf"])
        self.assertTrue(torch.equal(sliced["structure_coords"], torch.tensor([[1.0], [2.0]])))

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
            repetition_penalty=1.0,
            eval_do_sample=False,
            eval_temperature=0.1,
            eval_top_p=0.9,
            eval_top_k=20,
            min_new_tokens=1,
            max_new_tokens=32,
            min_p=0.0,
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
        self.assertFalse(model.generate_calls[0]["do_sample"])
        self.assertNotIn("temperature", model.generate_calls[0])
        self.assertNotIn("top_p", model.generate_calls[0])
        self.assertNotIn("top_k", model.generate_calls[0])
        self.assertEqual(model.generate_calls[0]["repetition_penalty"], 1.0)
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
        self.assertIn('"loss_policy_ratio_mean"', source)
        self.assertIn('"loss_policy_ratio_max"', source)
        self.assertIn('"loss_learning_rate"', source)
        self.assertIn('"loss_grad_norm"', source)
        self.assertIn('"eval_reward"', source)
        self.assertIn('"train_skipped_update"', source)
        self.assertIn('diagnostic/{reward_name}', source)
        self.assertIn("compute_batch_relative_advantages(", source)
        self.assertIn("compute_old_policy_sequence_log_probs(", source)
        self.assertIn("slice_rollout_group(", source)
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

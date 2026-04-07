from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "train_protein_llm.py"
WRAPPER_PATH = ROOT / "scripts" / "sh_train_protein_qwen_staged.sh"
MODEL_PATH = ROOT / "bioreason2" / "models" / "protein_llm.py"


class TrainProteinLLMTrackingContractsTest(unittest.TestCase):
    def test_train_sft_uses_weave_trace_for_generation(self):
        source = TRAIN_PATH.read_text()
        self.assertIn('@weave.op(name="train_sft_generation_trace")', source)
        self.assertIn("maybe_trace_sft_generation(", source)
        self.assertIn("prefer_original_generate=True", source)
        self.assertIn('"failure_reason"', source)
        self.assertIn('"assistant_marker_found"', source)
        self.assertNotIn("wandb.Table(", source)
        self.assertNotIn('step_id = f"gen_', source)

    def test_train_sft_logs_core_metrics_to_wandb(self):
        source = TRAIN_PATH.read_text()
        for expected in [
            '"train_loss"',
            '"train_loss_epoch"',
            '"val_loss"',
            '"val_loss_epoch"',
            '"lr_step"',
            '"lr_epoch"',
            '"trainer/global_step"',
        ]:
            self.assertIn(expected, source)

    def test_prefer_original_generate_uses_manual_decode_for_sample_traces(self):
        source = MODEL_PATH.read_text()
        self.assertIn("if prefer_original_generate and not do_sample:", source)
        self.assertIn("generated_inputs_embeds = text_inputs_embeds.clone()", source)
        self.assertIn("outputs = self.text_model(", source)
        self.assertIn("generated_ids = torch.cat([generated_ids, next_token], dim=1)", source)
        self.assertIn("generated_inputs_embeds = torch.cat([generated_inputs_embeds, next_token_embeds], dim=1)", source)

    def test_sft_wrapper_passes_weave_and_logs_more_frequently(self):
        wrapper = WRAPPER_PATH.read_text()
        self.assertIn('WEAVE_PROJECT=${WEAVE_PROJECT:-""}', wrapper)
        self.assertIn("WEAVE_TRACE_BUDGET=${WEAVE_TRACE_BUDGET:-64}", wrapper)
        self.assertIn('STAGE2_LOG_EVERY_N_STEPS=${STAGE2_LOG_EVERY_N_STEPS:-10}', wrapper)
        self.assertIn(
            'STAGE2_SAMPLE_GENERATION_EVERY_N_STEPS=${STAGE2_SAMPLE_GENERATION_EVERY_N_STEPS:-500}',
            wrapper,
        )
        self.assertIn('--weave_project "$WEAVE_PROJECT"', wrapper)
        self.assertIn('--weave_trace_budget "$WEAVE_TRACE_BUDGET"', wrapper)
        self.assertIn('--every_n_train_steps "$STAGE2_SAMPLE_GENERATION_EVERY_N_STEPS"', wrapper)


if __name__ == "__main__":
    unittest.main()

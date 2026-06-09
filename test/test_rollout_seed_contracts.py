from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCE_PATH = ROOT / "train.py"
VLLM_SOURCE_PATH = ROOT / "bioreason2" / "models" / "protein_vllm.py"


class RolloutSeedContractsTest(unittest.TestCase):
    def test_rollout_group_seed_changes_by_rank_and_generation(self):
        source = TRAIN_SOURCE_PATH.read_text()

        self.assertIn("generation_seed = int(self.args.seed)", source)
        self.assertIn("int(self.runtime.rank) * 100003", source)
        self.assertIn("self._generation_counter += 1", source)
        self.assertIn('"seed": generation_seed', source)

    def test_vllm_rollout_requests_get_distinct_sampling_seeds(self):
        source = VLLM_SOURCE_PATH.read_text()

        self.assertIn("base_seed_raw = generation_kwargs.get(\"seed\")", source)
        self.assertIn("for i in range(batch_size):", source)
        self.assertIn("per_req_kwargs[\"seed\"] = (base_seed + i) & 0xFFFFFFFF", source)
        self.assertIn("sampling_params_list.append(SamplingParams(**per_req_kwargs))", source)
        self.assertIn("sampling_params=sampling_params_list", source)


if __name__ == "__main__":
    unittest.main()

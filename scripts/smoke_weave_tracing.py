"""Emit a real Weave trace tree using the RunTracker, no model required.

This script exercises the RL training tracker against a live Weave project so
you can visually verify the ``train_rl_step`` / ``train_rl_query`` /
``train_rl_rollout_generate`` / ``train_rl_rollout_item`` /
``train_rl_reward_score`` / ``train_rl_reward_item`` tree in the Weave UI
without having to spin up a GPU or a SUNK allocation.

Usage (from repo root):

    export WANDB_API_KEY=...          # must be authenticated
    python scripts/smoke_weave_tracing.py \
        --weave_project wandb-healthcare/bioreason-pro-custom

The script fabricates a couple of proteins, a few fake completions per
rollout, and a trivial reward function (1.0 if the target GO id appears in the
completion, 0.0 otherwise). Everything else (span nesting, attribute
propagation, per-rollout/per-reward child emission) is the real code path
from ``train_protein_grpo.py``.

By default the script bypasses ``wandb.init`` so it only talks to Weave;
pass ``--use_wandb`` if you also want a W&B run created.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _import_grpo_module():
    sys.path.insert(0, str(_repo_root()))
    import train_protein_grpo as grpo  # type: ignore

    return grpo


def parse_smoke_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weave_project",
        required=True,
        help="Weave/W&B project, e.g. 'wandb-healthcare/bioreason-pro-custom'.",
    )
    parser.add_argument(
        "--run_name",
        default="rl-weave-tracing-smoke",
        help="Run name surfaced on every span.",
    )
    parser.add_argument("--num_steps", type=int, default=2)
    parser.add_argument("--queries_per_step", type=int, default=1)
    parser.add_argument("--rollouts_per_query", type=int, default=4)
    parser.add_argument(
        "--reasoning_prompt_style",
        default="paper_native_tight",
        help="Attached to every span as an attribute so you can filter by prompt variant.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Also create a live W&B run; without this flag wandb init is bypassed.",
    )
    return parser.parse_args(argv)


SAMPLE_PROTEINS = [
    ("Q8IWT5", ["GO:0008150", "GO:0003674"]),
    ("P04637", ["GO:0005515", "GO:0003700"]),
    ("P01116", ["GO:0007165"]),
    ("P15056", ["GO:0004672", "GO:0005524"]),
]


def _build_fake_completions(target_go_ids, rollouts_per_query):
    completions = []
    for rollout_idx in range(rollouts_per_query):
        if rollout_idx == 0:
            body = f"<think>Analyzing the sample.</think><|FINAL_ANSWER|>{target_go_ids[0]}<|/FINAL_ANSWER|>"
        elif rollout_idx == 1 and len(target_go_ids) >= 2:
            body = (
                "<think>Predicts two terms.</think>"
                f"<|FINAL_ANSWER|>{target_go_ids[0]},{target_go_ids[1]}<|/FINAL_ANSWER|>"
            )
        elif rollout_idx == rollouts_per_query - 1:
            body = "<think>Best guess is wrong.</think><|FINAL_ANSWER|>GO:0000000<|/FINAL_ANSWER|>"
        else:
            body = f"<think>Step {rollout_idx}.</think><|FINAL_ANSWER|>{target_go_ids[0]}<|/FINAL_ANSWER|>"
        completions.append(body)
    return completions


def _fake_reward(completion: str, target_go_ids) -> float:
    matches = sum(1 for term in target_go_ids if term in completion)
    return float(matches) / float(len(target_go_ids))


def run_smoke(smoke_args: argparse.Namespace) -> None:
    GRPO = _import_grpo_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "smoke-output"
        output_dir.mkdir(parents=True, exist_ok=True)

        cli_args = GRPO.parse_args(
            [
                "--text_model_name",
                "/tmp/demo-model",
                "--output_dir",
                str(output_dir),
            ]
        )
        cli_args.trace_rollouts_to_weave = "true"
        cli_args.weave_project = smoke_args.weave_project
        cli_args.run_name = smoke_args.run_name
        cli_args.reasoning_prompt_style = smoke_args.reasoning_prompt_style
        total_rollouts = smoke_args.num_steps * smoke_args.queries_per_step * smoke_args.rollouts_per_query
        cli_args.weave_trace_budget = max(total_rollouts, 4)
        cli_args.weave_trace_full_group_count = smoke_args.num_steps * smoke_args.queries_per_step
        cli_args.weave_trace_full_rollouts_per_group = smoke_args.rollouts_per_query

        runtime = GRPO.DistributedRuntime(
            enabled=False,
            rank=0,
            world_size=1,
            local_rank=0,
            device="cpu",
        )

        patches = []
        if not smoke_args.use_wandb:
            patches.append(
                mock.patch.object(
                    GRPO.RunTracker,
                    "_maybe_init_wandb",
                    return_value=None,
                )
            )

        with _stack_context(patches):
            tracker = GRPO.RunTracker(
                args=cli_args,
                config={"smoke_test": True},
                output_dir=output_dir,
                runtime=runtime,
            )

            if tracker.weave_client is None:
                raise SystemExit(
                    "Weave did not initialize — check that WANDB_API_KEY is set "
                    "and that --weave_project looks like '<entity>/<project>'."
                )

            print(
                f"Emitting {smoke_args.num_steps} steps × "
                f"{smoke_args.queries_per_step} queries × "
                f"{smoke_args.rollouts_per_query} rollouts to "
                f"https://wandb.ai/{smoke_args.weave_project}/weave/calls",
                flush=True,
            )

            try:
                for step_idx in range(smoke_args.num_steps):
                    _emit_step(
                        GRPO,
                        tracker,
                        step_idx=step_idx,
                        queries_per_step=smoke_args.queries_per_step,
                        rollouts_per_query=smoke_args.rollouts_per_query,
                    )
            finally:
                tracker.finish()

        print(
            "Done. Open the link above and filter calls by "
            "'stage=step' to see the per-step tree.",
            flush=True,
        )


def _stack_context(patches):
    """Return a context manager that enters a list of patches (possibly empty)."""

    from contextlib import ExitStack

    stack = ExitStack()
    for patcher in patches:
        stack.enter_context(patcher)
    return stack


def _emit_step(GRPO, tracker, *, step_idx, queries_per_step, rollouts_per_query):
    with tracker.weave_step_span(
        step=step_idx + 1,
        queries_per_step=queries_per_step,
        rollouts_per_query=rollouts_per_query,
    ):
        for query_idx in range(queries_per_step):
            protein_id, target_go_ids = SAMPLE_PROTEINS[
                (step_idx * queries_per_step + query_idx) % len(SAMPLE_PROTEINS)
            ]
            completions = _build_fake_completions(target_go_ids, rollouts_per_query)

            sample_meta = {
                "protein_id": protein_id,
                "split": "train",
                "go_bp": ",".join(target_go_ids),
            }
            query = GRPO.PreparedQuery(
                input_ids=None,
                attention_mask=None,
                protein_sequences=[f"MSEQ_{protein_id}"],
                batch_idx_map=[0],
                structure_coords=None,
                go_aspects=["bp"],
                sample_meta=sample_meta,
                prompt_text=f"Predict GO terms for {protein_id}.",
                multimodal_cache=None,
            )
            sampling = GRPO.SamplingSpec(
                temperature=1.0,
                top_k=-1,
                top_p=1.0,
                min_p=0.0,
                repetition_penalty=1.0,
                max_new_tokens=512,
            )

            with tracker.weave_query_span(
                step=step_idx + 1,
                split="train",
                protein_id=protein_id,
                rollouts_per_query=rollouts_per_query,
            ):
                outputs = tracker.trace_rollout_call(
                    step=step_idx + 1,
                    split="train",
                    query=query,
                    repeat_count=rollouts_per_query,
                    sampling=sampling,
                    generator=lambda captured=list(completions): list(captured),
                    tokenizer=None,
                )

                rewards = [_fake_reward(c, target_go_ids) for c in outputs]
                tracker.trace_reward_call(
                    step=step_idx + 1,
                    split="train",
                    query=query,
                    completions=outputs,
                    callback=lambda r=list(rewards): list(r),
                )

            print(
                f"  step={step_idx + 1} protein={protein_id} "
                f"rewards={['%.2f' % r for r in rewards]}",
                flush=True,
            )


def main(argv=None) -> None:
    if not os.environ.get("WANDB_API_KEY"):
        print(
            "⚠️  WANDB_API_KEY is not set — weave.init will fail. "
            "Run `wandb login` or `export WANDB_API_KEY=...` first.",
            file=sys.stderr,
        )
    args = parse_smoke_args(argv)
    run_smoke(args)


if __name__ == "__main__":
    main()

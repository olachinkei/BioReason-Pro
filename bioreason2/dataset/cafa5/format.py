from typing import Any, Dict
from pprint import pprint


def _combine_paper_contract_sections(reasoning: str, answer: str) -> str:
    reasoning = (reasoning or "").strip()
    answer = (answer or "").strip()
    if not reasoning:
        return answer
    if not answer:
        return reasoning
    return f"{reasoning}\n{answer}"


def format_cafa5_for_protein_llm(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a CAFA5 example into the required chat format for Protein-LLM.
    """
    assistant_reasoning = example["prompt"]["assistant_reasoning"].strip()
    assistant_answer = example["prompt"]["assistant_answer"].strip()
    preserve_paper_contract = (
        "<|REASONING|>" in assistant_reasoning
        or "<|FINAL_ANSWER|>" in assistant_answer
    )

    assistant_message: Dict[str, Any] = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": (
                    _combine_paper_contract_sections(assistant_reasoning, assistant_answer)
                    if preserve_paper_contract
                    else assistant_answer
                ),
            },
        ],
    }
    if not preserve_paper_contract:
        assistant_message["reasoning_content"] = assistant_reasoning

    return {
        "prompt": [
            # {
            #     "role": "system",
            #     "content": example["prompt"]["system"].strip(),
            # },
            {
                "role": "user",
                "content": [
                    {"type": "protein", "text": None},
                    {"type": "go_graph", "text": None},
                    {
                        "type": "text",
                        "text": f"{example['prompt']['system'].strip()}\n\n{example['prompt']['user'].strip()}",
                    },
                ],
            },
            assistant_message,
        ],
        "protein_sequences": [
            example["sequence"],
        ],
        "structure_path": example.get("structure_path"),
        "go_aspect": example.get("go_aspect"),
        "answer": example["prompt"]["assistant_answer"].strip(),
        "ground_truth_go_terms": example.get("ground_truth_go_terms", ""),
    }


if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset("wanglab/cafa5", name="cafa5_reasoning", cache_dir="cafa5_reasoning_cache")
    train_df = ds["train"].to_pandas()

    # Get a specific protein
    prot_id = "A0A078CGE6"
    protein_data = train_df[train_df["protein_id"] == prot_id].iloc[0]

    # Format the training example
    formatted_example = format_cafa5_for_protein_llm(protein_data)
    pprint(formatted_example)

"""Shared dataset loaders for all three tasks.

Each loader returns (train, val, test) datasets with columns:
  - prompt: list of message dicts (user turn only)
  - teacher_prompt: list of message dicts (user turn with demonstration)
  - messages: list of message dicts (user + assistant, for SFT)
"""

import json
from string import Template
from datasets import Dataset, load_dataset


def load_tooluse(seed=42):
    """Load ToolAlpaca tool-use dataset.

    Returns:
        train (4046 examples), val (None), test (68 examples)
    """
    train_data = json.load(open("data/tooluse_data/train_data.json"))
    eval_data = json.load(open("data/tooluse_data/eval_data.json"))

    teacher_template = Template("""$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

    def format_examples(examples):
        formatted = []
        for ex in examples:
            golden_resp = ex.get("golden_response", [])
            if isinstance(golden_resp, list):
                resp_text = "\n".join(golden_resp)
            else:
                resp_text = str(golden_resp)

            formatted.append({
                "prompt": [{"role": "user", "content": ex["prompt"]}],
                "teacher_prompt": [{"role": "user", "content": teacher_template.substitute(
                    orig_content=ex["prompt"], output_text=resp_text
                )}],
                "messages": [
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": resp_text},
                ],
            })
        return formatted

    train = Dataset.from_list(format_examples(train_data)).shuffle(seed=seed)
    test = Dataset.from_list(format_examples(eval_data))
    return train, None, test


def load_science(seed=42):
    """Load SciKnowEval Chemistry L-3 subset.

    Returns:
        train (~75%), val (~5%), test (~20%)
    """
    ds = load_dataset("hicai-zju/SciKnowEval", split="test")
    df = ds.to_pandas()

    df["level"] = df["details"].apply(
        lambda d: d.get("level", "") if isinstance(d, dict) else ""
    )
    chem_l3 = df[(df["domain"] == "Chemistry") & (df["level"] == "L3")]

    formatted = []
    for _, row in chem_l3.iterrows():
        prompt_template = row["prompt"]
        if isinstance(prompt_template, dict):
            prompt_template = prompt_template.get("default", "")
        question = row["question"]
        answer = row.get("answerKey", "")

        choices = row.get("choices", {})
        if isinstance(choices, dict):
            choice_text = choices.get("text", [])
            choice_labels = choices.get("label", [])
            choices_str = "\n".join(
                f"{l}. {t}" for l, t in zip(choice_labels, choice_text)
            )
        else:
            choices_str = ""

        full_prompt = f"{prompt_template}\n\nQuestion: {question}\n{choices_str}"

        teacher_content = f"""{full_prompt}

This is an example for a response to the question:
The correct answer is {answer}.

Now answer with a response of your own."""

        formatted.append({
            "prompt": [{"role": "user", "content": full_prompt}],
            "teacher_prompt": [{"role": "user", "content": teacher_content}],
            "messages": [
                {"role": "user", "content": full_prompt},
                {"role": "assistant", "content": f"The correct answer is {answer}."},
            ],
            "_answer": answer,
        })

    dataset = Dataset.from_list(formatted).shuffle(seed=seed)
    n = len(dataset)
    train_n = int(n * 0.75)
    val_n = int(n * 0.05)
    train = dataset.select(range(train_n))
    val = dataset.select(range(train_n, train_n + val_n))
    test = dataset.select(range(train_n + val_n, n))
    return train, val, test


def load_medical(seed=42):
    """Load HuatuoGPT-o1 English medical dataset.

    Returns:
        train (19000), val (500), test (500)
    """
    ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    ds = ds.shuffle(seed=seed)

    formatted = []
    for ex in ds:
        question = ex["Question"]
        response = ex["Response"]

        teacher_content = f"""{question}

This is an example for a response to the question:
{response}

Now answer with a response of your own, including the thinking process.
"""
        formatted.append({
            "prompt": [{"role": "user", "content": question}],
            "teacher_prompt": [{"role": "user", "content": teacher_content}],
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ],
        })

    dataset = Dataset.from_list(formatted)
    train = dataset.select(range(min(19000, len(dataset))))
    val = dataset.select(range(19000, min(19500, len(dataset))))
    test = dataset.select(range(19500, min(20000, len(dataset))))
    return train, val, test


TASK_LOADERS = {
    "tooluse": load_tooluse,
    "science": load_science,
    "medical": load_medical,
}

"""Science Q&A evaluation via exact match on answer letter.

Generates with vLLM (greedy), extracts answer letter (A/B/C/D) via regex,
compares against ground truth.
"""

import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams

from src.data import load_science


def extract_answer_letter(text: str) -> str:
    """Extract answer letter from model response.

    Tries multiple patterns in order of specificity:
    1. "The correct answer is X"
    2. "The answer is X"
    3. "Answer: X"
    4. Standalone letter at end of response
    """
    text = text.strip()

    patterns = [
        r'[Tt]he\s+correct\s+answer\s+is\s*[:\s]*([A-D])',
        r'[Tt]he\s+answer\s+is\s*[:\s]*([A-D])',
        r'[Aa]nswer\s*:\s*([A-D])',
        r'\b([A-D])\s*\.\s*$',
        r'\b([A-D])\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()

    return ""


def eval_science(model_path, seed=42, output_file=None, gpu_memory_utilization=0.9):
    """Run science Q&A evaluation.

    Args:
        model_path: Path to model checkpoint or HF model name.
        seed: Random seed for dataset split.
        output_file: Path to save results JSON.
        gpu_memory_utilization: vLLM GPU memory fraction.

    Returns:
        Dict with accuracy and per-example results.
    """
    _, _, test = load_science(seed=seed)
    print(f"Science test set: {len(test)} examples")

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(temperature=0, max_tokens=512)

    formatted_prompts = []
    for ex in test:
        messages = ex["prompt"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(text)

    outputs = llm.generate(formatted_prompts, sampling_params)

    correct = 0
    results = []
    for i, (output, ex) in enumerate(zip(outputs, test)):
        generated_text = output.outputs[0].text
        pred_answer = extract_answer_letter(generated_text)
        gold_answer = ex["_answer"]
        match = pred_answer == gold_answer

        if match:
            correct += 1

        results.append({
            "id": i,
            "predicted": pred_answer,
            "gold": gold_answer,
            "correct": match,
            "generated_text": generated_text[:500],
        })

    accuracy = correct / len(test) * 100
    print(f"Science Accuracy: {correct}/{len(test)} = {accuracy:.1f}%")

    output_data = {
        "model_path": model_path,
        "task": "science",
        "num_examples": len(test),
        "accuracy": accuracy,
        "correct": correct,
        "results": results,
    }

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {output_file}")

    return output_data

"""Tool-use evaluation via regex matching + pass@k.

Moved from root eval_tooluse.py (unchanged logic).
"""

import json
import math
import re

import numpy as np
from vllm import LLM, SamplingParams


def parse_action(text: str) -> list[dict]:
    """Extract Action and Action_Input from model output."""
    results = []
    action_pattern = r'Action:\s*(\w+)\s*\n\s*Action[_ ]?Input:\s*(\{.*?\}|\S+)'
    matches = re.findall(action_pattern, text, re.DOTALL)
    for action, action_input in matches:
        try:
            parsed_input = json.loads(action_input)
        except json.JSONDecodeError:
            parsed_input = action_input.strip()
        results.append({"Action": action.strip(), "Action_Input": parsed_input})
    return results


def normalize_action_input(action_input):
    """Normalize action input for comparison (handles argument ordering)."""
    if isinstance(action_input, dict):
        return {k: normalize_action_input(v) for k, v in sorted(action_input.items())}
    if isinstance(action_input, str):
        try:
            parsed = json.loads(action_input)
            return normalize_action_input(parsed)
        except (json.JSONDecodeError, TypeError):
            return action_input
    return action_input


def actions_match(pred_actions: list[dict], gold_actions: list[dict]) -> bool:
    """Check if predicted actions match golden actions (order-insensitive for args)."""
    if len(pred_actions) != len(gold_actions):
        return False
    for pred, gold in zip(pred_actions, gold_actions):
        if pred.get("Action", "").strip() != gold.get("Action", "").strip():
            return False
        pred_input = normalize_action_input(pred.get("Action_Input", {}))
        gold_input = normalize_action_input(gold.get("Action_Input", {}))
        if pred_input != gold_input:
            return False
    return True


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _normalize_gold(gold_actions):
    """Normalize golden answer Action_Input fields."""
    normalized = []
    for ga in gold_actions:
        ga_copy = dict(ga)
        if isinstance(ga_copy.get("Action_Input"), str):
            try:
                ga_copy["Action_Input"] = json.loads(ga_copy["Action_Input"])
            except (json.JSONDecodeError, TypeError):
                pass
        normalized.append(ga_copy)
    return normalized


def eval_greedy(llm, tokenizer, eval_data):
    """Greedy decoding accuracy (temperature=0)."""
    sampling_params = SamplingParams(temperature=0, max_tokens=1024)

    formatted_prompts = []
    for ex in eval_data:
        messages = [{"role": "user", "content": ex["prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(text)

    outputs = llm.generate(formatted_prompts, sampling_params)

    correct = 0
    results = []
    for i, (output, ex) in enumerate(zip(outputs, eval_data)):
        generated_text = output.outputs[0].text
        pred_actions = parse_action(generated_text)
        gold_actions = _normalize_gold(ex["golden_answer"])
        match = actions_match(pred_actions, gold_actions)
        if match:
            correct += 1
        results.append({
            "id": i,
            "instruction": ex.get("instruction", ""),
            "tool_name": ex.get("name", ""),
            "predicted_actions": pred_actions,
            "golden_actions": gold_actions,
            "generated_text": generated_text[:500],
            "correct": match,
        })

    accuracy = correct / len(eval_data) * 100
    return accuracy, correct, results


def eval_pass_at_k(llm, tokenizer, eval_data, n_samples=20, k_values=(1, 5, 10)):
    """pass@k evaluation with temperature=1.0, top-p=0.95."""
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        max_tokens=1024,
        n=n_samples,
    )

    formatted_prompts = []
    for ex in eval_data:
        messages = [{"role": "user", "content": ex["prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(text)

    outputs = llm.generate(formatted_prompts, sampling_params)

    pass_at_k_results = {k: [] for k in k_values}

    for i, (output, ex) in enumerate(zip(outputs, eval_data)):
        gold_actions = _normalize_gold(ex["golden_answer"])
        n_correct = 0
        for sample in output.outputs:
            pred_actions = parse_action(sample.text)
            if actions_match(pred_actions, gold_actions):
                n_correct += 1

        for k in k_values:
            if k <= n_samples:
                pak = pass_at_k(n_samples, n_correct, k)
                pass_at_k_results[k].append(pak)

    avg_pass_at_k = {}
    for k in k_values:
        if pass_at_k_results[k]:
            avg_pass_at_k[k] = float(np.mean(pass_at_k_results[k])) * 100

    return avg_pass_at_k, pass_at_k_results


def run_eval(model_path, eval_data_path="data/tooluse_data/eval_data.json",
             output_file=None, gpu_memory_utilization=0.9,
             n_samples=20, k_values=(1, 5, 10), skip_pass_at_k=False):
    """Run full tool-use evaluation. Returns results dict."""
    with open(eval_data_path) as f:
        eval_data = json.load(f)
    print(f"Loaded {len(eval_data)} evaluation examples")

    print(f"Loading model from {model_path}...")
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    print("\n>>> Greedy Accuracy (temperature=0) <<<")
    accuracy, correct, greedy_results = eval_greedy(llm, tokenizer, eval_data)
    print(f"Tool-Use Accuracy: {correct}/{len(eval_data)} = {accuracy:.1f}%")

    pass_at_k_scores = {}
    if not skip_pass_at_k:
        print(f"\n>>> pass@k (n={n_samples}, k={list(k_values)}, temp=1.0, top_p=0.95) <<<")
        pass_at_k_scores, _ = eval_pass_at_k(
            llm, tokenizer, eval_data, n_samples=n_samples, k_values=k_values
        )
        for k, score in pass_at_k_scores.items():
            print(f"  pass@{k}: {score:.1f}%")

    output = {
        "model_path": model_path,
        "num_examples": len(eval_data),
        "greedy_accuracy": accuracy,
        "greedy_correct": correct,
        "pass_at_k": {str(k): v for k, v in pass_at_k_scores.items()},
        "n_samples": n_samples,
        "greedy_results": greedy_results,
    }

    if output_file:
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return output

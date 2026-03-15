"""Medical evaluation: two-stage generate-then-judge.

Stage 1 (GPU): Generate responses with vLLM, save to JSON.
Stage 2 (CPU): GPT judge scores correctness via OpenAI API.

Supports --generate_only and --judge_only modes.
"""

import json
import os
from pathlib import Path

from src.data import load_medical


JUDGE_PROMPT = """You are an expert medical evaluator assessing whether a model's response correctly answers a medical question. Your task is to compare the model's response to the reference answer and determine if the model's response is:
1. CORRECT: The response contains the key medical information from the reference answer, even if phrased differently or includes additional correct medical details.
2. INCORRECT: The response is medically wrong, misses the main point, or provides incorrect medical information.
Focus on medical accuracy and completeness, not on writing style or verbosity.

[Medical Question]
{question}
[Reference Answer]
{reference_answer}
[Model Response]
{model_response}
Evaluate the model's response. Output ONLY one of: "CORRECT" or "INCORRECT"."""


def generate_responses(model_path, seed=42, output_file=None, gpu_memory_utilization=0.9):
    """Stage 1: Generate responses for medical test set."""
    from vllm import LLM, SamplingParams

    _, _, test = load_medical(seed=seed)
    print(f"Medical test set: {len(test)} examples")

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(temperature=0, max_tokens=1024)

    formatted_prompts = []
    for ex in test:
        messages = ex["prompt"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(text)

    outputs = llm.generate(formatted_prompts, sampling_params)

    results = []
    for i, (output, ex) in enumerate(zip(outputs, test)):
        generated_text = output.outputs[0].text
        # Extract question from prompt messages
        question = ex["prompt"][0]["content"]
        # Reference answer from messages (assistant turn)
        reference = ex["messages"][1]["content"]

        results.append({
            "id": i,
            "question": question,
            "reference_answer": reference,
            "model_response": generated_text,
        })

    output_data = {
        "model_path": model_path,
        "task": "medical",
        "stage": "generate",
        "num_examples": len(test),
        "results": results,
    }

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Generated responses saved to {output_file}")

    return output_data


def judge_responses(generations_file, output_file=None, judge_model="gpt-4o-mini"):
    """Stage 2: Use GPT judge to score responses.

    Requires OPENAI_API_KEY environment variable.
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Cannot run judge.")
        print("Set it with: export OPENAI_API_KEY=your_key")
        return None

    client = OpenAI(api_key=api_key)

    with open(generations_file) as f:
        data = json.load(f)

    results = data["results"]
    correct = 0
    judged = []

    for i, r in enumerate(results):
        prompt = JUDGE_PROMPT.format(
            question=r["question"],
            reference_answer=r["reference_answer"],
            model_response=r["model_response"],
        )

        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            verdict = response.choices[0].message.content.strip().upper()
        except Exception as e:
            print(f"  Error on example {i}: {e}")
            verdict = "ERROR"

        is_correct = "CORRECT" in verdict
        if is_correct:
            correct += 1

        judged.append({
            **r,
            "verdict": verdict,
            "correct": is_correct,
        })

        if (i + 1) % 50 == 0:
            print(f"  Judged {i+1}/{len(results)}, running accuracy: {correct/(i+1)*100:.1f}%")

    accuracy = correct / len(results) * 100
    print(f"\nMedical Accuracy (GPT judge): {correct}/{len(results)} = {accuracy:.1f}%")

    output_data = {
        "model_path": data["model_path"],
        "task": "medical",
        "stage": "judge",
        "judge_model": judge_model,
        "num_examples": len(results),
        "accuracy": accuracy,
        "correct": correct,
        "results": judged,
    }

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Judge results saved to {output_file}")

    return output_data

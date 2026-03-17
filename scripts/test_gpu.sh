#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64000M
#SBATCH --time=0-00:10
#SBATCH --output=logs/test-gpu-%j.out
#SBATCH --error=logs/test-gpu-%j.err
#SBATCH --job-name=test-gpu

cd ~/Self-Distillation
source setup_env.sh --job

python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('Memory:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', local_files_only=True)
print('Tokenizer loaded OK')

from vllm import LLM, SamplingParams
print('Loading model with vLLM...')
llm = LLM(model='Qwen/Qwen2.5-7B-Instruct', gpu_memory_utilization=0.8, trust_remote_code=True)
out = llm.generate(['Hello, world!'], SamplingParams(max_tokens=32, temperature=0))
print('vLLM output:', out[0].outputs[0].text[:100])

from datasets import load_from_disk
ds = load_from_disk('data/tooluse_data/eval_data')
print(f'Eval data loaded: {len(ds)} examples')

print()
print('=== ALL TESTS PASSED ===')
"

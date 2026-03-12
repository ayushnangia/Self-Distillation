#!/bin/bash
# Setup virtualenv for Self-Distillation on Vulcan
# Run once: bash setup_env.sh
#
# NOTE: Must load arrow and opencv modules BEFORE creating the virtualenv.
# vllm is installed with --no-deps to avoid dummy wheel blockers for
# opencv-python-headless and pyarrow (provided by modules instead).

set -e

module load StdEnv/2023 gcc/12.3 arrow/18.1.0 opencv/4.11.0 python/3.12 cuda/12.6

# Create virtualenv
virtualenv --no-download ~/sdft_env
source ~/sdft_env/bin/activate
pip install --no-index --upgrade pip

# Core packages (no arrow/opencv transitive deps)
pip install --no-index torch transformers accelerate peft trl deepspeed \
    numpy scipy matplotlib pandas rich wandb openai tqdm flashinfer-python

# datasets pinned to 4.3.0 (compatible with pyarrow 18 from arrow module)
pip install --no-index --no-deps datasets==4.3.0

# vllm with --no-deps (opencv-python-headless comes from module)
pip install --no-index --no-deps vllm

# Sub-dependencies
pip install --no-index safetensors tokenizers huggingface_hub regex \
    dill multiprocess xxhash py-cpuinfo hjson pynvml einops msgpack \
    msgspec lark compressed-tensors gguf mistral_common \
    prometheus-client prometheus-fastapi-instrumentator \
    uvicorn uvloop fastapi aiohttp cloudpickle blake3 ninja psutil \
    partial-json-parser

# Link module-provided site-packages into virtualenv
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
echo "/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/arrow/18.1.0/lib/python3.12/site-packages" > "$SITE/arrow.pth"
echo "/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcc12/opencv/4.11.0/lib/python3.12/site-packages" > "$SITE/opencv.pth"

echo ""
echo "Environment setup complete at ~/sdft_env"
echo "Activate with: source ~/sdft_env/bin/activate"

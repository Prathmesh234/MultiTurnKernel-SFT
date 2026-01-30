#!/bin/bash
# =============================================================================
# GPT-OSS-120B Server Launch Script
# =============================================================================
# Runs gpt-oss-120b with vLLM on 2 H100 GPUs using tensor parallelism
# Optimized for high reasoning mode for generating Triton kernel traces
# Uses uv for dependency management
# =============================================================================

set -e

# Configuration
MODEL_NAME="openai/gpt-oss-120b"
TENSOR_PARALLEL_SIZE=2  # Use 2 H100 GPUs
PORT=8000
HOST="0.0.0.0"

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1  # Use first 2 GPUs

# vLLM settings for reasoning model
MAX_MODEL_LEN=32768      # Allow long reasoning traces
GPU_MEMORY_UTILIZATION=0.95

echo "=============================================="
echo "GPT-OSS-120B Server for Triton Trace Generation"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Tensor Parallel: $TENSOR_PARALLEL_SIZE GPUs"
echo "Port: $PORT"
echo "Max sequence length: $MAX_MODEL_LEN"
echo "=============================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv --python 3.10
fi

# Activate the environment
source .venv/bin/activate

# Install vLLM with gpt-oss support using uv
echo "Installing vLLM with gpt-oss support..."
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

# Check GPU availability
echo ""
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Start vLLM server with tensor parallelism
echo "Starting vLLM server..."
uv run vllm serve $MODEL_NAME \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --trust-remote-code \
    --enable-prefix-caching \
    --dtype auto

#!/bin/bash
# =============================================================================
# GPT-OSS-120B Trace Generation Pipeline
# =============================================================================
# 1. Deploy Modal containers for H100 benchmarking
# 2. Start vLLM server for gpt-oss-120b locally
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# vLLM Configuration
MODEL_NAME="openai/gpt-oss-120b"
TENSOR_PARALLEL_SIZE=2
PORT=8000
HOST="0.0.0.0"
MAX_TOKENS=16384
MAX_MODEL_LEN=$MAX_TOKENS
GPU_MEMORY_UTILIZATION=0.9
TEMPERATURE=0.7
REASONING_LEVEL=high

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1

echo "=============================================="
echo "GPT-OSS-120B Trace Generation Pipeline"
echo "=============================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# Sync project dependencies
echo "Syncing project dependencies..."
uv sync

# Install vLLM (required for the server step)
echo "Installing vLLM..."
uv pip install vllm "huggingface-hub<1.0"

# ============================================
# STEP 1: Deploy Modal Containers
# ============================================
echo ""
echo "=============================================="
echo "STEP 1: Deploying Modal H100 Containers"
echo "=============================================="

# Check if modal is configured
# We use 'modal token info' which returns 0 if valid, non-zero usually if not.
# Or simpler: just try to list volumes or something lightweight.
# actually 'modal config show' might be better, but let's stick to checking if we can run a simple command.
if ! uv run --no-sync modal token info &> /dev/null; then
    echo ""
    echo "ERROR: Modal not configured!"
    echo "Please run: uv run --no-sync modal token set --token-id <ID> --token-secret <SECRET>"
    echo "Get your token from: https://modal.com/settings"
    exit 1
fi

# Deploy the Modal app
echo "Deploying Modal app for H100 benchmarking..."
uv run --no-sync modal deploy "$SCRIPT_DIR/modal_app.py"

echo ""
echo "Modal deployment complete!"
echo "Functions available:"
echo "  - benchmark_triton_kernel.remote(...)"
echo "  - benchmark_single.remote(...)"
echo "  - benchmark_parallel_remote.remote(...)"
echo ""

# ============================================
# STEP 2: Start vLLM Server
# ============================================
echo "=============================================="
echo "STEP 2: Starting vLLM Server"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Tensor Parallel: $TENSOR_PARALLEL_SIZE GPUs"
echo "Port: $PORT"
echo "Max sequence length: $MAX_MODEL_LEN"
echo "=============================================="

# Check GPU availability
echo ""
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Start vLLM server with tensor parallelism
echo "Starting vLLM server..."
uv run --no-sync vllm serve $MODEL_NAME \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --trust-remote-code \
    --enable-prefix-caching \
    --dtype auto \
    --reasoning-parser openai_gptoss \
    --tool-call-parser harmony

#!/bin/bash
# =============================================================================
# GLM-4.5-Air Trace Generation Pipeline
# =============================================================================
# 1. Deploy Modal containers for H100 benchmarking
# 2. Start vLLM server for GLM-4.5-Air locally
# =============================================================================
#
# Model: THUDM/GLM-4.5-Air-0414
#   - 106B total params, 12B active (MoE architecture)
#   - 128K context window
#   - Supports reasoning via <think> tags (vLLM reasoning parser: glm45)
#   - Native Multi-Token Prediction (MTP) layer for speculative decoding
#
# GPU Requirements:
#   - FP8:  2x H100 (80GB) or 1x H200 (141GB)
#   - BF16: 4x H100 (80GB) or 2x H200 (141GB)
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# vLLM Configuration
MODEL_NAME="zai-org/GLM-4.5-Air"
TENSOR_PARALLEL_SIZE=4
PORT=8000
HOST="0.0.0.0"
MAX_TOKENS=32768
MAX_MODEL_LEN=$MAX_TOKENS
GPU_MEMORY_UTILIZATION=0.9
TEMPERATURE=0.7
OUTPUT_FILE="reasoning_traces_glm45.json"

# GPU configuration (4 GPUs for BF16 with 106B MoE model)
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=============================================="
echo "GLM-4.5-Air Trace Generation Pipeline"
echo "=============================================="
echo ""
echo "Model: $MODEL_NAME"
echo "  - 106B total params, 12B active (MoE)"
echo "  - Reasoning parser: glm45"
echo "  - Tensor Parallel: $TENSOR_PARALLEL_SIZE GPUs"
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
echo "STEP 2: Starting vLLM Server for GLM-4.5-Air"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Tensor Parallel: $TENSOR_PARALLEL_SIZE GPUs"
echo "Port: $PORT"
echo "Max sequence length: $MAX_MODEL_LEN"
echo "Reasoning parser: glm45"
echo "=============================================="

# Check GPU availability
echo ""
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Deploy Modal app (ensures utilities.py and latest code are mounted)
echo ""
echo "Deploying Modal app..."
uv run --no-sync modal deploy modal_app.py
echo "Modal app deployed successfully."
echo ""

# Start vLLM server with tensor parallelism
# GLM-4.5-Air uses:
#   --reasoning-parser glm45       : extracts reasoning_content from <think> tags
#   --trust-remote-code            : required for GLM-4.5 architecture
#   --enable-prefix-caching        : for efficient prompt caching
#   --dtype auto                   : auto-detect dtype (BF16 on H100)
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
    --reasoning-parser glm45 \
    --tool-call-parser glm45 \
    --enable-auto-tool-choice

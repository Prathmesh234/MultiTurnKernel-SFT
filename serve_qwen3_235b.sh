#!/bin/bash
# =============================================================================
# Qwen3-235B-A22B Trace Generation Pipeline
# =============================================================================
# 1. Deploy Modal containers for H100 benchmarking
# 2. Start vLLM server for Qwen3-235B-A22B-Instruct (FP8) locally
# =============================================================================
#
# Model: Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
#   - 235B total params, 22B active (MoE: 128 experts, top-8 routing)
#   - 131072 context window
#   - Supports reasoning via <think> tags (vLLM reasoning parser: qwen3)
#   - FP8 quantized — fits on 8x H100 (80GB) with NVLink
#
# GPU Requirements:
#   - FP8:  8x H100 (80GB) with NVLink
#   - BF16: 16x H100 (would need pipeline parallel)
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# vLLM Configuration
MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
TENSOR_PARALLEL_SIZE=8
PORT=8000
HOST="0.0.0.0"
MAX_MODEL_LEN=131072           # Full 128K context window
GPU_MEMORY_UTILIZATION=0.92    # Higher util — FP8 leaves headroom on H100
MAX_NUM_SEQS=64                # Max concurrent sequences in-flight
SWAP_SPACE=16                  # 16 GB CPU swap for KV cache overflow
TEMPERATURE=0.7
OUTPUT_FILE="reasoning_traces_qwen3_235b.json"

# GPU configuration (8 GPUs for FP8 with 235B MoE model)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# vLLM v1 engine — async scheduling, better throughput
export VLLM_USE_V1=1

# Fast GPU-side safetensors deserialization (skips CPU→GPU copy)
export SAFETENSORS_FAST_GPU=1

# Required for multi-GPU: spawn workers to avoid CUDA re-init issues with fork
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=============================================="
echo "Qwen3-235B-A22B Trace Generation Pipeline"
echo "=============================================="
echo ""
echo "Model: $MODEL_NAME"
echo "  - 235B total params, 22B active (MoE)"
echo "  - Reasoning parser: qwen3"
echo "  - Tensor Parallel: $TENSOR_PARALLEL_SIZE GPUs"
echo "  - Context length: $MAX_MODEL_LEN"
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
echo "  - benchmark_kernelbench.remote(...)"
echo "  - benchmark_batch.remote(...)"
echo ""

# ============================================
# STEP 2: Start vLLM Server
# ============================================
echo "=============================================="
echo "STEP 2: Starting vLLM Server for Qwen3-235B"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Tensor Parallel: $TENSOR_PARALLEL_SIZE GPUs"
echo "Port: $PORT"
echo "Max sequence length: $MAX_MODEL_LEN"
echo "Reasoning parser: qwen3"
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

# Start vLLM server with tensor parallelism + expert parallelism
# Qwen3-235B uses:
#   --tensor-parallel-size 8       : shard across all 8 H100s
#   --enable-expert-parallel       : distribute MoE experts across TP ranks
#   --max-model-len 131072         : full 128K context window
#   --gpu-memory-utilization 0.92  : high util since FP8 leaves headroom
#   --max-num-seqs 64              : max concurrent sequences
#   --swap-space 16                : 16 GB CPU swap for KV cache overflow
#   --enable-reasoning             : enable thinking/reasoning extraction
#   --reasoning-parser qwen3       : native Qwen3 parser (extracts thinking into reasoning_content)
#   --trust-remote-code            : required for Qwen3 architecture
echo "Starting vLLM server..."
uv run --no-sync vllm serve $MODEL_NAME \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --enable-expert-parallel \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-num-seqs $MAX_NUM_SEQS \
    --swap-space $SWAP_SPACE \
    --enable-reasoning \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-prefix-caching \
    --dtype auto

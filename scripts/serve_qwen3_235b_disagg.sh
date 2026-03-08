#!/bin/bash
# =============================================================================
# Qwen3-235B-A22B Disaggregated Prefill/Decode Serving (4P + 4D)
# =============================================================================
# Splits the 8x H100 node into two vLLM instances:
#   - Prefill instance  (GPUs 0-3): port 8100, kv_producer
#   - Decode  instance  (GPUs 4-7): port 8000, kv_consumer
#
# KV cache is computed on prefill GPUs and transferred to decode GPUs over
# NVLink via PyNcclConnector (ships with vLLM, no extra deps needed).
# Orchestrator and multi-turn pipeline point to port 8000 (decode) unchanged.
#
# Model: Qwen/Qwen3-235B-A22B-Thinking-2507-FP8
#   - 235B total params, 22B active (MoE: 128 experts, top-8 routing)
#   - Always generates reasoning via <think> tags (reasoning parser: qwen3)
#   - FP8 quantized model weights
#
# GPU layout:
#   GPUs 0-3 -> Prefill  (TP=4, EP=4, port 8100)  role: kv_producer
#   GPUs 4-7 -> Decode   (TP=4, EP=4, port 8000)  role: kv_consumer
#
# --- Memory math ---
# FP8 weights at TP=4: ~58.75 GB/GPU. At 0.85 util (68 GB usable):
#   KV headroom = 68 - 58.75 = ~9.25 GB per GPU
#   Total decode-side KV (4 GPUs, FP8 KV cache): 9.25 GB x 4 / 46.875 KB per token
#   ~= 786K tokens total pool on decode side.
#   FP8 KV cache is REQUIRED -- BF16 halves the pool to ~393K which is borderline.
#
# --- max_model_len: 98304 (96K) ---
# 131K was used for EP8 (8 GPUs, ~38.6 GB KV headroom/GPU). With only 9.25 GB
# headroom per GPU on the decode side, 131K is too tight for safe block pool
# initialization. 96K is chosen because:
#   - Covers worst-case multi-turn context: 4 turns x ~20K tokens/turn
#     + system/user prompt = ~81K input by turn 4, with 15K to spare.
#   - Our traces run 16K-30K output tokens -- well within this limit.
#   - Pool can hold ~786K / 96K = ~8 concurrent max-length sequences (vs 6 at 131K).
#   - Meaningfully reduces block pool pressure vs 131K.
#
# --- Orchestrator / multi-turn compatibility ---
# No changes needed to orchestrator.py or multi_turn_queue.py:
#   - VLLM_BASE_URL = "http://localhost:8000/v1" already points to the decode instance.
#   - MAX_MODEL_LEN = 131072 in orchestrator is only used for max_tokens budget math,
#     bounded by MAX_COMPLETION_TOKENS=32768 -- no risk of over-requesting.
#   - MultiTurnQueue is pure Python logic with no network config.
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME="Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"
TENSOR_PARALLEL_SIZE=4           # TP=4 per instance (EP=4 via --enable-expert-parallel)
MAX_MODEL_LEN=98304              # 96K: safe for multi-turn, avoids tight block alloc at 131K
GPU_MEMORY_UTILIZATION=0.85      # Lowered from 0.92 -- TP=4 leaves only ~9.25 GB KV headroom
MAX_NUM_SEQS=16                  # Conservative: tight KV budget on 4-GPU decode side

PREFILL_PORT=8100
DECODE_PORT=8000
HOST="0.0.0.0"

KV_PORT=14579                    # NCCL rendezvous port for prefill<->decode KV transfer

# ---------------------------------------------------------------------------
# Environment vars (applied to both instances via export before exec)
# ---------------------------------------------------------------------------

# vLLM v1 engine -- async scheduling, better throughput
export VLLM_USE_V1=1

# Fast GPU-side safetensors deserialization (skips CPU->GPU copy)
export SAFETENSORS_FAST_GPU=1

# Required for multi-GPU: spawn workers to avoid CUDA re-init issues with fork
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Disable flashinfer CUTLASS/TRTLLM MoE -- FP8 block scaling needs CUDA 12.8+
export VLLM_USE_FLASHINFER_MOE_FP8=0

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Qwen3-235B Disaggregated Prefill/Decode Serving  (4P + 4D)"
echo "============================================================"
echo ""
echo "Model:           $MODEL_NAME"
echo "Prefill GPUs:    0,1,2,3  -> port $PREFILL_PORT  (kv_producer)"
echo "Decode  GPUs:    4,5,6,7  -> port $DECODE_PORT  (kv_consumer)"
echo "TP per instance: $TENSOR_PARALLEL_SIZE  (EP=$TENSOR_PARALLEL_SIZE via --enable-expert-parallel)"
echo "Max model len:   $MAX_MODEL_LEN  (96K -- safe for 4-turn multi-turn traces)"
echo "GPU mem util:    $GPU_MEMORY_UTILIZATION"
echo "Max num seqs:    $MAX_NUM_SEQS per instance"
echo "KV connector:    PyNcclConnector (NVLink, rendezvous port $KV_PORT)"
echo "KV dtype:        fp8_e4m3 (required for viable KV capacity on 4-GPU decode)"
echo "Reasoning:       qwen3 parser"
echo ""
echo "Orchestrator: point to port $DECODE_PORT (unchanged -- same as EP8 setup)"
echo "  python orchestrator.py --multi-turn --batch-size <N>"
echo ""

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

echo "Syncing project dependencies..."
uv sync

echo "Installing vLLM..."
uv pip install vllm "huggingface-hub<1.0"

# ---------------------------------------------------------------------------
# STEP 1: Deploy Modal Containers
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "STEP 1: Deploying Modal H100 Containers"
echo "============================================================"

if ! uv run --no-sync modal token info &> /dev/null; then
    echo ""
    echo "ERROR: Modal not configured!"
    echo "Please run: uv run --no-sync modal token set --token-id <ID> --token-secret <SECRET>"
    echo "Get your token from: https://modal.com/settings"
    exit 1
fi

echo "Deploying Modal app for H100 benchmarking..."
uv run --no-sync modal deploy "$SCRIPT_DIR/../modal_app.py"

echo ""
echo "Modal deployment complete!"
echo "Functions available:"
echo "  - benchmark_triton_kernel.remote(...)"
echo "  - benchmark_kernelbench.remote(...)"
echo "  - benchmark_batch.remote(...)"
echo ""

# ---------------------------------------------------------------------------
# STEP 2: GPU Availability Check
# ---------------------------------------------------------------------------
echo "============================================================"
echo "STEP 2: Checking GPU availability"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# ---------------------------------------------------------------------------
# STEP 3: Start Prefill Instance (GPUs 0-3, port 8100, kv_producer)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "STEP 3: Starting Prefill Instance (GPUs 0-3)"
echo "============================================================"
echo "Port: $PREFILL_PORT | kv_role: kv_producer | kv_rank: 0"
echo ""

CUDA_VISIBLE_DEVICES=0,1,2,3 \
uv run --no-sync vllm serve $MODEL_NAME \
    --host $HOST \
    --port $PREFILL_PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --enable-expert-parallel \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-num-seqs $MAX_NUM_SEQS \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-prefix-caching \
    --kv-cache-dtype fp8_e4m3 \
    --calculate-kv-scales True \
    --dtype auto \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_port":'"$KV_PORT"'}' &

PREFILL_PID=$!
echo ""
echo "Prefill instance started (PID $PREFILL_PID)."
echo "Waiting 90s for model weights to load before starting decode instance..."
echo "(Qwen3-235B FP8 at TP=4 takes ~60-90s to load)"
sleep 90

# Trap: kill prefill if this script exits (Ctrl-C or decode crash)
trap "echo ''; echo 'Shutting down prefill instance (PID $PREFILL_PID)...'; kill $PREFILL_PID 2>/dev/null; wait $PREFILL_PID 2>/dev/null; echo 'Prefill stopped.'" EXIT

# ---------------------------------------------------------------------------
# STEP 4: Start Decode Instance (GPUs 4-7, port 8000, kv_consumer)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "STEP 4: Starting Decode Instance (GPUs 4-7)"
echo "============================================================"
echo "Port: $DECODE_PORT | kv_role: kv_consumer | kv_rank: 1"
echo "Orchestrator and multi-turn pipeline target port $DECODE_PORT -- no changes needed."
echo ""

CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run --no-sync vllm serve $MODEL_NAME \
    --host $HOST \
    --port $DECODE_PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --enable-expert-parallel \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-num-seqs $MAX_NUM_SEQS \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-prefix-caching \
    --kv-cache-dtype fp8_e4m3 \
    --calculate-kv-scales True \
    --dtype auto \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_port":'"$KV_PORT"'}'

# ---------------------------------------------------------------------------
# Reached only if decode exits cleanly (trap handles Ctrl-C / crash)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Decode instance stopped. Prefill will be shut down by trap."
echo "============================================================"

#!/bin/bash
# =============================================================================
# Serve Trinity-Mini with vLLM + LoRA Support using UV
#
# This script uses UV for proper dependency management and virtual environment.
# vLLM provides an OpenAI-compatible API server with LoRA adapter support.
#
# The adapter used here targets attention layers only (q_proj, k_proj, v_proj, o_proj),
# which is fully supported by vLLM.
# =============================================================================

set -e

# --- Configuration ---
# Load .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

MODEL_ID=${TRINITY_MODEL_ID:-"arcee-ai/Trinity-Mini"}
# Construct path to final_model inside the local adapter path
LOCAL_BASE=${TRINITY_ADAPTER_PATH_LOCAL:-"./sft-reasoning/trinity-triton-sft-vllm"}
LORA_PATH="${LOCAL_BASE}/final_model"
ADAPTER_NAME=${SGLANG_ADAPTER_NAME:-"trinity-reasoning-vllm"}
PORT=8000
TP_SIZE=1 # H100 (80GB) can handle Trinity-Mini (26B) with TP=1

# Check that LoRA weights exist
if [ ! -d "$LORA_PATH" ]; then
    echo "LoRA adapter not found at $LORA_PATH"
    echo "Run 'modal token new' to authenticate with Modal, then"
    echo "run 'uv run python inference/vllm/download_lora_vllm.py' to download the adapter."
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create UV virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating UV virtual environment..."
    uv venv .venv --python 3.10
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install vLLM and all dependencies with UV
echo "Installing vLLM and dependencies with UV..."
uv pip install \
    "vllm" \
    "transformers==4.57.1" \
    "torch==2.9.1" \
    "numpy<2.0" \
    "scipy>=1.10.0" \
    "scikit-learn>=1.2.0"

echo ""
echo "Starting vLLM server on port $PORT..."
echo "Base Model: $MODEL_ID (via HuggingFace)"
echo "LoRA Adapter: $LORA_PATH as '$ADAPTER_NAME'"
echo ""

# vLLM Launch Command (OpenAI-compatible server)
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --enable-lora \
  --lora-modules "$ADAPTER_NAME=$LORA_PATH" \
  --tensor-parallel-size "$TP_SIZE" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 32768

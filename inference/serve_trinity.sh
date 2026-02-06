#!/bin/bash
# =============================================================================
# Serve Trinity-Mini with SGLang + LoRA Support
#
# SGLang supports MoE + LoRA (which vLLM currently lacks for fusedMoe)
# =============================================================================

set -e

# --- Configuration ---
MODEL_ID="arcee-ai/Trinity-Mini"
LORA_PATH="./sft-reasoning/checkpoint-40" 
ADAPTER_NAME="trinity-reasoning"
PORT=8000
TP_SIZE=1 # H100 (80GB) can handle Trinity-Mini (26B) with TP=1

# Check that LoRA weights exist
if [ ! -d "$LORA_PATH" ]; then
    echo "LoRA adapter not found at $LORA_PATH"
    echo "Run 'uv run python inference/download_lora_from_modal.py' first."
    exit 1
fi

# Check that LoRA weights exist
if [ ! -d "$LORA_PATH" ]; then
    echo "LoRA adapter not found at $LORA_PATH"
    echo "Run 'uv run python inference/download_lora_from_modal.py' first."
    exit 1
fi

# Install SGLang if not present (or upgrade for MoE LoRA support)
echo "Ensuring SGLang is installed..."
pip install "sglang[all]>=0.4.0" --upgrade

# Serve with SGLang
echo ""
echo "Starting SGLang server on port $PORT..."
echo "Base Model: $MODEL_ID (via HuggingFace)"
echo "LoRA Adapter: $LORA_PATH as '$ADAPTER_NAME'"
echo ""

# SGLang Launch Command
python3 -m sglang.launch_server \
  --model-path "$MODEL_ID" \
  --lora-paths "$ADAPTER_NAME=$LORA_PATH" \
  --enable-lora \
  --tp-size "$TP_SIZE" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --trust-remote-code \
  --context-length 32768


# Setup Commands

## Prerequisites

### 1. Modal Authentication

```bash
modal token set --token-id <YOUR_MODAL_TOKEN_ID> --token-secret <YOUR_MODAL_TOKEN_SECRET>
```

Get tokens from: https://modal.com/settings

Verify:
```bash
modal token info
```

### 2. Environment Variables

Create a `.env` file (use `.env.example` as a template):

```bash
HF_TOKEN=your_huggingface_token_here
```

**Note:** The Modal token is set via CLI, NOT in `.env`.

### 3. GPU Requirements

| Model | GPUs | VRAM | Script |
|-------|------|------|--------|
| GPT-OSS-120B | 2x H100 | ~160 GB | `run_gpt_oss.sh` |
| GLM-4.5-Air (BF16) | 4x H100 | ~320 GB | `run_glm45_air.sh` |
| Qwen3-235B (FP8) | 8x H100 (NVLink) | ~640 GB | `serve_qwen3_235b.sh` |

## Quick Start (Recommended)

### Option A: All-in-One Scripts

Each script handles: `uv sync` → vLLM install → Modal deploy → server launch.

```bash
# GLM-4.5-Air (4x H100)
bash run_glm45_air.sh

# GPT-OSS-120B (2x H100)
bash run_gpt_oss.sh

# Qwen3-235B (8x H100, FP8)
bash serve_qwen3_235b.sh
```

Keep the server running. Then in a **new terminal**:

```bash
# Single-turn generation
python orchestrator.py

# Multi-turn generation (iterative refinement)
python orchestrator.py --multi-turn --max-turns 4
```

### Option B: Manual Steps

#### Step 1: Install Dependencies

```bash
uv sync
uv pip install vllm "huggingface-hub<1.0"
```

#### Step 2: Deploy Modal

```bash
uv run --no-sync modal deploy modal_app.py
```

#### Step 3: Start vLLM Server

**GLM-4.5-Air (4x H100):**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

uv run --no-sync vllm serve zai-org/GLM-4.5-Air \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enable-prefix-caching \
    --dtype auto \
    --reasoning-parser glm45 \
    --tool-call-parser glm45 \
    --enable-auto-tool-choice
```

**GPT-OSS-120B (2x H100):**
```bash
export CUDA_VISIBLE_DEVICES=0,1

uv run --no-sync vllm serve openai/gpt-oss-120b \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enable-prefix-caching \
    --dtype auto \
    --reasoning-parser openai_gptoss \
    --tool-call-parser harmony
```

**Qwen3-235B-A22B (8x H100, FP8):**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=1
export SAFETENSORS_FAST_GPU=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

uv run --no-sync vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    --swap-space 16 \
    --enable-reasoning \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-prefix-caching \
    --dtype auto
```

#### Step 4: Generate Traces

```bash
# Single-turn
python orchestrator.py --output reasoning_traces.json

# Multi-turn
python orchestrator.py \
    --multi-turn \
    --max-turns 4 \
    --batch-size 5 \
    --output reasoning_traces_multiturn.json
```

## Orchestrator CLI Reference

```bash
python orchestrator.py \
    --vllm-url http://localhost:8000/v1 \
    --output reasoning_traces.json \
    --kernelbook-samples 1500 \
    --kernelbench-samples 1000 \
    --batch-size 10 \
    --save-interval 10 \
    --multi-turn \
    --max-turns 4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--vllm-url` | `http://localhost:8000/v1` | vLLM server URL |
| `--output` | `reasoning_traces.json` | Output JSON file |
| `--kernelbook-samples` | 1500 | KernelBook samples to process |
| `--kernelbench-samples` | 1000 | KernelBench samples to process |
| `--batch-size` | 10 | Concurrent generation requests |
| `--save-interval` | 10 | Save every N samples (single-turn) |
| `--multi-turn` | off | Enable multi-turn refinement |
| `--max-turns` | 4 | Max turns per sample (multi-turn) |

## Testing

### Multi-Turn E2E Test

Validates the full pipeline (Modal benchmark + MultiTurnQueue) without a vLLM server:

```bash
python test_multi_turn.py
```

Runs 4 turns: buggy kernel → CUDA crash → correct but slow → optimized fused kernel.

## Workflow Summary

1. **Set Modal token** (one-time):
   ```bash
   modal token set --token-id <ID> --token-secret <SECRET>
   ```

2. **Start model server** (keep running):
   ```bash
   bash serve_qwen3_235b.sh   # or run_glm45_air.sh / run_gpt_oss.sh
   ```

3. **Generate traces** (new terminal):
   ```bash
   python orchestrator.py --multi-turn --max-turns 4
   ```

4. **Output** saved to `reasoning_traces.json` (or `--output` path).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Modal not configured" | `modal token set --token-id <ID> --token-secret <SECRET>` |
| GPU OOM | Reduce `--gpu-memory-utilization` or `--max-model-len` |
| vLLM won't connect | Check: `curl http://localhost:8000/v1/models` |
| Qwen3 multi-GPU errors | Ensure `VLLM_WORKER_MULTIPROC_METHOD=spawn` is set |
| Slow Qwen3 startup | FP8 model is ~120 GB download on first run; uses safetensors fast GPU path after |

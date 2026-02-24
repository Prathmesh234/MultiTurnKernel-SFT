# KernelBench Triton

A benchmarking and trace generation pipeline for Triton GPU kernels on Modal H100 GPUs. Models generate Triton kernels from PyTorch code, which are validated for correctness and performance on remote H100s. Supports single-turn and multi-turn iterative refinement.

Adapted from the [KernelBench](https://github.com/ScalingIntelligence/KernelBench) project by Stanford's Scaling Intelligence Lab.

## Supported Models

| Model | Params (total / active) | Architecture | GPUs Required | Run Script |
|-------|------------------------|--------------|---------------|------------|
| GLM-4.5-Air | 106B / 12B | MoE | 4x H100 (BF16) | `run_glm45_air.sh` |
| GPT-OSS-120B | 120B | Dense | 2x H100 | `run_gpt_oss.sh` |
| Qwen3-235B-A22B | 235B / 22B | MoE (128 experts) | 8x H100 (FP8) | `serve_qwen3_235b.sh` |

## Quick Start

### 1. Prerequisites

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Authenticate with Modal
modal token set --token-id <ID> --token-secret <SECRET>
```

### 2. Start a Model Server

Each run script handles everything: dependency install, Modal deploy, and vLLM server launch.

```bash
# GLM-4.5-Air (4x H100, BF16)
bash run_glm45_air.sh

# GPT-OSS-120B (2x H100)
bash run_gpt_oss.sh

# Qwen3-235B (8x H100, FP8)
bash serve_qwen3_235b.sh
```

Keep the server running in its terminal.

### 3. Generate Traces

In a **new terminal**, run the orchestrator:

```bash
# Single-turn (default) — one shot per sample
python orchestrator.py

# Multi-turn — iterative refinement with feedback (up to 4 turns)
python orchestrator.py --multi-turn
```

## Running Modes

### Single-Turn

The default mode. Each sample gets one generation attempt. Fast but no self-correction.

```bash
python orchestrator.py \
    --output reasoning_traces.json \
    --kernelbook-samples 1500 \
    --kernelbench-samples 1000 \
    --batch-size 10
```

### Multi-Turn (Iterative Refinement)

Failed or slow kernels get feedback and retry up to `--max-turns` times. This is where the pipeline shines — the model can fix compilation errors, correctness bugs, and optimize performance across turns.

```bash
python orchestrator.py \
    --multi-turn \
    --max-turns 4 \
    --batch-size 5 \
    --output reasoning_traces_multiturn.json
```

**Multi-turn flow:**
1. Model generates a Triton kernel
2. Kernel is benchmarked on Modal H100 (correctness + speedup)
3. If incorrect or slow, feedback is sent back to the model
4. Model generates an improved kernel
5. Repeat until correct + fast, or max turns reached

See [`multi_turn.md`](./multi_turn.md) for the full architecture spec.

### Orchestrator CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--vllm-url` | `http://localhost:8000/v1` | vLLM server URL |
| `--output` | `reasoning_traces.json` | Output JSON file |
| `--kernelbook-samples` | 1500 | Number of KernelBook samples |
| `--kernelbench-samples` | 1000 | Number of KernelBench samples |
| `--batch-size` | 10 | Concurrent generation requests |
| `--save-interval` | 10 | Save traces every N samples (single-turn) |
| `--multi-turn` | off | Enable multi-turn iterative refinement |
| `--max-turns` | 4 | Max turns per sample (multi-turn only) |

## Testing

### Multi-Turn End-to-End Test

A standalone test script exercises the full pipeline without a live vLLM server:

```bash
python test_multi_turn.py
```

This runs 4 turns against the deployed Modal function:
1. **Turn 1:** Buggy kernel (`x * y` instead of `relu(x + y)`) → `correctness=False`
2. **Turn 2:** Illegal memory access (no bounds mask) → CUDA error
3. **Turn 3:** Correct but slow (two separate kernels) → `correctness=True`, `speedup < 1.0`
4. **Turn 4:** Optimized fused kernel (add+relu in one pass) → `correctness=True`, `speedup > 1.0`

Validates: `MultiTurnQueue` state management, Modal benchmarking, feedback routing, and trace building.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  vLLM Server    │     │   Orchestrator   │     │   Modal H100    │
│  (local GPU)    │◄───►│  orchestrator.py │◄───►│  modal_app.py   │
│                 │     │                  │     │                 │
│  GLM / GPT /   │     │  - Generation    │     │  - Correctness  │
│  Qwen3         │     │  - Extraction    │     │  - Performance  │
│                 │     │  - Multi-turn    │     │  - Speedup      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │ MultiTurnQueue   │
                        │ (passive proxy)  │
                        │                  │
                        │ - Deque mgmt     │
                        │ - Turn counting  │
                        │ - Feedback build │
                        │ - Trace finalize │
                        └──────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `orchestrator.py` | Main pipeline: generation, extraction, validation loop |
| `multi_turn_queue.py` | Passive proxy: deque, turns, feedback, traces |
| `modal_app.py` | Modal functions for H100 benchmarking |
| `dataloader.py` | Loads KernelBook + KernelBench samples |
| `utilities.py` | Input extraction helpers (mounted into Modal) |
| `run_glm45_air.sh` | GLM-4.5-Air server launcher (4x H100) |
| `run_gpt_oss.sh` | GPT-OSS-120B server launcher (2x H100) |
| `serve_qwen3_235b.sh` | Qwen3-235B server launcher (8x H100) |
| `test_multi_turn.py` | End-to-end multi-turn test (no vLLM needed) |

## Modal Benchmarking

### Deploy

```bash
modal deploy modal_app.py
```

### Available Functions

| Function | Use Case |
|----------|----------|
| `benchmark_triton_kernel` | Generic benchmark with `input_shapes` dict |
| `benchmark_kernelbench` | KernelBench `nn.Module` pattern with `get_inputs()` |
| `benchmark_batch` | Sequential batch on same container |

### Benchmark API

```python
import modal

benchmark = modal.Function.from_name("kernelbench-triton", "benchmark_triton_kernel")

result = benchmark.remote(
    kernel_code=triton_code,
    reference_torch_code=reference_code,
    input_shapes={"x": {"shape": [1024, 1024], "dtype": "float32"}},
    n_correctness=5,
    n_trials=50,
    kernel_name="my_kernel",
)
```

### Metrics

| Metric | Description |
|--------|-------------|
| `correctness` | Whether the kernel produces correct output |
| `speedup` | `reference_time / kernel_time` |
| `fast_0` | Correct (same as correctness) |
| `fast_1` | Correct AND faster than reference (speedup > 1.0) |
| `fast_2` | Correct AND at least 2x faster (speedup >= 2.0) |

## Kernel Code Requirements

### Triton Kernel

Must define `triton_kernel_wrapper` that takes the same inputs as the reference:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * 2, mask=mask)

def triton_kernel_wrapper(x):
    output = torch.empty_like(x)
    n = x.numel()
    grid = ((n + 1023) // 1024,)
    my_kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output
```

### PyTorch Reference

Must define `reference_impl` with matching signature:

```python
import torch

def reference_impl(x):
    return x * 2
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Modal not configured" | `modal token set --token-id <ID> --token-secret <SECRET>` |
| "triton_kernel_wrapper not found" | Ensure kernel code defines `triton_kernel_wrapper` function |
| GPU OOM | Reduce `--gpu-memory-utilization` or `--max-model-len` |
| vLLM connection error | Check server is running: `curl http://localhost:8000/v1/models` |
| CUDA illegal memory access | Modal auto-retries on fresh container (see `modal_app.py`) |

## Resources

- [Modal Documentation](https://modal.com/docs)
- [Triton Documentation](https://triton-lang.org/)
- [KernelBench Paper](https://scalingintelligence.stanford.edu/blogs/kernelbench/)
- [KernelBench GitHub](https://github.com/ScalingIntelligence/KernelBench)
- [vLLM Documentation](https://docs.vllm.ai/)

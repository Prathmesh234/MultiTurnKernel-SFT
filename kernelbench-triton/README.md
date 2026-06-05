# KernelBench Triton

A benchmarking and trace generation pipeline for Triton GPU kernels on Modal H100/H200 GPUs. Reasoning-enabled LLMs generate Triton kernels from PyTorch code, which are validated for correctness and performance on remote GPUs. Supports single-turn and multi-turn iterative refinement with automatic feedback routing, prefix caching, and reasoning trace capture.

Adapted from the [KernelBench](https://github.com/ScalingIntelligence/KernelBench) project by Stanford's Scaling Intelligence Lab.

## Supported Models

| Model | Params (total / active) | Architecture | GPUs Required | Quantization | Run Script |
|-------|------------------------|--------------|---------------|--------------|------------|
| Qwen3-235B-A22B | 235B / 22B | MoE (128 experts, top-8) | 8x H100 (FP8) | FP8 | `scripts/serve_qwen3_235b.sh` |
| GLM-4.5-Air | 106B / 12B | MoE | 4x H100 (BF16) | BF16 | `scripts/run_glm45_air.sh` |
| GPT-OSS-120B | 120B | Dense | 2x H100 | BF16 | `scripts/run_gpt_oss.sh` |

The primary model used for trace generation is **Qwen3-235B-A22B-Thinking-2507-FP8**, which supports internal chain-of-thought reasoning via `<think>` tags (vLLM reasoning parser: `qwen3`) and has a 131,072 token context window.

## Quick Start

### 1. Prerequisites

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Authenticate with Modal
modal token set --token-id <ID> --token-secret <SECRET>
```

### 2. Start a Model Server

Run scripts are located in the `scripts/` directory and handle dependency installation, Modal deploy, and vLLM server launch.

```bash
# Qwen3-235B (8x H100, FP8) — primary model
bash scripts/serve_qwen3_235b.sh

# Qwen3-235B disaggregated (4P + 4D split)
bash scripts/serve_qwen3_235b_disagg.sh

# GLM-4.5-Air (4x H100, BF16)
bash scripts/run_glm45_air.sh

# GPT-OSS-120B (2x H100)
bash scripts/run_gpt_oss.sh
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
    --kernelbook-samples 3000 \
    --kernelbench-samples 500 \
    --batch-size 128
```

### Multi-Turn (Iterative Refinement)

Failed or slow kernels get feedback and retry up to `--max-turns` times. The model can fix compilation errors, correctness bugs, and optimize performance across turns. Prefix caching achieves 68-78% hit rates in multi-turn workloads, reducing effective prefill compute by 3.3x.

```bash
python orchestrator.py \
    --multi-turn \
    --max-turns 4 \
    --batch-size 128 \
    --modal-parallel 32 \
    --output reasoning_traces_multiturn.json
```

**Multi-turn flow:**
1. Model generates a Triton kernel (with `<think>` reasoning and `<triton>` code tags)
2. Code is pre-validated locally (syntax, required functions, forbidden imports)
3. Kernel is benchmarked on Modal H100 (correctness + speedup)
4. If incorrect or slow, structured feedback is sent back to the model
5. Model generates an improved kernel with full conversation history
6. Repeat until correct + fast, or max turns reached
7. Completed traces saved to output; failed traces saved separately with `_failed` suffix

See [`documentation/multi_turn.md`](./documentation/multi_turn.md) for the full architecture spec.

### Orchestrator CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--vllm-url` | `http://localhost:8000/v1` | vLLM server URL |
| `--output` | `reasoning_traces.json` | Output JSON file |
| `--kernelbook-samples` | 3000 | Number of KernelBook samples |
| `--kernelbench-samples` | 500 | Number of KernelBench samples |
| `--batch-size` | 128 | Concurrent generation requests |
| `--multi-turn` | off | Enable multi-turn iterative refinement |
| `--max-turns` | 4 | Max turns per sample (multi-turn only) |
| `--modal-parallel` | 32 | Max parallel Modal validations |

## Testing

### Multi-Turn End-to-End Test

A standalone test script exercises the full pipeline without a live vLLM server:

```bash
python tests/test_multi_turn.py
```

This runs 4 turns against the deployed Modal function:
1. **Turn 1:** Buggy kernel (`x * y` instead of `relu(x + y)`) → `correctness=False`
2. **Turn 2:** Illegal memory access (no bounds mask) → CUDA error
3. **Turn 3:** Correct but slow (two separate kernels) → `correctness=True`, `speedup < 1.0`
4. **Turn 4:** Optimized fused kernel (add+relu in one pass) → `correctness=True`, `speedup > 1.0`

Validates: `MultiTurnQueue` state management, Modal benchmarking, feedback routing, and trace building.

### vLLM Reasoning Validation

```bash
python tests/test_reasoning.py
```

Verifies that the vLLM server returns reasoning traces via the `reasoning_content` field and that the `--reasoning-parser qwen3` flag is correctly set.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  vLLM Server    │     │   Orchestrator   │     │   Modal H100    │
│  (local GPU)    │◄───►│  orchestrator.py │◄───►│  modal_app.py   │
│                 │     │                  │     │                 │
│  Qwen3-235B    │     │  - Generation    │     │  - Correctness  │
│  (FP8, TP=8)   │     │  - Extraction    │     │  - Performance  │
│                 │     │  - Pre-validate  │     │  - Speedup      │
│  Reasoning:     │     │  - Multi-turn    │     │  - Crash recover│
│  <think> tags   │     │  - Resume support│     │                 │
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
                        │ - Failed requeue │
                        └──────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `orchestrator.py` | Main pipeline: generation, extraction, pre-validation, benchmarking loop |
| `multi_turn_queue.py` | Passive proxy: deque, turns, feedback routing, trace finalization |
| `modal_app.py` | Modal functions for H100 benchmarking with crash recovery |
| `dataloader.py` | Loads KernelBook (up to 4000 samples) + KernelBench (up to 270 samples) |
| `utilities.py` | Code pre-validation, input extraction, solved-key loading |
| `scripts/serve_qwen3_235b.sh` | Qwen3-235B server (8x H100, TP=8, FP8) |
| `scripts/serve_qwen3_235b_disagg.sh` | Qwen3-235B disaggregated serving (4P+4D) |
| `scripts/run_glm45_air.sh` | GLM-4.5-Air server launcher (4x H100) |
| `scripts/run_gpt_oss.sh` | GPT-OSS-120B server launcher (2x H100) |
| `tests/test_multi_turn.py` | End-to-end multi-turn test (no vLLM needed) |
| `tests/test_reasoning.py` | vLLM reasoning trace validation |

## Serving Configurations

### Standard (8x H100, TP=8)

The default configuration runs all 8 GPUs as a single tensor-parallel group:

- **TP=8, EP=8** — full tensor/expert parallelism
- **max_model_len=131,072** — full 128K context window
- **gpu_memory_utilization=0.92**
- **max_num_seqs=128** — high concurrency
- **Prefix caching enabled** — critical for multi-turn efficiency

### Disaggregated (4P + 4D)

Splits 8 GPUs into dedicated prefill (GPUs 0-3) and decode (GPUs 4-7) instances:

- **Prefill** (port 8100): TP=4, kv_producer — handles prompt processing
- **Decode** (port 8000): TP=4, kv_consumer — handles token generation
- KV cache transferred via NVLink (PyNcclConnector)
- **max_model_len=98,304** (96K), KV cache dtype=fp8_e4m3, max_num_seqs=16

Use disaggregated serving when prefill and decode workloads need independent tuning.

## Modal Benchmarking

### Deploy

```bash
modal deploy modal_app.py
```

### Available Functions

| Function | Use Case |
|----------|----------|
| `benchmark_triton_kernel` | Generic benchmark with `input_shapes` dict |
| `benchmark_kernelbench` | KernelBench `nn.Module` pattern with `get_inputs()` / `get_init_inputs()` |
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

### Error Recovery

- **Fatal GPU crashes** → container dies, Modal retries on fresh container with clean CUDA context
- **Non-fatal GPU faults** → `modal.experimental.stop_fetching_inputs()` drains container gracefully
- **Module cache** evicted between runs to prevent stale code
- **Temp files** cleaned up in `finally` blocks

### Metrics

| Metric | Description |
|--------|-------------|
| `correctness` | Whether the kernel produces correct output |
| `speedup` | `reference_time / kernel_time` |
| `fast_0` | Correct (same as correctness) |
| `fast_1` | Correct AND faster than reference (speedup > 1.0) |
| `fast_2` | Correct AND at least 2x faster (speedup >= 2.0) |

## Performance Findings

### H100 Batch Size Sweep (Qwen3-235B, TP=8, EP=8)

| Batch Size | Throughput (tok/s) | Interactivity (tok/s/user) | TTFT p50 | Notes |
|------------|-------------------|---------------------------|----------|-------|
| B=4 | 79 | — | — | GPU 53% utilized, high jitter |
| B=8 | 268 | 57.1 | — | GPU 100%, sweet spot for latency |
| B=16 | 392 | 57.1 | 0.61s | Best TTFT, same interactivity as B=8 |
| B=32 | 753 | 55.6 | — | TTFT p90 = 6.2s, quality concern |

### H200 at B=128

| Metric | Value |
|--------|-------|
| Throughput | ~1,930 tok/s (2.6x H100 B=32) |
| Per-GPU throughput | 225 tok/s/gpu (2.5x H100) |
| Interactivity | 26.7 tok/s/user (async-friendly) |
| KV cache usage | 23-26% median, peaks 47% |
| Energy efficiency | 0.51 tok/W (1.9x more efficient than H100) |
| Prefix cache hit rate | 68-78% |

### Multi-Turn Observations

- **Prefix cache hit rate: 68-78%** — reduces effective prefill compute by 3.3x
- Context grows per turn: T1 ≈ 500 tokens → T4 ≈ 9,500 tokens
- KV cache accumulates: 4-turn user occupies 4x KV blocks of single-turn
- Zero preemptions across all batch sizes (paged attention working well)
- Throughput variance: 47-58% stdev (bursty due to multi-turn lifecycle)

### Error Analysis (Single-Turn Baseline)

From 479 traces analyzed (42 correct, **8.77% baseline success rate**):

| Error Category | % of Failures | Root Cause |
|----------------|--------------|------------|
| CUDA Memory Errors | 39.8% | Missing boundary masks, stride miscalculations |
| Triton Compilation | 23.3% | `arange` requires constexpr, shape mismatches |
| Type/Reference Errors | 17.6% | Missing imports, NameError, AttributeError |
| Runtime Errors | 5.3% | Uninitialized global state |
| Other | 13.1% | Timeouts, assertion errors, missing deps |

Multi-turn refinement improves on this baseline by allowing the model to fix compilation errors, correctness bugs, and optimize performance iteratively.

See [`documentation/ERROR_ANALYSIS.md`](./documentation/ERROR_ANALYSIS.md) for the full breakdown and fix patterns.

## SFT Fine-Tuning

Verified reasoning traces (correctness=True, fast_0=True) are used to fine-tune **Trinity-Mini** (26B MoE, 3B active) via QLoRA:

- **LoRA config:** r=64, alpha=128, dropout=0.05
- **Training:** lr=2e-4, batch_size=1, grad_accum=8, 3 epochs, max_seq_len=4096
- **Format:** Conversational — User provides PyTorch, Assistant responds with `<think>reasoning</think><triton>code</triton>`
- **Deployment:** Modal inference server with SGLang (supports LoRA + MoE)

See [`sft-reasoning/README.md`](./sft-reasoning/README.md) for setup and training details.

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
| arange requires constexpr | Use fixed BLOCK_SIZE with masking instead of runtime values |
| High TTFT at large batch | Lower `--batch-size` or use disaggregated serving |
| Prefix cache misses | Ensure `--enable-prefix-caching` is set in vLLM args |

## Resources

- [Modal Documentation](https://modal.com/docs)
- [Triton Documentation](https://triton-lang.org/)
- [KernelBench Paper](https://scalingintelligence.stanford.edu/blogs/kernelbench/)
- [KernelBench GitHub](https://github.com/ScalingIntelligence/KernelBench)
- [vLLM Documentation](https://docs.vllm.ai/)

## Directory Structure

```
MultiTurnKernel-SFT/
├── orchestrator.py                # Main generation pipeline (single-turn & multi-turn)
├── modal_app.py                   # Modal H100 benchmarking with crash recovery
├── multi_turn_queue.py            # Multi-turn queue: feedback routing, trace finalization
├── dataloader.py                  # Dataset loader for KernelBook + KernelBench
├── utilities.py                   # Code pre-validation, input extraction, resume helpers
├── pyproject.toml                 # Project dependencies
├── uv.lock                        # Locked dependency versions
│
├── scripts/                       # Model serving scripts
│   ├── run_glm45_air.sh           #   GLM-4.5-Air (4x H100, BF16)
│   ├── run_gpt_oss.sh             #   GPT-OSS-120B (2x H100)
│   ├── serve_qwen3_235b.sh        #   Qwen3-235B standard (8x H100, TP=8, FP8)
│   ├── serve_qwen3_235b_disagg.sh #   Qwen3-235B disaggregated (4P+4D)
│   └── config-disagg.md           #   Disaggregated serving configuration notes
│
├── tests/                         # Test suite
│   ├── test_multi_turn.py         #   End-to-end multi-turn test (no vLLM needed)
│   └── test_reasoning.py          #   vLLM reasoning trace validation
│
├── traces/                        # Generated reasoning traces (JSON)
│   ├── reasoning_traces_glm45*.json           # GLM-4.5-Air traces
│   ├── reasoning_traces_qwen3_multiturn*.json # Qwen3 multi-turn traces (various batch sizes)
│   └── reasoning_traces_multiturn.json        # General multi-turn traces
│
├── analysis_multiturn/            # Performance analysis & visualization
│   ├── ANALYSIS.md                #   Full analysis report (H100/H200 batch sweep)
│   ├── analyze_traces.py          #   Trace analysis script (success rates, errors, speedups)
│   ├── chart_batch_comparison.py  #   Batch size comparison charts
│   ├── chart_h200.py              #   H200 performance charts
│   ├── kernel_journey.py          #   Per-kernel improvement visualization
│   └── charts/                    #   Generated chart images (PNG)
│
├── metrics/                       # vLLM server metrics collection
│   ├── README.md                  #   Metrics reference (InferenceX format)
│   ├── collect_metrics.py         #   Prometheus polling script (10s intervals)
│   ├── trim_prometheus.py         #   Prometheus data compaction
│   └── metrics-batch*/            #   Collected metrics per batch size (JSONL + Prometheus)
│
├── documentation/                 # Extended documentation
│   ├── DOCUMENTATION.md           #   Modal benchmarking reference
│   ├── DOCUMENTATION_CORRECTIONS.md #   Corrections and errata
│   ├── ERROR_ANALYSIS.md          #   Error categorization (39+ categories, fixes)
│   ├── SETUP_COMMANDS.md          #   Environment setup commands
│   ├── SGLANG_SETUP.md            #   SGLang serving setup guide
│   ├── AGENTS.md                  #   Agent configuration reference
│   ├── multi_turn.md              #   Multi-turn architecture spec
│   ├── async_orchestrator.md      #   Async orchestrator design doc
│   └── disaggregated_serving.md   #   Disaggregated serving architecture
│
├── sft-reasoning/                 # SFT fine-tuning pipeline
│   ├── README.md                  #   Trinity-Mini finetuning & deployment guide
│   ├── arcee-trinity-mini-sft.ipynb # Training notebook (QLoRA)
│   ├── pyproject.toml             #   SFT dependencies
│   └── uv.lock                    #   Locked SFT dependencies
│
└── exploration/                   # Early exploration & prototyping
    ├── README.md                  #   Exploration notes
    ├── benchmark_example.py       #   Benchmark usage examples
    ├── pyproject.toml             #   Exploration dependencies
    └── uv.lock                    #   Locked exploration dependencies
```

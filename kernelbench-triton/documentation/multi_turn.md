# Multi-Turn Reasoning & Kernel Generation

## Overview

Multi-turn iterative refinement for Triton kernel generation. The model generates a kernel, it gets benchmarked on Modal H100, and if it fails or is slow, feedback is sent back for another attempt. Up to `max_turns` (default 4) per sample.

**Output file**: `reasoning_traces_glm45_multiturn.json` (configurable via `--output`)

## Running

### Prerequisites

1. Modal deployed: `modal deploy modal_app.py`
2. vLLM server running (any supported model):

```bash
# Pick one:
bash run_glm45_air.sh      # GLM-4.5-Air (4x H100)
bash run_gpt_oss.sh         # GPT-OSS-120B (2x H100)
bash serve_qwen3_235b.sh    # Qwen3-235B (8x H100, FP8)
```

### Single-Turn (Default)

One generation attempt per sample. No feedback loop.

```bash
python orchestrator.py
```

### Multi-Turn

Iterative refinement with feedback. Failed/slow kernels get retried.

```bash
python orchestrator.py --multi-turn --max-turns 4 --batch-size 5
```

### Full Example with Qwen3

```bash
# Terminal 1: Start Qwen3 server
bash serve_qwen3_235b.sh

# Terminal 2: Run multi-turn generation
python orchestrator.py \
    --multi-turn \
    --max-turns 4 \
    --batch-size 5 \
    --output reasoning_traces_qwen3_multiturn.json \
    --kernelbook-samples 1500 \
    --kernelbench-samples 1000
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--multi-turn` | off | Enable multi-turn mode |
| `--max-turns` | 4 | Max turns per sample |
| `--batch-size` | 10 | Concurrent generation requests |
| `--output` | `reasoning_traces.json` | Output file |
| `--kernelbook-samples` | 1500 | KernelBook sample count |
| `--kernelbench-samples` | 1000 | KernelBench sample count |
| `--vllm-url` | `http://localhost:8000/v1` | vLLM server URL |

## Architecture

### Single Deque Flow (FIFO)

`MultiTurnQueue` is a **passive proxy** — it holds items, tracks turns, and builds feedback strings. All actual work (API calls, Modal validation, code extraction) is driven by `TraceOrchestrator`.

```
                    ┌──────────────────────────────────────────┐
                    │              DEQUE (FIFO)                 │
                    │  popleft() <── items ──> append()         │
                    └──────────────────────────────────────────┘
                                    │
                              popleft()
                                    │
                                    ▼
                    ┌──────────────────────────────────────────┐
                    │         Model (vLLM)                      │
                    │  reasoning_content + <triton>code</triton>│
                    │  (batch of N before validation)           │
                    └──────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────────────┐
                    │         Modal H100 Validation             │
                    │  correctness + speedup                    │
                    └──────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────────────┐
                    │         Decision Logic                    │
                    │  - Success (correct + fast)? → Finalize  │
                    │  - Max turns reached? → Finalize         │
                    │  - Correct but slow? → Requeue           │
                    │  - Error/incorrect? → Requeue            │
                    └──────────────────────────────────────────┘
                                    │
                          (if retry) │
                                    ▼
                          deque.append(item)
                          (back of the line)
```

### MultiTurnQueue API

```python
class MultiTurnQueue:
    def __init__(self, max_turns=4)
    def add(self, item: dict)                    # Push item onto deque
    def pop(self) -> dict | None                 # Pop next item (FIFO)
    def __len__(self) -> int                     # Items remaining
    def should_stop(turn_num, result) -> (bool, str)  # Check if done
    def build_feedback(result) -> str            # Build feedback string
    def requeue_with_feedback(item, feedback, completion)  # Retry
    def finalize(item, reason) -> dict           # Build final trace
```

**Does NOT**: call the model, call Modal, extract code, manage sessions.

### Stop Conditions

| Condition | `should_stop` returns |
|-----------|----------------------|
| `correctness=True` AND `speedup >= 1.0` | `True, "success_fast"` |
| `turn_num >= max_turns` | `True, "max_turns_reached"` |
| Otherwise | `False, "continue"` |

### Feedback Templates

| Scenario | Feedback sent |
|----------|---------------|
| Compilation/runtime error | `"Compilation/runtime error: {error}. Fix and regenerate..."` |
| Incorrect output | `"Incorrect output: Output mismatch with PyTorch reference..."` |
| Correct but slow | `"Correct but slow ({speedup}x PyTorch). Optimize..."` |
| Correct and fast | `"Great work! The kernel is correct and fast."` |

## Conversation Flow

The model sees standard multi-turn chat. Roles naturally delineate turns:

| Turn | Role | Content | reasoning_content |
|------|------|---------|-------------------|
| 1 | system | System prompt + multi-turn addendum | - |
| 1 | user | "Convert this PyTorch code: ```...```" | - |
| 1 | assistant | `<triton>code</triton>` | step-by-step reasoning (extracted by vLLM) |
| 2 | user | "Your kernel FAILED. Error: ... Fix and regenerate." | - |
| 2 | assistant | `<triton>fixed code</triton>` | fix analysis (extracted by vLLM) |
| 3 | user | "Correct but slow (0.8x). Optimize..." | - |
| 3 | assistant | `<triton>optimized code</triton>` | perf analysis (extracted by vLLM) |

**Note:** The model still generates `<think>...</think>` tokens internally, but vLLM's reasoning parser (`glm45`, `qwen3`, etc.) automatically extracts them into the `reasoning_content` field of the API response. The `content` field only contains the non-thinking output.

## Item Structure

Each item in the deque:

```python
{
    "sample_key": str,           # e.g. "kernelbench_80"
    "sample": dict,              # Original sample metadata
    "pytorch_code": str,         # PyTorch code to convert
    "messages": list[dict],      # Full conversation history
    "turn_num": int,             # Current turn (1-based)
    "turns_history": list[dict], # All previous turn results
}
```

## Output Format

Each completed sample produces a trace:

```python
{
    "sample_key": str,
    "source": str,                # "kernelbook" or "kernelbench"
    "level": int | None,
    "name": str | None,
    "problem_id": int | None,
    "pytorch_code": str,
    "num_turns": int,             # How many turns (1-4)
    "stop_reason": str,           # "success_fast" or "max_turns_reached"
    "final_triton_code": str,
    "final_result": {
        "correctness": bool,
        "speedup": float,
        "fast_0": bool, "fast_1": bool, "fast_2": bool,
        "error": str | None,
    },
    "turns": [                    # ALL turns recorded
        {
            "turn": 1,
            "thinking": str,
            "model_reasoning": str,
            "triton_code": str,
            "full_completion": str,
            "feedback_given": str | None,  # null on last turn
            "result": { ... },
        },
    ],
    "full_messages": [ ... ],     # Complete conversation for SFT training
    "timestamp": str,
}
```

## End-to-End Test

`test_multi_turn.py` validates the full pipeline without a live vLLM server. Uses fake generation with deliberately broken/slow/optimized kernels and calls the deployed Modal function.

```bash
python test_multi_turn.py
```

### Test Flow (4 Turns)

| Turn | Kernel | Expected Result |
|------|--------|-----------------|
| 1 | Buggy: `x * y` instead of `relu(x + y)` | `correctness=False` |
| 2 | Illegal memory access (no bounds mask, huge OOB offset) | CUDA error |
| 3 | Correct but slow (two separate kernel launches) | `correctness=True`, `speedup < 1.0` |
| 4 | Optimized fused `add+relu` (single kernel pass) | `correctness=True`, `speedup > 1.0` |

Exercises all feedback paths:
- **Correctness failure** → "Incorrect output: Output mismatch..."
- **Runtime/CUDA error** → "Compilation/runtime error: ..."
- **Correct but slow** → "Correct but slow (0.84x PyTorch). Optimize..."
- **Success** → `stop_reason="success_fast"`, trace finalized

### Test Assertions

- Turn 1: `correctness=False`, no runtime error, feedback given
- Turn 2: `correctness=False`, runtime error present, feedback given
- Turn 3: `correctness=True`, `speedup < 1.0`
- Turn 4: `correctness=True`, `speedup >= 1.0`, `stop_reason="success_fast"`

## Context Window Budget

With `--max-model-len 32768` (GLM-4.5-Air) or `131072` (Qwen3):

| Component | Tokens |
|-----------|--------|
| System prompt | ~1.5K |
| PyTorch code | ~0.5-2K |
| Per turn (response + feedback) | ~2.5-5.5K |
| **Total for 4 turns** | ~15-30K |

Fits within 32K for GLM-4.5-Air. Qwen3's 128K context gives plenty of headroom.

## Model-Specific Notes

### GLM-4.5-Air
- Reasoning parser: `glm45`
- 4x H100 (BF16), `--max-model-len 32768`
- Native `<think>` tag support

### GPT-OSS-120B
- Reasoning parser: `openai_gptoss`
- 2x H100, `--max-model-len 16384`

### Qwen3-235B-A22B (FP8)
- Reasoning parser: `qwen3` (native parser, available since vLLM 0.9.0)
- 8x H100 (FP8), `--max-model-len 131072`
- Requires: `VLLM_USE_V1=1`, `SAFETENSORS_FAST_GPU=1`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`
- `--enable-expert-parallel` for MoE sharding across 8 GPUs
- `--max-num-seqs 64`, `--swap-space 16` for throughput

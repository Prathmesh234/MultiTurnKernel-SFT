# KernelBench Triton - Modal Benchmarking Documentation

## Overview

Modal app (`modal_app.py`) benchmarks Triton GPU kernels against PyTorch reference implementations on H100 GPUs. It measures correctness and performance, returning structured results with `fast_0/1/2` flags.

## Deployment

```bash
modal deploy modal_app.py
```

## Functions

| Function | Purpose |
|---|---|
| `benchmark_triton_kernel` | Generic kernel benchmark with explicit `input_shapes` dict |
| `benchmark_kernelbench` | KernelBench `nn.Module` pattern (uses `get_inputs()` / `get_init_inputs()`) |
| `benchmark_batch` | Sequential batch of `benchmark_kernelbench` calls on the same container |

## Calling Deployed Functions

Use `modal.Function.from_name()` to call the deployed app from any Python script:

```python
import modal

benchmark = modal.Function.from_name("kernelbench-triton", "benchmark_triton_kernel")

result = benchmark.remote(
    kernel_code=kernel_code_string,
    reference_torch_code=reference_code_string,
    input_shapes={"x": {"shape": [4096, 4096], "dtype": "float32"}},
    n_correctness=5,
    n_trials=50,
    kernel_name="my_kernel",
)
```

Run with the Python environment that has `modal` installed:

```bash
/opt/miniconda3/bin/python my_script.py
```

> **Note:** `modal run` only works for scripts that define a Modal app with `@app.local_entrypoint()`. For client scripts that call deployed functions, use plain `python`.

## Result Schema

```python
{
    "kernel_name": str,
    "timestamp": str,          # ISO format
    "correctness": bool,       # All correctness checks passed
    "speedup": float,          # reference_time / kernel_time
    "reference_time_ms": float,
    "kernel_time_ms": float,
    "fast_0": bool,            # Correct
    "fast_1": bool,            # Correct AND faster than reference
    "fast_2": bool,            # Correct AND >= 2x faster
    "error": str | None,       # Error message if any
}
```

## GPU Error Recovery Architecture

### Previous approach (deprecated)

The original design used `multiprocessing.get_context("spawn")` to run each kernel in a fresh subprocess with its own CUDA context. This provided isolation but added complexity.

### Current approach: Modal's built-in retries

We replaced subprocess isolation with Modal's infrastructure-level recovery (commit `6ebe69e`). Each `@app.function` is configured with:

```python
retries=modal.Retries(
    max_retries=2,
    backoff_coefficient=1.0,
    initial_delay=1.0,
)
```

**How it works:**

1. **Fatal GPU errors** (e.g. illegal memory access): The container crashes. Modal automatically reschedules the function on a **fresh container** with a clean CUDA context.
2. **Non-fatal GPU faults**: Caught as `RuntimeError` in our code, which calls `modal.experimental.stop_fetching_inputs()` to drain the container gracefully so no further work is routed to a poisoned GPU.
3. **Non-GPU errors** (import failures, shape mismatches, etc.): Caught by the general `except Exception` handler and returned in the result dict without draining the container.

**Why this is better than subprocess isolation:**

- Simpler code — no multiprocessing, no queues, no IPC
- Modal handles the hard parts (container lifecycle, GPU health monitoring)
- Each `.remote()` call is already container-isolated by Modal
- Retries land on a fresh container with guaranteed clean CUDA state

### Key isolation points

| Point | Isolation mechanism |
|-------|-------------------|
| **Between `.remote()` calls** | Each call may land on a different container |
| **After GPU crash** | Container dies, Modal retries on fresh container |
| **After non-fatal fault** | `stop_fetching_inputs()` drains container, future calls go elsewhere |
| **Module cache** | `sys.modules.pop()` in `finally` block prevents stale modules within a container |

## Verified Recovery Behavior (2026-02-23)

Tested by running a deliberately broken kernel followed immediately by a correct kernel against the deployed app.

### Test 1: Good kernel (vector add)

A correct Triton element-wise add kernel on 4096x4096 float32 tensors:

```python
@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

**Result:**
```
Kernel:      good_vector_add
Correctness: True
Speedup:     0.99x
Ref time:    0.0696 ms
Kernel time: 0.0701 ms
fast_0:      True
fast_1:      False
fast_2:      False
```

Speedup ~1.0x is expected — PyTorch's `x + y` is already a highly optimized elementwise op.

### Test 2: Bad kernel (out-of-bounds memory access)

A deliberately broken kernel with no masking and offsets multiplied by 1,000,000:

```python
@triton.jit
def _oob_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE * 1000000 + tl.arange(0, BLOCK_SIZE)
    # No mask — every load/store hits invalid memory
    x = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, x * 2.0)
```

**Result:**
```
Kernel:      bad_oob_kernel
Correctness: False
Speedup:     0.00x
fast_0:      False
fast_1:      False
fast_2:      False
ERROR:       RuntimeError: CUDA error: operation not supported on global/shared address space
```

The CUDA error was caught, the container was drained via `stop_fetching_inputs()`, and the error was returned in the result dict.

### Test 3: Good kernel AFTER bad kernel (recovery test)

The same correct vector-add kernel called immediately after the bad kernel:

**Result:**
```
Kernel:      good_vector_add_after_crash
Correctness: True
Speedup:     0.99x
Ref time:    0.0697 ms
Kernel time: 0.0703 ms
fast_0:      True
fast_1:      False
fast_2:      False
```

**Confirms:** A poisoned CUDA context does NOT leak into subsequent calls. Modal routed the follow-up call to a fresh container with clean GPU state.

## Kernel Code Format

### For `benchmark_triton_kernel`

**Kernel code** must define `triton_kernel_wrapper`:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _my_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * 2.0, mask=mask)

def triton_kernel_wrapper(x):
    output = torch.empty_like(x)
    n = output.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _my_kernel[grid](x, output, n, BLOCK_SIZE=BLOCK_SIZE)
    return output
```

**Reference code** must define `reference_impl`:

```python
import torch

def reference_impl(x):
    return x * 2.0
```

**Input shapes** dict:

```python
input_shapes = {
    "x": {"shape": [4096, 4096], "dtype": "float32"},
}
```

### For `benchmark_kernelbench`

**PyTorch code** must define an `nn.Module` class, `get_inputs()`, and `get_init_inputs()`:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.linear(x)

def get_inputs():
    return [torch.randn(32, 256, dtype=torch.float32)]

def get_init_inputs():
    return [256]
```

**Triton code** must define `triton_kernel_wrapper` that accepts the same inputs (and optionally model weights/biases — the benchmarker auto-fills these based on parameter count).

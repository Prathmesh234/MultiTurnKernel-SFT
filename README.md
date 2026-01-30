# KernelBench Triton

A benchmarking environment for Triton kernels on Modal H100 GPUs. This is adapted from the [KernelBench](https://github.com/ScalingIntelligence/KernelBench) project by Stanford's Scaling Intelligence Lab.

**Note: This is for BENCHMARKING ONLY, not for reward/training.**

## Features

- Benchmark Triton kernels against PyTorch reference implementations
- Run on NVIDIA H100 GPUs (Hopper architecture) via Modal
- Measure correctness with multiple random inputs
- Measure performance with proper warmup and synchronization
- Calculate speedup metrics (fast_0, fast_1, fast_2)
- Persistent storage for all benchmark results

## Prerequisites

1. **Create a Modal account**: Sign up at [modal.com](https://modal.com)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Authenticate with Modal**:
   ```bash
   modal setup
   ```

## Quick Start

### Deploy the Application

```bash
modal deploy modal_app.py
```

### Run the Example

```bash
modal run modal_app.py
```

This runs a simple vector addition benchmark.

## Usage

### Basic Benchmark

```python
import modal

# Lookup the deployed app
benchmark_fn = modal.Function.lookup("kernelbench-triton", "benchmark_triton_kernel")

# Define your Triton kernel
triton_kernel_code = '''
import torch
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * 2  # Example: double each element
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_kernel_wrapper(x):
    """Wrapper that invokes the Triton kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    my_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output
'''

# Define PyTorch reference
reference_code = '''
import torch

def reference_impl(x):
    return x * 2
'''

# Specify input shapes
input_shapes = {
    "x": {"shape": [1024 * 1024], "dtype": "float32"},
}

# Run benchmark
result = benchmark_fn.remote(
    kernel_code=triton_kernel_code,
    reference_torch_code=reference_code,
    input_shapes=input_shapes,
    n_correctness=10,
    n_trials=100,
    kernel_name="double_elements",
)

print(f"Correctness: {result['correctness']}")
print(f"Speedup: {result['speedup']:.2f}x")
print(f"fast_0: {result['fast_0']}")
print(f"fast_1: {result['fast_1']}")
print(f"fast_2: {result['fast_2']}")
```

### Batch Benchmarking

```python
import modal

benchmark_batch_fn = modal.Function.lookup("kernelbench-triton", "benchmark_batch")

kernels = [
    {
        "kernel_code": kernel1_code,
        "reference_torch_code": ref1_code,
        "input_shapes": shapes1,
        "kernel_name": "kernel_1",
    },
    {
        "kernel_code": kernel2_code,
        "reference_torch_code": ref2_code,
        "input_shapes": shapes2,
        "kernel_name": "kernel_2",
    },
]

results = benchmark_batch_fn.remote(kernels)
for r in results:
    print(f"{r['kernel_name']}: speedup={r['speedup']:.2f}x, correct={r['correctness']}")
```

### Get All Results

```python
import modal

get_results_fn = modal.Function.lookup("kernelbench-triton", "get_all_results")
results = get_results_fn.remote()

for r in results:
    print(f"{r['timestamp']}: {r['kernel_name']} - {r['speedup']:.2f}x")
```

### Get Summary Statistics

```python
import modal

get_stats_fn = modal.Function.lookup("kernelbench-triton", "get_summary_statistics")
stats = get_stats_fn.remote()

print(f"Total benchmarks: {stats['total_benchmarks']}")
print(f"Correctness rate: {stats['correctness_rate']:.2%}")
print(f"fast_1 rate: {stats['fast_1_rate']:.2%}")
print(f"fast_2 rate: {stats['fast_2_rate']:.2%}")
print(f"Average speedup: {stats['average_speedup']:.2f}x")
```

### Clear Results

```python
import modal

clear_fn = modal.Function.lookup("kernelbench-triton", "clear_results")
result = clear_fn.remote()
print(f"Deleted {result['deleted_files']} result files")
```

## Kernel Code Requirements

### Triton Kernel Code

Your kernel code **must** define a function named `triton_kernel_wrapper` that:
- Takes the same keyword arguments as defined in `input_shapes`
- Returns a single output tensor
- Handles all Triton kernel invocation internally

```python
def triton_kernel_wrapper(x, y):
    """Wrapper function that invokes the Triton kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    my_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

### Reference Code

Your reference code **must** define a function named `reference_impl` that:
- Takes the same keyword arguments as defined in `input_shapes`
- Returns a single output tensor using PyTorch operations

```python
def reference_impl(x, y):
    """Reference PyTorch implementation."""
    return x + y
```

## Input Shapes Format

```python
input_shapes = {
    "tensor_name": {
        "shape": [dim1, dim2, ...],  # Required
        "dtype": "float32",           # Optional, defaults to float32
    },
}
```

Supported dtypes: `float32`, `float16`, `bfloat16`, `int32`, `int64`

## Metrics

| Metric | Description |
|--------|-------------|
| `correctness` | Whether the kernel produces correct output |
| `speedup` | `reference_time / kernel_time` |
| `fast_0` | Correct (same as correctness) |
| `fast_1` | Correct AND faster than reference (speedup > 1.0) |
| `fast_2` | Correct AND at least 2x faster than reference (speedup >= 2.0) |

## Configuration

### Benchmark Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_correctness` | 10 | Number of correctness checks with random inputs |
| `n_trials` | 100 | Number of timing trials |
| `n_warmup` | 10 | Number of warmup runs before timing |
| `rtol` | 1e-5 | Relative tolerance for correctness check |
| `atol` | 1e-5 | Absolute tolerance for correctness check |

### GPU Configuration

The environment is configured for H100 GPUs:
- `GPU_CONFIG`: H100
- `TORCH_CUDA_ARCH_LIST`: 9.0 (Hopper architecture)

## Result Storage

All benchmark results are saved to a persistent Modal Volume named `triton-benchmark-results`. Each result is saved as a JSON file with a timestamp-based filename.

Example result structure:
```json
{
  "kernel_name": "vector_add",
  "timestamp": "2024-01-15T10:30:00.123456",
  "correctness": true,
  "speedup": 1.45,
  "reference_time_ms": 0.5234,
  "kernel_time_ms": 0.3610,
  "fast_0": true,
  "fast_1": true,
  "fast_2": false,
  "error": null,
  "input_shapes": {...},
  "n_correctness": 10,
  "n_trials": 100
}
```

## Example Kernels

### Vector Addition

```python
triton_kernel_code = '''
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_kernel_wrapper(x, y):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
'''

reference_code = '''
import torch
def reference_impl(x, y):
    return x + y
'''

input_shapes = {
    "x": {"shape": [1024 * 1024], "dtype": "float32"},
    "y": {"shape": [1024 * 1024], "dtype": "float32"},
}
```

### Matrix Multiplication

```python
triton_kernel_code = '''
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)

def triton_kernel_wrapper(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), triton.cdiv(N, meta["BLOCK_SIZE_N"]))
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
    )
    return c
'''

reference_code = '''
import torch
def reference_impl(a, b):
    return torch.matmul(a, b)
'''

input_shapes = {
    "a": {"shape": [512, 512], "dtype": "float32"},
    "b": {"shape": [512, 512], "dtype": "float32"},
}
```

## Troubleshooting

### Common Issues

1. **"triton_kernel_wrapper not found"**: Ensure your kernel code defines a function named `triton_kernel_wrapper`.

2. **"reference_impl not found"**: Ensure your reference code defines a function named `reference_impl`.

3. **Correctness failures**: Check that your kernel handles boundary conditions correctly (use masks).

4. **Low speedup**: Triton may not always be faster than PyTorch for simple operations. Complex fused operations typically show better speedups.

### Debug Tips

- Set `n_correctness=1` and `n_trials=1` for quick debugging
- Check the `error` field in results for detailed error messages
- Use smaller input shapes when debugging

## Resources

- [Modal Documentation](https://modal.com/docs)
- [Triton Documentation](https://triton-lang.org/)
- [KernelBench Paper](https://scalingintelligence.stanford.edu/blogs/kernelbench/)
- [KernelBench GitHub](https://github.com/ScalingIntelligence/KernelBench)

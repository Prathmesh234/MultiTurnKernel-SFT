"""
KernelBench Triton - Modal Application for Benchmarking Triton Kernels on H100 GPUs

This is a BENCHMARKING ONLY environment, not for reward/training.
It accepts Triton kernel code as input, compares against PyTorch reference,
and measures correctness + performance.
"""

import modal
import json
from datetime import datetime
from typing import Optional

# Create Modal app
app = modal.App("kernelbench-triton")

# Create persistent volume for storing benchmark results
volume = modal.Volume.from_name("triton-benchmark-results", create_if_missing=True)

# Define container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .env({
        "GPU_CONFIG": "H100",
        "TORCH_CUDA_ARCH_LIST": "9.0",  # Hopper architecture
        "TRITON_CACHE_DIR": "/tmp/triton_cache",
    })
    .apt_install("git", "build-essential")
    .pip_install(
        "torch>=2.1.0",
        "triton>=2.1.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "datasets>=2.14.0",
    )
    .run_commands(
        "git clone https://github.com/ScalingIntelligence/KernelBench.git /KernelBench",
        "cd /KernelBench && pip install -e .",
    )
)


@app.function(
    gpu="H100",
    image=image,
    volumes={"/results": volume},
    timeout=3600,
)
def benchmark_triton_kernel(
    kernel_code: str,
    reference_torch_code: str,
    input_shapes: dict,
    n_correctness: int = 10,
    n_trials: int = 100,
    n_warmup: int = 10,
    kernel_name: Optional[str] = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> dict:
    """
    Benchmark a Triton kernel against PyTorch reference implementation.

    This is for BENCHMARKING ONLY, not for reward/training.

    Args:
        kernel_code: Complete Triton kernel source code as string.
                     Must define a function named 'triton_kernel_wrapper' that takes
                     the same inputs as the reference and returns the output tensor.
        reference_torch_code: PyTorch reference implementation as string.
                              Must define a function named 'reference_impl' that takes
                              input tensors and returns the output tensor.
        input_shapes: Dictionary specifying input tensor shapes and dtypes.
                      Format: {"input_name": {"shape": [dim1, dim2, ...], "dtype": "float32"}}
        n_correctness: Number of correctness checks with random inputs.
        n_trials: Number of timing trials for performance measurement.
        n_warmup: Number of warmup runs before timing.
        kernel_name: Optional name for the kernel (for logging/results).
        rtol: Relative tolerance for correctness check.
        atol: Absolute tolerance for correctness check.

    Returns:
        Dictionary containing:
        - correctness: bool - Whether kernel produces correct output
        - speedup: float - reference_time / kernel_time
        - reference_time_ms: float - Average reference execution time in ms
        - kernel_time_ms: float - Average kernel execution time in ms
        - fast_0: bool - Correct (same as correctness)
        - fast_1: bool - Correct AND faster than reference
        - fast_2: bool - Correct AND at least 2x faster than reference
        - timestamp: str - ISO format timestamp
        - kernel_name: str - Name of the kernel
        - error: str - Error message if any
    """
    import torch
    import time
    import traceback

    # Initialize result dict
    result = {
        "kernel_name": kernel_name or "unnamed_kernel",
        "timestamp": datetime.now().isoformat(),
        "correctness": False,
        "speedup": 0.0,
        "reference_time_ms": None,
        "kernel_time_ms": None,
        "fast_0": False,
        "fast_1": False,
        "fast_2": False,
        "error": None,
        "input_shapes": input_shapes,
        "n_correctness": n_correctness,
        "n_trials": n_trials,
    }

    try:
        import tempfile
        import importlib.util
        import os
        import sys

        # Write kernel code to a temporary file so Triton can access source
        with tempfile.NamedTemporaryFile(mode='w', suffix='_kernel.py', delete=False) as f:
            f.write(kernel_code)
            kernel_file = f.name

        # Write reference code to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='_reference.py', delete=False) as f:
            f.write(reference_torch_code)
            reference_file = f.name

        try:
            # Import kernel module
            spec = importlib.util.spec_from_file_location("kernel_module", kernel_file)
            kernel_module = importlib.util.module_from_spec(spec)
            sys.modules["kernel_module"] = kernel_module
            spec.loader.exec_module(kernel_module)
            
            if not hasattr(kernel_module, "triton_kernel_wrapper"):
                raise ValueError("Kernel code must define 'triton_kernel_wrapper' function")
            triton_wrapper = kernel_module.triton_kernel_wrapper

            # Import reference module
            spec = importlib.util.spec_from_file_location("reference_module", reference_file)
            reference_module = importlib.util.module_from_spec(spec)
            sys.modules["reference_module"] = reference_module
            spec.loader.exec_module(reference_module)
            
            if not hasattr(reference_module, "reference_impl"):
                raise ValueError("Reference code must define 'reference_impl' function")
            reference_impl = reference_module.reference_impl

            # Helper to generate random inputs
            def generate_inputs():
                inputs = {}
                for name, spec in input_shapes.items():
                    shape = spec["shape"]
                    dtype_str = spec.get("dtype", "float32")
                    dtype_map = {
                        "float32": torch.float32,
                        "float16": torch.float16,
                        "bfloat16": torch.bfloat16,
                        "int32": torch.int32,
                        "int64": torch.int64,
                    }
                    dtype = dtype_map.get(dtype_str, torch.float32)

                    if dtype in [torch.float32, torch.float16, torch.bfloat16]:
                        inputs[name] = torch.randn(shape, dtype=dtype, device="cuda")
                    else:
                        inputs[name] = torch.randint(0, 100, shape, dtype=dtype, device="cuda")
                return inputs

            # Check correctness
            print(f"Running {n_correctness} correctness checks...")
            correctness = True
            for i in range(n_correctness):
                inputs = generate_inputs()

                # Run reference
                ref_output = reference_impl(**inputs)

                # Run Triton kernel
                kernel_output = triton_wrapper(**inputs)

                # Compare outputs
                if not torch.allclose(ref_output, kernel_output, rtol=rtol, atol=atol):
                    correctness = False
                    max_diff = (ref_output - kernel_output).abs().max().item()
                    print(f"Correctness check {i+1} failed. Max diff: {max_diff}")
                    break

            result["correctness"] = correctness
            result["fast_0"] = correctness

            # Measure performance only if correct
            if correctness:
                print(f"Running performance benchmark ({n_warmup} warmup, {n_trials} trials)...")

                # Use fixed inputs for timing
                timing_inputs = generate_inputs()

                # Warmup
                for _ in range(n_warmup):
                    _ = reference_impl(**timing_inputs)
                    _ = triton_wrapper(**timing_inputs)
                torch.cuda.synchronize()

                # Time reference
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(n_trials):
                    _ = reference_impl(**timing_inputs)
                torch.cuda.synchronize()
                reference_time = (time.perf_counter() - start) / n_trials * 1000  # ms

                # Time Triton kernel
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(n_trials):
                    _ = triton_wrapper(**timing_inputs)
                torch.cuda.synchronize()
                kernel_time = (time.perf_counter() - start) / n_trials * 1000  # ms

                speedup = reference_time / kernel_time if kernel_time > 0 else 0.0

                result["reference_time_ms"] = reference_time
                result["kernel_time_ms"] = kernel_time
                result["speedup"] = speedup
                result["fast_1"] = speedup > 1.0
                result["fast_2"] = speedup >= 2.0

                print(f"Reference time: {reference_time:.4f} ms")
                print(f"Kernel time: {kernel_time:.4f} ms")
                print(f"Speedup: {speedup:.2f}x")

        finally:
            # Cleanup temporary files
            try:
                os.unlink(kernel_file)
                os.unlink(reference_file)
            except:
                pass

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"Error during benchmark: {result['error']}")

    # Save result to persistent storage
    result_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    result_path = f"/results/{result_id}.json"

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    volume.commit()
    print(f"Result saved to {result_path}")

    return result


@app.function(
    gpu="H100",
    image=image,
    volumes={"/results": volume},
    timeout=7200,
)
def benchmark_batch(kernels: list[dict]) -> list[dict]:
    """
    Benchmark multiple Triton kernels in batch (SEQUENTIAL - same container).

    Args:
        kernels: List of kernel specifications, each containing:
            - kernel_code: Triton kernel source code
            - reference_torch_code: PyTorch reference implementation
            - input_shapes: Input tensor specifications
            - kernel_name (optional): Name for the kernel
            - n_correctness (optional): Number of correctness checks
            - n_trials (optional): Number of timing trials

    Returns:
        List of benchmark results
    """
    results = []
    for i, kernel_spec in enumerate(kernels):
        print(f"\n{'='*50}")
        print(f"Benchmarking kernel {i+1}/{len(kernels)}: {kernel_spec.get('kernel_name', 'unnamed')}")
        print(f"{'='*50}")

        result = benchmark_triton_kernel.local(
            kernel_code=kernel_spec["kernel_code"],
            reference_torch_code=kernel_spec["reference_torch_code"],
            input_shapes=kernel_spec["input_shapes"],
            n_correctness=kernel_spec.get("n_correctness", 10),
            n_trials=kernel_spec.get("n_trials", 100),
            kernel_name=kernel_spec.get("kernel_name"),
        )
        results.append(result)

    return results


# =============================================================================
# PARALLEL EXECUTION - Multiple H100 Containers
# =============================================================================
# This is the recommended approach for high-throughput benchmarking.
# Uses Modal's .map() to distribute kernels across N containers.
# Total time = max(kernel_times) instead of sum(kernel_times)
# =============================================================================

# Default number of parallel containers
DEFAULT_PARALLEL_CONTAINERS = 4


@app.function(
    gpu="H100",
    image=image,
    volumes={"/results": volume},
    timeout=3600,
    concurrency_limit=8,  # Allow up to 8 parallel containers
)
def benchmark_single(kernel_spec: dict) -> dict:
    """
    Benchmark a single Triton kernel (for parallel execution).
    
    This function is designed to be called via .map() for parallel execution.
    Each call runs on a separate H100 container.
    
    Args:
        kernel_spec: Kernel specification containing:
            - kernel_code: Triton kernel source code
            - reference_torch_code: PyTorch reference implementation
            - input_shapes: Input tensor specifications
            - kernel_name (optional): Name for the kernel
            - n_correctness (optional): Number of correctness checks
            - n_trials (optional): Number of timing trials
            - rtol (optional): Relative tolerance for correctness
            - atol (optional): Absolute tolerance for correctness
    
    Returns:
        Benchmark result dictionary
    """
    return benchmark_triton_kernel.local(
        kernel_code=kernel_spec["kernel_code"],
        reference_torch_code=kernel_spec["reference_torch_code"],
        input_shapes=kernel_spec["input_shapes"],
        n_correctness=kernel_spec.get("n_correctness", 10),
        n_trials=kernel_spec.get("n_trials", 100),
        kernel_name=kernel_spec.get("kernel_name"),
        rtol=kernel_spec.get("rtol", 1e-5),
        atol=kernel_spec.get("atol", 1e-5),
    )


def benchmark_parallel(kernels: list[dict], n_containers: int = DEFAULT_PARALLEL_CONTAINERS) -> list[dict]:
    """
    Benchmark multiple Triton kernels in PARALLEL across multiple H100 containers.
    
    This is the recommended approach for high-throughput benchmarking.
    Uses Modal's .map() to distribute kernels across N containers.
    
    Benefits:
        - 4x+ throughput for large batches (with n_containers=4)
        - Total time = max(kernel_times) instead of sum(kernel_times)
        - Essential for fast RL iteration
    
    Args:
        kernels: List of kernel specifications (see benchmark_single for format)
        n_containers: Maximum number of parallel containers (default: 4)
    
    Returns:
        List of benchmark results in the same order as input kernels
    
    Example:
        >>> kernels = [
        ...     {"kernel_code": code1, "reference_torch_code": ref1, ...},
        ...     {"kernel_code": code2, "reference_torch_code": ref2, ...},
        ... ]
        >>> results = benchmark_parallel(kernels, n_containers=4)
    """
    print(f"\n{'='*60}")
    print(f"PARALLEL BENCHMARK: {len(kernels)} kernels on up to {n_containers} H100 containers")
    print(f"{'='*60}\n")
    
    # Use Modal's .map() for parallel execution
    results = list(benchmark_single.map(kernels))
    
    # Print summary
    correct = sum(1 for r in results if r.get("correctness", False))
    fast_1 = sum(1 for r in results if r.get("fast_1", False))
    fast_2 = sum(1 for r in results if r.get("fast_2", False))
    
    print(f"\n{'='*60}")
    print(f"PARALLEL BENCHMARK COMPLETE")
    print(f"  Total kernels: {len(results)}")
    print(f"  Correct: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    print(f"  fast_1 (correct & faster): {fast_1}/{len(results)}")
    print(f"  fast_2 (correct & 2x faster): {fast_2}/{len(results)}")
    print(f"{'='*60}\n")
    
    return results


@app.function(
    image=image,
    volumes={"/results": volume},
)
def benchmark_parallel_remote(kernels: list[dict], n_containers: int = DEFAULT_PARALLEL_CONTAINERS) -> list[dict]:
    """
    Remote-callable version of benchmark_parallel.
    
    Use this to trigger parallel benchmarking from outside Modal:
        results = benchmark_parallel_remote.remote(kernels, n_containers=4)
    """
    return benchmark_parallel(kernels, n_containers)


@app.function(
    image=image,
    volumes={"/results": volume},
)
def get_all_results() -> list[dict]:
    """
    Retrieve all benchmark results from persistent storage.

    Returns:
        List of all benchmark results as dictionaries.
    """
    import os

    volume.reload()  # Refresh volume to get latest results

    results = []
    results_dir = "/results"

    if os.path.exists(results_dir):
        for filename in sorted(os.listdir(results_dir)):
            if filename.endswith(".json"):
                filepath = os.path.join(results_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        results.append(json.load(f))
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    return results


@app.function(
    image=image,
    volumes={"/results": volume},
)
def get_summary_statistics() -> dict:
    """
    Calculate summary statistics across all benchmarks.

    Returns:
        Dictionary containing:
        - total_benchmarks: Total number of benchmarks
        - correctness_rate: Fraction of correct kernels
        - fast_1_rate: Fraction of kernels that are correct AND faster
        - fast_2_rate: Fraction of kernels that are correct AND 2x faster
        - average_speedup: Average speedup of correct kernels
        - speedup_distribution: Min, median, max speedups
    """
    import statistics

    results = get_all_results.local()

    if not results:
        return {
            "total_benchmarks": 0,
            "correctness_rate": 0.0,
            "fast_1_rate": 0.0,
            "fast_2_rate": 0.0,
            "average_speedup": 0.0,
            "speedup_distribution": {"min": 0.0, "median": 0.0, "max": 0.0},
        }

    total = len(results)
    correct = sum(1 for r in results if r.get("correctness", False))
    fast_1 = sum(1 for r in results if r.get("fast_1", False))
    fast_2 = sum(1 for r in results if r.get("fast_2", False))

    speedups = [r["speedup"] for r in results if r.get("correctness", False) and r.get("speedup")]
    avg_speedup = statistics.mean(speedups) if speedups else 0.0

    if speedups:
        speedup_dist = {
            "min": min(speedups),
            "median": statistics.median(speedups),
            "max": max(speedups),
        }
    else:
        speedup_dist = {"min": 0.0, "median": 0.0, "max": 0.0}

    return {
        "total_benchmarks": total,
        "correctness_rate": correct / total,
        "fast_1_rate": fast_1 / total,
        "fast_2_rate": fast_2 / total,
        "average_speedup": avg_speedup,
        "speedup_distribution": speedup_dist,
    }


@app.function(
    image=image,
    volumes={"/results": volume},
)
def clear_results() -> dict:
    """
    Clear all benchmark results from persistent storage.

    Returns:
        Dictionary with count of deleted files.
    """
    import os

    volume.reload()

    deleted = 0
    results_dir = "/results"

    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(results_dir, filename)
                try:
                    os.remove(filepath)
                    deleted += 1
                except Exception as e:
                    print(f"Error deleting {filename}: {e}")

    volume.commit()

    return {"deleted_files": deleted}


@app.local_entrypoint()
def main():
    """
    Example usage of the KernelBench Triton benchmarking system.
    """
    # Example: Sum reduction along dimension 1
    # Production-max size: [64, 256, 512, 512] = 4.3B elements
    triton_kernel_code = '''
import torch
import triton
import triton.language as tl


@triton.jit
def triton_sum_dim1_kernel(
    in_ptr, out_ptr,
    batch_size, inner_size,
    reduce_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Sum reduction along dim=1. Each thread block handles BLOCK_SIZE output elements."""
    pid = tl.program_id(0)
    
    # Each program handles BLOCK_SIZE output elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_output = batch_size * inner_size
    mask = offsets < total_output
    
    # Calculate batch and inner indices for each output element
    batch_idx = offsets // inner_size
    inner_idx = offsets % inner_size
    
    # Input stride: [batch, reduce, h, w] flattened
    # Stride for batch = reduce_dim * inner_size
    # Stride for reduce = inner_size
    batch_stride = reduce_dim * inner_size
    
    # Accumulate sum over reduce dimension
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Use a while loop instead of for loop for dynamic reduce_dim
    # This avoids Triton trying to unroll a large loop
    r = 0
    while r < reduce_dim:
        in_offset = batch_idx * batch_stride + r * inner_size + inner_idx
        val = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
        acc += val
        r += 1
    
    tl.store(out_ptr + offsets, acc, mask=mask)


def triton_kernel_wrapper(x):
    """Wrapper: sum 4D tensor along dim=1."""
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dim() == 4, "Input must be 4D"
    
    batch, reduce_dim, h, w = x.shape
    inner_size = h * w
    
    # Output shape: [batch, h, w]
    output = torch.empty((batch, h, w), dtype=x.dtype, device=x.device)
    
    # Total output elements
    total_output = batch * inner_size
    
    # Config
    BLOCK_SIZE = 1024
    grid = ((total_output + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Flatten input to 1D for pointer arithmetic
    x_flat = x.contiguous().view(-1)
    out_flat = output.view(-1)
    
    triton_sum_dim1_kernel[grid](
        x_flat, out_flat,
        batch, inner_size,
        reduce_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )
    
    return output
'''

    # Reference PyTorch implementation
    reference_code = '''
import torch

def reference_impl(x):
    """Reference PyTorch implementation: sum along dim=1."""
    return torch.sum(x, dim=1)
'''

    # Medium-large input: 536 million elements = ~2.1 GB
    input_shapes = {
        "x": {"shape": [64, 128, 256, 256], "dtype": "float32"},
    }

    print("Running KernelBench Triton benchmark example...")
    print("=" * 60)

    # Run benchmark with higher tolerance for large reductions
    # (summing 128 numbers causes accumulated float errors)
    result = benchmark_triton_kernel.remote(
        kernel_code=triton_kernel_code,
        reference_torch_code=reference_code,
        input_shapes=input_shapes,
        n_correctness=5,
        n_trials=50,
        kernel_name="sum_dim1_medium_large",
        rtol=1e-4,  # Higher tolerance for large reductions
        atol=1e-4,
    )

    print("\nBenchmark Results:")
    print("-" * 40)
    print(f"Kernel: {result['kernel_name']}")
    print(f"Correctness: {result['correctness']}")
    print(f"Speedup: {result['speedup']:.2f}x")
    print(f"Reference time: {result['reference_time_ms']:.4f} ms" if result['reference_time_ms'] else "Reference time: N/A")
    print(f"Kernel time: {result['kernel_time_ms']:.4f} ms" if result['kernel_time_ms'] else "Kernel time: N/A")
    print(f"fast_0 (correct): {result['fast_0']}")
    print(f"fast_1 (correct & faster): {result['fast_1']}")
    print(f"fast_2 (correct & 2x faster): {result['fast_2']}")

    if result['error']:
        print(f"\nError: {result['error']}")

    # Get summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics (all benchmarks):")
    print("-" * 40)

    stats = get_summary_statistics.remote()
    print(f"Total benchmarks: {stats['total_benchmarks']}")
    print(f"Correctness rate: {stats['correctness_rate']:.2%}")
    print(f"fast_1 rate: {stats['fast_1_rate']:.2%}")
    print(f"fast_2 rate: {stats['fast_2_rate']:.2%}")
    print(f"Average speedup: {stats['average_speedup']:.2f}x")
    print(f"Speedup range: {stats['speedup_distribution']['min']:.2f}x - {stats['speedup_distribution']['max']:.2f}x")

"""
KernelBench Triton - Modal Application for Benchmarking Triton Kernels on H100 GPUs

This is a BENCHMARKING ONLY environment, not for reward/training.
It accepts Triton kernel code as input, compares against PyTorch reference,
and measures correctness + performance.

Functions:
    benchmark_triton_kernel  - Single kernel benchmark (generic input_shapes dict)
    benchmark_kernelbench    - Single kernel benchmark (KernelBench nn.Module pattern)
    benchmark_batch          - Sequential batch on same container

GPU error recovery:
    Modal handles CUDA crashes automatically. If a kernel triggers an illegal memory
    access, the container crashes and Modal reschedules on a fresh container with a
    clean CUDA context (via retries). For non-fatal GPU faults, we call
    modal.experimental.stop_fetching_inputs() to drain the container gracefully.
    See: https://modal.com/docs/guide/gpu-health

For parallel execution and results management, see AGENTS.md "Deferred Features".
"""

import modal
import json
from datetime import datetime
from typing import Optional

# Import utilities (will be available locally, imported inside functions for Modal)
try:
    from utilities import extract_inputs, get_shapes
except ImportError:
    pass

# Create Modal app
app = modal.App("kernelbench-triton")

# Create persistent volume for storing benchmark results
volume = modal.Volume.from_name("triton-benchmark-results", create_if_missing=True)

# Define container image with all dependencies
# NOTE: KernelBook was generated with PyTorch 2.5.0, so we need that version
#       for torch._inductor.runtime.triton_heuristics.grid to be available
# IMPORTANT: Install KernelBench first, then upgrade PyTorch to 2.5+ after
image = (
    modal.Image.debian_slim(python_version="3.10")
    .env({
        "GPU_CONFIG": "H100",
        "TORCH_CUDA_ARCH_LIST": "9.0",  # Hopper architecture
        "TRITON_CACHE_DIR": "/tmp/triton_cache",
    })
    .apt_install("git", "build-essential")
    .pip_install(
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "datasets>=2.14.0",
    )
    .run_commands(
        "git clone https://github.com/ScalingIntelligence/KernelBench.git /KernelBench",
        "cd /KernelBench && pip install -e .",
    )
    # Install PyTorch 2.5+ AFTER KernelBench to ensure correct version
    .pip_install(
        "torch==2.5.0",  # Exact version KernelBook was generated with
        "triton==3.1.0",  # Match triton version with PyTorch 2.5
    )
    # Mount utilities module so it's importable inside remote functions
    .add_local_file("utilities.py", "/root/utilities.py")
)


def _make_error_result(kernel_name, error_msg, **extra):
    """Build a standard error result dict."""
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
        "error": error_msg,
    }
    result.update(extra)
    return result


@app.function(
    gpu="H100",
    image=image,
    volumes={"/results": volume},
    timeout=600,
    retries=modal.Retries(
        max_retries=2,
        backoff_coefficient=1.0,
        initial_delay=1.0,
    ),
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

    GPU error recovery is handled by Modal's infrastructure:
    - Container crashes (e.g. CUDA illegal memory access) are retried on a fresh
      container with a clean CUDA context via modal.Retries.
    - Non-fatal GPU faults trigger modal.experimental.stop_fetching_inputs() to
      drain the container gracefully.

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
    import sys
    import torch
    import time
    import traceback
    import tempfile
    import importlib.util
    import os

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
        # Write code to temp files (Triton JIT needs real files on disk)
        with tempfile.NamedTemporaryFile(mode='w', suffix='_kernel.py', delete=False) as f:
            f.write(kernel_code)
            kernel_file = f.name

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

            # Helper to generate random inputs from input_shapes dict
            def generate_inputs():
                inputs = {}
                for name, shape_spec in input_shapes.items():
                    shape = shape_spec["shape"]
                    dtype_str = shape_spec.get("dtype", "float32")
                    dtype_map = {
                        "float32": torch.float32,
                        "float16": torch.float16,
                        "bfloat16": torch.bfloat16,
                        "int32": torch.int32,
                        "int64": torch.int64,
                    }
                    dtype = dtype_map.get(dtype_str, torch.float32)
                    if dtype in (torch.float32, torch.float16, torch.bfloat16):
                        inputs[name] = torch.randn(shape, dtype=dtype, device="cuda")
                    else:
                        inputs[name] = torch.randint(0, 100, shape, dtype=dtype, device="cuda")
                return inputs

            # --- Correctness phase ---
            print(f"Running {n_correctness} correctness checks...")
            correctness = True
            for i in range(n_correctness):
                inputs = generate_inputs()
                ref_output = reference_impl(**inputs)
                kernel_output = triton_wrapper(**inputs)

                if not torch.allclose(ref_output, kernel_output, rtol=rtol, atol=atol):
                    correctness = False
                    max_diff = (ref_output - kernel_output).abs().max().item()
                    print(f"Correctness check {i+1} failed. Max diff: {max_diff}")
                    break

            result["correctness"] = correctness
            result["fast_0"] = correctness

            # --- Performance phase (only if correct) ---
            if correctness:
                print(f"Running performance benchmark ({n_warmup} warmup, {n_trials} trials)...")
                timing_inputs = generate_inputs()

                # Warmup
                for _ in range(n_warmup):
                    reference_impl(**timing_inputs)
                    triton_wrapper(**timing_inputs)
                torch.cuda.synchronize()

                # Time reference
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(n_trials):
                    reference_impl(**timing_inputs)
                torch.cuda.synchronize()
                reference_time = (time.perf_counter() - start) / n_trials * 1000

                # Time Triton kernel
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(n_trials):
                    triton_wrapper(**timing_inputs)
                torch.cuda.synchronize()
                kernel_time = (time.perf_counter() - start) / n_trials * 1000

                speedup = reference_time / kernel_time if kernel_time > 0 else 0.0
                result["reference_time_ms"] = reference_time
                result["kernel_time_ms"] = kernel_time
                result["speedup"] = speedup
                result["fast_1"] = speedup > 1.0
                result["fast_2"] = speedup >= 2.0

                print(f"Reference: {reference_time:.4f} ms | Kernel: {kernel_time:.4f} ms | Speedup: {speedup:.2f}x")

        finally:
            try:
                os.unlink(kernel_file)
                os.unlink(reference_file)
            except Exception:
                pass
            # Evict cached modules so the next invocation doesn't get stale code
            sys.modules.pop("kernel_module", None)
            sys.modules.pop("reference_module", None)

    except RuntimeError as e:
        # GPU faults (e.g. illegal memory access) surface as RuntimeError.
        # Signal Modal to stop sending work to this container so it gets replaced.
        modal.experimental.stop_fetching_inputs()
        result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"GPU fault — draining container: {result['error']}")

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"Error: {result['error']}")

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
    timeout=600,
    retries=modal.Retries(
        max_retries=2,
        backoff_coefficient=1.0,
        initial_delay=1.0,
    ),
)
def benchmark_kernelbench(
    triton_code: str,
    pytorch_code: str,
    n_correctness: int = 5,
    n_trials: int = 20,
    kernel_name: Optional[str] = None,
    entry_point: str = "Model",
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> dict:
    """
    Benchmark a Triton kernel against PyTorch reference for KernelBench problems.

    This function handles the KernelBench/KernelBook pattern where:
    - pytorch_code defines a nn.Module class with parameters (weights, biases)
    - pytorch_code defines get_inputs() and get_init_inputs() functions
    - triton_code defines triton_kernel_wrapper() that takes model inputs

    GPU error recovery is handled by Modal's infrastructure:
    - Container crashes (e.g. CUDA illegal memory access) are retried on a fresh
      container with a clean CUDA context via modal.Retries.
    - Non-fatal GPU faults trigger modal.experimental.stop_fetching_inputs() to
      drain the container gracefully.

    Args:
        triton_code: Complete Triton kernel source code as string.
                     Must define a function named 'triton_kernel_wrapper'.
        pytorch_code: PyTorch code with a nn.Module class,
                      get_inputs(), and get_init_inputs() functions.
        n_correctness: Number of correctness checks with random inputs.
        n_trials: Number of timing trials for performance measurement.
        kernel_name: Optional name for the kernel (for logging/results).
        entry_point: Name of the nn.Module class in pytorch_code (default: "Model").
        rtol: Relative tolerance for correctness check.
        atol: Absolute tolerance for correctness check.

    Returns:
        Benchmark result dictionary with correctness, speedup, fast_0/1/2 flags.
    """
    import sys
    sys.path.insert(0, "/root")  # Modal mounts utilities.py at /root

    import torch
    import time
    import traceback
    import tempfile
    import importlib.util
    import os
    import inspect

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
    }

    try:
        # Write code strings to temp files so importlib can load them as modules.
        # Triton needs real files on disk (not exec'd strings) to compile kernels.
        with tempfile.NamedTemporaryFile(mode='w', suffix='_pytorch.py', delete=False) as f:
            f.write(pytorch_code)
            pytorch_file = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='_triton.py', delete=False) as f:
            f.write(triton_code)
            triton_file = f.name

        try:
            # --- Load the PyTorch reference module ---
            spec = importlib.util.spec_from_file_location("pytorch_module", pytorch_file)
            pytorch_module = importlib.util.module_from_spec(spec)
            sys.modules["pytorch_module"] = pytorch_module
            spec.loader.exec_module(pytorch_module)

            # Find the nn.Module class: try entry_point first, then "Model",
            # then scan for any nn.Module subclass as a last resort.
            if hasattr(pytorch_module, entry_point):
                ModelClass = getattr(pytorch_module, entry_point)
            elif hasattr(pytorch_module, "Model"):
                ModelClass = pytorch_module.Model
            else:
                import torch.nn as nn
                ModelClass = None
                for name, obj in vars(pytorch_module).items():
                    if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
                        ModelClass = obj
                        print(f"Found nn.Module class: {name}")
                        break
                if ModelClass is None:
                    raise ValueError(f"PyTorch code must define '{entry_point}' class or an nn.Module subclass")

            # Smoke-test get_inputs() to catch shape/dtype problems early
            from utilities import extract_inputs
            test_inputs = extract_inputs(pytorch_code)
            print(f"Extracted {len(test_inputs)} inputs: {[tuple(t.shape) for t in test_inputs if hasattr(t, 'shape')]}")

            get_inputs = pytorch_module.get_inputs
            get_init_inputs = getattr(pytorch_module, "get_init_inputs", lambda: [])

            # --- Load the Triton kernel module ---
            spec = importlib.util.spec_from_file_location("triton_module", triton_file)
            triton_module = importlib.util.module_from_spec(spec)
            sys.modules["triton_module"] = triton_module
            spec.loader.exec_module(triton_module)  # JIT compilation happens here

            if not hasattr(triton_module, "triton_kernel_wrapper"):
                raise ValueError("Triton code must define 'triton_kernel_wrapper' function")
            triton_kernel_wrapper = triton_module.triton_kernel_wrapper

            # Inspect the wrapper signature so we know how many args to pass
            sig = inspect.signature(triton_kernel_wrapper)
            wrapper_params = list(sig.parameters.keys())
            print(f"triton_kernel_wrapper expects: {wrapper_params}")

            # --- Instantiate the reference model ---
            # get_init_inputs() may return [], [args...], or ([args], {kwargs})
            init_inputs = get_init_inputs()
            if isinstance(init_inputs, (list, tuple)) and len(init_inputs) == 2:
                if isinstance(init_inputs[0], (list, tuple)) and isinstance(init_inputs[1], dict):
                    # ([positional], {keyword}) format
                    model = ModelClass(*init_inputs[0], **init_inputs[1]).cuda()
                else:
                    model = ModelClass(*init_inputs).cuda()
            elif isinstance(init_inputs, (list, tuple)):
                model = ModelClass(*init_inputs).cuda()
            else:
                model = ModelClass().cuda()
            model.eval()

            def _call_wrapper(cuda_inputs):
                """Call triton_kernel_wrapper with the right number of args.

                Kernels vary in what they expect:
                  - just the input tensor(s) matching get_inputs()
                  - input tensor(s) + model weight/bias parameters
                We match by parameter count.
                """
                if len(wrapper_params) == 1:
                    return triton_kernel_wrapper(cuda_inputs[0])
                elif len(wrapper_params) == len(cuda_inputs):
                    return triton_kernel_wrapper(*cuda_inputs)
                else:
                    # Kernel wants more args than get_inputs() provides —
                    # append model parameters (weights then biases) to fill the gap.
                    kernel_args = list(cuda_inputs)
                    for module in model.modules():
                        if hasattr(module, 'weight') and module.weight is not None:
                            kernel_args.append(module.weight.data)
                        if hasattr(module, 'bias') and module.bias is not None:
                            kernel_args.append(module.bias.data)
                    return triton_kernel_wrapper(*kernel_args[:len(wrapper_params)])

            # --- Correctness phase ---
            print(f"Running {n_correctness} correctness checks...")
            correctness = True
            for i in range(n_correctness):
                inputs = get_inputs()
                if not isinstance(inputs, (list, tuple)):
                    inputs = [inputs]
                cuda_inputs = [x.cuda() if hasattr(x, 'cuda') else x for x in inputs]

                with torch.no_grad():
                    ref_output = model(*cuda_inputs)

                try:
                    kernel_output = _call_wrapper(cuda_inputs)
                except Exception as first_err:
                    # Signature inference failed — try the explicit weight/bias fallback
                    try:
                        kernel_args = list(cuda_inputs)
                        for module in model.modules():
                            if hasattr(module, 'weight') and module.weight is not None:
                                kernel_args.append(module.weight.data)
                            if hasattr(module, 'bias') and module.bias is not None:
                                kernel_args.append(module.bias.data)
                        kernel_output = triton_kernel_wrapper(*kernel_args[:len(wrapper_params)])
                    except Exception:
                        raise first_err  # both attempts failed; surface the original error

                if not torch.allclose(ref_output, kernel_output, rtol=rtol, atol=atol):
                    correctness = False
                    max_diff = (ref_output - kernel_output).abs().max().item()
                    print(f"Correctness check {i+1} failed. Max diff: {max_diff}")
                    print(f"  ref shape: {ref_output.shape}, kernel shape: {kernel_output.shape}")
                    break  # no need to run remaining checks

            result["correctness"] = correctness
            result["fast_0"] = correctness  # fast_0 == "at least as correct as ref"

            # --- Performance phase (only if correct) ---
            if correctness:
                print(f"Running performance benchmark ({n_trials} trials)...")

                # Use fixed inputs for timing so variance comes from the kernel, not data gen
                inputs = get_inputs()
                if not isinstance(inputs, (list, tuple)):
                    inputs = [inputs]
                cuda_inputs = [x.cuda() if hasattr(x, 'cuda') else x for x in inputs]
                kernel_call = lambda: _call_wrapper(cuda_inputs)

                # Warmup: let CUDA JIT settle and caches warm up before we start the clock
                for _ in range(10):
                    with torch.no_grad():
                        model(*cuda_inputs)
                    kernel_call()
                torch.cuda.synchronize()

                # Time the PyTorch reference
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(n_trials):
                    with torch.no_grad():
                        model(*cuda_inputs)
                torch.cuda.synchronize()  # wait for all GPU work to finish before stopping clock
                reference_time = (time.perf_counter() - start) / n_trials * 1000  # ms per call

                # Time the Triton kernel
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(n_trials):
                    kernel_call()
                torch.cuda.synchronize()
                kernel_time = (time.perf_counter() - start) / n_trials * 1000

                speedup = reference_time / kernel_time if kernel_time > 0 else 0.0
                result["reference_time_ms"] = reference_time
                result["kernel_time_ms"] = kernel_time
                result["speedup"] = speedup
                result["fast_1"] = speedup > 1.0   # faster than PyTorch
                result["fast_2"] = speedup >= 2.0  # at least 2x faster

                print(f"Reference: {reference_time:.4f} ms | Kernel: {kernel_time:.4f} ms | Speedup: {speedup:.2f}x")

        finally:
            # Always clean up temp files even if the kernel crashed mid-execution
            try:
                os.unlink(pytorch_file)
                os.unlink(triton_file)
            except Exception:
                pass
            # Evict cached modules so the next invocation doesn't get stale code
            sys.modules.pop("pytorch_module", None)
            sys.modules.pop("triton_module", None)

    except RuntimeError as e:
        # GPU faults (e.g. illegal memory access) surface as RuntimeError.
        # Signal Modal to stop sending work to this container so it gets replaced.
        modal.experimental.stop_fetching_inputs()
        result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"GPU fault — draining container: {result['error']}")

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"Error: {result['error']}")

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
    retries=modal.Retries(
        max_retries=2,
        backoff_coefficient=1.0,
        initial_delay=1.0,
    ),
)
def benchmark_batch(kernels: list[dict]) -> list[dict]:
    """
    Benchmark multiple Triton kernels in batch (SEQUENTIAL - same container).

    Each kernel is benchmarked via benchmark_kernelbench.local(). If a kernel
    causes a fatal CUDA error, the container crashes and Modal retries on a
    fresh container. Non-fatal GPU faults drain the container gracefully.

    Args:
        kernels: List of kernel specifications, each containing:
            - triton_code: Triton kernel source code
            - pytorch_code: PyTorch reference implementation
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

        result = benchmark_kernelbench.local(
            triton_code=kernel_spec["triton_code"],
            pytorch_code=kernel_spec["pytorch_code"],
            n_correctness=kernel_spec.get("n_correctness", 5),
            n_trials=kernel_spec.get("n_trials", 20),
            kernel_name=kernel_spec.get("kernel_name"),
            entry_point=kernel_spec.get("entry_point", "Model"),
            rtol=kernel_spec.get("rtol", 1e-4),
            atol=kernel_spec.get("atol", 1e-4),
        )
        results.append(result)

    return results


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


if __name__ == "__main__":
    main()

"""
KernelBench Triton - Modal Application for Benchmarking Triton Kernels on H100 GPUs

This is a BENCHMARKING ONLY environment, not for reward/training.
It accepts Triton kernel code as input, compares against PyTorch reference,
and measures correctness + performance.

Functions:
    benchmark_triton_kernel  - Single kernel benchmark (generic input_shapes dict)
    benchmark_kernelbench    - Single kernel benchmark (KernelBench nn.Module pattern)
    benchmark_batch          - Sequential batch on same container with CUDA recovery

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


def _check_cuda_health():
    """Check if CUDA context is still healthy by running a small operation."""
    import torch
    try:
        t = torch.zeros(1, device='cuda')
        t = t + 1
        del t
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def _attempt_cuda_recovery():
    """Attempt to recover CUDA state after an error. Returns True if successful."""
    import torch
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    return _check_cuda_health()


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
        # Attempt CUDA cleanup so batch callers have a chance to recover
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass

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
    timeout=3600,
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
    import torch
    import time
    import traceback
    import tempfile
    import importlib.util
    import os
    import sys
    import inspect

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
    }

    try:
        # Write pytorch code to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='_pytorch.py', delete=False) as f:
            f.write(pytorch_code)
            pytorch_file = f.name

        # Write triton code to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='_triton.py', delete=False) as f:
            f.write(triton_code)
            triton_file = f.name

        try:
            # Import pytorch module
            spec = importlib.util.spec_from_file_location("pytorch_module", pytorch_file)
            pytorch_module = importlib.util.module_from_spec(spec)
            sys.modules["pytorch_module"] = pytorch_module
            spec.loader.exec_module(pytorch_module)

            # Get required functions and classes
            # First try the specified entry_point, then fall back to finding any nn.Module
            if hasattr(pytorch_module, entry_point):
                ModelClass = getattr(pytorch_module, entry_point)
            elif hasattr(pytorch_module, "Model"):
                ModelClass = pytorch_module.Model
            else:
                # Find any class that inherits from nn.Module
                import torch.nn as nn
                ModelClass = None
                for name, obj in vars(pytorch_module).items():
                    if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
                        ModelClass = obj
                        print(f"Found nn.Module class: {name}")
                        break
                if ModelClass is None:
                    raise ValueError(f"PyTorch code must define '{entry_point}' class or an nn.Module subclass")

            # Use utilities to extract get_inputs function
            from utilities import extract_inputs
            
            # Test that get_inputs works by extracting once
            test_inputs = extract_inputs(pytorch_code)
            print(f"Extracted {len(test_inputs)} inputs with shapes: {[tuple(t.shape) for t in test_inputs if hasattr(t, 'shape')]}")
            
            # Keep reference to get_inputs for repeated calls
            get_inputs = pytorch_module.get_inputs
            get_init_inputs = getattr(pytorch_module, "get_init_inputs", lambda: [])

            # Import triton module
            spec = importlib.util.spec_from_file_location("triton_module", triton_file)
            triton_module = importlib.util.module_from_spec(spec)
            sys.modules["triton_module"] = triton_module
            spec.loader.exec_module(triton_module)

            if not hasattr(triton_module, "triton_kernel_wrapper"):
                raise ValueError("Triton code must define 'triton_kernel_wrapper' function")
            triton_kernel_wrapper = triton_module.triton_kernel_wrapper

            # Analyze triton_kernel_wrapper signature to understand what inputs it expects
            sig = inspect.signature(triton_kernel_wrapper)
            wrapper_params = list(sig.parameters.keys())
            print(f"triton_kernel_wrapper expects: {wrapper_params}")

            # Create model instance
            init_inputs = get_init_inputs()
            if isinstance(init_inputs, (list, tuple)) and len(init_inputs) == 2:
                # Check if it's [args, kwargs] format
                if isinstance(init_inputs[0], (list, tuple)) and isinstance(init_inputs[1], dict):
                    model = ModelClass(*init_inputs[0], **init_inputs[1]).cuda()
                else:
                    model = ModelClass(*init_inputs).cuda()
            elif isinstance(init_inputs, (list, tuple)):
                model = ModelClass(*init_inputs).cuda()
            else:
                model = ModelClass().cuda()

            model.eval()

            # Check correctness
            print(f"Running {n_correctness} correctness checks...")
            correctness = True

            for i in range(n_correctness):
                try:
                    # Generate inputs using get_inputs()
                    inputs = get_inputs()
                    if not isinstance(inputs, (list, tuple)):
                        inputs = [inputs]

                    # Move inputs to CUDA with error handling
                    try:
                        cuda_inputs = [x.cuda() if hasattr(x, 'cuda') else x for x in inputs]
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        if "illegal memory access" in str(e).lower() or "cuda error" in str(e).lower():
                            print(f"CUDA error detected during input transfer: {e}")
                            print("Attempting to reset GPU state...")
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            # Try to reset the device
                            try:
                                torch.cuda.reset_peak_memory_stats()
                            except:
                                pass
                            raise  # Re-raise to mark as failed
                        raise

                    # Run reference (PyTorch model)
                    with torch.no_grad():
                        ref_output = model(*cuda_inputs)

                except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as e:
                    error_str = str(e).lower()
                    if "illegal memory access" in error_str or "cuda error" in error_str:
                        print(f"CUDA error in correctness check {i+1}: {e}")
                        print("GPU state corrupted - marking kernel as incorrect")
                        result["error"] = f"CUDA illegal memory access during correctness check: {str(e)}"
                        correctness = False
                        # Reset GPU state for next kernel
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        except:
                            pass
                        break
                    else:
                        # Some other error - re-raise
                        raise

                # Prepare inputs for triton kernel
                # The triton kernel may expect different arguments depending on implementation
                # Common pattern: kernel expects (x, weight, bias) or just (x)
                try:
                    try:
                        # Try calling with just the input tensors first
                        if len(wrapper_params) == 1:
                            kernel_output = triton_kernel_wrapper(cuda_inputs[0])
                        elif len(wrapper_params) == len(cuda_inputs):
                            kernel_output = triton_kernel_wrapper(*cuda_inputs)
                        else:
                            # Kernel might expect model parameters too
                            # Try to extract them from the model
                            kernel_args = list(cuda_inputs)

                            # Look for common parameter patterns
                            for name, param in model.named_parameters():
                                kernel_args.append(param.data)

                            # Try calling with all args
                            kernel_output = triton_kernel_wrapper(*kernel_args[:len(wrapper_params)])

                    except Exception as e:
                        error_str = str(e).lower()
                        # Don't retry on CUDA errors — GPU state is corrupted
                        if "illegal memory access" in error_str or "cuda error" in error_str:
                            raise
                        # If that fails, try with explicit parameter extraction
                        print(f"First attempt failed: {e}")
                        # For nn.Linear based models, extract weight and bias
                        kernel_args = list(cuda_inputs)
                        for module in model.modules():
                            if hasattr(module, 'weight') and module.weight is not None:
                                kernel_args.append(module.weight.data)
                            if hasattr(module, 'bias') and module.bias is not None:
                                kernel_args.append(module.bias.data)

                        kernel_output = triton_kernel_wrapper(*kernel_args[:len(wrapper_params)])

                except Exception as e:
                    error_str = str(e).lower()
                    if "illegal memory access" in error_str or "cuda error" in error_str:
                        print(f"CUDA error from triton kernel in check {i+1}: {e}")
                        print("GPU state corrupted - marking kernel as incorrect")
                        result["error"] = f"CUDA error from triton kernel: {str(e)}"
                        correctness = False
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                        break
                    raise

                # Compare outputs
                if not torch.allclose(ref_output, kernel_output, rtol=rtol, atol=atol):
                    correctness = False
                    max_diff = (ref_output - kernel_output).abs().max().item()
                    print(f"Correctness check {i+1} failed. Max diff: {max_diff}")
                    print(f"  ref shape: {ref_output.shape}, kernel shape: {kernel_output.shape}")
                    break

            result["correctness"] = correctness
            result["fast_0"] = correctness

            # Measure performance only if correct
            if correctness:
                print(f"Running performance benchmark ({n_trials} trials)...")

                # Generate fixed inputs for timing
                inputs = get_inputs()
                if not isinstance(inputs, (list, tuple)):
                    inputs = [inputs]
                cuda_inputs = [x.cuda() if hasattr(x, 'cuda') else x for x in inputs]

                # Prepare kernel args once
                if len(wrapper_params) == 1:
                    kernel_call = lambda: triton_kernel_wrapper(cuda_inputs[0])
                elif len(wrapper_params) == len(cuda_inputs):
                    kernel_call = lambda: triton_kernel_wrapper(*cuda_inputs)
                else:
                    kernel_args = list(cuda_inputs)
                    for module in model.modules():
                        if hasattr(module, 'weight') and module.weight is not None:
                            kernel_args.append(module.weight.data)
                        if hasattr(module, 'bias') and module.bias is not None:
                            kernel_args.append(module.bias.data)
                    kernel_call = lambda: triton_kernel_wrapper(*kernel_args[:len(wrapper_params)])

                # Warmup
                for _ in range(10):
                    with torch.no_grad():
                        _ = model(*cuda_inputs)
                    _ = kernel_call()
                torch.cuda.synchronize()

                # Time reference
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(n_trials):
                    with torch.no_grad():
                        _ = model(*cuda_inputs)
                torch.cuda.synchronize()
                reference_time = (time.perf_counter() - start) / n_trials * 1000

                # Time triton kernel
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(n_trials):
                    _ = kernel_call()
                torch.cuda.synchronize()
                kernel_time = (time.perf_counter() - start) / n_trials * 1000

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
                os.unlink(pytorch_file)
                os.unlink(triton_file)
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
    cuda_corrupted = False

    for i, kernel_spec in enumerate(kernels):
        print(f"\n{'='*50}")
        print(f"Benchmarking kernel {i+1}/{len(kernels)}: {kernel_spec.get('kernel_name', 'unnamed')}")
        print(f"{'='*50}")

        # If CUDA was corrupted by a previous kernel, skip remaining
        if cuda_corrupted:
            print("Skipping: CUDA context corrupted by a previous kernel")
            results.append({
                "kernel_name": kernel_spec.get("kernel_name", "unnamed_kernel"),
                "timestamp": datetime.now().isoformat(),
                "correctness": False,
                "speedup": 0.0,
                "reference_time_ms": None,
                "kernel_time_ms": None,
                "fast_0": False,
                "fast_1": False,
                "fast_2": False,
                "error": "Skipped: CUDA context corrupted by a previous kernel",
            })
            continue

        result = benchmark_triton_kernel.local(
            kernel_code=kernel_spec["kernel_code"],
            reference_torch_code=kernel_spec["reference_torch_code"],
            input_shapes=kernel_spec["input_shapes"],
            n_correctness=kernel_spec.get("n_correctness", 10),
            n_trials=kernel_spec.get("n_trials", 100),
            kernel_name=kernel_spec.get("kernel_name"),
        )
        results.append(result)

        # Check if this kernel corrupted CUDA state
        if result.get("error") and ("cuda" in result["error"].lower() or "illegal memory" in result["error"].lower()):
            print("CUDA error detected, attempting recovery...")
            if not _attempt_cuda_recovery():
                print("CUDA recovery failed - remaining kernels will be skipped")
                cuda_corrupted = True

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
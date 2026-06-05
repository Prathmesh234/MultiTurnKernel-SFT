"""
Simple Triton Kernel Benchmark Example

This demonstrates the RL pipeline flow:
1. Take Python/PyTorch code from KernelBook
2. Generate optimized Triton kernel (what LLM will do)
3. Benchmark on Modal H100

Run with: modal run benchmark_example.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modal_app import app, benchmark_kernelbench


# ============================================================================
# EXAMPLE 1: Simple Log-Cosh Loss
# Original PyTorch code from KernelBook
# ============================================================================

PYTORCH_CODE_LOG_LOSS = '''
import torch
import torch.nn as nn

class Log_Loss(nn.Module):
    def __init__(self):
        super(Log_Loss, self).__init__()

    def forward(self, ytrue, ypred):
        delta = ypred - ytrue
        return torch.mean(torch.log(torch.cosh(delta)))

def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]

def get_init_inputs():
    return [[], {}]
'''

# Hand-written optimized Triton kernel for Log_Loss
# This is what we want the LLM to generate!
TRITON_CODE_LOG_LOSS = '''
import torch
import triton
import triton.language as tl

@triton.jit
def log_cosh_loss_kernel(
    output_ptr,
    ytrue_ptr,
    ypred_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused log-cosh loss kernel: mean(log(cosh(ypred - ytrue)))"""
    pid = tl.program_id(0)
    
    # Compute block start
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    ytrue = tl.load(ytrue_ptr + offsets, mask=mask, other=0.0)
    ypred = tl.load(ypred_ptr + offsets, mask=mask, other=0.0)
    
    # Compute: log(cosh(delta))
    delta = ypred - ytrue
    
    # cosh(x) = (exp(x) + exp(-x)) / 2
    exp_delta = tl.exp(delta)
    exp_neg_delta = tl.exp(-delta)
    cosh_val = (exp_delta + exp_neg_delta) / 2.0
    
    # log(cosh(x))
    log_cosh = tl.log(cosh_val)
    
    # Sum for mean calculation (will divide by n_elements later)
    sum_val = tl.sum(log_cosh, axis=0)
    
    # Atomic add to output for reduction
    tl.atomic_add(output_ptr, sum_val / n_elements)


def triton_kernel_wrapper(ytrue, ypred):
    """Wrapper that matches PyTorch forward() signature."""
    assert ytrue.shape == ypred.shape
    assert ytrue.is_cuda and ypred.is_cuda
    
    n_elements = ytrue.numel()
    
    # Allocate output
    output = torch.zeros(1, device=ytrue.device, dtype=ytrue.dtype)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    log_cosh_loss_kernel[grid](
        output,
        ytrue.contiguous(),
        ypred.contiguous(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.squeeze()
'''


# ============================================================================
# EXAMPLE 2: Simple ReLU
# ============================================================================

PYTORCH_CODE_RELU = '''
import torch
import torch.nn as nn

class SimpleReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(1024, 1024, device='cuda')]

def get_init_inputs():
    return [[], {}]
'''

TRITON_CODE_RELU = '''
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_kernel_wrapper(x):
    """Wrapper matching forward() signature."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
'''


@app.local_entrypoint()
def run_examples():
    """Run benchmark examples."""
    print("=" * 70)
    print("TRITON KERNEL BENCHMARK EXAMPLES")
    print("=" * 70)
    
    examples = [
        ("Log_Loss", PYTORCH_CODE_LOG_LOSS, TRITON_CODE_LOG_LOSS),
        ("SimpleReLU", PYTORCH_CODE_RELU, TRITON_CODE_RELU),
    ]
    
    for name, pytorch_code, triton_code in examples:
        print(f"\n{'='*70}")
        print(f"Benchmarking: {name}")
        print("=" * 70)
        
        try:
            result = benchmark_kernelbench.remote(
                triton_code=triton_code,
                pytorch_code=pytorch_code,
                kernel_name=name,
                entry_point=name.replace("_", ""),
                n_correctness=3,
                n_trials=20,
            )
            
            if result.get("error"):
                print(f"❌ Error: {result['error'][:200]}...")
            else:
                print(f"✅ Correctness: {result['correctness']}")
                if result['correctness']:
                    print(f"   Reference time: {result['reference_time_ms']:.4f} ms")
                    print(f"   Kernel time: {result['kernel_time_ms']:.4f} ms")
                    print(f"   Speedup: {result['speedup']:.2f}x")
                    
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

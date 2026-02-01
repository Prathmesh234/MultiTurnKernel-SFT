#!/bin/bash
# =============================================================================
# KernelBench Triton - Modal Deployment Script
# =============================================================================
# Deploys and runs the KernelBench benchmarking system on Modal H100 GPUs
# Supports parallel execution across multiple containers for high throughput
#
# Features:
#   - Parallel benchmark execution across up to 8 H100 containers
#   - Persistent result storage in Modal volumes
#   - Both single-kernel and batch benchmarking modes
#
# Usage:
#   ./run_gpt_oss.sh              # Run example benchmark
#   ./run_gpt_oss.sh deploy       # Deploy as persistent endpoint
#   ./run_gpt_oss.sh test         # Run quick test with single kernel
#   ./run_gpt_oss.sh parallel N   # Run parallel benchmark with N containers
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODAL_APP="modal_app.py"
DEFAULT_PARALLEL_CONTAINERS=4

echo "=============================================="
echo "KernelBench Triton - Modal H100 Deployment"
echo "=============================================="
echo "Script directory: $SCRIPT_DIR"
echo "Modal app: $MODAL_APP"
echo "=============================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# Sync project dependencies from pyproject.toml
echo "Syncing project dependencies..."
uv sync

# Check if modal is configured
if ! uv run modal token show &> /dev/null; then
    echo ""
    echo "ERROR: Modal not configured!"
    echo "Please run: uv run modal token set --token-id <ID> --token-secret <SECRET>"
    echo "Get your token from: https://modal.com/settings"
    exit 1
fi

# Parse command line arguments
ACTION="${1:-run}"
N_CONTAINERS="${2:-$DEFAULT_PARALLEL_CONTAINERS}"

case "$ACTION" in
    "run")
        echo ""
        echo "Running KernelBench benchmark example..."
        echo "This will spin up H100 containers on Modal."
        echo ""
        uv run modal run "$SCRIPT_DIR/$MODAL_APP"
        ;;
    
    "deploy")
        echo ""
        echo "Deploying KernelBench to Modal as persistent endpoint..."
        echo "This will keep the functions available for remote calls."
        echo ""
        uv run modal deploy "$SCRIPT_DIR/$MODAL_APP"
        echo ""
        echo "Deployment complete!"
        echo "You can now call functions remotely:"
        echo "  - benchmark_triton_kernel.remote(...)"
        echo "  - benchmark_single.remote(...)"
        echo "  - benchmark_parallel_remote.remote(kernels, n_containers=$N_CONTAINERS)"
        ;;
    
    "test")
        echo ""
        echo "Running quick test with single kernel..."
        echo ""
        uv run python -c "
import modal
from modal_app import app, benchmark_triton_kernel

# Quick test kernel
kernel_code = '''
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

def triton_kernel_wrapper(x, y):
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta[\"BLOCK_SIZE\"]),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output
'''

reference_code = '''
import torch
def reference_impl(x, y):
    return x + y
'''

input_shapes = {
    'x': {'shape': [1024, 1024], 'dtype': 'float32'},
    'y': {'shape': [1024, 1024], 'dtype': 'float32'},
}

with app.run():
    result = benchmark_triton_kernel.remote(
        kernel_code=kernel_code,
        reference_torch_code=reference_code,
        input_shapes=input_shapes,
        kernel_name='vector_add_test',
        n_correctness=3,
        n_trials=10,
    )
    print(f'Correctness: {result[\"correctness\"]}')
    print(f'Speedup: {result[\"speedup\"]:.2f}x')
"
        ;;
    
    "parallel")
        echo ""
        echo "Running parallel benchmark with $N_CONTAINERS H100 containers..."
        echo ""
        uv run python -c "
import modal
from modal_app import app, benchmark_parallel

# Example kernels for parallel testing
kernels = []
for i in range(${N_CONTAINERS}):
    kernels.append({
        'kernel_code': '''
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

def triton_kernel_wrapper(x, y):
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta[\"BLOCK_SIZE\"]),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output
''',
        'reference_torch_code': '''
import torch
def reference_impl(x, y):
    return x + y
''',
        'input_shapes': {
            'x': {'shape': [2048, 2048], 'dtype': 'float32'},
            'y': {'shape': [2048, 2048], 'dtype': 'float32'},
        },
        'kernel_name': f'parallel_test_{i}',
        'n_correctness': 3,
        'n_trials': 10,
    })

with app.run():
    results = benchmark_parallel(kernels, n_containers=${N_CONTAINERS})
    
    print('\\n' + '='*60)
    print('RESULTS SUMMARY')
    print('='*60)
    for r in results:
        status = 'PASS' if r['correctness'] else 'FAIL'
        speedup = r.get('speedup', 0)
        print(f\"  {r['kernel_name']}: {status} (speedup: {speedup:.2f}x)\")
"
        ;;
    
    "stats")
        echo ""
        echo "Getting summary statistics from all benchmarks..."
        echo ""
        uv run python -c "
import modal
from modal_app import app, get_summary_statistics

with app.run():
    stats = get_summary_statistics.remote()
    print('Summary Statistics:')
    print(f'  Total benchmarks: {stats[\"total_benchmarks\"]}')
    print(f'  Correctness rate: {stats[\"correctness_rate\"]:.1%}')
    print(f'  fast_1 rate: {stats[\"fast_1_rate\"]:.1%}')
    print(f'  fast_2 rate: {stats[\"fast_2_rate\"]:.1%}')
    print(f'  Average speedup: {stats[\"average_speedup\"]:.2f}x')
    print(f'  Speedup range: {stats[\"speedup_distribution\"][\"min\"]:.2f}x - {stats[\"speedup_distribution\"][\"max\"]:.2f}x')
"
        ;;
    
    "clear")
        echo ""
        echo "Clearing all benchmark results..."
        echo ""
        uv run python -c "
import modal
from modal_app import app, clear_results

with app.run():
    result = clear_results.remote()
    print(f'Deleted {result[\"deleted_files\"]} result files.')
"
        ;;
    
    *)
        echo "Unknown action: $ACTION"
        echo ""
        echo "Usage:"
        echo "  ./run_gpt_oss.sh              # Run example benchmark"
        echo "  ./run_gpt_oss.sh deploy       # Deploy as persistent endpoint"
        echo "  ./run_gpt_oss.sh test         # Run quick test with single kernel"
        echo "  ./run_gpt_oss.sh parallel N   # Run parallel benchmark with N containers"
        echo "  ./run_gpt_oss.sh stats        # Get summary statistics"
        echo "  ./run_gpt_oss.sh clear        # Clear all stored results"
        exit 1
        ;;
esac

echo ""
echo "Done!"

# KernelBench Triton - Agent Documentation

## Overview

This is a **benchmarking environment** for Triton kernels running on Modal's H100 GPUs.

### Current Objective

**Goal**: Use `gpt-oss-120b` to generate reasoning traces on KernelBench problems, then use these verified traces to finetune our Arcee model for Triton kernel generation.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      PHASE 1: TRACE GENERATION                         │
│                                                                         │
│  Input:  KernelBench problems (PyTorch code)                           │
│  Model:  gpt-oss-120b (strong reasoning model)                         │
│  Output: (PyTorch → <think>reasoning</think> → Triton) traces          │
│                                                                         │
│  For each KernelBench problem:                                         │
│    1. gpt-oss-120b generates Triton kernel with reasoning              │
│    2. Modal H100 verifies correctness + measures speedup               │
│    3. Keep only traces where correctness=True AND speedup > 1.0        │
│    4. Save verified traces as training data                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PHASE 2: ARCEE FINETUNING                          │
│                                                                         │
│  Dataset: Verified traces from Phase 1                                 │
│  Model:   Arcee model (16B MoE, 3B active)                            │
│  Method:  SFT on (PyTorch → Thinking → Triton) examples               │
│                                                                         │
│  Result: Arcee model learns to generate optimized Triton kernels       │
│          with reasoning, trained on verified-correct examples          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PHASE 3: RL REFINEMENT (Future)                   │
│                                                                         │
│  Further optimize the finetuned Arcee model using RL                  │
│  Reward: correctness + speedup on held-out KernelBench L3+L4          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This Approach?

| Challenge | Solution |
|-----------|----------|
| No reasoning traces in existing data | Generate with gpt-oss-120b |
| Need verified-correct examples | Modal H100 validates every trace |
| Avoid memorization | Only use traces that pass verification |
| Speedup matters | Filter for speedup > 1.0 |

## Modal Platform

### What is Modal?
Modal is a serverless cloud platform that allows you to run Python code on remote GPUs with minimal setup. Key features:

- **On-demand GPU access**: Spin up H100s instantly without managing infrastructure
- **Container-based**: Each function runs in an isolated container with specified dependencies
- **Pay-per-use**: Only pay for actual compute time (billed per second)
- **Persistent volumes**: Store results across runs

### Our Modal Configuration

```python
# Container image setup
image = modal.Image.debian_slim(python_version="3.10")
    .env({
        "GPU_CONFIG": "H100",
        "TORCH_CUDA_ARCH_LIST": "9.0",  # Hopper architecture
    })
    .pip_install("torch>=2.1.0", "triton>=2.1.0", ...)
    .run_commands("git clone .../KernelBench.git && pip install -e .")

# GPU allocation
@app.function(gpu="H100", timeout=3600)
def benchmark_triton_kernel(...):
    ...
```

### Key Modal Commands

| Command | Description |
|---------|-------------|
| `modal run modal_app.py` | Run the example benchmark |
| `modal deploy modal_app.py` | Deploy as persistent endpoint |
| `modal token set --token-id X --token-secret Y` | Set API credentials |

---

## Implementation Details

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       YOUR LOCAL MACHINE                        │
│  - Sends kernel code (string) + reference code + input shapes  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODAL CLOUD (H100 GPU)                       │
│  1. Writes kernel code to temp file (Triton JIT requirement)   │
│  2. Imports and executes kernel                                 │
│  3. Compares output against PyTorch reference                   │
│  4. Measures execution time with CUDA sync                      │
│  5. Saves results to persistent volume                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               PERSISTENT VOLUME: triton-benchmark-results       │
│  - JSON files with timestamp-based filenames                    │
│  - Survives across container restarts                           │
└─────────────────────────────────────────────────────────────────┘
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `benchmark_triton_kernel()` | Main benchmark function - runs single kernel |
| `benchmark_batch()` | Sequential batch execution on same container |
| `get_all_results()` | Retrieve all stored benchmark results |
| `get_summary_statistics()` | Aggregate stats across all benchmarks |
| `clear_results()` | Delete all stored results |

### Why Temp Files for Kernel Code?

Triton's `@jit` decorator requires source code to be in a real Python file (it calls `inspect.getsourcelines()`). Using `exec()` on strings fails because there's no file to inspect. Our solution:

1. Write kernel code to a temp file
2. Import it as a module using `importlib`
3. Execute the kernel
4. Clean up temp files

---

## Benchmark Metrics

### Result Structure

```json
{
  "kernel_name": "vector_add",
  "timestamp": "2026-01-30T07:59:21.695407",
  "correctness": true,
  "speedup": 2.45,
  "reference_time_ms": 0.0534,
  "kernel_time_ms": 0.0218,
  "fast_0": true,
  "fast_1": true,
  "fast_2": true,
  "error": null,
  "input_shapes": {...},
  "n_correctness": 10,
  "n_trials": 100
}
```

### Metric Definitions

| Metric | Definition | RL Reward Suggestion |
|--------|------------|---------------------|
| **correctness** | Kernel output matches reference within tolerance (rtol=1e-5, atol=1e-5) | Base requirement |
| **speedup** | `reference_time_ms / kernel_time_ms` - how much faster than PyTorch | Continuous reward signal |
| **fast_0** | Same as `correctness` - kernel produces correct output | +1 |
| **fast_1** | Correct AND `speedup > 1.0` (faster than reference) | +2 |
| **fast_2** | Correct AND `speedup >= 2.0` (at least 2x faster) | +3 |
| **error** | Error message if kernel crashed, syntax error, or timeout | -1 (penalize failures) |

### Benchmark Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_correctness` | 10 | Number of random inputs to test correctness |
| `n_trials` | 100 | Number of timing trials (averaged) |
| `n_warmup` | 10 | Warmup runs before timing (JIT compilation) |
| `rtol` | 1e-5 | Relative tolerance for `torch.allclose()` |
| `atol` | 1e-5 | Absolute tolerance for `torch.allclose()` |

---

## Kernel Code Requirements

### Triton Kernel Format

Your kernel code string **must** define:

```python
# 1. The Triton kernel with @triton.jit
@triton.jit
def my_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...

# 2. A wrapper function named exactly "triton_kernel_wrapper"
def triton_kernel_wrapper(x):
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    my_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output
```

### Reference Code Format

```python
# Must define a function named exactly "reference_impl"
def reference_impl(x):
    return x * 2  # PyTorch implementation
```

### Input Shapes Format

```python
input_shapes = {
    "tensor_name": {
        "shape": [dim1, dim2, ...],
        "dtype": "float32",  # float32, float16, bfloat16, int32, int64
    }
}
```

---

## RL Pipeline Integration

### Task: PyTorch → Triton Translation

The RL policy learns to convert PyTorch code into optimized Triton kernels:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              INPUT                                      │
│  PyTorch Code:                                                          │
│    def pytorch_kernel(x):                                               │
│        return torch.sum(x, dim=1)                                       │
│  + Input Shapes (from environment, NOT chosen by policy)                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          POLICY (LLM)                                   │
│  Generates: Complete Triton kernel + wrapper + configs                  │
│    - Algorithm implementation                                           │
│    - BLOCK_SIZE, num_warps, num_stages                                  │
│    - Memory access patterns                                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT (Modal H100)                             │
│  - Executes Triton kernel                                               │
│  - Compares against PyTorch reference                                   │
│  - Returns: correctness, speedup, fast_0/1/2                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Training Loop

```python
for sample in kernelbook_dataset:
    # PyTorch code from dataset
    pytorch_code = sample["python_code"]
    
    # Policy generates Triton translation
    triton_code = policy.generate(pytorch_code)
    
    # Benchmark on Modal (shapes provided by curriculum, not policy)
    result = benchmark_triton_kernel.remote(
        kernel_code=triton_code,
        reference_torch_code=pytorch_code,
        input_shapes=current_curriculum_shapes,
    )
    
    # Reward = correctness + speedup
    reward = compute_reward(result)
    policy.update(pytorch_code, triton_code, reward)
```

### Input Size Curriculum (Production-Aware)

**Target Production Setup:** 16B MoE model with 3B active parameters

| Memory Budget | Amount |
|---------------|--------|
| H100 Total | 80 GB |
| Model (3B active, bfloat16) | ~6 GB |
| MoE routing buffers | ~2 GB |
| KV cache | ~4 GB |
| Activations | ~4 GB |
| CUDA overhead | ~5 GB |
| Safety headroom | ~5 GB |
| **Available for tensors** | **~54 GB** |

**Capped Curriculum (won't cause OOM with model loaded):**

| Phase | Input Shape | Elements | Memory | Notes |
|-------|-------------|----------|--------|-------|
| 1 | `[32, 64, 128, 128]` | 33M | 134 MB | Fast iteration, warmup |
| 2 | `[64, 128, 128, 128]` | 134M | 536 MB | Small |
| 3 | `[64, 128, 256, 256]` | 536M | 2.1 GB | Medium |
| 4 | `[64, 256, 256, 256]` | 1.1B | 4.3 GB | Large |
| 5 | `[64, 256, 512, 512]` | 4.3B | **17 GB** | **MAX** (final phase) |

```python
# Production-safe curriculum (with 16B MoE model overhead)
SIZE_CURRICULUM = [
    {"shape": [32, 64, 128, 128], "episodes": 500},    # 134 MB - warmup
    {"shape": [64, 128, 128, 128], "episodes": 1000},  # 536 MB - small
    {"shape": [64, 128, 256, 256], "episodes": 2000},  # 2.1 GB - medium
    {"shape": [64, 256, 256, 256], "episodes": 3000},  # 4.3 GB - large
    {"shape": [64, 256, 512, 512], "episodes": 5000},  # 17 GB - MAX (production safe)
]
# DO NOT GO HIGHER - leaves ~35GB headroom for model + intermediates
```

### Why Environment Controls Shapes (Not Policy)

- **Prevents gaming**: Policy can't pick tiny sizes to artificially inflate speedup
- **Fair comparison**: All kernels tested on same sizes
- **Realistic benchmark**: Production-sized tensors
- **Curriculum learning**: Gradually increase difficulty

### Batching Options

**Sequential (current)**: One container, kernels run one after another
```python
results = benchmark_batch.remote(kernels=[...])
```

**Parallel (TODO)**: Multiple containers, kernels run simultaneously
```python
results = benchmark_parallel.remote(kernels=[...], n_gpus=4)
```

---

## Next TODO

### 1. Parallel Execution with 4 H100 Containers

Implement `benchmark_parallel()` function that:
- Accepts a list of kernels
- Spins up N H100 containers in parallel (default: 4)
- Distributes kernels across containers
- Returns all results when complete

```python
@app.function(gpu="H100", image=image, concurrency_limit=4)
def benchmark_single(kernel_spec: dict) -> dict:
    return benchmark_triton_kernel.local(**kernel_spec)

def benchmark_parallel(kernels: list[dict], n_gpus: int = 4) -> list[dict]:
    # Use Modal's .map() for parallel execution
    return list(benchmark_single.map(kernels))
```

Benefits:
- 4x throughput for large batches
- Total time = max(kernel_times) instead of sum(kernel_times)
- Essential for fast RL iteration

### 2. Future Enhancements
- [ ] Keep containers warm to reduce cold start latency
- [ ] Add kernel caching to skip re-benchmarking identical code
- [ ] Implement timeout tuning for faster failure detection
- [ ] Add more granular error categorization (syntax vs runtime vs correctness)

---

## Dataset Strategy: KernelBench vs KernelBook

### Why We Chose KernelBench for RL Evaluation

| Dataset | Purpose | Contains Solutions? | Use Case |
|---------|---------|---------------------|----------|
| **GPUMODE/KernelBook** | SFT Training | ✅ Yes (PyTorch → Triton pairs) | Pre-training / warmup |
| **ScalingIntelligence/KernelBench** | RL Environment | ❌ No (just PyTorch reference) | Evaluation / RL training |

**Reasoning:**
1. **No train-test contamination**: KernelBench has NO Triton solutions, so the model can't memorize answers
2. **Verifiable rewards**: We can check correctness against PyTorch reference
3. **Measurable speedup**: Compare Triton kernel time vs PyTorch baseline
4. **Standardized benchmark**: ICML '25 paper with established metrics
5. **Multiple difficulty levels**: Level 1-4 provides natural curriculum

### KernelBench Dataset Structure

```
ScalingIntelligence/KernelBench
├── level_1 (100 samples) - Single-kernel ops (convs, matmuls, norms)
├── level_2 (100 samples) - Fusion patterns (Conv+ReLU, Matmul+Scale)
├── level_3 (50 samples)  - Full architectures (ResNet, MobileNet, MiniGPT)
└── level_4 (20 samples)  - HuggingFace models
```

**Sample format:**
```python
{
    "code": """
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, A, B):
        return torch.triu(torch.matmul(A, B))

def get_inputs():
    A = torch.triu(torch.rand(4096, 4096))
    B = torch.triu(torch.rand(4096, 4096))
    return [A, B]
""",
    "level": 1,
    "name": "6_Triu_Matmul",
    "problem_id": 6
}
```

### Integration Pseudocode

```python
from datasets import load_dataset

# Load KernelBench as RL environment
kernelbench = load_dataset('ScalingIntelligence/KernelBench', split='level_1')

for episode in range(num_episodes):
    # Sample a problem
    sample = kernelbench[episode % len(kernelbench)]
    
    # Extract PyTorch reference (the problem)
    pytorch_code = sample['code']
    
    # Policy generates Triton kernel translation
    triton_code = policy.generate(pytorch_code)
    
    # Adapter: Convert KernelBench format to our benchmark format
    reference_code = f'''
{pytorch_code}

# Adapter for benchmark
_model = Model()
def reference_impl(*inputs):
    return _model(*inputs)
'''
    
    # Extract input shapes from get_inputs()
    input_shapes = parse_get_inputs(pytorch_code)
    
    # Benchmark on Modal H100
    result = benchmark_triton_kernel.remote(
        kernel_code=triton_code,
        reference_torch_code=reference_code,
        input_shapes=input_shapes,
    )
    
    # Compute reward
    if result['error']:
        reward = -1.0  # Crash penalty
    elif not result['correctness']:
        reward = 0.0   # Incorrect output
    else:
        # Correct! Reward based on speedup
        speedup = result['speedup']
        reward = 1.0 + max(0, log(speedup))  # Bonus for faster
    
    # Update policy
    policy.update(pytorch_code, triton_code, reward)
```

### Train/Test Split Strategy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          SFT WARMUP                                     │
│  Dataset: GPUMODE/KernelBook (has solutions)                           │
│  Purpose: Teach model basic Triton syntax                              │
│  Samples: ~500-1000                                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          RL TRAINING                                    │
│  Dataset: ScalingIntelligence/KernelBench Level 1 + 2                  │
│  Purpose: Learn to generate correct, fast kernels                      │
│  Samples: 200 problems × many episodes                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION                                     │
│  Dataset: ScalingIntelligence/KernelBench Level 3 + 4                  │
│  Purpose: Test generalization to harder problems                       │
│  Samples: 70 held-out problems                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

This ensures:
- ✅ No solution leakage during RL
- ✅ Clean train/test split (L1+L2 vs L3+L4)
- ✅ Increasing difficulty for curriculum

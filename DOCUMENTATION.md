# KernelBench Triton - Architecture Documentation

## CUDA State Safety & Process Isolation

### Overview

This document explains the critical architecture decisions around process isolation and CUDA state safety in KernelBench Triton. The key insight is: **each kernel benchmark runs in a completely fresh subprocess to guarantee clean GPU state and avoid stale module pollution.**

---

## Table of Contents

1. [The Problem: State Pollution](#the-problem-state-pollution)
2. [The Solution: Process Isolation with `spawn`](#the-solution-process-isolation-with-spawn)
3. [Why Not Pool() or ThreadPoolExecutor?](#why-not-pool-or-threadpoolexecutor)
4. [How `mp.get_context("spawn")` Works](#how-mpget_contextspawn-works)
5. [CUDA Context Isolation](#cuda-context-isolation)
6. [Module Cache Cleanup](#module-cache-cleanup)
7. [Timeout & Crash Recovery](#timeout--crash-recovery)
8. [End-to-End Example](#end-to-end-example)

---

## The Problem: State Pollution

### CUDA Context Inheritance (fork)

If we naively used Python's default `fork()` approach:

```python
# Parent process
import torch
torch.cuda.init()  # Creates CUDA context X

# Parent forks a child
p = mp.Process(target=benchmark_kernel)
p.start()
```

**What happens in the child:**
- Child inherits the parent's **entire memory image**, including CUDA context X
- When child imports torch and calls `.cuda()`, the GPU driver sees:
  - "This process already has a GPU context"
  - "Trying to initialize again???"
- GPU state corruption, illegal memory access, process crash

**Result:** One bad kernel crashes the GPU for all future benchmarks in the parent.

### Module Cache Pollution (reused workers)

If we used `mp.Pool()` (worker process reuse):

```python
# Worker process 1, Kernel A
triton_module = importlib.util.module_from_spec(spec)
sys.modules["triton_module"] = triton_module  # Cached!
spec.loader.exec_module(triton_module)

# Worker process 1, Kernel B (same process!)
# Python sees "triton_module" already in sys.modules
# Reuses old module instead of reloading from new file
```

**Result:** Stale Triton JIT cache, wrong kernel runs, incorrect benchmarks.

---

## The Solution: Process Isolation with `spawn`

We use `spawn` to create a **completely fresh Python interpreter** for each kernel:

```python
import multiprocessing as mp

# In benchmark_kernelbench() or benchmark_triton_kernel()
ctx = mp.get_context("spawn")      # Use spawn backend
result_queue = ctx.Queue()         # IPC channel for results

p = ctx.Process(
    target=_kernelbench_worker,    # Worker function
    args=(triton_code, pytorch_code, n_correctness, n_trials, ...),
    daemon=True,
)

p.start()
p.join(timeout=300)  # 5-minute hard timeout
```

**Key benefits:**
- ✅ Fresh Python interpreter (no inherited memory)
- ✅ No pre-existing CUDA context (clean GPU state)
- ✅ Empty `sys.modules` (no stale module cache)
- ✅ Isolated timings (no carryover from previous kernels)
- ✅ Hard timeout support (kill hung processes)
- ✅ Parent survives child crashes (process boundary = safety boundary)

---

## Why Not Pool() or ThreadPoolExecutor?

### ThreadPoolExecutor ❌

**Problem:** Threads share the same CUDA context within a process.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    future = executor.submit(_kernelbench_worker, triton_code, pytorch_code)
```

**Why it fails:**
- All 4 threads share one CUDA context
- If Thread 1's kernel corrupts GPU, Threads 2-4 are affected
- Python's GIL limits true parallelism anyway
- **Not suitable for untrusted/generated code**

### mp.Pool() ⚠️

**Problem:** Worker processes are reused between tasks.

```python
with mp.Pool(processes=4) as pool:
    result = pool.apply_async(_kernelbench_worker, args=(triton_code,))
```

**Why it's suboptimal:**
1. **State carryover:** Same worker process runs Kernel A, then Kernel B
   - Triton's JIT cache persists (old kernels served)
   - CUDA context persists (GPU memory fragmentation, state pollution)
   - `sys.modules` still has old modules

2. **Delayed respawn:** If a worker crashes:
   - Pool respawns a replacement (adds latency)
   - GPU context might still be corrupted during transition

3. **Awkward timeout:** `result.get(timeout=300)` at task level, not process level

**Better than threads, but not ideal for isolation.**

### Manual mp.Process ✅

**Why we chose it:**
- **Complete isolation:** Fresh Python interpreter per kernel
- **Clean GPU state:** No inherited CUDA context
- **No module pollution:** Empty `sys.modules` each time
- **Hard timeout:** `p.join(timeout=300)` + `p.kill()` if hung
- **Parent survives crashes:** Process boundary protects container

---

## How `mp.get_context("spawn")` Works

### The Three Start Methods

Python's multiprocessing supports three ways to create child processes:

#### 1. fork (default on Unix, ❌ broken for CUDA)

```python
ctx = mp.get_context("fork")
```

**Mechanism:**
- Calls `os.fork()` → duplicate entire parent process
- Child inherits memory, file handles, **CUDA context**

**Visual:**
```
Parent (GPU context X, sys.modules={torch, triton, ...})
           ↓ fork()
Child (GPU context X copy, sys.modules={torch, triton, ...})
           ↓
GPU driver sees two processes with same context → corruption
```

**For CUDA:** ❌ Broken — inherited context causes GPU crashes

---

#### 2. spawn (what we use, ✅ correct for CUDA)

```python
ctx = mp.get_context("spawn")
```

**Mechanism:**
- Starts a fresh Python interpreter (`python -c "..."`)
- Child receives only serialized arguments (pickled)
- No inherited memory, no inherited CUDA context

**Visual:**
```
Parent (GPU context X, sys.modules={torch, triton, ...})
           ↓ spawn (fresh interpreter)
Child (empty memory, NO GPU context, sys.modules={})
           ↓
Child imports torch → creates fresh GPU context Y
           ↓
Isolated: Parent has X, Child has Y (different CUDA contexts)
```

**For CUDA:** ✅ Perfect — each process gets its own context

---

#### 3. forkserver (Unix only, rarely needed)

```python
ctx = mp.get_context("forkserver")
```

**Mechanism:**
- Maintains a clean "server" process
- Server forks itself (not the parent) to create workers
- Somewhere between fork and spawn in terms of cleanliness

**For CUDA:** ⚠️ Better than fork, but spawn is still preferred

---

### What Gets Passed to the Child?

With `spawn`, **only arguments get pickled and transmitted**:

```python
# Parent
p = ctx.Process(
    target=_kernelbench_worker,
    args=(triton_code, pytorch_code, n_correctness, ...)
)
p.start()

# Child receives
def _kernelbench_worker(triton_code, pytorch_code, n_correctness, ...):
    # triton_code = unpickled string
    # pytorch_code = unpickled string
    # n_correctness = unpickled integer
    # Everything else? Fresh and clean!
```

**Child inherits:** ✓ Arguments only

**Child does NOT inherit:** ✗ GPU context, modules, memory, file handles

---

## CUDA Context Isolation

### The Execution Flow

```python
# modal_app.py: benchmark_kernelbench()

ctx = mp.get_context("spawn")
p = ctx.Process(
    target=_kernelbench_worker,
    args=(triton_code, pytorch_code, ...)
)
p.start()     # ← Spawns fresh Python interpreter
p.join(timeout=300)

# Inside child process (_kernelbench_worker):
# 1. Fresh Python interpreter starts
# 2. torch is NOT imported yet
# 3. CUDA is NOT initialized yet

import torch  # Load torch module

# Create temp files (Triton needs real files on disk)
with tempfile.NamedTemporaryFile(...) as f:
    f.write(pytorch_code)
    pytorch_file = f.name

with tempfile.NamedTemporaryFile(...) as f:
    f.write(triton_code)
    triton_file = f.name

# Dynamically load PyTorch module
spec = importlib.util.spec_from_file_location("pytorch_module", pytorch_file)
pytorch_module = importlib.util.module_from_spec(spec)
sys.modules["pytorch_module"] = pytorch_module
spec.loader.exec_module(pytorch_module)

# PyTorch model instantiated
model = ModelClass().cuda()  # ← First CUDA call creates fresh context Y

# Dynamically load Triton module
spec = importlib.util.spec_from_file_location("triton_module", triton_file)
triton_module = importlib.util.module_from_spec(spec)
sys.modules["triton_module"] = triton_module
spec.loader.exec_module(triton_module)  # ← Triton JIT compiles in context Y

# Benchmark runs entirely within context Y

# Cleanup
finally:
    os.unlink(pytorch_file)
    os.unlink(triton_file)
    sys.modules.pop("pytorch_module", None)   # ← Module cache cleanup
    sys.modules.pop("triton_module", None)    # ← (added in recent fix)

# Child process exits → context Y destroyed
# Parent's context X remains untouched
```

### Key Isolation Points

| Point | Isolation |
|-------|-----------|
| **Process start** | Fresh Python interpreter |
| **CUDA init** | `.cuda()` call creates context only in child |
| **Triton JIT** | Compilation happens in child's context only |
| **Module cache** | `sys.modules` is empty, populated fresh |
| **Process exit** | Child's context Y destroyed; parent's context X unaffected |

---

## Module Cache Cleanup

### The Problem (Before Fix)

```python
# Child process, Kernel A
sys.modules["triton_module"] = triton_module  # Cached!
spec.loader.exec_module(triton_module)

# If worker function called again in same process:
# Child process, Kernel B (hypothetically, with Pool)
spec.loader.exec_module(triton_module)  # Reuses cached module!
```

### The Solution (Our Fix)

```python
finally:
    try:
        os.unlink(pytorch_file)
        os.unlink(triton_file)
    except Exception:
        pass

    # NEW: Evict cached modules
    sys.modules.pop("pytorch_module", None)
    sys.modules.pop("triton_module", None)
```

**Why both approaches?**
1. **With spawn (current):** Each kernel gets a fresh process, so modules are already isolated. But this is belt-and-suspenders for safety.
2. **If we ever add process reuse:** The cleanup ensures stale modules don't persist.

**Where cleanup happens:**
- `modal_app.py:217-220` — `_generic_worker` cleanup
- `modal_app.py:457-461` — `_kernelbench_worker` cleanup

---

## Timeout & Crash Recovery

### Hard Timeout

```python
p.start()
p.join(timeout=300)  # 5-minute hard cap

if p.is_alive():
    p.kill()  # Force kill subprocess
    p.join()
    result = {
        "kernel_name": kernel_name,
        "error": "Timeout: kernel exceeded 300s limit",
        "correctness": False,
        ...
    }
```

**Why this is safe:**
- `p.kill()` terminates the child process
- Child's CUDA context Y is destroyed
- Parent's context X remains alive
- Next `.remote()` call gets a healthy GPU

**Without process isolation:** Killing a thread/worker might leave GPU in bad state.

### Crash Recovery

```python
elif p.exitcode != 0:
    # Non-zero exit = subprocess crashed
    result = {
        "kernel_name": kernel_name,
        "error": f"Worker crashed (exit code {p.exitcode}) — likely CUDA illegal memory access",
        "correctness": False,
        ...
    }
```

**Why this is safe:**
- If child crashes (CUDA illegal memory access → SIGABRT = exit code -6)
- Only child process dies
- Child's GPU context is freed
- Parent Modal container remains alive
- Next kernel runs on clean GPU

**Without process isolation:** One crash would corrupt the container's GPU permanently.

---

## End-to-End Example

Let's walk through benchmarking a simple kernel:

### PyTorch Reference Code
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
    return torch.randn(32, 256, dtype=torch.float32)

def get_init_inputs():
    return [256]
```

### Triton Kernel Code
```python
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(x_ptr, w_ptr, out_ptr, M, N, K, ...):
    pid_m = tl.program_id(0)
    # ... kernel logic ...

def triton_kernel_wrapper(x, weight, bias=None):
    out = torch.empty(...)
    grid = (...)
    matmul_kernel[grid](...)
    return out
```

### Execution Flow

#### Step 1: Parent spawns subprocess
```
benchmark_kernelbench(triton_code=..., pytorch_code=...)
  ↓
ctx = mp.get_context("spawn")
p = ctx.Process(target=_kernelbench_worker, args=(...))
p.start()  ← Fresh Python interpreter starts
```

#### Step 2: Child process initializes
```
[Child Process - Fresh Interpreter]

import torch  # First torch import in this process
# CUDA not initialized yet!

with tempfile.NamedTemporaryFile(...) as f:
    f.write(pytorch_code)
    pytorch_file = "/tmp/tmp123_pytorch.py"

with tempfile.NamedTemporaryFile(...) as f:
    f.write(triton_code)
    triton_file = "/tmp/tmp456_triton.py"
```

#### Step 3: Load PyTorch module
```
spec = importlib.util.spec_from_file_location("pytorch_module", pytorch_file)
pytorch_module = importlib.util.module_from_spec(spec)
sys.modules["pytorch_module"] = pytorch_module
spec.loader.exec_module(pytorch_module)

# Now pytorch_module has:
#   - Model class
#   - get_inputs() function
#   - get_init_inputs() function

model = Model(256).cuda()  # ← Creates GPU context Y in child
```

#### Step 4: Load Triton module
```
spec = importlib.util.spec_from_file_location("triton_module", triton_file)
triton_module = importlib.util.module_from_spec(spec)
sys.modules["triton_module"] = triton_module
spec.loader.exec_module(triton_module)  # ← Triton JIT compiles here in context Y

triton_kernel_wrapper = triton_module.triton_kernel_wrapper
```

#### Step 5: Run benchmark
```
for i in range(n_correctness):
    inputs = get_inputs()  # Fresh random tensors
    cuda_inputs = [x.cuda() for x in inputs]  # All in context Y

    ref_output = model(*cuda_inputs)
    kernel_output = _call_wrapper(cuda_inputs)

    assert torch.allclose(ref_output, kernel_output)
    print("✓ Correct")

# Performance phase
for _ in range(n_trials):
    kernel_output = _call_wrapper(cuda_inputs)

print(f"Speedup: {reference_time / kernel_time:.2f}x")
```

#### Step 6: Cleanup
```
finally:
    os.unlink(pytorch_file)  # Delete /tmp/tmp123_pytorch.py
    os.unlink(triton_file)   # Delete /tmp/tmp456_triton.py
    sys.modules.pop("pytorch_module", None)
    sys.modules.pop("triton_module", None)

result_queue.put(result)
# Child process exits → GPU context Y destroyed
```

#### Step 7: Parent receives result
```
[Parent Process - Still intact]

p.join()  # Wait for child to finish
result = result_queue.get_nowait()  # Grab result dict

# Save to disk
with open(f"/results/{result_id}.json", "w") as f:
    json.dump(result, f)

volume.commit()
return result
```

#### Step 8: Next kernel runs
```
# Parent still has GPU available and healthy
# Next kernel spawns a fresh child (Step 1 repeats)
p = ctx.Process(target=_kernelbench_worker, args=(...))
p.start()  ← Another fresh interpreter
```

---

## Summary

| Aspect | How We Handle It |
|--------|-----------------|
| **CUDA context isolation** | `spawn` creates fresh interpreter, no inherited GPU context |
| **Module pollution** | Fresh `sys.modules` per kernel + explicit cleanup |
| **Process reuse** | Each kernel gets a new process (no state carryover) |
| **Crash handling** | Process crash doesn't affect parent; next kernel runs on clean GPU |
| **Timeout** | `p.join(timeout=300)` + `p.kill()` for hung kernels |
| **Result passing** | Queue-based IPC (`result_queue.put()` / `.get()`) |
| **Triton JIT cache** | Fresh cache per process (loaded from file each time) |

This architecture ensures **robustness and correctness** even when benchmarking untrusted/generated kernel code.

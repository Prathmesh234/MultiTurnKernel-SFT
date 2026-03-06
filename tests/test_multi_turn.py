"""
End-to-end test for the multi-turn pipeline.

Exercises: fake model generation → Modal benchmark → MultiTurnQueue feedback → retry.

Turn 1: deliberately buggy kernel (x * y instead of relu(x + y)) → correctness=False
Turn 2: illegal memory access kernel (no mask, massive OOB) → CUDA error
Turn 3: correct but naive kernel (separate add + relu) → correctness=True, speedup < 1.0
Turn 4: optimized fused kernel (add + relu in one pass) → correctness=True, speedup >= 1.0

Run: /opt/miniconda3/bin/python test_multi_turn.py
"""

import json
import re
import modal
from multi_turn_queue import MultiTurnQueue

# ---------------------------------------------------------------------------
# Kernel code strings
# ---------------------------------------------------------------------------

# Turn 1: wrong operation entirely — multiplies instead of add+relu
BUGGY_KERNEL = '''
import torch
import triton
import triton.language as tl

@triton.jit
def vec_op_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # BUG: multiplies instead of add+relu
    out = x * y
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_kernel_wrapper(x, y):
    output = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    vec_op_kernel[grid](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)
    return output
'''

# Turn 2: illegal memory access — no bounds mask, huge offset → CUDA crash
ILLEGAL_MEM_KERNEL = '''
import torch
import triton
import triton.language as tl

@triton.jit
def bad_access_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # BUG: massive offset with no bounds check — reads/writes way past buffer end
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + 1000000000
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    out = x + y
    tl.store(out_ptr + offsets, out)

def triton_kernel_wrapper(x, y):
    output = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    bad_access_kernel[grid](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)
    return output
'''

# Turn 3: correct but slow — two separate Triton kernels (no fusion)
CORRECT_SLOW_KERNEL = '''
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

@triton.jit
def relu_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.maximum(x, 0.0), mask=mask)

def triton_kernel_wrapper(x, y):
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    # Two separate kernel launches — no fusion
    tmp = torch.empty_like(x)
    add_kernel[grid](x, y, tmp, n, BLOCK_SIZE=BLOCK_SIZE)
    output = torch.empty_like(x)
    relu_kernel[grid](tmp, output, n, BLOCK_SIZE=BLOCK_SIZE)
    return output
'''

# Turn 4: optimized — fused add+relu in a single kernel pass
OPTIMIZED_KERNEL = '''
import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_relu_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Fused: one memory read, one write, no intermediate allocation
    out = tl.maximum(x + y, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_kernel_wrapper(x, y):
    output = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_add_relu_kernel[grid](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)
    return output
'''

# Reference: two separate PyTorch ops (add then relu) — Triton fusion should beat this
REFERENCE_CODE = '''
import torch

def reference_impl(x, y):
    return torch.relu(x + y)
'''

INPUT_SHAPES = {
    "x": {"shape": [2048, 2048], "dtype": "float32"},
    "y": {"shape": [2048, 2048], "dtype": "float32"},
}

# ---------------------------------------------------------------------------
# Fake generator — returns progressively better kernels each turn
# ---------------------------------------------------------------------------

KERNELS_BY_TURN = {
    1: ("I will write a kernel for this fused add+relu operation.", BUGGY_KERNEL),
    2: (
        "The previous kernel was incorrect — output mismatch.\n"
        "Let me try a different approach with direct memory access.",
        ILLEGAL_MEM_KERNEL,
    ),
    3: (
        "The previous kernel caused a CUDA crash due to illegal memory access.\n"
        "I need to add proper bounds masking. Using two separate kernels for now.",
        CORRECT_SLOW_KERNEL,
    ),
    4: (
        "The kernel is correct but slow (two separate kernel launches).\n"
        "I will fuse add+relu into a single kernel pass to eliminate the intermediate buffer.",
        OPTIMIZED_KERNEL,
    ),
}


def fake_generate(turn_num: int, feedback: str | None) -> str:
    """Simulate model output with <think>/<triton> tags.

    Turn 1: buggy kernel (wrong op)
    Turn 2: illegal memory access (CUDA crash)
    Turn 3: correct but slow (two kernels)
    Turn 4+: optimized fused kernel
    """
    # Clamp to max defined turn so turns beyond 4 reuse the optimized kernel
    thinking, code = KERNELS_BY_TURN.get(turn_num, KERNELS_BY_TURN[4])
    return f"<think>\n{thinking}\n</think>\n\n<triton>\n{code}\n</triton>"


# ---------------------------------------------------------------------------
# Main test loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MULTI-TURN END-TO-END TEST")
    print("=" * 60)

    # Look up the deployed Modal function
    benchmark = modal.Function.from_name("kernelbench-triton", "benchmark_triton_kernel")

    # Set up the queue (max 4 turns)
    queue = MultiTurnQueue(max_turns=5)

    # Build the initial item — mirrors orchestrator.run_multi_turn() format
    item = {
        "sample_key": "test_fused_add_relu",
        "sample": {
            "source": "test",
            "level": "1",
            "name": "fused_add_relu",
            "problem_id": "test_001",
        },
        "pytorch_code": REFERENCE_CODE,
        "messages": [
            {"role": "system", "content": "You are a Triton kernel engineer."},
            {"role": "user", "content": "Write a Triton kernel that computes relu(x + y)."},
        ],
        "turn_num": 1,
        "turns_history": [],
    }
    queue.add(item)

    # ---------- multi-turn loop ----------
    feedback = None

    while len(queue) > 0:
        item = queue.pop()
        turn = item["turn_num"]

        print(f"\n--- Turn {turn} ---")

        # 1. Fake-generate a kernel
        completion = fake_generate(turn, feedback)

        # 2. Extract code (simple tag extraction like orchestrator does)
        match = re.search(r"<triton>(.*?)</triton>", completion, re.DOTALL)
        triton_code = match.group(1).strip() if match else None
        assert triton_code, "Failed to extract Triton code"

        # 3. Benchmark on Modal
        print("Sending to Modal for benchmarking...")
        result = benchmark.remote(
            kernel_code=triton_code,
            reference_torch_code=REFERENCE_CODE,
            input_shapes=INPUT_SHAPES,
            n_correctness=5,
            n_trials=50,
            kernel_name=f"test_fused_add_relu_turn{turn}",
        )

        correct = result.get("correctness", False)
        speedup = result.get("speedup", 0.0)
        error = result.get("error")
        print(f"  correctness={correct}  speedup={speedup:.2f}x  error={error}")

        # 4. Record turn history
        item["turns_history"].append({
            "turn": turn,
            "triton_code": triton_code,
            "result": result,
            "feedback_given": None,
        })

        # 5. Should we stop?
        stop, reason = queue.should_stop(turn, result)

        if stop:
            print(f"  Stopping: {reason}")
            trace = queue.finalize(item, reason)
            break
        else:
            # 6. Build feedback and requeue
            feedback = queue.build_feedback(result)
            item["turns_history"][-1]["feedback_given"] = feedback
            print(f"  Feedback: {feedback.strip()}")
            queue.requeue_with_feedback(item, feedback, completion)

    # ---------- print final trace ----------
    assert len(queue.completed_traces) == 1, "Expected exactly one completed trace"
    trace = queue.completed_traces[0]

    print("\n" + "=" * 60)
    print("FINAL TRACE")
    print("=" * 60)
    print(json.dumps(trace, indent=2, default=str))

    # ---------- assertions ----------
    # Turn 1: incorrect (buggy kernel does x*y)
    assert trace["turns"][0]["result"]["correctness"] is False, "Turn 1 should be incorrect"
    assert trace["turns"][0]["result"].get("error") is None, "Turn 1 should NOT have a runtime error"
    assert trace["turns"][0]["feedback_given"] is not None, "Turn 1 should have feedback"

    # Turn 2: CUDA illegal memory access — should return an error
    assert trace["turns"][1]["result"]["correctness"] is False, "Turn 2 should be incorrect"
    assert trace["turns"][1]["result"].get("error") is not None, "Turn 2 should have a runtime error"
    assert trace["turns"][1]["feedback_given"] is not None, "Turn 2 should have feedback"
    print(f"  Turn 2 error (expected): {trace['turns'][1]['result']['error'][:80]}...")

    # Turn 3: correct but slow (two separate kernels, no fusion)
    assert trace["turns"][2]["result"]["correctness"] is True, "Turn 3 should be correct"
    assert trace["turns"][2]["result"]["speedup"] < 1.0, "Turn 3 should be slow"

    # Turn 4: optimized fused kernel — should hit success_fast
    assert trace["num_turns"] == 4, f"Expected 4 turns, got {trace['num_turns']}"
    assert trace["stop_reason"] == "success_fast", f"Expected success_fast, got {trace['stop_reason']}"
    assert trace["final_result"].get("correctness") is True, "Final kernel should be correct"
    assert trace["final_result"].get("speedup", 0) >= 1.0, "Final kernel should be fast"

    print(f"\nAll assertions passed. ({trace['num_turns']} turns, stop_reason={trace['stop_reason']})")


if __name__ == "__main__":
    main()

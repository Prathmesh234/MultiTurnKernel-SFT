# Multi-Turn Reasoning & Kernel Generation Specification

## Overview

This document specifies a **single-deque FIFO architecture** for multi-turn iterative refinement of Triton kernel generation using GLM-4.5-Air. The system allows the model to self-correct and optimize kernels over multiple turns (up to 4) based on validation feedback.

**Output file**: `reasoning_traces_glm45_multiturn.json`

## Architecture

### Single Deque Flow (FIFO)

A `MultiTurnQueue` class manages a single `collections.deque`. It is a **passive data structure / proxy** - it holds items, tracks turns, and builds feedback strings. All actual work (API calls, Modal validation, extraction) is done by the existing `TraceOrchestrator`.

```
                    ┌──────────────────────────────────────────┐
                    │              DEQUE (FIFO)                 │
                    │  popleft() <── items ──> append()         │
                    └──────────────────────────────────────────┘
                                    │
                              popleft()
                                    │
                                    ▼
                    ┌──────────────────────────────────────────┐
                    │         GLM-4.5-Air (vLLM)               │
                    │  Generate <think>...<triton>...           │
                    │  (batch of 5 before validation)          │
                    └──────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────────────┐
                    │         Modal H100 Validation             │
                    │  correctness + speedup                    │
                    └──────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────────────┐
                    │         Decision Logic                    │
                    │  - Success (correct + fast)? → Save      │
                    │  - Turn 4 reached? → Save                │
                    │  - Correct but slow? → deque.append()    │
                    │  - Error/incorrect? → deque.append()     │
                    └──────────────────────────────────────────┘
                                    │
                          (if retry) │
                                    ▼
                          deque.append(item)
                          (back of the line)
```

## MultiTurnQueue - The Proxy

`MultiTurnQueue` is just a container. It does NOT make API calls, validate kernels, or manage sessions. The orchestrator drives the loop and calls into the queue for bookkeeping.

```python
from collections import deque
from datetime import datetime

class MultiTurnQueue:
    """
    Passive proxy that manages:
    - The deque (add, pop, re-queue)
    - Turn counting
    - Feedback string construction
    - Building the final trace dict

    Does NOT: call GLM-4.5-Air, call Modal, extract code, manage sessions.
    """

    def __init__(self, max_turns: int = 4):
        self.max_turns = max_turns
        self.queue = deque()
        self.completed_traces = []

    def add(self, item: dict):
        """Push an item onto the deque."""
        self.queue.append(item)

    def pop(self) -> dict | None:
        """Pop next item (FIFO). Returns None if empty."""
        return self.queue.popleft() if self.queue else None

    def __len__(self) -> int:
        return len(self.queue)

    def should_stop(self, turn_num: int, result: dict) -> tuple[bool, str]:
        """Check if this item is done or needs another turn."""
        if turn_num >= self.max_turns:
            return True, "max_turns_reached"
        if result.get("correctness") and result.get("speedup", 0) >= 1.0:
            return True, "success_fast"
        return False, "continue"

    def build_feedback(self, result: dict) -> str:
        """Build the feedback string to send back to the model."""
        error = result.get("error")
        correctness = result.get("correctness", False)
        speedup = result.get("speedup", 0.0)

        if error:
            return FEEDBACK_COMPILATION_FAILED.format(error_message=error)
        elif not correctness:
            return FEEDBACK_CORRECTNESS_FAILED.format(
                error_message="Output mismatch with PyTorch reference"
            )
        elif speedup < 1.0:
            return FEEDBACK_NEEDS_OPTIMIZATION.format(speedup=speedup)
        else:
            return "Great work! The kernel is correct and fast."

    def requeue_with_feedback(self, item: dict, feedback: str, completion: str):
        """Append assistant response + feedback to messages and re-queue."""
        item["messages"].append({"role": "assistant", "content": completion})
        item["messages"].append({"role": "user", "content": feedback})
        item["turn_num"] += 1
        self.queue.append(item)

    def finalize(self, item: dict, reason: str) -> dict:
        """Build the final trace dict and add to completed list."""
        last_turn = item["turns_history"][-1]

        trace = {
            "sample_key": item["sample_key"],
            "source": item["sample"].get("source"),
            "level": item["sample"].get("level"),
            "name": item["sample"].get("name"),
            "problem_id": item["sample"].get("problem_id"),
            "pytorch_code": item["pytorch_code"],
            "num_turns": item["turn_num"],
            "stop_reason": reason,
            "final_triton_code": last_turn.get("triton_code"),
            "final_result": last_turn.get("result", {}),
            "turns": item["turns_history"],
            "full_messages": item["messages"],
            "timestamp": datetime.now().isoformat(),
        }

        self.completed_traces.append(trace)
        return trace
```

### What the Orchestrator Does (unchanged except noted)

The orchestrator still owns the entire control flow. The only change needed is that `generate_completion()` accepts a `messages` list directly (instead of building messages from `pytorch_code` internally). Everything else stays the same.

```python
# In orchestrator.py - the multi-turn loop (inside existing run() method)

async def run_multi_turn(self, batch_size=5):
    samples = self.dataloader.load_all()
    queue = MultiTurnQueue(max_turns=4)

    # Orchestrator builds initial items and adds to queue
    for sample in samples:
        sample_key = self._get_sample_key(sample)
        item = {
            "sample_key": sample_key,
            "sample": sample,
            "pytorch_code": sample["pytorch_code"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT + MULTI_TURN_ADDENDUM},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                    pytorch_code=sample["pytorch_code"]
                )},
            ],
            "turn_num": 1,
            "turns_history": [],
        }
        queue.add(item)

    # Existing Modal context + aiohttp session - unchanged
    with modal_app.run():
        async with aiohttp.ClientSession() as session:
            while len(queue) > 0:
                # Pop a batch
                batch = []
                for _ in range(min(batch_size, len(queue))):
                    batch.append(queue.pop())

                # Generate completions concurrently (existing method)
                gen_tasks = [
                    self.generate_completion(item["messages"], session)
                    for item in batch
                ]
                responses = await asyncio.gather(*gen_tasks)

                # Process each: extract, validate, route
                for item, response in zip(batch, responses):
                    # Handle generation failure
                    if not response or not response.get("content"):
                        item["turns_history"].append({
                            "turn": item["turn_num"],
                            "triton_code": None,
                            "result": {"correctness": False, "error": "Generation failed"},
                            "feedback_given": None,
                        })
                        stop, reason = queue.should_stop(item["turn_num"], {"error": "Generation failed"})
                        if stop:
                            queue.finalize(item, reason)
                        else:
                            feedback = queue.build_feedback({"error": "Generation failed"})
                            item["turns_history"][-1]["feedback_given"] = feedback
                            queue.requeue_with_feedback(item, feedback, "")
                        continue

                    completion = response["content"]
                    reasoning = response.get("reasoning")

                    # Extract (existing methods)
                    triton_code = self.extract_triton_code(completion)
                    thinking = self.extract_thinking(completion)

                    if not triton_code:
                        # Extraction failed - treat as error turn
                        result = {"correctness": False, "error": "Triton code extraction failed"}
                    else:
                        # Validate on Modal (existing method)
                        result = await self.validate_on_modal(
                            triton_code, item["pytorch_code"], item["sample"]
                        )

                    # Record turn
                    turn_result = {
                        "turn": item["turn_num"],
                        "thinking": thinking,
                        "model_reasoning": reasoning,
                        "triton_code": triton_code,
                        "full_completion": completion,
                        "result": result,
                        "feedback_given": None,  # set below if retrying
                    }
                    item["turns_history"].append(turn_result)

                    # Route
                    stop, reason = queue.should_stop(item["turn_num"], result)
                    if stop:
                        queue.finalize(item, reason)
                    else:
                        feedback = queue.build_feedback(result)
                        turn_result["feedback_given"] = feedback
                        queue.requeue_with_feedback(item, feedback, completion)

                # Save after every batch (incremental, same as current single-turn)
                # Writes ALL completed traces so far, not just this batch
                self._save_multiturn_traces(queue.completed_traces)
                print(f"Saved {len(queue.completed_traces)} traces to {OUTPUT_FILE}")

    return queue.completed_traces
```

### Required Additions to Orchestrator

**`_save_multiturn_traces()`** - same pattern as existing `_save_traces()`, just writes to the multiturn output file:

```python
def _save_multiturn_traces(self, traces: list[dict]):
    """Save completed multi-turn traces to JSON (overwrites with full list each time)."""
    with open(OUTPUT_FILE, "w") as f:
        json.dump(traces, f, indent=2, default=str)
```

### Required Change to `generate_completion()`

The only orchestrator method that needs a tweak: accept `messages` directly instead of building them from `pytorch_code`.

```python
# Current signature:
async def generate_completion(self, pytorch_code: str, session) -> dict

# Updated signature:
async def generate_completion(self, messages: list[dict], session) -> dict
```

The body stays the same (payload construction, API call, response parsing). Just remove the internal message-building lines and use the passed-in `messages` directly.

## Prompt Design (Multi-Turn)

### System Prompt Update

The existing system prompt in `orchestrator.py` is extended with a multi-turn paragraph. No `<turn>` tags in the prompt - the chat message roles naturally delineate turns:

```python
# Appended to the existing SYSTEM_PROMPT:
MULTI_TURN_ADDENDUM = """
=== MULTI-TURN FEEDBACK ===

You may receive feedback on your generated kernel if it fails validation
or is slower than PyTorch. When you receive feedback, analyze the error
or performance issue, then generate an improved version. Always respond
with the same <think>...</think> <triton>...</triton> format.
"""
```

### Conversation Flow (Natural Chat Roles)

The model sees standard multi-turn chat. No custom markup needed - roles handle turn separation:

| Turn | Role | Content |
|------|------|---------|
| 1 | system | System prompt (with multi-turn addendum) |
| 1 | user | "Convert this PyTorch code: ```...```" |
| 1 | assistant | `<think>reasoning</think><triton>code</triton>` |
| 2 | user | "Your kernel FAILED to compile. Error: ... Fix and regenerate." |
| 2 | assistant | `<think>updated reasoning</think><triton>fixed code</triton>` |
| 3 | user | "Correct but slow (0.8x). Optimize for ..." |
| 3 | assistant | `<think>perf analysis</think><triton>optimized code</triton>` |

## Item Structure

Each item in the deque contains:

```python
{
    "sample_key": str,           # Unique identifier (e.g., "kernelbench_80")
    "sample": dict,              # Original sample metadata
    "pytorch_code": str,         # The PyTorch code to convert
    "messages": list[dict],      # Conversation history for GLM-4.5-Air
    "turn_num": int,             # Current turn number (1-4)
    "turns_history": list[dict], # All previous turn results
}
```

## Feedback Templates

```python
FEEDBACK_COMPILATION_FAILED = """Your Triton kernel FAILED to compile or execute.

Error:
{error_message}

Common issues:
- Using torch.* inside @triton.jit (use tl.* instead)
- Non-constexpr in tl.arange() arguments
- Shape mismatches in tl.dot()
- Missing masks for boundary conditions

Fix the issue and regenerate.
Respond with the same <think>...</think> <triton>...</triton> format."""

FEEDBACK_CORRECTNESS_FAILED = """Your generated Triton kernel produced INCORRECT results.

Error details:
{error_message}

The kernel failed correctness validation against the PyTorch reference.
Please analyze what went wrong and regenerate a corrected version.
Respond with the same <think>...</think> <triton>...</triton> format."""

FEEDBACK_NEEDS_OPTIMIZATION = """Your Triton kernel is CORRECT but slower than PyTorch.

Performance:
- Current speedup: {speedup:.2f}x
- Target: >1.0x (faster than PyTorch)

Analyze the performance bottleneck and optimize. Consider:
- Memory access patterns (coalescing)
- Block size tuning (try 128, 256, 512, 1024)
- Reducing redundant loads
- Better use of shared memory

Respond with the same <think>...</think> <triton>...</triton> format."""
```

## Output Format

**File**: `reasoning_traces_glm45_multiturn.json`

Each completed sample produces a trace grouped by the original PyTorch question, containing the full trajectory (all turns) plus the final result:

```python
trace = {
    "sample_key": str,
    "source": str,              # "kernelbook" or "kernelbench"
    "level": int | None,
    "name": str | None,
    "problem_id": int | None,
    "pytorch_code": str,

    # Multi-turn metadata
    "num_turns": int,           # How many turns it took (1-4)
    "stop_reason": str,         # "success_fast" or "max_turns_reached"

    # Final result (from last turn)
    "final_triton_code": str,
    "final_result": {
        "correctness": bool,
        "speedup": float,
        "fast_0": bool,
        "fast_1": bool,
        "fast_2": bool,
        "error": str | None,
    },

    # ALL turns (including successful ones)
    "turns": [
        {
            "turn": 1,
            "thinking": str,
            "model_reasoning": str,
            "triton_code": str,
            "full_completion": str,
            "feedback_given": str | None,   # feedback sent AFTER this turn (null on last turn)
            "result": {
                "correctness": bool,
                "speedup": float,
                "fast_0": bool,
                "fast_1": bool,
                "fast_2": bool,
                "error": str | None,
            }
        },
        # ... up to 4 turns
    ],

    # Full conversation history (for SFT training)
    "full_messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Convert this PyTorch code: ..."},
        {"role": "assistant", "content": "...", "reasoning": "..."},
        {"role": "user", "content": "Your kernel FAILED to compile..."},
        {"role": "assistant", "content": "...", "reasoning": "..."},
        # ... up to 4 assistant responses
    ],

    "timestamp": str,
}
```

**Key points:**
- Every turn is recorded in `turns`, including ones that succeeded
- `feedback_given` shows what feedback was sent AFTER this turn's result (null on the last turn)
- `final_triton_code` and `final_result` duplicate the last turn's data for easy access
- `full_messages` is the complete conversation for SFT training

## Context Window Management

GLM-4.5-Air has a **128K context window**, but we set `--max-model-len 32768` for the vLLM server.

**Budget per sample (4 turns):**
- System prompt: ~1.5K tokens
- PyTorch code: ~0.5-2K tokens
- Per turn:
  - Model response: ~2-5K tokens
  - Feedback: ~0.2-0.5K tokens
- **Total for 4 turns**: ~15-30K tokens

This fits comfortably within the 32K limit.

## Configuration

```python
# multi_turn_queue.py
MAX_TURNS = 4
GENERATION_BATCH_SIZE = 5   # Generate 5 completions before validating
OUTPUT_FILE = "reasoning_traces_glm45_multiturn.json"
```

## Usage

```bash
# Start vLLM server (already running via run_glm45_air.sh)
# Then run orchestrator with multi-turn:
uv run --no-sync python orchestrator.py \
    --output reasoning_traces_glm45_multiturn.json \
    --multi-turn \
    --max-turns 4 \
    --batch-size 5
```

## Dry-Run Trace (kernelbench_80)

Walking through the real `kernelbench_80` sample (GEMM + Max + Subtract + GELU) to verify the flow:

```
TURN 1:
  Orchestrator builds item: messages=[system, user(pytorch_code)], turn_num=1
  queue.add(item)
  queue.pop() → item
  orchestrator.generate_completion(item["messages"], session) → response
  orchestrator.extract_triton_code(response) → triton_code (bad signature)
  orchestrator.validate_on_modal(triton_code, ...) → {error: "TypeError: missing arg 'max_dim'"}
  queue.should_stop(1, result) → False
  queue.build_feedback(result) → FEEDBACK_COMPILATION_FAILED with TypeError
  turn_result["feedback_given"] = feedback
  queue.requeue_with_feedback(item, feedback, completion) → turn_num=2, back of deque

TURN 2:
  queue.pop() → item (messages now has 4 entries: system, user, assistant, user-feedback)
  orchestrator.generate_completion(item["messages"], session) → fixed response
  extract + validate → {correctness: true, speedup: 0.85}
  queue.should_stop(2, result) → False (correct but slow)
  queue.build_feedback(result) → FEEDBACK_NEEDS_OPTIMIZATION (0.85x)
  requeue → turn_num=3

TURN 3:
  queue.pop() → item (messages now has 6 entries)
  generate → optimized code
  validate → {correctness: true, speedup: 1.3}
  queue.should_stop(3, result) → True, "success_fast"
  queue.finalize(item, "success_fast") → trace saved to completed_traces
  orchestrator._save_traces(queue.completed_traces) → writes JSON
```

## Future Enhancements

1. **Adaptive turn limits**: Allow more turns for harder samples
2. **Turn budget**: Stop early if same error repeats 2+ times
3. **Feedback diversity**: Randomly sample from multiple feedback templates
4. **Checkpointing**: Save deque state for resume after crash
5. **Metrics dashboard**: Real-time view of deque depth and success rates

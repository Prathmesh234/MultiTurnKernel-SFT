"""
MultiTurnQueue - passive proxy for multi-turn iterative refinement.

Manages a deque of items, turn counting, feedback construction,
and final trace building. Does NOT make API calls, validate kernels,
or manage sessions - the orchestrator drives all actual work.
"""

from collections import deque
from datetime import datetime


# Feedback templates
FEEDBACK_COMPILATION_FAILED = """Compilation/runtime error:
{error_message}

Fix and regenerate using <think>...</think> <triton>...</triton> format."""

FEEDBACK_CORRECTNESS_FAILED = """Incorrect output: {error_message}

Fix and regenerate using <think>...</think> <triton>...</triton> format."""

FEEDBACK_NEEDS_OPTIMIZATION = """Correct but slow ({speedup:.2f}x PyTorch). Optimize and regenerate using <think>...</think> <triton>...</triton> format."""


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
        self.queue: deque[dict] = deque()
        self.completed_traces: list[dict] = []

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

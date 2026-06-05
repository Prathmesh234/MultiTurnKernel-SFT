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

Fix and regenerate the corrected code inside <triton>...</triton> tags."""

FEEDBACK_CORRECTNESS_FAILED = """Incorrect output: {error_message}

Fix and regenerate the corrected code inside <triton>...</triton> tags."""

FEEDBACK_NEEDS_OPTIMIZATION = """Correct but slow ({speedup:.2f}x PyTorch). Optimize and regenerate the improved code inside <triton>...</triton> tags."""


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
        self.failed_tasks: list[dict] = []
        self._pending_validation_count: int = 0

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

    @property
    def pending_validation_count(self) -> int:
        """Number of items currently awaiting Modal validation."""
        return self._pending_validation_count

    def submit_for_validation(self, item: dict, triton_code, completion: str, reasoning):
        """Mark an item as awaiting Modal validation (no blocking).

        The orchestrator fires the actual Modal call as a background task
        and calls resolve_validation() when it finishes.
        """
        item["_validation_state"] = {
            "triton_code": triton_code,
            "completion": completion,
            "reasoning": reasoning,
        }
        self._pending_validation_count += 1

    def resolve_validation(self, item: dict, result: dict) -> str | None:
        """Process a Modal validation result: finalize or requeue.

        Returns the sample_key if finalized, None if requeued.
        """
        state = item.pop("_validation_state")
        self._pending_validation_count -= 1

        turn_result = {
            "turn": item["turn_num"],
            "reasoning": state["reasoning"],
            "triton_code": state["triton_code"],
            "full_completion": state["completion"],
            "result": result,
            "feedback_given": None,
        }
        item["turns_history"].append(turn_result)

        stop, reason = self.should_stop(item["turn_num"], result)
        if stop:
            self.finalize(item, reason)
            return item["sample_key"]
        else:
            feedback = self.build_feedback(result)
            turn_result["feedback_given"] = feedback
            self.requeue_with_feedback(item, feedback, state["completion"])
            return None

    def requeue_with_feedback(self, item: dict, feedback: str, completion: str):
        """Append assistant response + feedback to messages and re-queue.

        Uses appendleft so the item gets its next turn immediately (depth-first),
        rather than waiting for all 1770+ items to cycle through (breadth-first).
        """
        item["messages"].append({"role": "assistant", "content": completion})
        item["messages"].append({"role": "user", "content": feedback})
        item["turn_num"] += 1
        self.queue.appendleft(item)

    def finalize(self, item: dict, reason: str) -> dict:
        """Build the final trace dict and add to completed list.

        If all turns failed (no correctness across any turn) and the item has
        not already been requeued once, saves the trace to failed_tasks and
        requeues the item fresh at the end of the main queue for one more round.
        """
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

        all_failed = (
            reason == "max_turns_reached"
            and not any(t.get("result", {}).get("correctness") for t in item["turns_history"])
        )

        if all_failed and not item.get("requeued_after_failure"):
            self.failed_tasks.append(trace)
            # Reset item to initial state and requeue at the end.
            # Reassignment (not mutation) so the trace keeps its data.
            item["messages"] = item["messages"][:2]
            item["turn_num"] = 1
            item["turns_history"] = []
            item["requeued_after_failure"] = True
            self.queue.append(item)
        else:
            self.completed_traces.append(trace)

        return trace

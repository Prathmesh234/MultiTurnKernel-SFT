#!/usr/bin/env python3
"""
Generate additional reasoning traces using OpenRouter API.

Phase 1: Resume incomplete traces (max 4 total turns)
Phase 2: Continue failed traces (+3 additional turns on top of original 4 = 7 max)

Uses the same MultiTurnQueue, Modal validation pipeline, and system prompt
as the original orchestrator, but calls OpenRouter instead of local vLLM.

Model: qwen/qwen3-235b-a22b-thinking-2507  (same Qwen3-235B-A22B FP8)

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python scripts/run_openrouter_traces.py --phase incomplete
    python scripts/run_openrouter_traces.py --phase failed
    python scripts/run_openrouter_traces.py --phase all
"""

import json
import os
import sys
import re
import asyncio
import aiohttp
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path so we can import modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from multi_turn_queue import MultiTurnQueue
from utilities import pre_validate_triton_code
import modal
from modal_app import app as modal_app, benchmark_kernelbench
from orchestrator import (
    SYSTEM_PROMPT,
    MULTI_TURN_ADDENDUM,
    USER_PROMPT_TEMPLATE,
    MAX_COMPLETION_TOKENS,
    TEMPERATURE,
)

# ---------------------------------------------------------------------------
# OpenRouter config — same model as vLLM (Qwen3-235B-A22B FP8), zero variance
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "qwen/qwen3-235b-a22b-thinking-2507"
MAX_MODEL_LEN = 131072  # Same context window as vLLM setup

# ---------------------------------------------------------------------------
# File paths (relative to PROJECT_ROOT)
# ---------------------------------------------------------------------------
INCOMPLETE_FILE = PROJECT_ROOT / "incomplete" / "incomplete_traces.json"
FAILED_FILE = PROJECT_ROOT / "traces" / "reasoning_traces_multiturn_failed.json"
QWEN_TRACES_FILE = PROJECT_ROOT / "traces" / "reasoning_traces_multiturn_qwen.json"
CORRECT_TRACES_FILE = PROJECT_ROOT / "traces" / "reasoning_traces_multiturn.json"

# Phase limits
INCOMPLETE_MAX_TURNS = 4   # Total turns for incomplete traces
FAILED_EXTRA_TURNS = 3     # Additional turns on top of original for failed traces


# ---------------------------------------------------------------------------
# OpenRouter API
# ---------------------------------------------------------------------------

async def generate_completion(
    messages: list[dict],
    session: aiohttp.ClientSession,
    headers: dict,
    retries: int = 3,
) -> Optional[dict]:
    """
    Generate a completion via OpenRouter. Mirrors orchestrator.generate_completion
    but hits the OpenRouter endpoint instead of local vLLM.
    """
    total_input_chars = sum(len(m["content"]) for m in messages)
    estimated_input_tokens = int(total_input_chars / 3.5) + 50
    max_tokens = min(MAX_COMPLETION_TOKENS, MAX_MODEL_LEN - estimated_input_tokens)
    max_tokens = max(max_tokens, 1024)

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
    }

    for attempt in range(retries):
        try:
            async with session.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=900),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    message = data["choices"][0]["message"]
                    raw_content = message.get("content") or ""

                    # Extract reasoning — OpenRouter may use different field names
                    reasoning = (
                        message.get("reasoning")
                        or message.get("reasoning_content")
                    )
                    # Fallback: vLLM-style </think> tag left in content
                    if not reasoning and "</think>" in raw_content:
                        parts = raw_content.split("</think>", 1)
                        reasoning = parts[0].strip()
                        raw_content = parts[1].strip()
                    # Fallback: <think>...</think> wrapping
                    if not reasoning and "<think>" in raw_content:
                        think_match = re.search(
                            r"<think>(.*?)</think>", raw_content, re.DOTALL
                        )
                        if think_match:
                            reasoning = think_match.group(1).strip()
                            raw_content = raw_content[think_match.end():].strip()

                    return {
                        "content": raw_content,
                        "reasoning": reasoning,
                        "usage": data.get("usage"),
                    }
                elif response.status == 429:
                    # Rate limited — back off
                    wait = 2 ** (attempt + 1)
                    error = await response.text()
                    print(f"  Rate limited (429), retrying in {wait}s ... ({error[:120]})")
                    await asyncio.sleep(wait)
                    continue
                else:
                    error = await response.text()
                    print(f"  API error: {response.status} - {error[:200]}")
                    if attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return None
        except Exception as e:
            print(f"  Request failed: {type(e).__name__}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return None

    return None


def extract_triton_code(completion: str) -> Optional[str]:
    """Extract Triton code from model completion (same logic as orchestrator)."""
    triton_match = re.search(r"<triton>(.*?)</triton>", completion, re.DOTALL)
    if triton_match:
        return triton_match.group(1).strip()

    code_blocks = re.findall(r"```python\n(.*?)```", completion, re.DOTALL)
    for block in code_blocks:
        if "triton" in block.lower() and "@triton.jit" in block:
            return block.strip()

    if "@triton.jit" in completion:
        idx = completion.find("import torch")
        if idx == -1:
            idx = completion.find("import triton")
        if idx != -1:
            return completion[idx:].strip()

    return None


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> list:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def save_json(path: Path, data: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def append_traces(path: Path, new_traces: list[dict]):
    """Load existing traces, append new ones, save back."""
    existing = load_json(path)
    existing_keys = {t["sample_key"] for t in existing}
    added = 0
    for t in new_traces:
        if t["sample_key"] not in existing_keys:
            existing.append(t)
            existing_keys.add(t["sample_key"])
            added += 1
    save_json(path, existing)
    print(f"  Appended {added} traces to {path.name} (total: {len(existing)})")


# ---------------------------------------------------------------------------
# Phase 1: Incomplete traces
# ---------------------------------------------------------------------------

def build_incomplete_queue_items(max_turns: int) -> list[dict]:
    """
    Load incomplete traces and convert to queue items.

    Incomplete traces have full_messages ending with a user feedback message,
    ready for the next generation call.
    """
    raw = load_json(INCOMPLETE_FILE)
    if not raw:
        print("No incomplete traces found.")
        return []

    items = []
    for trace in raw:
        # Skip if already at or past max turns
        current_turn = trace["current_turn"]
        if current_turn > max_turns:
            continue

        item = {
            "sample_key": trace["sample_key"],
            "sample": {
                "source": trace.get("source"),
                "level": trace.get("level"),
                "name": trace.get("name"),
                "problem_id": trace.get("problem_id"),
                "entry_point": "Model",
            },
            "pytorch_code": trace["pytorch_code"],
            "messages": trace["full_messages"],  # Already has conversation history
            "turn_num": current_turn,
            "turns_history": trace.get("turns", []),
        }
        items.append(item)

    print(f"Loaded {len(items)} incomplete traces (max_turns={max_turns})")
    return items


# ---------------------------------------------------------------------------
# Phase 2: Failed traces
# ---------------------------------------------------------------------------

def build_failed_queue_items(extra_turns: int) -> list[dict]:
    """
    Load failed traces and prepare them for additional turns.

    Failed traces have 4 turns completed. The full_messages end at turn 3's
    feedback (8 messages). Turn 4's response is in turns[3] but NOT in
    full_messages, so we reconstruct the conversation:
      1. Take full_messages (ends at turn 3 feedback)
      2. Append turn 4 assistant response
      3. Build + append feedback for turn 4 result
      4. Set turn_num = original_turns + 1
    """
    raw = load_json(FAILED_FILE)
    if not raw:
        print("No failed traces found.")
        return []

    # Also check which keys are already solved in the qwen traces
    qwen_traces = load_json(QWEN_TRACES_FILE)
    solved_keys = {
        t["sample_key"]
        for t in qwen_traces
        if t.get("stop_reason") == "success_fast"
    }
    correct_traces = load_json(CORRECT_TRACES_FILE)
    solved_keys |= {t["sample_key"] for t in correct_traces}

    items = []
    skipped = 0
    for trace in raw:
        if trace["sample_key"] in solved_keys:
            skipped += 1
            continue

        original_turns = trace["num_turns"]  # Typically 4
        max_turns = original_turns + extra_turns

        # Reconstruct messages: append turn 4 response + feedback
        messages = list(trace["full_messages"])  # Copy
        turns = trace.get("turns", [])

        if turns:
            last_turn = turns[-1]
            # Append the last turn's assistant response
            if last_turn.get("full_completion"):
                messages.append({
                    "role": "assistant",
                    "content": last_turn["full_completion"],
                })
                # Build feedback for the last turn's result
                from multi_turn_queue import MultiTurnQueue
                temp_q = MultiTurnQueue()
                feedback = temp_q.build_feedback(last_turn.get("result", {}))
                messages.append({"role": "user", "content": feedback})

        item = {
            "sample_key": trace["sample_key"],
            "sample": {
                "source": trace.get("source"),
                "level": trace.get("level"),
                "name": trace.get("name"),
                "problem_id": trace.get("problem_id"),
                "entry_point": "Model",
            },
            "pytorch_code": trace["pytorch_code"],
            "messages": messages,
            "turn_num": original_turns + 1,  # Next turn after original
            "turns_history": list(turns),  # Preserve original turn history
            "_max_turns": max_turns,  # Store per-item max turns
        }
        items.append(item)

    print(f"Loaded {len(items)} failed traces for continuation (+{extra_turns} turns, skipped {skipped} already solved)")
    return items


# ---------------------------------------------------------------------------
# Main processing loop — reuses MultiTurnQueue
# ---------------------------------------------------------------------------

async def process_phase(
    items: list[dict],
    max_turns: int,
    batch_size: int,
    modal_parallel: int,
    phase_name: str,
) -> tuple[list[dict], list[dict]]:
    """
    Process a list of queue items through the multi-turn pipeline.

    Returns (successful_traces, failed_traces).
    """
    if not items:
        return [], []

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    queue = MultiTurnQueue(max_turns=max_turns)
    for item in items:
        queue.add(item)

    print(f"\n{'='*60}")
    print(f"PHASE: {phase_name}")
    print(f"{'='*60}")
    print(f"Queue size: {len(queue)}, max_turns: {max_turns}, batch_size: {batch_size}")

    with modal_app.run():
        async with aiohttp.ClientSession() as session:
            in_flight_batch: list[dict] = []
            try:
                while len(queue) > 0:
                    # Pop a batch
                    batch = []
                    for _ in range(min(batch_size, len(queue))):
                        batch.append(queue.pop())
                    in_flight_batch = list(batch)

                    print(f"\nBatch of {len(batch)} | Queue remaining: {len(queue)} | "
                          f"Completed: {len(queue.completed_traces)} | Failed: {len(queue.failed_tasks)}")

                    # For failed traces with per-item max_turns, temporarily
                    # override queue.max_turns per item in should_stop check
                    # We handle this by checking _max_turns on each item

                    # Generate completions concurrently
                    responses = await asyncio.gather(*[
                        generate_completion(item["messages"], session, headers)
                        for item in batch
                    ])

                    # Extract code, split into modal-needed vs locally-resolved
                    modal_items = []
                    resolved_items = []

                    for idx, (item, response) in enumerate(zip(batch, responses)):
                        if not response or not response.get("content"):
                            result = {"correctness": False, "error": "Generation failed"}
                            resolved_items.append((idx, item, result, None, None))
                            continue

                        completion = response["content"]
                        reasoning = response.get("reasoning")
                        triton_code = extract_triton_code(completion)

                        if not triton_code:
                            result = {"correctness": False, "error": "Triton code extraction failed"}
                            resolved_items.append((idx, item, result, completion, reasoning))
                        else:
                            pre_error = pre_validate_triton_code(triton_code)
                            if pre_error:
                                print(f"  {item['sample_key']} turn {item['turn_num']} pre-validation: {pre_error[:80]}")
                                result = {"correctness": False, "speedup": 0.0, "error": pre_error}
                                resolved_items.append((idx, item, result, completion, reasoning))
                            else:
                                modal_items.append((idx, item, triton_code, completion, reasoning))

                    # Validate on Modal in parallel
                    modal_results = {}
                    for chunk_start in range(0, len(modal_items), modal_parallel):
                        chunk = modal_items[chunk_start:chunk_start + modal_parallel]
                        print(f"  Validating {len(chunk)} kernels on Modal H100...")

                        starmap_inputs = [
                            (
                                triton_code,
                                item["pytorch_code"],
                                5,       # n_correctness
                                20,      # n_trials
                                item["sample"].get("name", "generated_kernel"),
                                item["sample"].get("entry_point", "Model"),
                                1e-4,    # rtol
                                1e-4,    # atol
                            )
                            for _, item, triton_code, _, _ in chunk
                        ]

                        results_iter = await asyncio.to_thread(
                            lambda inputs=starmap_inputs: list(
                                benchmark_kernelbench.starmap(
                                    inputs,
                                    return_exceptions=True,
                                    wrap_returned_exceptions=False,
                                )
                            )
                        )

                        for (idx, item, triton_code, _, _), result in zip(chunk, results_iter):
                            if isinstance(result, Exception):
                                modal_results[idx] = {
                                    "correctness": False,
                                    "speedup": 0.0,
                                    "error": str(result),
                                }
                            else:
                                modal_results[idx] = result

                        print(f"  Modal chunk done ({len(chunk)} results)")

                    # Merge all results
                    all_items = []
                    for idx, item, result, completion, reasoning in resolved_items:
                        all_items.append((idx, item, result, None, completion, reasoning))
                    for idx, item, triton_code, completion, reasoning in modal_items:
                        result = modal_results[idx]
                        all_items.append((idx, item, result, triton_code, completion, reasoning))
                    all_items.sort(key=lambda x: x[0])

                    # Route results
                    for _, item, result, triton_code, completion, reasoning in all_items:
                        turn_result = {
                            "turn": item["turn_num"],
                            "reasoning": reasoning,
                            "triton_code": triton_code,
                            "full_completion": completion,
                            "result": result,
                            "feedback_given": None,
                        }
                        item["turns_history"].append(turn_result)

                        # Per-item max_turns override for failed traces
                        effective_max = item.get("_max_turns", queue.max_turns)
                        if item["turn_num"] >= effective_max:
                            stop, reason = True, "max_turns_reached"
                        else:
                            stop, reason = queue.should_stop(item["turn_num"], result)

                        if stop:
                            queue.finalize(item, reason)
                            correctness = result.get("correctness", False)
                            speedup = result.get("speedup", 0.0)
                            print(f"  {item['sample_key']} DONE ({reason}): "
                                  f"correct={correctness}, speedup={speedup:.2f}x, "
                                  f"turns={item['turn_num']}")
                        else:
                            feedback = queue.build_feedback(result)
                            turn_result["feedback_given"] = feedback
                            queue.requeue_with_feedback(item, feedback, completion or "")
                            print(f"  {item['sample_key']} turn {item['turn_num'] - 1} -> retrying")

                    in_flight_batch = []

            finally:
                print(f"\n{phase_name} summary: "
                      f"{len(queue.completed_traces)} completed, "
                      f"{len(queue.failed_tasks)} failed")

    # Separate successful vs failed
    successful = [
        t for t in queue.completed_traces
        if t.get("stop_reason") == "success_fast"
    ]
    failed = [
        t for t in queue.completed_traces
        if t.get("stop_reason") != "success_fast"
    ] + queue.failed_tasks

    return successful, failed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run_incomplete(batch_size: int, modal_parallel: int):
    """Phase 1: Resume incomplete traces."""
    items = build_incomplete_queue_items(INCOMPLETE_MAX_TURNS)
    successful, failed = await process_phase(
        items,
        max_turns=INCOMPLETE_MAX_TURNS,
        batch_size=batch_size,
        modal_parallel=modal_parallel,
        phase_name="INCOMPLETE TRACES",
    )

    if successful:
        append_traces(QWEN_TRACES_FILE, successful)
        append_traces(CORRECT_TRACES_FILE, successful)
    if failed:
        append_traces(FAILED_FILE, failed)

    return successful, failed


async def run_failed(batch_size: int, modal_parallel: int):
    """Phase 2: Continue failed traces with additional turns."""
    items = build_failed_queue_items(FAILED_EXTRA_TURNS)

    # max_turns is per-item (_max_turns), but we set queue max to the highest
    max_possible = max((it.get("_max_turns", 7) for it in items), default=7)
    successful, failed = await process_phase(
        items,
        max_turns=max_possible,
        batch_size=batch_size,
        modal_parallel=modal_parallel,
        phase_name="FAILED TRACES (continuation)",
    )

    if successful:
        append_traces(QWEN_TRACES_FILE, successful)
        append_traces(CORRECT_TRACES_FILE, successful)
    if failed:
        # Update the failed file (these are still unsolved after extra turns)
        append_traces(FAILED_FILE, failed)

    return successful, failed


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate traces via OpenRouter API"
    )
    parser.add_argument(
        "--phase",
        choices=["incomplete", "failed", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Concurrent API requests per batch (default: 4, keep low for rate limits)",
    )
    parser.add_argument(
        "--modal-parallel",
        type=int,
        default=16,
        help="Max parallel Modal validation calls (default: 16)",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        print("  export OPENROUTER_API_KEY='sk-or-...'")
        sys.exit(1)

    if args.phase in ("incomplete", "all"):
        s, f = await run_incomplete(args.batch_size, args.modal_parallel)
        print(f"\nIncomplete phase: {len(s)} successful, {len(f)} failed")

    if args.phase in ("failed", "all"):
        s, f = await run_failed(args.batch_size, args.modal_parallel)
        print(f"\nFailed phase: {len(s)} successful, {len(f)} failed")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())

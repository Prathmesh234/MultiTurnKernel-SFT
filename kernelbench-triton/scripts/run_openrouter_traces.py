#!/usr/bin/env python3
"""
Generate additional reasoning traces using OpenRouter API.

Continues failed traces (+3 additional turns on top of original 4 = 7 max)

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python scripts/run_openrouter_traces.py
"""

import json
import os
import sys
import re
import asyncio
from pathlib import Path
from typing import Optional

import openai
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from multi_turn_queue import MultiTurnQueue
from utilities import pre_validate_triton_code, extract_triton_code
from modal_app import app as modal_app, benchmark_kernelbench

MAX_MODEL_LEN = 131072       # Must match --max-model-len on vLLM server (Qwen3-235B = 131072)
MAX_COMPLETION_TOKENS = 32768  # Upper bound; dynamically capped per request
TEMPERATURE = 0.7

# ---------------------------------------------------------------------------
# OpenRouter config
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "qwen/qwen3-235b-a22b-thinking-2507"

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
FAILED_FILE = PROJECT_ROOT / "traces" / "reasoning_traces_multiturn_failed.json"
QWEN_TRACES_FILE = PROJECT_ROOT / "traces" / "reasoning_traces_multiturn_qwen.json"
CORRECT_TRACES_FILE = PROJECT_ROOT / "traces" / "reasoning_traces_correct_multiturn.json"

FAILED_EXTRA_TURNS = 3


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
    """Append or update traces in a JSON file.

    If a trace with the same sample_key already exists, it is replaced
    (so that continuation turns overwrite the old entry).  Otherwise the
    new trace is appended.
    """
    existing = load_json(path)
    key_to_idx = {t["sample_key"]: i for i, t in enumerate(existing)}
    added, updated = 0, 0
    for t in new_traces:
        if t["sample_key"] in key_to_idx:
            existing[key_to_idx[t["sample_key"]]] = t
            updated += 1
        else:
            key_to_idx[t["sample_key"]] = len(existing)
            existing.append(t)
            added += 1
    save_json(path, existing)
    print(f"  Saved to {path.name}: {added} added, {updated} updated (total: {len(existing)})")


def remove_traces(path: Path, keys_to_remove: set[str]):
    """Remove traces by sample_key from a JSON file."""
    if not keys_to_remove:
        return
    existing = load_json(path)
    before = len(existing)
    existing = [t for t in existing if t["sample_key"] not in keys_to_remove]
    removed = before - len(existing)
    if removed > 0:
        save_json(path, existing)
        print(f"  Removed {removed} solved traces from {path.name} (remaining: {len(existing)})")


# ---------------------------------------------------------------------------
# OpenRouter API
# ---------------------------------------------------------------------------

async def generate_completion(
    messages: list[dict],
    client: AsyncOpenAI,
    retries: int = 3,
) -> Optional[dict]:
    total_input_chars = sum(
        len(m["content"]) if isinstance(m["content"], str)
        else sum(len(b.get("text", "")) for b in m["content"])
        for m in messages
    )
    estimated_input_tokens = int(total_input_chars / 3.5) + 50
    max_tokens = min(MAX_COMPLETION_TOKENS, MAX_MODEL_LEN - estimated_input_tokens)
    max_tokens = max(max_tokens, 1024)

    # Add cache_control to system prompt (messages[0]) and first user message (messages[1]).
    # These are the largest repeated blocks — system prompt is identical across all requests,
    # first user message (pytorch code) is identical across all turns of the same problem.
    cached_messages = []
    for i, msg in enumerate(messages):
        if i < 2:
            cached_messages.append({
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"], "cache_control": {"type": "ephemeral"}}],
            })
        else:
            cached_messages.append(msg)

    for attempt in range(retries):
        try:
            completion = await client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=cached_messages,
                max_tokens=max_tokens,
                temperature=TEMPERATURE,
                extra_body={
                    "provider": {
                        "order": ["DeepInfra"],
                        "allow_fallbacks": False,
                        "quantizations": ["fp8"],
                    },
                },
                timeout=900,
            )
            message = completion.choices[0].message
            raw_content = message.content or ""

            reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
            if not reasoning and "</think>" in raw_content:
                parts = raw_content.split("</think>", 1)
                reasoning = parts[0].strip()
                raw_content = parts[1].strip()
            if not reasoning and "<think>" in raw_content:
                think_match = re.search(r"<think>(.*?)</think>", raw_content, re.DOTALL)
                if think_match:
                    reasoning = think_match.group(1).strip()
                    raw_content = raw_content[think_match.end():].strip()

            return {"content": raw_content, "reasoning": reasoning}

        except openai.RateLimitError as e:
            wait = 2 ** (attempt + 1)
            print(f"  Rate limited (429), retrying in {wait}s ... ({str(e)[:120]})")
            await asyncio.sleep(wait)
        except Exception as e:
            print(f"  Request failed: {type(e).__name__}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return None

    return None


# ---------------------------------------------------------------------------
# Failed traces
# ---------------------------------------------------------------------------

def build_failed_queue_items(extra_turns: int) -> list[dict]:
    raw = load_json(FAILED_FILE)
    if not raw:
        print("No failed traces found.")
        return []

    queue = MultiTurnQueue()
    items = []
    for trace in raw:
        original_turns = trace["num_turns"]
        messages = list(trace["full_messages"])
        turns = trace.get("turns", [])

        if turns:
            last_turn = turns[-1]
            if last_turn.get("full_completion"):
                messages.append({"role": "assistant", "content": last_turn["full_completion"]})
                messages.append({"role": "user", "content": queue.build_feedback(last_turn.get("result", {}))})

        items.append({
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
            "turn_num": original_turns + 1,
            "turns_history": list(turns),
            "_max_turns": original_turns + extra_turns,
            "requeued_after_failure": True,  # Prevent finalize() from resetting and requeuing
        })

    print(f"Loaded {len(items)} failed traces for continuation (+{extra_turns} turns)")
    return items


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

async def process_phase(
    items: list[dict],
    max_turns: int,
    batch_size: int,
    modal_parallel: int,
    phase_name: str,
    success_paths: list[Path],
    failed_path: Path,
) -> tuple[list[dict], list[dict]]:
    if not items:
        return [], []

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")

    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )

    queue = MultiTurnQueue(max_turns=max_turns)
    for item in items:
        queue.add(item)

    print(f"\n{'='*60}")
    print(f"PHASE: {phase_name}")
    print(f"{'='*60}")
    print(f"Queue size: {len(queue)}, max_turns: {max_turns}, batch_size: {batch_size}")

    with modal_app.run():
        in_flight_batch: list[dict] = []
        try:
            while len(queue) > 0:
                batch = [queue.pop() for _ in range(min(batch_size, len(queue)))]
                in_flight_batch = list(batch)

                print(f"\nBatch of {len(batch)} | Queue remaining: {len(queue)} | "
                      f"Completed: {len(queue.completed_traces)} | Failed: {len(queue.failed_tasks)}")

                responses = await asyncio.gather(*[
                    generate_completion(item["messages"], client)
                    for item in batch
                ])

                modal_items = []
                resolved_items = []

                for idx, (item, response) in enumerate(zip(batch, responses)):
                    if not response or not response.get("content"):
                        resolved_items.append((idx, item, {"correctness": False, "error": "Generation failed"}, None, None))
                        continue

                    completion = response["content"]
                    reasoning = response.get("reasoning")
                    triton_code = extract_triton_code(completion)

                    if not triton_code:
                        resolved_items.append((idx, item, {"correctness": False, "error": "Triton code extraction failed"}, completion, reasoning))
                    else:
                        pre_error = pre_validate_triton_code(triton_code)
                        if pre_error:
                            print(f"  {item['sample_key']} turn {item['turn_num']} pre-validation: {pre_error[:80]}")
                            resolved_items.append((idx, item, {"correctness": False, "speedup": 0.0, "error": pre_error}, completion, reasoning))
                        else:
                            modal_items.append((idx, item, triton_code, completion, reasoning))

                modal_results = {}
                for chunk_start in range(0, len(modal_items), modal_parallel):
                    chunk = modal_items[chunk_start:chunk_start + modal_parallel]
                    print(f"  Validating {len(chunk)} kernels on Modal H100...")

                    starmap_inputs = [
                        (
                            triton_code,
                            item["pytorch_code"],
                            5, 20,
                            item["sample"].get("name", "generated_kernel"),
                            item["sample"].get("entry_point", "Model"),
                            1e-4, 1e-4,
                        )
                        for _, item, triton_code, _, _ in chunk
                    ]

                    results_iter = await asyncio.to_thread(
                        lambda inputs=starmap_inputs: list(
                            benchmark_kernelbench.starmap(inputs, return_exceptions=True, wrap_returned_exceptions=False)
                        )
                    )

                    for (idx, item, triton_code, _, _), result in zip(chunk, results_iter):
                        modal_results[idx] = (
                            {"correctness": False, "speedup": 0.0, "error": str(result)}
                            if isinstance(result, Exception) else result
                        )

                    print(f"  Modal chunk done ({len(chunk)} results)")

                all_items = sorted(
                    [(idx, item, result, None, completion, reasoning) for idx, item, result, completion, reasoning in resolved_items] +
                    [(idx, item, modal_results[idx], triton_code, completion, reasoning) for idx, item, triton_code, completion, reasoning in modal_items],
                    key=lambda x: x[0],
                )

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

                    # Check success FIRST — a correct+fast result on the last turn
                    # should still be classified as success_fast, not max_turns_reached
                    if result.get("correctness") and result.get("speedup", 0) >= 1.0:
                        stop, reason = True, "success_fast"
                    else:
                        effective_max = item.get("_max_turns", queue.max_turns)
                        if item["turn_num"] >= effective_max:
                            stop, reason = True, "max_turns_reached"
                        else:
                            stop, reason = False, "continue"

                    if stop:
                        prev_completed = len(queue.completed_traces)
                        prev_failed = len(queue.failed_tasks)
                        queue.finalize(item, reason)
                        if len(queue.completed_traces) > prev_completed:
                            trace = queue.completed_traces[-1]
                            if trace["stop_reason"] == "success_fast":
                                for path in success_paths:
                                    append_traces(path, [trace])
                                # Remove from failed file since it's now solved
                                remove_traces(failed_path, {trace["sample_key"]})
                            else:
                                append_traces(failed_path, [trace])
                        elif len(queue.failed_tasks) > prev_failed:
                            append_traces(failed_path, [queue.failed_tasks[-1]])
                        print(f"  {item['sample_key']} DONE ({reason}): "
                              f"correct={result.get('correctness', False)}, speedup={result.get('speedup', 0.0):.2f}x, "
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

    successful = [t for t in queue.completed_traces if t.get("stop_reason") == "success_fast"]
    failed = [t for t in queue.completed_traces if t.get("stop_reason") != "success_fast"] + queue.failed_tasks
    return successful, failed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run_failed(batch_size: int, modal_parallel: int):
    items = build_failed_queue_items(FAILED_EXTRA_TURNS)
    max_possible = max((it.get("_max_turns", 7) for it in items), default=7)
    successful, failed = await process_phase(
        items, max_possible, batch_size, modal_parallel, "FAILED TRACES (continuation)",
        success_paths=[QWEN_TRACES_FILE, CORRECT_TRACES_FILE],
        failed_path=FAILED_FILE,
    )
    return successful, failed


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate traces via OpenRouter API")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--modal-parallel", type=int, default=16)
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        print("  export OPENROUTER_API_KEY='sk-or-...'")
        sys.exit(1)

    s, f = await run_failed(args.batch_size, args.modal_parallel)
    print(f"\nFailed phase: {len(s)} successful, {len(f)} failed")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())

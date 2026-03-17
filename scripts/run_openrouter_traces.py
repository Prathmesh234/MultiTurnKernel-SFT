#!/usr/bin/env python3
"""
Generate additional reasoning traces using OpenRouter API.

Phase 1: Resume incomplete traces (max 4 total turns)
Phase 2: Continue failed traces (+3 additional turns on top of original 4 = 7 max)

Subclasses TraceOrchestrator — only overrides __init__ (skip dataloader) and
generate_completion (OpenRouter endpoint). Reuses extract_triton_code,
validate_on_modal, Modal starmap, and the full batch processing pipeline.

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
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from multi_turn_queue import MultiTurnQueue
from utilities import pre_validate_triton_code
import modal
from modal_app import app as modal_app, benchmark_kernelbench
from orchestrator import (
    TraceOrchestrator,
    SYSTEM_PROMPT,
    MULTI_TURN_ADDENDUM,
    USER_PROMPT_TEMPLATE,
    OUTPUT_FILE_MULTITURN,
    MAX_MODEL_LEN,
    MAX_COMPLETION_TOKENS,
    TEMPERATURE,
)

# ---------------------------------------------------------------------------
# OpenRouter config — same model as vLLM, zero token variance
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "qwen/qwen3-235b-a22b-thinking-2507"

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
INCOMPLETE_FILE = PROJECT_ROOT / "incomplete" / "incomplete_traces.json"
FAILED_FILE = PROJECT_ROOT / "traces" / "reasoning_traces_multiturn_failed.json"
QWEN_TRACES_FILE = PROJECT_ROOT / "traces" / "reasoning_traces_multiturn_qwen.json"
CORRECT_TRACES_FILE = PROJECT_ROOT / "traces" / "reasoning_traces_multiturn.json"

# Phase limits
INCOMPLETE_MAX_TURNS = 4
FAILED_EXTRA_TURNS = 3


# ---------------------------------------------------------------------------
# Subclass — only generate_completion differs (OpenRouter vs local vLLM)
# ---------------------------------------------------------------------------

class OpenRouterOrchestrator(TraceOrchestrator):
    """TraceOrchestrator with OpenRouter as the LLM backend."""

    def __init__(self):
        # Bypass parent __init__ (no dataloader/output file needed)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY environment variable")

        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # Parent fields that inherited methods may reference
        self.vllm_base_url = OPENROUTER_BASE_URL
        self.traces = []
        self.processed_keys = set()

    async def generate_completion(
        self,
        messages: list[dict],
        session: aiohttp.ClientSession,
        retries: int = 3,
    ) -> Optional[dict]:
        """
        Override: hit OpenRouter instead of local vLLM.

        Same payload shape (OpenAI-compatible), adds auth headers and
        retry-with-backoff for 429 rate limits.
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
                    headers=self._headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=900),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        message = data["choices"][0]["message"]
                        raw_content = message.get("content") or ""

                        # Reasoning extraction — handle OpenRouter + vLLM variants
                        reasoning = (
                            message.get("reasoning")
                            or message.get("reasoning_content")
                        )
                        if not reasoning and "</think>" in raw_content:
                            parts = raw_content.split("</think>", 1)
                            reasoning = parts[0].strip()
                            raw_content = parts[1].strip()
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
                        wait = 2 ** (attempt + 1)
                        print(f"  Rate limited (429), retrying in {wait}s ...")
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


# ---------------------------------------------------------------------------
# File helpers
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
    """Append new traces (deduplicated by sample_key)."""
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
# Build queue items from existing trace files
# ---------------------------------------------------------------------------

def build_incomplete_items(max_turns: int) -> list[dict]:
    """Load incomplete traces → queue items (messages ready for next generation)."""
    raw = load_json(INCOMPLETE_FILE)
    items = []
    for trace in raw:
        if trace["current_turn"] > max_turns:
            continue
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
            "messages": trace["full_messages"],
            "turn_num": trace["current_turn"],
            "turns_history": trace.get("turns", []),
        })
    print(f"Loaded {len(items)} incomplete traces (max_turns={max_turns})")
    return items


def build_failed_items(extra_turns: int) -> list[dict]:
    """
    Load failed traces → queue items with conversation reconstructed.

    full_messages ends at turn N-1 feedback. We append turn N's assistant
    response + build fresh feedback so the model can continue.
    """
    raw = load_json(FAILED_FILE)

    # Skip already-solved keys
    solved_keys = {
        t["sample_key"]
        for t in load_json(QWEN_TRACES_FILE)
        if t.get("stop_reason") == "success_fast"
    }
    solved_keys |= {t["sample_key"] for t in load_json(CORRECT_TRACES_FILE)}

    feedback_builder = MultiTurnQueue()  # Only used for build_feedback()
    items = []
    skipped = 0

    for trace in raw:
        if trace["sample_key"] in solved_keys:
            skipped += 1
            continue

        original_turns = trace["num_turns"]
        turns = trace.get("turns", [])

        # Reconstruct: append last turn's response + feedback
        messages = list(trace["full_messages"])
        if turns:
            last_turn = turns[-1]
            if last_turn.get("full_completion"):
                messages.append({"role": "assistant", "content": last_turn["full_completion"]})
                messages.append({"role": "user", "content": feedback_builder.build_feedback(last_turn.get("result", {}))})

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
        })

    print(f"Loaded {len(items)} failed traces (+{extra_turns} turns, skipped {skipped} solved)")
    return items


# ---------------------------------------------------------------------------
# Batch processing loop — uses inherited orchestrator methods
# ---------------------------------------------------------------------------

async def process_items(
    orchestrator: OpenRouterOrchestrator,
    items: list[dict],
    max_turns: int,
    batch_size: int,
    modal_parallel: int,
    phase_name: str,
) -> tuple[list[dict], list[dict]]:
    """
    Run multi-turn pipeline on pre-built queue items.

    Uses orchestrator.extract_triton_code() and orchestrator.validate_on_modal()
    from the parent class. Modal validation uses .starmap() for parallelism
    (same as TraceOrchestrator.run_multi_turn).
    """
    if not items:
        return [], []

    queue = MultiTurnQueue(max_turns=max_turns)
    for item in items:
        queue.add(item)

    print(f"\n{'='*60}")
    print(f"PHASE: {phase_name}")
    print(f"{'='*60}")
    print(f"Queue: {len(queue)}, max_turns: {max_turns}, batch_size: {batch_size}")

    with modal_app.run():
        async with aiohttp.ClientSession() as session:
            try:
                while len(queue) > 0:
                    batch = [queue.pop() for _ in range(min(batch_size, len(queue)))]

                    print(f"\nBatch {len(batch)} | Queue: {len(queue)} | "
                          f"Done: {len(queue.completed_traces)} | Failed: {len(queue.failed_tasks)}")

                    # --- Generate completions (OpenRouter) ---
                    responses = await asyncio.gather(*[
                        orchestrator.generate_completion(item["messages"], session)
                        for item in batch
                    ])

                    # --- Extract code, pre-validate locally ---
                    modal_items = []
                    resolved_items = []

                    for idx, (item, resp) in enumerate(zip(batch, responses)):
                        if not resp or not resp.get("content"):
                            resolved_items.append((idx, item, {"correctness": False, "error": "Generation failed"}, None, None))
                            continue

                        completion = resp["content"]
                        reasoning = resp.get("reasoning")
                        triton_code = orchestrator.extract_triton_code(completion)

                        if not triton_code:
                            resolved_items.append((idx, item, {"correctness": False, "error": "Triton extraction failed"}, completion, reasoning))
                        else:
                            pre_error = pre_validate_triton_code(triton_code)
                            if pre_error:
                                print(f"  {item['sample_key']} t{item['turn_num']} pre-val: {pre_error[:80]}")
                                resolved_items.append((idx, item, {"correctness": False, "speedup": 0.0, "error": pre_error}, completion, reasoning))
                            else:
                                modal_items.append((idx, item, triton_code, completion, reasoning))

                    # --- Validate on Modal via .starmap() (inherited pattern) ---
                    modal_results = {}
                    for chunk_start in range(0, len(modal_items), modal_parallel):
                        chunk = modal_items[chunk_start:chunk_start + modal_parallel]
                        print(f"  Validating {len(chunk)} kernels on Modal H100...")

                        starmap_inputs = [
                            (tc, it["pytorch_code"], 5, 20,
                             it["sample"].get("name", "generated_kernel"),
                             it["sample"].get("entry_point", "Model"), 1e-4, 1e-4)
                            for _, it, tc, _, _ in chunk
                        ]
                        results_iter = await asyncio.to_thread(
                            lambda inputs=starmap_inputs: list(
                                benchmark_kernelbench.starmap(inputs, return_exceptions=True, wrap_returned_exceptions=False)
                            )
                        )
                        for (idx, *_rest), result in zip(chunk, results_iter):
                            if isinstance(result, Exception):
                                modal_results[idx] = {"correctness": False, "speedup": 0.0, "error": str(result)}
                            else:
                                modal_results[idx] = result

                    # --- Route results through MultiTurnQueue ---
                    all_results = (
                        [(idx, it, res, None, comp, reas) for idx, it, res, comp, reas in resolved_items]
                        + [(idx, it, modal_results[idx], tc, comp, reas) for idx, it, tc, comp, reas in modal_items]
                    )
                    all_results.sort(key=lambda x: x[0])

                    for _, item, result, triton_code, completion, reasoning in all_results:
                        turn_result = {
                            "turn": item["turn_num"],
                            "reasoning": reasoning,
                            "triton_code": triton_code,
                            "full_completion": completion,
                            "result": result,
                            "feedback_given": None,
                        }
                        item["turns_history"].append(turn_result)

                        # Per-item max_turns for failed traces
                        effective_max = item.get("_max_turns", queue.max_turns)
                        if item["turn_num"] >= effective_max:
                            stop, reason = True, "max_turns_reached"
                        else:
                            stop, reason = queue.should_stop(item["turn_num"], result)

                        if stop:
                            queue.finalize(item, reason)
                            print(f"  {item['sample_key']} DONE ({reason}): "
                                  f"correct={result.get('correctness')}, "
                                  f"speedup={result.get('speedup', 0):.2f}x")
                        else:
                            feedback = queue.build_feedback(result)
                            turn_result["feedback_given"] = feedback
                            queue.requeue_with_feedback(item, feedback, completion or "")
                            print(f"  {item['sample_key']} t{item['turn_num'] - 1} -> retry")

            finally:
                print(f"\n{phase_name}: {len(queue.completed_traces)} completed, {len(queue.failed_tasks)} failed")

    successful = [t for t in queue.completed_traces if t.get("stop_reason") == "success_fast"]
    failed = [t for t in queue.completed_traces if t.get("stop_reason") != "success_fast"] + queue.failed_tasks
    return successful, failed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate traces via OpenRouter API")
    parser.add_argument("--phase", choices=["incomplete", "failed", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=4, help="Concurrent API requests (default: 4)")
    parser.add_argument("--modal-parallel", type=int, default=16, help="Max parallel Modal calls (default: 16)")
    args = parser.parse_args()

    orchestrator = OpenRouterOrchestrator()

    if args.phase in ("incomplete", "all"):
        items = build_incomplete_items(INCOMPLETE_MAX_TURNS)
        successful, failed = await process_items(
            orchestrator, items, INCOMPLETE_MAX_TURNS,
            args.batch_size, args.modal_parallel, "INCOMPLETE TRACES",
        )
        if successful:
            append_traces(QWEN_TRACES_FILE, successful)
            append_traces(CORRECT_TRACES_FILE, successful)
        if failed:
            append_traces(FAILED_FILE, failed)
        print(f"\nIncomplete: {len(successful)} successful, {len(failed)} failed")

    if args.phase in ("failed", "all"):
        items = build_failed_items(FAILED_EXTRA_TURNS)
        max_possible = max((it.get("_max_turns", 7) for it in items), default=7)
        successful, failed = await process_items(
            orchestrator, items, max_possible,
            args.batch_size, args.modal_parallel, "FAILED TRACES (continuation)",
        )
        if successful:
            append_traces(QWEN_TRACES_FILE, successful)
            append_traces(CORRECT_TRACES_FILE, successful)
        if failed:
            append_traces(FAILED_FILE, failed)
        print(f"\nFailed: {len(successful)} successful, {len(failed)} failed")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())

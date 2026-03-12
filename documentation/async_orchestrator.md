# Async Orchestrator — Continuous vLLM Feeding

## Problem

The current orchestrator has a synchronous batch loop:

```
Generate batch (128) → Wait for ALL Modal validations → Requeue → Repeat
```

This creates GPU dead time ("valleys") between rounds. When the first 128 sequences
finish generating, vLLM sits idle waiting for all 128 Modal H100 validation containers
to return results. Observed impact: throughput dropped from ~2950 tok/s to ~330 tok/s
during the Modal validation window (~9x regression).

Root cause: `asyncio.gather` on generation + synchronous wait on all Modal results
before any new work is dispatched.

---

## Fix: Fully Async Pipeline

Decouple generation from validation entirely. Keep a live "in-flight" pool of
sequences so vLLM always has `max_num_seqs` requests to process.

### Core idea

Instead of batch rounds, run two concurrent async loops:

1. **Generator loop** — constantly pops items from the queue and fires generation
   requests at vLLM, up to a concurrency cap (`max_num_seqs`). Does not wait for
   Modal validation before sending the next request.

2. **Validator loop** — as generation completes, immediately fires Modal validation
   as a background task. When Modal returns, routes the result back:
   - Success → finalize trace
   - Failure → build feedback, requeue item, generator loop picks it up immediately

```
Queue ──► Generator (async) ──► vLLM
              │                   │
              │              (response)
              │                   │
              ▼                   ▼
         In-flight pool    Validator (async, background)
                                  │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
               finalize trace              requeue w/ feedback
                                           (goes back to Queue)
```

### Key implementation changes

**Replace the while loop in `run_multi_turn`:**

```python
# Current (blocking):
while len(queue) > 0:
    batch = [queue.pop() for _ in range(min(batch_size, len(queue)))]
    responses = await asyncio.gather(*[generate(item) for item in batch])
    # ... process all responses ...
    # ... wait for all Modal validations ...
    # ... only NOW requeue ...

# New (async pipeline):
# Use an asyncio.Semaphore to cap in-flight generations at max_num_seqs
# Each item independently: generate -> validate -> (finalize | requeue)
# The semaphore ensures vLLM is always saturated without overloading it
```

**Pseudocode:**

```python
sem = asyncio.Semaphore(max_concurrent)  # e.g. 128, matches max_num_seqs

async def process_item(item):
    async with sem:
        # 1. Generate
        response = await generate_completion(item["messages"], session)

        # 2. Extract + pre-validate
        triton_code = extract_triton_code(response["content"])

        # 3. Validate on Modal (non-blocking — runs while sem releases for next item)
        result = await validate_on_modal(triton_code, ...)

        # 4. Route
        if should_stop(item["turn_num"], result):
            queue.finalize(item, reason)
        else:
            feedback = queue.build_feedback(result)
            queue.requeue_with_feedback(item, feedback, response["content"])
            # Requeued item will be picked up by the task spawner below

async def run_pipeline():
    pending = set()

    while len(queue) > 0 or pending:
        # Spawn new tasks up to semaphore limit
        while len(queue) > 0:
            item = queue.pop()
            task = asyncio.create_task(process_item(item))
            pending.add(task)

        # Wait for at least one to finish before checking queue again
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        # Save progress after each completion
        save_traces(queue.completed_traces)
```

### Why this keeps vLLM fed

- The semaphore controls how many sequences are in-flight at vLLM at once
- As soon as one sequence finishes generation (and moves to Modal validation),
  the semaphore slot frees and a new generation request is immediately dispatched
- Modal validation runs as a background coroutine — it does not block the generator
- vLLM always sees `~max_concurrent` active requests → no valleys

### Tuning

| Parameter | Recommended | Notes |
|-----------|------------|-------|
| `max_concurrent` (semaphore) | 128 | Match `max_num_seqs` on vLLM server |
| Modal validation concurrency | unbounded | Modal handles its own container scaling |
| Save interval | every completed task | Fine-grained, no data loss on crash |

### Expected outcome

- vLLM throughput sustained at ~2950 tok/s throughout the run (no valleys)
- GPU utilization stays at 100% continuously
- KV cache usage stays elevated (good — GPU is always fed)
- Total wall-clock time to generate 3500 traces significantly reduced

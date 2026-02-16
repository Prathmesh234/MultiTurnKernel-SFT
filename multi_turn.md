# Multi-Turn Reasoning & Kernel Generation Specification

## Overview

This document specifies a **two-queue architecture** for multi-turn iterative refinement of Triton kernel generation using GLM-4.5-Air. The system allows the model to self-correct and optimize kernels over multiple turns (up to 4) based on validation feedback.

## Architecture

### Queue-Based Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    PROCESSING QUEUE                          │
│  Items: {pytorch_code, messages[], turn_num, sample_key}    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  GLM-4.5    │ (vLLM server)
                    │  Generate   │
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Modal     │ (H100 validation)
                    │  Validate   │
                    └─────────────┘
                           │
                           ▼
                  ┌────────────────────┐
                  │  Decision Logic    │
                  │  - Success? → Save │
                  │  - Turn 4? → Save  │
                  │  - Else → TURN QUEUE│
                  └────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       TURN QUEUE                             │
│  Items: {pytorch_code, messages[] + feedback, turn_num+1}   │
└─────────────────────────────────────────────────────────────┘
                           │
                           └──────► Back to PROCESSING QUEUE
```

## Core Components

### 1. Item Structure

Each item flowing through the queues contains:

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

### 2. Turn History Structure

Each turn's result is stored:

```python
{
    "turn": int,
    "completion": str,           # Full model response
    "reasoning": str,            # Internal reasoning_content from GLM-4.5-Air
    "triton_code": str,          # Extracted Triton kernel
    "result": {
        "correctness": bool,
        "speedup": float,
        "error": str | None,
    },
}
```

## Implementation

### Orchestrator Class Structure

```python
class MultiTurnOrchestrator:
    def __init__(self, max_turns=4):
        self.max_turns = max_turns
        self.processing_queue = asyncio.Queue()
        self.turn_queue = asyncio.Queue()
        self.completed_traces = []
        
        # Track active samples to prevent duplicates
        self.active_samples = set()
        
    async def run(self, batch_size=10):
        """Main entry point - starts all workers"""
        workers = [
            # Generation workers (call GLM-4.5-Air)
            *[self.generation_worker(i) for i in range(5)],
            
            # Validation workers (call Modal H100)
            *[self.validation_worker(i) for i in range(3)],
            
            # Turn router (decides: save or continue?)
            self.turn_router_worker(),
        ]
        
        # Seed initial samples
        await self._seed_initial_samples()
        
        # Run all workers concurrently
        await asyncio.gather(*workers)
```

### Worker: Generation

Pulls from processing queue, generates completion, validates:

```python
async def generation_worker(self, worker_id: int):
    """Generate Triton kernel from PyTorch code"""
    async with aiohttp.ClientSession() as session:
        while True:
            item = await self.processing_queue.get()
            
            try:
                # Call GLM-4.5-Air with current message history
                response = await self.generate_completion(
                    item["messages"], 
                    session
                )
                
                # Extract Triton code
                triton_code = self.extract_triton_code(response["content"])
                thinking = self.extract_thinking(response["content"])
                
                # Validate on Modal H100
                result = await self.validate_on_modal(
                    triton_code, 
                    item["pytorch_code"], 
                    item["sample"]
                )
                
                # Package turn result
                turn_result = {
                    "turn": item["turn_num"],
                    "completion": response["content"],
                    "reasoning": response["reasoning"],
                    "triton_code": triton_code,
                    "thinking": thinking,
                    "result": result,
                }
                
                # Add to history
                item["turns_history"].append(turn_result)
                
                # Send to turn router for decision
                await self.turn_queue.put(item)
                
            except Exception as e:
                print(f"Worker {worker_id} failed: {e}")
                # Mark as failed and save
                item["turns_history"].append({
                    "turn": item["turn_num"],
                    "error": str(e),
                })
                await self.turn_queue.put(item)
            finally:
                self.processing_queue.task_done()
```

### Worker: Turn Router

Decides whether to continue or save:

```python
async def turn_router_worker(self):
    """Routes items: either back to processing queue or to completed"""
    while True:
        item = await self.turn_queue.get()
        
        try:
            last_turn = item["turns_history"][-1]
            result = last_turn.get("result", {})
            
            # Check stop conditions
            should_stop, reason = self._should_stop(
                item["turn_num"], 
                result
            )
            
            if should_stop:
                # Save trace
                trace = self._build_final_trace(item, reason)
                self.completed_traces.append(trace)
                self.active_samples.discard(item["sample_key"])
                self._save_traces()
                
                print(f"✓ {item['sample_key']} completed: {reason}")
            else:
                # Build feedback and continue
                feedback = self._build_feedback(result)
                
                # Append assistant's response + user feedback to messages
                item["messages"].append({
                    "role": "assistant",
                    "content": last_turn["completion"]
                })
                item["messages"].append({
                    "role": "user",
                    "content": feedback
                })
                
                # Increment turn and re-queue
                item["turn_num"] += 1
                await self.processing_queue.put(item)
                
                print(f"↻ {item['sample_key']} → Turn {item['turn_num']}")
                
        except Exception as e:
            print(f"Turn router error: {e}")
        finally:
            self.turn_queue.task_done()
```

## Stop Conditions

```python
def _should_stop(self, turn_num: int, result: dict) -> tuple[bool, str]:
    """Determine if we should stop iterating"""
    
    # Hard limit: max turns reached
    if turn_num >= self.max_turns:
        return True, "max_turns_reached"
    
    # Success: correct AND faster than PyTorch
    if result.get("correctness") and result.get("speedup", 0) >= 1.0:
        return True, "success_fast"
    
    # Partial success: correct but slow (after turn 2)
    if result.get("correctness") and turn_num >= 2:
        return True, "success_correct_only"
    
    # Continue iterating
    return False, "continue"
```

## Feedback Templates

Different feedback based on failure mode:

```python
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

def _build_feedback(self, result: dict) -> str:
    """Select appropriate feedback template"""
    error = result.get("error")
    correctness = result.get("correctness", False)
    speedup = result.get("speedup", 0.0)
    
    if error:
        # Compilation or runtime error
        return FEEDBACK_COMPILATION_FAILED.format(error_message=error)
    elif not correctness:
        # Incorrect output
        return FEEDBACK_CORRECTNESS_FAILED.format(
            error_message="Output mismatch with PyTorch reference"
        )
    elif speedup < 1.0:
        # Correct but slow
        return FEEDBACK_NEEDS_OPTIMIZATION.format(speedup=speedup)
    else:
        # Should not reach here (would have stopped)
        return "Great work! The kernel is correct and fast."
```

## Final Trace Schema

Each completed sample produces a trace with the following structure. **Each turn's data is wrapped with `<turn>` tags** to clearly differentiate between iterations:

```python
trace = {
    "sample_key": str,
    "source": str,              # "kernelbook" or "kernelbench"
    "pytorch_code": str,
    
    # Multi-turn metadata
    "num_turns": int,           # How many turns it took
    "stop_reason": str,         # Why we stopped
    
    # Final result (from last successful turn)
    "triton_code": str,
    "thinking": str,
    "model_reasoning": str,
    "full_completion": str,
    "result": {
        "correctness": bool,
        "speedup": float,
        "fast_0": bool,
        "fast_1": bool,
        "fast_2": bool,
        "error": str | None,
    },
    
    # Turn-by-turn history (each turn wrapped with <turn> tags)
    "turns": [
        "<turn>",
        {
            "turn": 1,
            "thinking": str,
            "model_reasoning": str,
            "triton_code": str,
            "full_completion": str,
            "result": {
                "correctness": bool,
                "speedup": float,
                "error": str | None
            }
        },
        "</turn>",
        "<turn>",
        {
            "turn": 2,
            "thinking": str,
            "model_reasoning": str,
            "triton_code": str,
            "full_completion": str,
            "result": {
                "correctness": bool,
                "speedup": float,
                "error": str | None
            }
        },
        "</turn>",
        # ... up to 4 turns
    ],
    
    # Full conversation history for GLM-4.5-Air (for SFT training)
    "full_messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...", "reasoning": "..."},
        {"role": "user", "content": "FEEDBACK: ..."},
        {"role": "assistant", "content": "...", "reasoning": "..."},
        # ... up to 4 turns
    ],
    
    # Timestamp
    "timestamp": str,
}
```

**Note on `<turn>` Tags:**
- Each turn's JSON object is wrapped with string literals `"<turn>"` and `"</turn>"` in the `turns` array
- This matches the format used in the current single-turn `reasoning_traces_glm45.json` file
- Makes it easy to visually parse and differentiate between turns when inspecting the JSON
- The tags are stored as separate string elements in the array, not as part of the JSON object itself

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

**Dynamic max_tokens calculation:**

```python
async def generate_completion(self, messages, session):
    # Estimate input tokens
    total_input_chars = sum(len(m["content"]) for m in messages)
    estimated_input_tokens = int(total_input_chars / 3.5) + 50
    
    # Leave room for generation
    max_tokens = min(
        MAX_COMPLETION_TOKENS,  # 32768
        MAX_MODEL_LEN - estimated_input_tokens
    )
    max_tokens = max(max_tokens, 1024)  # Floor
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
    }
    # ... rest of API call
```

## Seeding Initial Samples

```python
async def _seed_initial_samples(self):
    """Load samples and add to processing queue"""
    samples = self.dataloader.load_all()
    
    for sample in samples:
        sample_key = self._get_sample_key(sample)
        
        # Skip if already processed
        if sample_key in self.active_samples:
            continue
        
        # Create initial item
        item = {
            "sample_key": sample_key,
            "sample": sample,
            "pytorch_code": sample["pytorch_code"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                    pytorch_code=sample["pytorch_code"]
                )},
            ],
            "turn_num": 1,
            "turns_history": [],
        }
        
        self.active_samples.add(sample_key)
        await self.processing_queue.put(item)
```

## Benefits of This Architecture

1. **Parallelism**: Multiple workers process different samples concurrently
2. **Self-Correction**: Model learns from validation feedback
3. **Rich Training Data**: Multi-turn traces show reasoning progression
4. **Scalability**: Easy to add more workers or adjust batch sizes
5. **Resilience**: Failed items don't block the pipeline
6. **Observability**: Clear separation of concerns (generate → validate → route)

## Configuration

```python
# orchestrator.py
MAX_TURNS = 4
GENERATION_WORKERS = 5  # Parallel GLM-4.5-Air calls
VALIDATION_WORKERS = 3  # Parallel Modal H100 validations
BATCH_SIZE = 10         # Initial samples to seed
```

## Usage

```bash
# Start vLLM server (already running via run_glm45_air.sh)
# Then run orchestrator with multi-turn enabled:
uv run --no-sync python orchestrator.py \
    --output reasoning_traces_multiturn.json \
    --max-turns 4 \
    --batch-size 10
```

## Future Enhancements

1. **Adaptive turn limits**: Allow more turns for harder samples
2. **Turn budget**: Stop early if same error repeats 2+ times
3. **Feedback diversity**: Randomly sample from multiple feedback templates
4. **Checkpointing**: Save queue state for resume after crash
5. **Metrics dashboard**: Real-time view of queue depths and success rates

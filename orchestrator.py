"""
Orchestrator for generating reasoning traces with GLM-4.5-Air.

This script:
1. Loads PyTorch code from KernelBook and KernelBench via dataloader
2. Sends each problem to GLM-4.5-Air (running locally via vLLM)
3. Extracts the generated Triton kernel code
4. Validates correctness + measures speedup on Modal H100
5. Saves verified traces to reasoning_traces.json
"""

import json
import os
import asyncio
import aiohttp
from datetime import datetime
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from dataloader import KernelDataLoader

# Import Modal app and function for remote execution
import modal
from modal_app import app as modal_app, benchmark_kernelbench

# Configuration
VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "zai-org/GLM-4.5-Air"
OUTPUT_FILE = "reasoning_traces.json"
MAX_MODEL_LEN = 32768  # Must match --max-model-len on vLLM server
MAX_COMPLETION_TOKENS = 32768  # Upper bound; dynamically capped per request
TEMPERATURE = 0.7

SYSTEM_PROMPT = """You are an expert GPU kernel engineer. Your task is to convert PyTorch code into optimized Triton kernels.

=== TRITON PRIMER: CORE CONCEPTS ===

1. SPMD PROGRAMMING MODEL
   Triton uses Single Program, Multiple Data: the SAME kernel code runs on many GPU threads simultaneously.
   Each thread (program instance) processes a different portion of data.
   
   Key insight: Use tl.program_id() to determine which data THIS instance should process.
   
   Example: To process array of size N with BLOCK_SIZE per program:
   - Program 0 processes elements [0, BLOCK_SIZE)
   - Program 1 processes elements [BLOCK_SIZE, 2*BLOCK_SIZE)
   - etc.

2. COMPILE-TIME vs RUNTIME
   Triton kernels are COMPILED before execution. Some values must be known at compile-time.
   
   COMPILE-TIME (tl.constexpr):
   - BLOCK_SIZE, num_warps, num_stages
   - Arguments to tl.arange(start, end) - both must be constants
   - Tensor shape parameters marked with : tl.constexpr
   
   RUNTIME:
   - Actual data values
   - Loop bounds (range(0, N, BLOCK_SIZE) is fine - N can be runtime)
   - Loaded tensor elements
   
   CRITICAL: tl.arange(0, BLOCK_SIZE) ✓  but  tl.arange(0, n) where n is runtime ✗
   Solution: Use fixed BLOCK_SIZE with masking for boundaries

3. MEMORY SAFETY
   GPU memory is accessed via pointers. Out-of-bounds access causes crashes.
   
   Always use MASKING:
   ```python
   offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
   mask = offsets < N  # Check boundaries
   data = tl.load(ptr + offsets, mask=mask, other=0.0)  # Safe
   ```
   
   The mask ensures we only touch valid memory locations.

4. TRITON LANGUAGE SCOPE
   Inside @triton.jit functions, you're in "Triton-land":
   - Use tl.* operations ONLY
   - No torch.* functions
   - No Python control flow on tensor data (use tl.where instead)
   
   Outside (in wrapper), you're in "Python-land":
   - Use torch.* freely
   - Allocate tensors, compute grid sizes
   - Launch kernels

5. MATRIX OPERATIONS
   tl.dot(A, B) performs matrix multiplication:
   - Requires A.shape = (M, K) and B.shape = (K, N)
   - Results in shape (M, N)
   - Use tl.trans(B) if B is (N, K) to get (K, N)
   
   Common pattern for GEMM:
   ```python
   # Load tiles
   a = tl.load(...)  # Shape: (BLOCK_M, BLOCK_K)
   b = tl.load(...)  # Shape: (BLOCK_N, BLOCK_K)
   # Transpose b to match dimensions
   b_t = tl.trans(b)  # Now: (BLOCK_K, BLOCK_N)
   # Multiply
   c = tl.dot(a, b_t)  # Result: (BLOCK_M, BLOCK_N)
   ```

=== YOUR TASK ===

For each PyTorch operation, you should:
1. Analyze the operation and memory access patterns
2. Think step-by-step about how to parallelize it
3. Choose appropriate BLOCK_SIZE and num_warps
4. Write the complete Triton kernel implementation

Your response MUST follow this format:

<think>
[Your step-by-step reasoning about the conversion]
- What operation is being performed?
- What are the input/output shapes?
- How should this be parallelized?
- What memory access pattern should be used?
- What BLOCK_SIZE and num_warps are optimal?
</think>

<triton>
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_name(...):
    # Your Triton kernel implementation
    ...

def triton_kernel_wrapper(input_tensors):
    # Wrapper that calls the kernel and returns output
    ...
</triton>

=== CRITICAL REQUIREMENTS ===

1. The wrapper function MUST be named `triton_kernel_wrapper`
2. The wrapper takes the SAME inputs as Model.forward() - just the input tensors, NOT model weights
3. If the model has weights (nn.Linear, nn.Conv2d, etc.), the wrapper should create random weights or accept them as additional parameters
4. **IMPORTANT**: If get_init_inputs() returns parameters (e.g., {'quantiles': 4, 'hidden_size': 128}), the wrapper MUST accept these as keyword arguments with defaults matching those values
5. **Triton API Limitations**: tl.tanh, tl.pow, tl.unsqueeze do NOT exist - use tl.exp for tanh, ** operator for pow, reshape for unsqueeze

=== TRITON KERNEL RULES - MUST FOLLOW ===

IMPORTS - Only use these:
```python
import torch
import triton
import triton.language as tl
```

INSIDE @triton.jit KERNELS - Use ONLY triton.language (tl) operations:
- tl.load(), tl.store() - memory access
- tl.arange(), tl.zeros(), tl.full() - tensor creation
- tl.sum(), tl.max(), tl.min() - reductions
- tl.exp(), tl.log(), tl.sqrt(), tl.abs() - math ops
- tl.maximum(), tl.minimum() - element-wise min/max
- tl.where() - conditional selection
- tl.program_id(), tl.num_programs() - grid info
- Standard operators: +, -, *, /, %, <, >, ==, &, |

NEVER use inside @triton.jit:
- torch.* functions (torch.sum, torch.mean, torch.relu, etc.)
- .pow(), .sqrt(), .exp() methods on tensors - use ** operator for pow, tl.sqrt(), tl.exp() for others
- Python classes or objects
- nn.* modules

CONSTEXPR RULES:
- tl.arange(start, end) - both start and end MUST be constants or tl.constexpr
- BLOCK_SIZE: tl.constexpr in kernel signature
- Use powers of 2: 64, 128, 256, 512, 1024

REDUCTION PATTERN:
```python
# CORRECT - accumulator matches block size
acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
for i in range(0, N, BLOCK_SIZE):
    x = tl.load(ptr + offsets, mask=mask)
    acc += x
result = tl.sum(acc)  # reduce at the end

# WRONG - shape mismatch in loop
acc = tl.zeros([1], dtype=tl.float32)
acc += tl.sum(x)  # ERROR: shape changes!
```

WRAPPER FUNCTION PATTERN:
```python
def triton_kernel_wrapper(x):
    # x is the input tensor from get_inputs()
    output = torch.empty_like(x)  # or appropriate shape

    # Grid and block configuration
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)

    # Launch kernel
    my_kernel[grid](x, output, x.numel(), BLOCK_SIZE=BLOCK_SIZE)

    return output
```

COMMON OPERATIONS:
- ReLU: tl.maximum(x, 0.0)
- Sigmoid: 1.0 / (1.0 + tl.exp(-x))
- Tanh: (tl.exp(2*x) - 1) / (tl.exp(2*x) + 1)
- Softmax: exp_x = tl.exp(x - tl.max(x)); exp_x / tl.sum(exp_x)
- Mean: tl.sum(x) / n_elements

USE ASCII ONLY - no unicode characters like – or —, use - instead.
"""


USER_PROMPT_TEMPLATE = """Convert the following PyTorch code to an optimized Triton kernel:

```python
{pytorch_code}
```

Generate a complete Triton implementation that produces the same output as the PyTorch code."""


class TraceOrchestrator:
    """
    Orchestrates the trace generation pipeline.
    """
    
    def __init__(
        self,
        vllm_base_url: str = VLLM_BASE_URL,
        output_file: str = OUTPUT_FILE,
        kernelbook_samples: int = 1500,
        kernelbench_samples: int = 1000,
    ):
        self.vllm_base_url = vllm_base_url
        self.output_file = Path(output_file)
        self.dataloader = KernelDataLoader(
            kernelbook_samples=kernelbook_samples,
            kernelbench_samples=kernelbench_samples,
        )
        
        # Load existing traces if resuming
        self.traces = self._load_existing_traces()
        self.processed_keys = {self._get_sample_key(t) for t in self.traces}
        
        # Ensure the output file exists or is updated immediately
        self._save_traces()
        
    def _load_existing_traces(self) -> list:
        """Load existing traces from output file if it exists."""
        if self.output_file.exists():
            with open(self.output_file, "r") as f:
                return json.load(f)
        return []
    
    def _save_traces(self):
        """Save traces to output file."""
        with open(self.output_file, "w") as f:
            json.dump(self.traces, f, indent=2, default=str)
    
    def _get_sample_key(self, sample: dict) -> str:
        """Generate unique key for a sample to detect duplicates."""
        source = sample.get("source", "unknown")
        if source == "kernelbench":
            return f"kernelbench_{sample.get('problem_id', 'unknown')}"
        else:
            return f"kernelbook_{sample.get('index', hash(sample.get('pytorch_code', '')[:100]))}"
    
    async def generate_completion(
        self,
        pytorch_code: str,
        session: aiohttp.ClientSession,
    ) -> Optional[dict]:
        """
        Generate a Triton kernel completion from GLM-4.5-Air.
        
        Args:
            pytorch_code: The PyTorch code to convert
            session: aiohttp session for requests
            
        Returns:
            Dict with 'content' and 'reasoning' fields, or None if failed
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(pytorch_code=pytorch_code)},
        ]
        
        # GLM-4.5-Air has thinking/reasoning enabled by default.
        # No reasoning_effort needed — it's a binary toggle (on/off).
        # To disable thinking, pass: extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        
        # Dynamically compute max_tokens to fit within the context window.
        # Estimate input tokens (~3.5 chars/token) and leave room for them.
        total_input_chars = sum(len(m["content"]) for m in messages)
        estimated_input_tokens = int(total_input_chars / 3.5) + 50  # safety margin
        max_tokens = min(MAX_COMPLETION_TOKENS, MAX_MODEL_LEN - estimated_input_tokens)
        max_tokens = max(max_tokens, 1024)  # floor so we always generate something
        
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
        }
        
        try:
            async with session.post(
                f"{self.vllm_base_url}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    message = data["choices"][0]["message"]
                    return {
                        "content": message.get("content"),
                        "reasoning": message.get("reasoning_content"),  # vLLM-specific field
                        "usage": data.get("usage")
                    }
                else:
                    error = await response.text()
                    print(f"API error: {response.status} - {error}")
                    return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    
    def extract_triton_code(self, completion: str) -> Optional[str]:
        """
        Extract Triton code from model completion.
        
        Args:
            completion: The full model response
            
        Returns:
            The extracted Triton code or None
        """
        import re
        
        # Try to find <triton>...</triton> block
        triton_match = re.search(r"<triton>(.*?)</triton>", completion, re.DOTALL)
        if triton_match:
            return triton_match.group(1).strip()
        
        # Fallback: try to find code block with triton imports
        code_blocks = re.findall(r"```python\n(.*?)```", completion, re.DOTALL)
        for block in code_blocks:
            if "triton" in block.lower() and "@triton.jit" in block:
                return block.strip()
        
        # Last resort: look for any @triton.jit decorated function
        if "@triton.jit" in completion:
            # Find the start of imports
            idx = completion.find("import torch")
            if idx == -1:
                idx = completion.find("import triton")
            if idx != -1:
                return completion[idx:].strip()
        
        return None
    
    def extract_thinking(self, completion: str) -> Optional[str]:
        """
        Extract thinking/reasoning from model completion.
        
        Args:
            completion: The full model response
            
        Returns:
            The extracted thinking or None
        """
        import re
        
        # Try to find <think>...</think> block
        think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        if think_match:
            return think_match.group(1).strip()
        
        # Fallback: everything before <triton> or first code block
        triton_idx = completion.find("<triton>")
        if triton_idx == -1:
            triton_idx = completion.find("```python")
        
        if triton_idx > 0:
            return completion[:triton_idx].strip()
        
        return None
    
    async def validate_on_modal(
        self,
        triton_code: str,
        pytorch_code: str,
        sample: dict,
    ) -> dict:
        """
        Validate the generated Triton kernel on Modal H100.

        Args:
            triton_code: The generated Triton kernel
            pytorch_code: The original PyTorch code
            sample: The sample dict with metadata

        Returns:
            Benchmark result dict
        """
        try:
            # Use the new benchmark_kernelbench function that properly handles
            # the KernelBench/KernelBook pattern with get_inputs() and get_init_inputs()
            # KernelBook samples have an entry_point field specifying the class name
            entry_point = sample.get("entry_point", "Model")

            # Run blocking Modal .remote() call in a thread so it doesn't
            # block the asyncio event loop — enables parallel validations
            result = await asyncio.to_thread(
                benchmark_kernelbench.remote,
                triton_code=triton_code,
                pytorch_code=pytorch_code,
                n_correctness=5,
                n_trials=20,
                kernel_name=sample.get("name", "generated_kernel"),
                entry_point=entry_point,
                rtol=1e-4,
                atol=1e-4,
            )
            return result
        except Exception as e:
            return {
                "correctness": False,
                "speedup": 0.0,
                "error": str(e),
            }

    async def process_sample(
        self,
        sample: dict,
        session: aiohttp.ClientSession,
    ) -> Optional[dict]:
        """
        Process a single sample: generate, extract, validate.
        
        Args:
            sample: The sample dict from dataloader
            session: aiohttp session
            
        Returns:
            Trace dict or None if failed
        """
        sample_key = self._get_sample_key(sample)
        
        # Skip if already processed
        if sample_key in self.processed_keys:
            return None
        
        pytorch_code = sample["pytorch_code"]
        
        # Step 1: Generate completion
        print(f"Generating completion for {sample_key}...")
        response = await self.generate_completion(pytorch_code, session)
        if not response:
            return None
        
        completion = response.get("content")
        model_reasoning = response.get("reasoning")  # Internal chain-of-thought from GLM-4.5-Air
        
        # Validate content is not None
        if not completion:
            print(f"API returned empty content for {sample_key}")
            return None
        
        # Step 2: Extract Triton code and thinking
        triton_code = self.extract_triton_code(completion)
        thinking = self.extract_thinking(completion)
        
        if not triton_code:
            print(f"Failed to extract Triton code for {sample_key}")
            return {
                "sample_key": sample_key,
                "error": "Triton extraction failed",
                "full_completion": completion,
                "model_reasoning": model_reasoning
            }
        
        # Step 3: Validate on Modal
        print(f"Validating {sample_key} on Modal H100...")
        result = await self.validate_on_modal(triton_code, pytorch_code, sample)
        print(f"Validation complete for {sample_key}: Correct={result.get('correctness')}, Speedup={result.get('speedup')}x")
        
        # Create trace
        trace = {
            "sample_key": sample_key,
            "source": sample.get("source"),
            "level": sample.get("level"),
            "name": sample.get("name"),
            "problem_id": sample.get("problem_id"),
            "pytorch_code": pytorch_code,
            "thinking": thinking,  # Extracted from <think> tags in completion
            "model_reasoning": model_reasoning,  # Internal CoT from GLM-4.5-Air (reasoning_content)
            "triton_code": triton_code,
            "full_completion": completion,
            "result": {
                "correctness": result.get("correctness", False),
                "speedup": result.get("speedup", 0.0),
                "fast_0": result.get("fast_0", False),
                "fast_1": result.get("fast_1", False),
                "fast_2": result.get("fast_2", False),
                "error": result.get("error"),
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        return trace
    
    async def run(
        self,
        batch_size: int = 10,
        save_interval: int = 10,
    ):
        """
        Run the orchestrator.

        Args:
            batch_size: Number of concurrent requests
            save_interval: Save traces every N samples
        """
        samples = self.dataloader.load_all()

        print(f"Total samples: {len(samples)}")
        print(f"Already processed: {len(self.processed_keys)}")
        print(f"Remaining: {len(samples) - len(self.processed_keys)}")
        print()

        # Run within Modal app context so .remote() calls work
        with modal_app.run():
            async with aiohttp.ClientSession() as session:
                # Process in batches
                processed = 0

                for i in tqdm(range(0, len(samples), batch_size), desc="Processing"):
                    batch = samples[i:i + batch_size]

                    # Process batch concurrently
                    tasks = [
                        self.process_sample(sample, session)
                        for sample in batch
                    ]
                    results = await asyncio.gather(*tasks)

                    # Add all traces to the list so user can see progress
                    for trace in results:
                        if trace:
                            self.traces.append(trace)
                            self.processed_keys.add(trace["sample_key"])
                            processed += 1

                            if trace.get("result", {}).get("correctness"):
                                print(f"  {trace['sample_key']} is CORRECT ({trace['result']['speedup']:.2f}x)")
                            else:
                                err = trace.get("result", {}).get("error") or trace.get("error", "Unknown error")
                                print(f"  {trace['sample_key']} FAILED: {err}")

                    # Save after every batch for better visibility during development
                    self._save_traces()
                    if processed > 0:
                        print(f"Saved {len(self.traces)} traces to {self.output_file}")

        # Final save
        self._save_traces()

        # Print summary
        correct_count = sum(1 for t in self.traces if t.get("result", {}).get("correctness"))
        fast_1_count = sum(1 for t in self.traces if t.get("result", {}).get("fast_1"))
        fast_2_count = sum(1 for t in self.traces if t.get("result", {}).get("fast_2"))

        print("\n" + "=" * 60)
        print("TRACE GENERATION COMPLETE")
        print("=" * 60)
        print(f"Total traces saved: {len(self.traces)}")
        print(f"Correct (fast_0): {correct_count}")
        print(f"Faster than PyTorch (fast_1): {fast_1_count}")
        print(f"2x faster (fast_2): {fast_2_count}")
        print(f"Output file: {self.output_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate reasoning traces for Triton kernels")
    parser.add_argument("--vllm-url", default=VLLM_BASE_URL, help="vLLM server URL")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output JSON file")
    parser.add_argument("--kernelbook-samples", type=int, default=1500)
    parser.add_argument("--kernelbench-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=10)
    
    args = parser.parse_args()
    
    orchestrator = TraceOrchestrator(
        vllm_base_url=args.vllm_url,
        output_file=args.output,
        kernelbook_samples=args.kernelbook_samples,
        kernelbench_samples=args.kernelbench_samples,
    )
    
    asyncio.run(orchestrator.run(
        batch_size=args.batch_size,
        save_interval=args.save_interval,
    ))


if __name__ == "__main__":
    main()
"""
Orchestrator for generating reasoning traces with gpt-oss-120b.

This script:
1. Loads PyTorch code from KernelBook and KernelBench via dataloader
2. Sends each problem to gpt-oss-120b (running locally via vLLM)
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

# Import Modal function for remote execution
import modal

# Configuration
VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "openai/gpt-oss-120b"
OUTPUT_FILE = "reasoning_traces.json"
MAX_TOKENS = 16384  # Long enough for reasoning + code
TEMPERATURE = 0.7
REASONING_LEVEL = "high"  # low, medium, high


SYSTEM_PROMPT = f"""Reasoning: {REASONING_LEVEL}

You are an expert GPU kernel engineer. Your task is to convert PyTorch code into optimized Triton kernels.

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

IMPORTANT:
- The wrapper function MUST be named `triton_kernel_wrapper`
- It should take the same inputs as the PyTorch Model.forward() method
- It should return the same output as the PyTorch implementation
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
    ) -> Optional[str]:
        """
        Generate a Triton kernel completion from gpt-oss-120b.
        
        Args:
            pytorch_code: The PyTorch code to convert
            session: aiohttp session for requests
            
        Returns:
            The model's response or None if failed
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(pytorch_code=pytorch_code)},
        ]
        
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
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
                    return data["choices"][0]["message"]["content"]
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
        # Import the Modal app and function
        from modal_app import benchmark_triton_kernel
        
        # For KernelBench: the pytorch_code contains get_inputs() which we'll use
        # For KernelBook: we need to construct input shapes from the code
        
        # Create reference code that includes get_inputs() for dynamic input generation
        # The reference uses Model.forward() as the reference implementation
        reference_code = f'''
{pytorch_code}

# Adapter for benchmark - use the Model class from the code
_model = Model(*get_init_inputs()) if 'get_init_inputs' in dir() else Model()

def reference_impl(*inputs):
    return _model(*inputs)

# Generate inputs for benchmarking
def generate_inputs():
    return get_inputs()
'''
        
        # For the triton kernel, we need to wrap it to accept the same inputs
        triton_code_with_inputs = f'''
{triton_code}

# Import the get_inputs from reference for input generation
'''
        
        # Since we have get_inputs() in the code, we don't need static input_shapes
        # The Modal container will execute get_inputs() to generate test tensors
        # 
        # BUT our current modal_app.py expects input_shapes dict...
        # We need to either:
        # 1. Parse input shapes from get_inputs() 
        # 2. Or create a new benchmark function that uses get_inputs()
        #
        # For now, let's try to extract a reasonable shape from the code
        input_shapes = self._extract_input_shapes_from_code(pytorch_code)
        
        try:
            result = benchmark_triton_kernel.remote(
                kernel_code=triton_code,
                reference_torch_code=reference_code,
                input_shapes=input_shapes,
                n_correctness=5,
                n_trials=20,
                kernel_name=sample.get("name", "generated_kernel"),
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
    
    def _extract_input_shapes_from_code(self, pytorch_code: str) -> dict:
        """
        Extract input shapes from get_inputs() function in PyTorch code.
        
        This is a best-effort extraction - for complex cases, we default to
        reasonable shapes.
        """
        import re
        
        # Try to find shape definitions in the code
        input_shapes = {}
        
        # Look for batch_size = N pattern
        batch_match = re.search(r'batch_size\s*=\s*(\d+)', pytorch_code)
        batch_size = int(batch_match.group(1)) if batch_match else 32
        
        # Look for input_shape = (N, M, ...) pattern
        shape_match = re.search(r'input_shape\s*=\s*\(([^)]+)\)', pytorch_code)
        if shape_match:
            dims = [int(d.strip()) for d in shape_match.group(1).split(',') if d.strip().isdigit()]
            if dims:
                input_shapes["x"] = {"shape": [batch_size] + dims, "dtype": "float32"}
                return input_shapes
        
        # Look for N = number patterns (common for matrix sizes)
        n_match = re.search(r'\bN\s*=\s*(\d+)', pytorch_code)
        n_size = int(n_match.group(1)) if n_match else 1024
        
        # Look for M = number
        m_match = re.search(r'\bM\s*=\s*(\d+)', pytorch_code)
        m_size = int(m_match.group(1)) if m_match else n_size
        
        # Look for torch.rand(N, M) or similar patterns
        rand_match = re.search(r'torch\.rand[n]?\(([^)]+)\)', pytorch_code)
        if rand_match:
            args = rand_match.group(1)
            # Try to parse the arguments
            if 'N' in args:
                input_shapes["x"] = {"shape": [n_size, m_size], "dtype": "float32"}
            else:
                # Try to parse numeric dimensions
                dims = re.findall(r'\d+', args)
                if dims:
                    input_shapes["x"] = {"shape": [int(d) for d in dims[:4]], "dtype": "float32"}
        
        # Default fallback
        if not input_shapes:
            input_shapes["x"] = {"shape": [batch_size, 1024], "dtype": "float32"}
        
        return input_shapes
    
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
        completion = await self.generate_completion(pytorch_code, session)
        if not completion:
            return None
        
        # Step 2: Extract Triton code and thinking
        triton_code = self.extract_triton_code(completion)
        thinking = self.extract_thinking(completion)
        
        if not triton_code:
            print(f"Failed to extract Triton code for {sample_key}")
            return None
        
        # Step 3: Validate on Modal
        result = await self.validate_on_modal(triton_code, pytorch_code, sample)
        
        # Create trace
        trace = {
            "sample_key": sample_key,
            "source": sample.get("source"),
            "level": sample.get("level"),
            "name": sample.get("name"),
            "problem_id": sample.get("problem_id"),
            "pytorch_code": pytorch_code,
            "thinking": thinking,
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
                
                # Add successful traces
                for trace in results:
                    if trace and trace["result"]["correctness"]:
                        self.traces.append(trace)
                        self.processed_keys.add(trace["sample_key"])
                        processed += 1
                
                # Save periodically
                if processed % save_interval == 0 and processed > 0:
                    self._save_traces()
                    print(f"\nSaved {len(self.traces)} traces (processed {processed})")
        
        # Final save
        self._save_traces()
        
        # Print summary
        correct_count = sum(1 for t in self.traces if t["result"]["correctness"])
        fast_1_count = sum(1 for t in self.traces if t["result"]["fast_1"])
        fast_2_count = sum(1 for t in self.traces if t["result"]["fast_2"])
        
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

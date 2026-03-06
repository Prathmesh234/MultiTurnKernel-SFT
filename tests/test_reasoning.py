"""
Quick test script to verify the vLLM server is returning reasoning traces.
Makes a single request using the same logic as orchestrator.py and prints
the reasoning_content (thinking tokens) alongside the response content.
"""

import asyncio
import aiohttp
import json

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"

TEST_PROMPT = """Convert the following PyTorch code to an optimized Triton kernel:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

def get_inputs():
    return [torch.randn(1024, 1024)]

def get_init_inputs():
    return []
```

Generate a complete Triton implementation that produces the same output as the PyTorch code."""


async def test_reasoning():
    messages = [
        {"role": "user", "content": TEST_PROMPT},
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 16384,
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": True},  # Same as orchestrator
    }

    print("=" * 60)
    print("Sending request to vLLM server...")
    print(f"URL: {VLLM_BASE_URL}/chat/completions")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as response:
            if response.status != 200:
                error = await response.text()
                print(f"ERROR {response.status}: {error}")
                return

            raw_text = await response.text()
            data = json.loads(raw_text)

    print("\n--- RAW RESPONSE SNIPPET (first 500 chars) ---")
    print(raw_text[:500])

    message = data["choices"][0]["message"]
    raw_content = message.get("content") or ""

    # vLLM 0.16.0 strips <think> but leaves </think> in content and does not
    # populate the reasoning field. Split manually.
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if not reasoning and "</think>" in raw_content:
        parts = raw_content.split("</think>", 1)
        reasoning = parts[0].strip()
        raw_content = parts[1].strip()

    content = raw_content
    usage = data.get("usage", {})

    print("\n--- RAW CHOICE (full) ---")
    print(json.dumps(data["choices"][0], indent=2)[:500])

    print("\n--- USAGE ---")
    print(json.dumps(usage, indent=2))

    print("\n--- REASONING TOKENS (reasoning_content) ---")
    if reasoning:
        print(f"[Length: {len(reasoning)} chars]\n")
        # Print first 2000 chars so it's readable
        print(reasoning[:2000])
        if len(reasoning) > 2000:
            print(f"\n... [truncated, {len(reasoning) - 2000} more chars]")
    else:
        print("WARNING: reasoning_content is None or empty!")
        print("This means the server is NOT returning reasoning traces.")
        print("Check that --reasoning-parser qwen3 is set on the vLLM server.")

    print("\n--- RESPONSE CONTENT ---")
    if content:
        print(f"[Length: {len(content)} chars]\n")
        print(content[:2000])
        if len(content) > 2000:
            print(f"\n... [truncated, {len(content) - 2000} more chars]")
    else:
        print("WARNING: content is None or empty!")

    print("\n--- SUMMARY ---")
    print(f"reasoning_content present: {reasoning is not None}")
    print(f"reasoning_content length:  {len(reasoning) if reasoning else 0} chars")
    print(f"content present:           {content is not None}")
    print(f"content length:            {len(content) if content else 0} chars")


if __name__ == "__main__":
    asyncio.run(test_reasoning())

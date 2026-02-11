#!/usr/bin/env python3
"""
Test request to the vLLM inference server running Trinity-Mini with LoRA.

This mirrors the exact same request pattern used in the SGLang deployment.
Both vLLM and SGLang expose OpenAI-compatible /v1/chat/completions endpoints,
so the request format is identical.

Usage:
    python inference/vllm/test_request.py
    python inference/vllm/test_request.py --base-url http://localhost:8000/v1
    python inference/vllm/test_request.py --model trinity-reasoning-vllm
"""

import argparse
import json
import requests


VLLM_BASE_URL = "http://localhost:8000/v1"
# Use the base model name by default (same as SGLang test in SGLANG_SETUP.md).
# To use the LoRA adapter, pass --model trinity-reasoning-vllm
MODEL_NAME = "arcee-ai/Trinity-Mini"


def test_chat_completion(base_url: str, model: str):
    """
    Send a chat completion request to the vLLM server.
    This is the same request pattern used in the SGLang deployment.
    """
    url = f"{base_url}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain kernels in ML."},
        ],
        "temperature": 0.7,
    }

    print(f"Sending request to {url}")
    print(f"Model: {model}")
    print("-" * 60)

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        message = data["choices"][0]["message"]

        print(f"\nResponse:")
        print(message.get("content", ""))

        if data.get("usage"):
            usage = data["usage"]
            print(f"\nUsage:")
            print(f"  Prompt tokens: {usage.get('prompt_tokens')}")
            print(f"  Completion tokens: {usage.get('completion_tokens')}")
            print(f"  Total tokens: {usage.get('total_tokens')}")

    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to {url}")
        print("Make sure the vLLM server is running:")
        print("  bash inference/vllm/serve_trinity_uv.sh")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"Error: {e}")


def test_triton_generation(base_url: str, model: str):
    """
    Send a Triton kernel generation request to the vLLM server.
    This mirrors the orchestrator.py request pattern used in the SGLang deployment.
    """
    url = f"{base_url}/chat/completions"

    # Example PyTorch code (same pattern as orchestrator.py)
    pytorch_code = """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(1024, 1024, device='cuda')]

def get_init_inputs():
    return []
"""

    system_prompt = "You are an expert GPU kernel engineer. Convert PyTorch code to optimized Triton kernels."

    user_prompt = f"""Convert the following PyTorch code to an optimized Triton kernel:

```python
{pytorch_code}
```

Generate a complete Triton implementation that produces the same output as the PyTorch code."""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 16384,
        "temperature": 0.7,
    }

    print(f"\nSending Triton generation request to {url}")
    print(f"Model: {model}")
    print("-" * 60)

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
        response.raise_for_status()

        data = response.json()
        message = data["choices"][0]["message"]
        content = message.get("content", "")

        print(f"\nResponse (first 500 chars):")
        print(content[:500])
        if len(content) > 500:
            print(f"\n... ({len(content)} total characters)")

        if data.get("usage"):
            usage = data["usage"]
            print(f"\nUsage:")
            print(f"  Prompt tokens: {usage.get('prompt_tokens')}")
            print(f"  Completion tokens: {usage.get('completion_tokens')}")
            print(f"  Total tokens: {usage.get('total_tokens')}")

    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to {url}")
        print("Make sure the vLLM server is running:")
        print("  bash inference/vllm/serve_trinity_uv.sh")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test vLLM Trinity-Mini inference server")
    parser.add_argument("--base-url", default=VLLM_BASE_URL, help="vLLM server base URL")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name to use in requests")
    parser.add_argument(
        "--triton",
        action="store_true",
        help="Also run a Triton kernel generation test (mirrors orchestrator.py)",
    )
    args = parser.parse_args()

    # Test 1: Basic chat completion (same as SGLANG_SETUP.md curl example)
    test_chat_completion(args.base_url, args.model)

    # Test 2: Triton kernel generation (same as orchestrator.py pattern)
    if args.triton:
        test_triton_generation(args.base_url, args.model)


if __name__ == "__main__":
    main()

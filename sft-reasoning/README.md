# SFT Reasoning: Finetuning Trinity-Mini on Triton Traces

This folder contains the script for finetuning the `arcee-ai/Trinity-Mini` model on verified reasoning traces for PyTorch → Triton kernel translation.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT: Reasoning Traces                        │
│                                                                         │
│  Source: gpt-oss-120b generating Triton kernels from PyTorch           │
│  Format: PyTorch → <think>reasoning</think> → Triton                   │
│  Quality: Verified correct AND compiled on Modal H100                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FILTERING CRITERIA                                 │
│                                                                         │
│  ✅ correctness == True    (kernel output matches reference)           │
│  ✅ fast_0 == True          (kernel compiles & produces correct output) │
│  ⬜ fast_1 (optional)       (speedup > 1.0 - nice to have)             │
│  ⬜ fast_2 (optional)       (speedup >= 2.0 - nice to have)            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         SFT TRAINING                                    │
│                                                                         │
│  Model:    arcee-ai/Trinity-Mini (26B MoE, 3B active)                  │
│  Method:   QLoRA (4-bit quantization + LoRA adapters)                  │
│  Format:   Conversational (User: PyTorch, Assistant: Think + Triton)  │
│  Trainer:  SFTTrainer from HuggingFace TRL                             │
└─────────────────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `finetune_trinity.py` | Main finetuning script (Jupyter notebook style) |
| `pyproject.toml` | Python dependencies (uv) |
| `README.md` | This file |

## Prerequisites

1. **Generate Traces**: Run the orchestrator to generate reasoning traces:
   ```bash
   cd ..
   python orchestrator.py
   ```
   This creates `reasoning_traces.json` with traces from gpt-oss-120b.

2. **GPU Requirements**: 
   - Minimum: A100 40GB (for QLoRA training)
   - Recommended: A100 80GB or H100 80GB

3. **Install Dependencies** (using uv):
   ```bash
   # Install uv if you don't have it
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Sync dependencies
   uv sync
   
   # Optional: Flash Attention 2 for faster training
   uv pip install flash-attn --no-build-isolation
   
   # Optional: Weights & Biases for experiment tracking
   uv sync --extra wandb
   ```

## Usage

### Quick Start

```python
from finetune_trinity import main

# Run the full training pipeline
main()
```

### Step-by-Step

```python
from finetune_trinity import (
    load_and_filter_traces,
    prepare_dataset,
    load_model_and_tokenizer,
    create_lora_config,
    create_training_config,
    train,
)

# 1. Load and filter traces
traces = load_and_filter_traces(
    "../reasoning_traces.json",
    require_correctness=True,
    require_fast_0=True,
    require_fast_1=False,  # Optional
    require_fast_2=False,  # Optional
)

# 2. Prepare dataset
dataset = prepare_dataset(traces)

# 3. Load model with quantization
model, tokenizer = load_model_and_tokenizer()

# 4. Configure LoRA
lora_config = create_lora_config()

# 5. Configure training
training_config = create_training_config()

# 6. Train!
trainer = train(model, tokenizer, dataset, training_config, lora_config)
```

### Inference with Finetuned Model

```python
from finetune_trinity import inference

result = inference("""
import torch

def softmax(x, dim=-1):
    x_max = x.max(dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
""")

print(result)
# Output: <think>reasoning...</think><triton>kernel code...</triton>
```

## Filtering Logic

The script applies the following filters to ensure training quality:

| Filter | Requirement | Purpose |
|--------|-------------|---------|
| `correctness` | **Must be True** | Kernel output matches PyTorch reference |
| `fast_0` | **Must be True** | Same as correctness - baseline requirement |
| `fast_1` | Optional | Kernel is faster than PyTorch (speedup > 1.0) |
| `fast_2` | Optional | Kernel is 2x faster (speedup >= 2.0) |

### Why these criteria?

1. **correctness + fast_0 = Minimum bar**: The kernel must produce correct output. An incorrect kernel, no matter how fast, is useless.

2. **fast_1 is optional**: We want the model to learn the *reasoning* process first. Speed optimization can come later through RL refinement.

3. **fast_2 is optional**: Only ~10-20% of kernels achieve 2x speedup. Requiring this would filter out too much training data.

## Configuration

Key hyperparameters in `finetune_trinity.py`:

```python
# Model
MODEL_NAME = "arcee-ai/Trinity-Mini"  # 26B MoE, 3B active
MAX_SEQ_LENGTH = 4096

# Training
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 3

# LoRA
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
```

## Output

After training, the model is saved to `./trinity-triton-sft/` with:
- LoRA adapter weights
- Tokenizer configuration
- Training logs (TensorBoard)

To use the finetuned model:

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("./trinity-triton-sft")
tokenizer = AutoTokenizer.from_pretrained("./trinity-triton-sft")
```

## Next Steps

After SFT training:

1. **Evaluate**: Test on held-out KernelBench Level 3+4 problems
2. **RL Refinement**: Use the SFT checkpoint as the base for GRPO/PPO training
3. **Merge Weights**: Optionally merge LoRA weights into base model for faster inference

## References

- [AGENTS.md](../AGENTS.md) - Full project documentation
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) - HuggingFace TRL documentation
- [PEFT LoRA](https://huggingface.co/docs/peft/task_guides/lora_based_methods) - Efficient finetuning
- [arcee-ai/Trinity-Mini](https://huggingface.co/arcee-ai/Trinity-Mini) - Model card

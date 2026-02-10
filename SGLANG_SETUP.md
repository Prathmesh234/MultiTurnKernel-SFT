# SGLang + Trinity-Mini Setup Commands

## Summary
Successfully set up SGLang 0.5.8 with proper dependencies using a UV virtual environment. SGLang is specifically used here because it **fully supports MoE + LoRA** (including MLP layers like `gate_proj`, `up_proj`, and `down_proj`), which is critical for the Trinity-Mini architecture.

---

## 1. Login to Modal

```bash
modal token new
```

This will open a browser authentication flow. After completion, your token is stored at `~/.modal.toml`.

---

## 2. Download LoRA Adapter from Modal

```bash
cd /home/ubuntu/Arcee-Mini-Kernel
source .venv/bin/activate
uv run python inference/download_lora_from_modal.py
```

Checkpoints are downloaded to `./sft-reasoning/checkpoint-40`.

---

## 3. Start SGLang Server

### Serving with LoRA (Recommended)

```bash
bash /home/ubuntu/Arcee-Mini-Kernel/inference/serve_trinity_uv.sh
```

**Note:** If you encounter a shape mismatch `AssertionError` during CUDA graph capture, it typically indicates a configuration alignment issue between the base model and the adapter's expected hidden dimensions in the MoE layers.

### Serving Base Model Only (Fallback)

```bash
bash /home/ubuntu/Arcee-Mini-Kernel/inference/serve_trinity_base.sh
```

---

## Environment Details

### Successfully Installed Packages
- ✅ **sglang** == 0.5.8 (Optimized MoE LoRA support)
- ✅ **transformers** == 4.57.1
- ✅ **torch** == 2.9.1
- ✅ **numpy** == 1.24.4 (Required for SciPy compatibility)

### Hardware Requirements
- Trinity-Mini (26B MoE) requires ~48GB VRAM for weights + KV cache.
- An H100 (80GB) or A100 (80GB) is recommended.

---

## Debugging the LoRA AssertionError

If the server crashes with `assert x.shape[-1] == K` in `chunked_sgmv_shrink.py`:
1. It confirms SGLang is attempting to apply the LoRA to the MoE layers.
2. The mismatch is likely because Trinity-Mini uses a specific Expert dimensionality that needs to be precisely matched in the SGLang kernel launch.
3. Ensure the `--trust-remote-code` flag is present to allow correct loading of the MoE architecture.

---

## Testing the Server

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "arcee-ai/Trinity-Mini",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain kernels in ML."}
    ],
    "temperature": 0.7
  }'
```

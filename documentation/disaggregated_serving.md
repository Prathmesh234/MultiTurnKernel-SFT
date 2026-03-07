# Disaggregated Serving Experiment — Qwen3-235B on 8x H100

Experiment comparing two vLLM serving configurations for Qwen3-235B-A22B-Thinking-2507-FP8
on a single 8x H100 (80GB NVLink) node, with and without FP8 KV cache quantization.

---

## Background

The current baseline (`scripts/serve_qwen3_235b.sh`) runs **Config A: EP8** — all 8 GPUs
handle both prefill and decode together with TP=8 + expert parallelism.

**Config B: Disaggregated Prefill/Decode (PD 4+4)** splits the node in half:
- 4 GPUs dedicated to prefill (prompt processing)
- 4 GPUs dedicated to decode (token generation)
- KV cache is computed on prefill GPUs and transferred to decode GPUs over NVLink

For a thinking model like Qwen3 that generates long reasoning chains (10K–30K tokens per
request), the decode phase dominates runtime. Dedicated decode GPUs avoid prefill and decode
competing for the same compute. The tradeoff is less GPU memory per role, which tightens KV
cache headroom — especially on the decode side.

---

## Memory Math

### Model weights (FP8 = 1 byte/param)

| Config | TP size | Weights/GPU | H100 usable (0.85 util) | KV headroom/GPU |
|--------|---------|-------------|--------------------------|-----------------|
| A: EP8 | 8 | ~29.4 GB | 68 GB | **~38.6 GB** |
| B: 4P+4D | 4 | ~58.75 GB | 68 GB | **~9.25 GB** |

> Note: `gpu_memory_utilization` is dropped to 0.85 for Config B (from 0.92) to avoid
> vLLM block pool allocation failures when weights consume most of the 80 GB.

### KV cache capacity (Qwen3-235B: 8 KV heads GQA, head_dim=128, 94 layers)

Total KV bytes per token = 2 × 8 heads × 128 dims × 94 layers = **~375 KB/token** (BF16)
With FP8 KV: **~187.5 KB/token** (1 byte instead of 2)

| Config | KV dtype | Total system KV capacity | At 80K tokens/req, max concurrent |
|--------|----------|--------------------------|------------------------------------|
| A: EP8 | BF16 | ~38.6 GB × 8 / 47 KB = ~6.6M tokens | ~82 |
| A: EP8 | **FP8** | doubles → **~13.2M tokens** | ~165 |
| B: decode (4 GPU) | BF16 | ~9.25 GB × 4 / 94 KB = ~393K tokens | ~5 |
| B: decode (4 GPU) | **FP8** | doubles → **~786K tokens** | ~10 |

**FP8 KV cache is required for Config B to be viable at any meaningful batch size.**

---

## FP8 KV Cache — Native vLLM Feature

`--kv-cache-dtype fp8_e4m3` is a **built-in vLLM flag**, completely independent of Mooncake
or any other external system. It has been available since vLLM v0.5.x.

- Stores K and V tensors as FP8 in the KV cache blocks; attention compute stays in BF16
- H100 has native FP8 tensor cores (Hopper) — dequantization overhead is near zero
- `--calculate-kv-scales True` enables dynamic per-tensor scale calibration at runtime
  (no offline calibration step required)
- Reported improvements: ~38% tokens/sec uplift for code generation on H100, up to 2x ITL
  improvement for large MoE models
- No measurable quality degradation for code generation tasks

---

## Config A — EP8 Baseline (with FP8 KV)

Add two flags to `scripts/serve_qwen3_235b.sh`:

```bash
uv run --no-sync vllm serve $MODEL_NAME \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    --swap-space 16 \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-prefix-caching \
    --kv-cache-dtype fp8_e4m3 \
    --calculate-kv-scales True \
    --dtype auto
```

Orchestrator (no changes needed):
```bash
python orchestrator.py --multi-turn --batch-size <N>
```

---

## Config B — PD 4+4 with PyNcclConnector + FP8 KV

Two separate vLLM processes on the same node. Run each in its own terminal.

### Prefill instance (GPUs 0–3)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_USE_V1=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
SAFETENSORS_FAST_GPU=1 \
uv run --no-sync vllm serve Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 \
    --host 0.0.0.0 \
    --port 8100 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 16 \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-prefix-caching \
    --kv-cache-dtype fp8_e4m3 \
    --calculate-kv-scales True \
    --dtype auto \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_port":14579}'
```

### Decode instance (GPUs 4–7) — orchestrator points here

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
VLLM_USE_V1=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
SAFETENSORS_FAST_GPU=1 \
uv run --no-sync vllm serve Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 16 \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enable-prefix-caching \
    --kv-cache-dtype fp8_e4m3 \
    --calculate-kv-scales True \
    --dtype auto \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_port":14579}'
```

Orchestrator points to port 8000 (decode instance) — no changes needed:
```bash
python orchestrator.py --multi-turn --batch-size <N>
```

---

## Experiment Matrix

Run both configs across these batch sizes. Config B without FP8 KV is included only as a
reference baseline to confirm OOM boundary.

| Config | KV dtype | `--batch-size` | Expected outcome |
|--------|----------|----------------|-----------------|
| A: EP8 | BF16 | 4 | Confirmed baseline |
| A: EP8 | BF16 | 8 | Safe |
| A: EP8 | BF16 | 16 | Safe |
| A: EP8 | BF16 | 32 | Swap pressure, watch logs |
| A: EP8 | **FP8** | 4 | Throughput baseline with FP8 |
| A: EP8 | **FP8** | 8 | Should be comfortable |
| A: EP8 | **FP8** | 16 | Should be comfortable |
| A: EP8 | **FP8** | 32 | Target ceiling for EP8 |
| B: 4P+4D | BF16 | 1 | OOM reference — expect failure above 2 |
| B: 4P+4D | BF16 | 4 | Likely OOM or heavy swap on decode |
| B: 4P+4D | **FP8** | 4 | Viable — primary comparison point |
| B: 4P+4D | **FP8** | 8 | Primary comparison point vs A FP8 |
| B: 4P+4D | **FP8** | 16 | Upper ceiling for Config B |

**Key comparison:** Config B FP8 batch=8 vs Config A FP8 batch=8 — same KV pressure, but
Config B should show lower TTFT (time-to-first-token) due to dedicated prefill GPUs.

---

## What to Measure

For each run, capture:

- **TTFT** — time from request submission to first token (vLLM logs or prometheus)
- **ITL** — inter-token latency during decode
- **Throughput** — total tokens/sec across all concurrent requests
- **Correctness rate** — from the reasoning traces (orchestrator output)
- **Speedup distribution** — fast_0 / fast_1 / fast_2 breakdown from `analysis_multiturn/`
- **OOM events** — any CUDA out-of-memory in vLLM stderr

Prometheus metrics are already collected via `metrics/collect_metrics.py`. Run it alongside
the orchestrator during each experiment run.

---

## Notes on Mooncake (Optional Upgrade)

`PyNcclConnector` (used above) ships with vLLM and uses NCCL for KV transfer over NVLink —
sufficient for single-node 4P+4D. If this experiment shows Config B is promising and you want
to push further:

- **MooncakeConnectorV1** replaces `PyNcclConnector` with lower-latency RDMA transfer (useful
  for multi-node), and adds **Mooncake Store** — a distributed KV cache pool that caches
  finished-request KVs for prefix reuse. Given the fixed 860-token system prompt shared across
  all kernelbench problems, Mooncake Store would compute that prefix KV once and serve it to
  every subsequent request, eliminating the prefill cost of the longest part of every prompt.
- Requires `pip install mooncake-transfer-engine` and a running `mooncake_master` process.
- Change `"kv_connector":"PyNcclConnector"` → `"kv_connector":"MooncakeConnectorV1"` and add
  `kv_connector_extra_config` with prefill/decode TP sizes.

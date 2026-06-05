# vLLM Metrics Collection — Reference Guide

> **Read-only, zero-interference** observer for a live vLLM server.
> Inspired by metrics from the [SemiAnalysis InferenceX v2](https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs) benchmarking framework.

---

## Quick Start

```bash
# Default: poll every 10s
uv run --no-sync python metrics/collect_metrics.py

# Custom interval
uv run --no-sync python metrics/collect_metrics.py --interval 30

# Custom output paths
uv run --no-sync python metrics/collect_metrics.py \
    --output metrics/run2.jsonl \
    --prometheus-output metrics/prometheus_run2.json

# Different server URL
uv run --no-sync python metrics/collect_metrics.py --url http://localhost:9000
```

The script connects only to `GET /metrics` and `nvidia-smi`. It **never sends inference requests** and does not affect the vLLM engine or the orchestrator.

---

## Output Files

| File | Format | Description |
|------|--------|-------------|
| `metrics/metrics.jsonl` | JSON Lines (1 object per poll) | Structured, derived, and processed metrics |
| `metrics/prometheus.json` | JSON Array | Raw Prometheus scrape ground truth — every single metric vLLM exposes |

### `metrics.jsonl` — Record Structure

Each line is a self-contained JSON object with the following top-level keys:

```json
{
  "timestamp":             "ISO-8601 UTC timestamp of this poll",
  "unix_ts":               1772763895.9,
  "collection_interval_s": 10,
  "poll_index":            42,

  "server_config":         { ... },
  "system_info":           { ... },
  "inferencex_metrics":    { ... },
  "rates":                 { ... },
  "vllm_metrics":          { ... },
  "gpu_stats":             [ ... ],
  "http_metrics":          { ... },
  "process_metrics":       { ... },
  "engine_state":          { ... },
  "cache_config":          { ... }
}
```

### `prometheus.json` — Ground Truth Structure

An array where each element represents one full scrape:

```json
[
  {
    "timestamp":  "ISO-8601",
    "unix_ts":    1772763895.9,
    "poll_index": 0,
    "raw_text":   "<full prometheus text exposition as returned by /metrics>",
    "parsed":     { "<metric_name>": { "<label_set>": <value>, "__help__": "...", "__type__": "..." } }
  }
]
```

---

## Section-by-Section Metric Reference

---

### 1. `server_config` — Static Server Configuration

Captured once and embedded in every record for self-contained analysis.

| Field | Value | Description |
|-------|-------|-------------|
| `model` | `Qwen/Qwen3-235B-A22B-Thinking-2507-FP8` | HuggingFace model ID |
| `model_type` | `MoE (235B total / 22B active, 128 experts top-8)` | Architecture summary |
| `quantization` | `FP8` | Weight quantization format |
| `tensor_parallel_size` | `8` | Number of GPUs the model is sharded across |
| `expert_parallel` | `true` | MoE expert parallelism enabled |
| `max_model_len` | `131072` | Maximum sequence length (tokens) |
| `gpu_memory_utilization` | `0.92` | Fraction of GPU VRAM reserved for KV cache |
| `max_num_seqs` | `64` | Maximum concurrent in-flight sequences |
| `swap_space_gb` | `16` | CPU RAM used as overflow KV cache |
| `reasoning_parser` | `qwen3` | Streaming reasoning (`<think>`) extractor |
| `prefix_caching` | `true` | Automatic prefix cache (APC) enabled |
| `dtype` | `auto` | Compute dtype (resolved to bf16/fp8 by vLLM) |
| `vllm_use_v1` | `true` | vLLM V1 async engine (better scheduling) |
| `workload_type` | `non-disaggregated` | Prefill and decode run on the same node/GPUs |

---

### 2. `system_info` — Host Hardware (collected once at startup)

| Field | Description |
|-------|-------------|
| `cpu_model` | CPU model name (e.g. Intel Xeon Platinum 8470) |
| `cpu_cores` | Total logical CPU cores |
| `ram_total_bytes` | Total system RAM in bytes |
| `ram_used_bytes` | Used system RAM at startup |
| `ram_available_bytes` | Available system RAM at startup |
| `kernel` | Linux kernel version |
| `gpu_type` | GPU model and count (e.g. `NVIDIA H100 80GB HBM3 x8 (NVLink)`) |
| `num_gpus` | Number of GPUs detected |
| `total_gpu_memory_mib` | Total VRAM across all GPUs (MiB) |
| `gpus[]` | Per-GPU static info: name, driver, PCI bus ID, VRAM, max PCIe gen/width |
| `gpu_topology` | Full `nvidia-smi topo -m` output showing GPU-GPU and GPU-NIC interconnect types |
| `nvlink` | Per-GPU NVLink info: number of links, per-link bandwidth (GB/s), total bandwidth |

**NVLink context for this setup:** 18 links × 26.56 GB/s = **478 GB/s** aggregate bidirectional bandwidth per GPU. This is what enables full-mesh high-speed all-to-all communication for EP (expert parallelism).

---

### 3. `inferencex_metrics` — Derived InferenceX-style Metrics

These are the primary analysis metrics, computed from raw Prometheus data and matching the SemiAnalysis InferenceX benchmark framework dimensions.

---

#### 3a. Throughput

| Metric | Unit | Description |
|--------|------|-------------|
| `output_throughput_tok_per_sec` | tok/s | Rate of output (generation) tokens produced — the core "how fast" metric |
| `prompt_throughput_tok_per_sec` | tok/s | Rate of prompt (prefill) tokens processed — non-zero only during active prefill |
| `output_throughput_tok_per_sec_per_gpu` | tok/s/GPU | Output throughput normalized by GPU count. Key InferenceX comparison metric |
| `prompt_throughput_tok_per_sec_per_gpu` | tok/s/GPU | Prefill throughput per GPU |
| `total_output_tokens` | tokens | Cumulative generation tokens since server start |
| `total_prompt_tokens` | tokens | Cumulative prefill tokens since server start |
| `total_requests_succeeded` | count | Cumulative successfully completed requests |
| `requests_by_finish_reason` | dict | Breakdown: `stop` / `length` / `abort` / other |
| `preemptions_total` | count | Cumulative KV cache preemptions (forced eviction when cache full) |

> **InferenceX note:** `output_throughput_tok_per_sec_per_gpu` is the primary InferenceX y-axis for comparing GPU efficiency. For this H100 x8 setup running Qwen3-235B FP8 non-disaggregated, you're achieving ~15–16 tok/s/GPU.

---

#### 3b. Latency Histograms

Every histogram metric below is stored as a dict with the following keys:

```json
{
  "count": 37,          // total number of completed requests observed
  "sum":   16.78,       // sum of all values (seconds)
  "mean":  0.454,       // arithmetic mean
  "p50":   0.034,       // 50th percentile (median)
  "p90":   0.062,       // 90th percentile
  "p95":   0.071,       // 95th percentile
  "p99":   0.081        // 99th percentile
}
```

All time values are in **seconds** unless noted.

| Metric Key | vLLM Prometheus Source | Description |
|------------|----------------------|-------------|
| `ttft` | `vllm:time_to_first_token_seconds` | **Time To First Token** — wall-clock time from request arrival to first generated token. Includes queue wait + prefill time. Critical for streaming UX. |
| `tpot` | `vllm:request_time_per_output_token_seconds` | **Time Per Output Token** — average wall-clock time per generated token for a request. Lower = more responsive. |
| `itl` | `vllm:inter_token_latency_seconds` | **Inter-Token Latency** — time between consecutive tokens. More fine-grained than TPOT; captures jitter. |
| `e2e_latency` | `vllm:e2e_request_latency_seconds` | **End-to-End latency** — total wall-clock time from request arrival to last token. |
| `queue_time` | `vllm:request_queue_time_seconds` | Time a request spent in WAITING state before prefill began. High values = server overloaded. |
| `prefill_time` | `vllm:request_prefill_time_seconds` | Time spent in the PREFILL phase (processing input tokens). Scales with prompt length. |
| `decode_time` | `vllm:request_decode_time_seconds` | Time spent in the DECODE phase (generating output tokens). Scales with output length. |
| `inference_time` | `vllm:request_inference_time_seconds` | Total time in RUNNING state (prefill + decode, excluding queue wait). |
| `request_prompt_tokens` | `vllm:request_prompt_tokens` | Distribution of prompt lengths across completed requests. |
| `request_generation_tokens` | `vllm:request_generation_tokens` | Distribution of output lengths across completed requests. |
| `request_max_gen_tokens` | `vllm:request_max_num_generation_tokens` | Distribution of requested `max_tokens` parameter values. |
| `request_max_tokens_param` | `vllm:request_params_max_tokens` | Distribution of `max_tokens` as passed by the client. |
| `iteration_tokens` | `vllm:iteration_tokens_total` | Tokens processed per engine step (prefill + decode tokens combined per scheduler iteration). |
| `prefill_kv_computed_tokens` | `vllm:request_prefill_kv_computed_tokens` | New KV tokens actually computed during prefill (i.e. not served from prefix cache). `prompt_tokens - cached_tokens`. |

---

#### 3c. Interactivity (tok/s/user) — Core InferenceX Metric

**Interactivity** is the SemiAnalysis InferenceX primary x-axis. It is defined as how fast each individual user receives tokens — the inverse of TPOT.

```
interactivity (tok/s/user) = 1 / TPOT (s/tok)
```

| Metric | Description |
|--------|-------------|
| `interactivity_p50_tok_per_sec_per_user` | Median interactivity across completed requests |
| `interactivity_p90_tok_per_sec_per_user` | 90th percentile interactivity |
| `interactivity_p95_tok_per_sec_per_user` | 95th percentile interactivity |
| `interactivity_p99_tok_per_sec_per_user` | 99th percentile interactivity |
| `interactivity_mean_tok_per_sec_per_user` | Mean interactivity |

> **Intuition:** Higher interactivity = faster perceived typing speed for each user. The fundamental tradeoff: increasing batch size increases total throughput but decreases per-user interactivity (each user gets tokens slower). InferenceX plots throughput vs interactivity as a Pareto curve.

---

#### 3d. Request Concurrency & Scheduling

| Metric | Description |
|--------|-------------|
| `requests_running` | Requests currently being processed by the GPU (prefill or decode in flight) |
| `requests_waiting` | Requests in the queue waiting for GPU availability |
| `effective_batch_size` | Approximation of current batch size (= `requests_running`) |

> **Context:** For non-disaggregated workloads, prefill and decode compete for GPU time. vLLM's scheduler interleaves them. High `requests_waiting` indicates GPU saturation.

---

#### 3e. KV Cache & Memory Pressure

| Metric | Description |
|--------|-------------|
| `kv_cache_usage_pct` | Fraction of KV cache slots occupied (%). 100% = fully occupied, triggers preemptions |
| `preemptions_total` | Cumulative KV cache preemptions — forced eviction of in-flight requests to free space. Non-zero = memory pressure |
| `preemptions_per_sec` | Rate of preemptions (should be 0 under normal load) |

> **KV cache sizing:** With `gpu_memory_utilization=0.92` and FP8 weights on 8x H100 (80GB), the remaining VRAM (~4-6 GB/GPU) is used for KV cache. The `kv_cache_usage_pct` tracks how full this is. For long reasoning traces (Qwen3 thinking mode), this can grow significantly.

---

#### 3f. Prefix Cache (Automatic Prefix Caching / APC)

| Metric | Description |
|--------|-------------|
| `prefix_cache_hit_rate_pct` | % of prompt tokens found in the prefix cache (saved recompute). 80%+ = excellent |
| `prefix_cache_hits_total` | Cumulative cached tokens served from local prefix cache |
| `prefix_cache_queries_total` | Cumulative tokens queried against the prefix cache |
| `external_prefix_cache_hit_rate_pct` | Hit rate for cross-instance KV cache sharing (via KV connector — 0% in non-disaggregated) |
| `prompt_tokens_cached_total` | Total prompt tokens served from cache (local + external) |
| `prompt_tokens_recomputed_total` | Tokens that had to be recomputed despite being in cache (cache invalidation events) |

> **High hit rate (80%+):** In multi-turn workloads with shared system prompts, prefix caching is extremely effective. Each turn shares the prefix of the conversation history, so only the new turn needs actual prefill compute.

---

#### 3g. Energy Efficiency (InferenceX-style)

| Metric | Unit | Description |
|--------|------|-------------|
| `total_power_draw_w` | Watts | Aggregate power draw across all 8 H100s |
| `avg_power_draw_per_gpu_w` | Watts | Average per-GPU power draw |
| `total_power_limit_w` | Watts | Aggregate TDP/power cap across all GPUs |
| `output_tokens_per_watt` | tok/W | **Primary energy efficiency metric.** Generated tokens per Watt of GPU power. Higher = more efficient. |
| `picojoules_per_output_token` | pJ/tok | Energy cost per generated token in picojoules (1 pJ = 10⁻¹² J). InferenceX charts this on a log scale. |
| `millijoules_per_output_token` | mJ/tok | Same metric in a more human-readable unit. |

> **Formula:** `pJ/token = (total_power_W / output_throughput_tok_per_s) × 10¹²`
>
> **InferenceX context:** The SemiAnalysis article tracks pJ/token as a key TCO metric. GB200 NVL72 achieves dramatically lower pJ/token than H100 due to both higher throughput and lower power draw per token. For non-disaggregated H100 FP8, expect ~10,000–15,000 mJ/token at moderate batch sizes.

---

#### 3h. Multi-Modal Cache

| Metric | Description |
|--------|-------------|
| `mm_cache_hit_rate_pct` | Hit rate for multi-modal (image/video) token cache. 0% for text-only workloads. |

---

### 4. `rates` — Per-Second Counter Deltas

Prometheus counters are monotonically increasing. The collector computes the **delta between consecutive polls** to give per-second rates:

| Metric | Description |
|--------|-------------|
| `generation_tokens_per_sec` | Output tokens generated per second (same as `output_throughput_tok_per_sec`) |
| `prompt_tokens_per_sec` | Prompt tokens processed per second |
| `request_success_per_sec` | Completed requests per second |
| `num_preemptions_per_sec` | Preemptions per second |
| `prefix_cache_hits_per_sec` | New cache hits per second |
| `prefix_cache_queries_per_sec` | New cache queries per second |
| `external_prefix_cache_hits_per_sec` | External cache hits per second |
| `external_prefix_cache_queries_per_sec` | External cache queries per second |

> **Note:** The first poll will show `null` for all rates since there is no previous value to diff against. Rates become valid from the second poll onwards.

---

### 5. `vllm_metrics` — All Raw vLLM Prometheus Metrics (Flat)

Every `vllm:*` metric exposed by the server, parsed into a flat dict. This includes all the above plus lower-level internals. Histogram metrics are stored as nested dicts keyed by their full label string (e.g. `'engine="0",le="0.04",model_name="..."'`).

Key metrics you'll see here that aren't promoted to `inferencex_metrics`:

| Metric | Description |
|--------|-------------|
| `cache_config_info` | KV cache block size, number of blocks, swap blocks |
| `engine_sleep_state` | Whether the engine is in sleep/offload mode |
| `request_params_n` | Distribution of the `n` (beam count) parameter |
| `mm_cache_hits_total` | Multi-modal cache raw counters |

---

### 6. `gpu_stats` — Per-GPU Runtime Metrics

Array of one dict per GPU (8 for this setup), queried via `nvidia-smi`:

| Field | Unit | Description |
|-------|------|-------------|
| `index` | int | GPU index (0–7) |
| `name` | str | GPU model name |
| `temperature_c` | °C | GPU die temperature |
| `utilization_gpu_pct` | % | SM (compute) utilization |
| `utilization_mem_pct` | % | Memory controller utilization |
| `memory_total_mib` | MiB | Total HBM3 capacity |
| `memory_used_mib` | MiB | Used HBM3 (model weights + KV cache) |
| `memory_free_mib` | MiB | Free HBM3 |
| `power_draw_w` | W | Current power draw |
| `power_limit_w` | W | Configured power cap (TDP) |
| `sm_clock_mhz` | MHz | Current SM (shader) clock speed |
| `mem_clock_mhz` | MHz | Current HBM3 memory clock speed |
| `pcie_gen_current` | int | Active PCIe generation (5 = PCIe 5.0) |
| `pcie_width_current` | int | Active PCIe link width (16 = x16) |

---

### 7. `http_metrics` — API Server HTTP Metrics

Metrics from the FastAPI/uvicorn server that fronts vLLM:

| Metric | Description |
|--------|-------------|
| `http_requests_total` | Total HTTP requests by method, status code, and handler |
| `http_request_duration_seconds` | Request latency histogram by handler (low-resolution) |
| `http_request_duration_highr_seconds` | High-resolution request latency histogram (many buckets) |
| `http_request_size_bytes` | Incoming request payload size |
| `http_response_size_bytes` | Outgoing response payload size |

---

### 8. `process_metrics` — Python Process Metrics

| Metric | Description |
|--------|-------------|
| `process_cpu_seconds_total` | Total CPU time (user + system) consumed by the vLLM process |
| `process_resident_memory_bytes` | RSS — physical RAM used by the vLLM process |
| `process_virtual_memory_bytes` | VSS — virtual memory mapped by the vLLM process |
| `process_open_fds` | Number of open file descriptors |
| `process_max_fds` | Maximum allowed file descriptors |
| `process_start_time_seconds` | Unix timestamp when the process started |

---

### 9. `engine_state` — vLLM Engine State

| Field | Description |
|-------|-------------|
| `awake` | `1` = engine is active, `0` = engine is sleeping (used in sleep/offload mode) |
| `weights_offloaded` | `1` = weights offloaded to CPU (sleep level 1) |
| `discard_all` | `1` = KV cache discarded (sleep level 2) |

---

### 10. `cache_config` — KV Cache Configuration

Exposed by vLLM's `vllm:cache_config_info` gauge. Contains block size, number of GPU/CPU blocks, swap strategy, and other cache configuration parameters from the engine initialization.

---

## Understanding the Throughput vs Interactivity Tradeoff

The core insight from SemiAnalysis InferenceX is that inference performance is a **curve**, not a single number:

```
High Throughput │  ████ (high batch, low interactivity)
                │ ██
                │ ██
Low Throughput  │                              ██ (batch 1, high interactivity)
                └──────────────────────────────────────
                  Low interactivity         High interactivity
                  (slow per user)           (fast per user, tok/s/user)
```

For your current **non-disaggregated** setup:
- You're running **batch size 4**, placing you in the **mid-range** of the curve
- **~29 tok/s/user** interactivity (p50 TPOT-derived)
- **~125 tok/s** total throughput across all 4 concurrent requests
- **~15.7 tok/s/GPU** — competitive for H100 FP8 on a 235B MoE non-disagg workload

To push toward **higher throughput** (lower cost/token): increase `--max-num-seqs` and batch more requests.
To push toward **higher interactivity** (better UX): serve fewer concurrent requests.

---

## Analyzing the Output

### Quick one-liner summaries

```bash
# Show all polls condensed
jq -r '[.timestamp, .inferencex_metrics.output_throughput_tok_per_sec, .inferencex_metrics.interactivity_p50_tok_per_sec_per_user, .inferencex_metrics.kv_cache_usage_pct, .inferencex_metrics.total_power_draw_w] | @tsv' metrics/metrics.jsonl

# Show TTFT p99 over time
jq -r '[.timestamp, .inferencex_metrics.ttft.p99] | @tsv' metrics/metrics.jsonl

# Show energy efficiency over time
jq -r '[.timestamp, .inferencex_metrics.millijoules_per_output_token] | @tsv' metrics/metrics.jsonl

# Show prefix cache hit rate trajectory
jq -r '[.timestamp, .inferencex_metrics.prefix_cache_hit_rate_pct] | @tsv' metrics/metrics.jsonl

# Show per-GPU utilization for GPU 0
jq -r '[.timestamp, (.gpu_stats[] | select(.index == 0) | .utilization_gpu_pct, .power_draw_w)] | @tsv' metrics/metrics.jsonl
```

### Load the full dataset in Python

```python
import json
from pathlib import Path

records = [json.loads(l) for l in Path("metrics/metrics.jsonl").read_text().splitlines() if l]

# Extract throughput time series
timestamps = [r["timestamp"] for r in records]
throughput  = [r["inferencex_metrics"]["output_throughput_tok_per_sec"] for r in records]
interactivity = [r["inferencex_metrics"]["interactivity_p50_tok_per_sec_per_user"] for r in records]
energy_mj   = [r["inferencex_metrics"]["millijoules_per_output_token"] for r in records]
```

---

## File Layout

```
metrics/
├── README.md                  ← This file
├── collect_metrics.py         ← The collector script
├── metrics.jsonl              ← Structured metrics (append-only, one record per poll)
└── prometheus.json            ← Raw Prometheus ground truth (all scrapes, full text + parsed)
```

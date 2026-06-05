# Disaggregated Serving — Configuration Matrix
## Qwen3-235B-A22B-FP8 | 4P + 4D | 8x H100 80GB

---

## Memory Baseline

| Parameter | Value |
|-----------|-------|
| Decode GPUs | 4x H100 80GB (GPUs 4-7) |
| FP8 weights at TP=4 | ~58.75 GB/GPU |
| GPU memory utilization | 0.85 → ~69.3 GB usable/GPU |
| KV headroom per decode GPU | ~9.25 GB |
| Total decode KV pool | ~37 GB (4 GPUs) |
| KV cache dtype | FP8 @ 46.875 KB/token |
| **Total decode token slots** | **~786K tokens** |
| max_model_len | 96K (98304) |
| MAX_COMPLETION_TOKENS (orchestrator) | 32K (32768) |

Prefill KV is **transient** — computed on GPUs 0-3 and immediately transferred to decode via NVLink, then freed. Prefill max_num_seqs can therefore be set 2x decode without KV pressure.

---

## Configuration Matrix

### Config 1 — Light (batch 4 baseline)
```
PREFILL  max_num_seqs = 8
DECODE   max_num_seqs = 8
Orchestrator batch_size = 4
```
| Metric | Value |
|--------|-------|
| Decode KV budget/seq | ~98K tokens |
| Coverage | Full max_model_len (96K) fits with headroom |
| Preemption risk | None |
| Swap pressure | None |
| Notes | Safest config for initial disagg validation. Good for debugging KV transfer. Matches batch-4 non-disagg run. |

---

### Config 2 — Moderate (batch 8)
```
PREFILL  max_num_seqs = 16
DECODE   max_num_seqs = 16
Orchestrator batch_size = 8
```
| Metric | Value |
|--------|-------|
| Decode KV budget/seq | ~49K tokens |
| Coverage | Covers MAX_COMPLETION_TOKENS (32K) + ~17K prompt headroom |
| Preemption risk | Very low |
| Swap pressure | None |
| Notes | Comfortable for multi-turn traces. Equivalent to current non-disagg default. Good comparison point for batch-8 non-disagg. |

---

### Config 3 — Balanced (batch 16)
```
PREFILL  max_num_seqs = 32
DECODE   max_num_seqs = 24
Orchestrator batch_size = 16
```
| Metric | Value |
|--------|-------|
| Decode KV budget/seq | ~32.7K tokens |
| Coverage | Matches MAX_COMPLETION_TOKENS (32K) ceiling |
| Preemption risk | Low — tight but workable for typical traces |
| Swap pressure | Light on worst-case multi-turn turn 4 |
| Notes | Practical sweet spot for disagg. Prefill runs ahead of decode (2x seqs → prefill never bottlenecks). Direct comparison to batch-16 non-disagg. |

---

### Config 4 — High Throughput (batch 32)
```
PREFILL  max_num_seqs = 48
DECODE   max_num_seqs = 32
Orchestrator batch_size = 32
```
| Metric | Value |
|--------|-------|
| Decode KV budget/seq | ~24.6K tokens |
| Coverage | Below MAX_COMPLETION_TOKENS — preemption expected on long reasoning traces |
| Preemption risk | Medium — long think blocks (~20K+ tokens) will trigger preemption |
| Swap pressure | Moderate — 16 GB swap will be used |
| Notes | Pushes decode utilization hard. vLLM scheduler will reorder/preempt to fit. Throughput gain over Config 3 depends on actual trace length distribution. Matches batch-32 non-disagg run. |

---

### Config 5 — Heavy / Max Pressure (batch 64)
```
PREFILL  max_num_seqs = 64
DECODE   max_num_seqs = 48
Orchestrator batch_size = 64
```
| Metric | Value |
|--------|-------|
| Decode KV budget/seq | ~16.4K tokens |
| Coverage | Well below MAX_COMPLETION_TOKENS — heavy preemption guaranteed on reasoning traces |
| Preemption risk | High — scheduler will preempt aggressively |
| Swap pressure | Heavy — 16 GB swap fully utilized |
| Notes | Maximum concurrency attempt. Throughput may plateau or regress vs Config 4 due to preemption overhead. Useful for finding the real ceiling. Matches batch-64 non-disagg run. Watch metrics closely — if preemption count spikes and token throughput drops, Config 4 was the actual peak. |

---

## Summary Table

| Config | Prefill max_num_seqs | Decode max_num_seqs | Batch size | Decode KV/seq | Preemption risk |
|--------|---------------------|---------------------|------------|---------------|-----------------|
| 1 — Light | 8 | 8 | 4 | ~98K | None |
| 2 — Moderate | 16 | 16 | 8 | ~49K | Very low |
| 3 — Balanced | 32 | 24 | 16 | ~32.7K | Low |
| 4 — High | 48 | 32 | 32 | ~24.6K | Medium |
| 5 — Heavy | 64 | 48 | 64 | ~16.4K | High |

---

## How to Change Configs

In `scripts/serve_qwen3_235b_disagg.sh`, update these two variables:

```bash
MAX_NUM_SEQS=16   # set decode value from table above
```

Since prefill and decode share `MAX_NUM_SEQS` in the current script, to set them independently override per-instance:

```bash
# Prefill instance launch — add --max-num-seqs <prefill_value>
# Decode instance launch  — add --max-num-seqs <decode_value>
```

And the orchestrator:

```bash
uv run --no-sync python orchestrator.py --multi-turn --batch-size <N>
```

---

## What to Watch in Metrics

When running heavy configs (4–5), monitor `metrics/collect_metrics.py` output for:

- `preemption_total` — rising fast = KV pressure too high
- `kv_cache_usage_pct` — sustained >95% = swap kicking in
- `output_token_throughput` — if this drops vs a lighter config, you've passed the peak
- `tpot_p99` — inter-token latency spikes signal decode-side stalling

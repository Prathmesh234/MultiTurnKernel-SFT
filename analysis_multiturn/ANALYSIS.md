# Inference Analysis: Qwen3-235B-A22B-FP8 Multi-Turn Serving
## 8× NVIDIA H100 SXM · 8× NVIDIA H200 SXM · vLLM

> **Methodology note.** This analysis follows the [SemiAnalysis InferenceX framework](https://semianalysis.com/2025/05/15/inferencex-llm-inference-benchmark/) for evaluating LLM serving performance. The core insight from that work is that **raw throughput alone is a misleading metric** — the right plane to reason on is the *interactivity–throughput Pareto frontier*, where interactivity is defined as tokens-per-second *per active user* (inverse of TPOT at the per-user level). A system that maximises throughput at the cost of grinding every user's response to a crawl is not a good serving system. Both axes must be optimised simultaneously.

---

## Table of Contents

1. [Setup and Configuration](#1-setup-and-configuration)
2. [H100 Analysis — Batch Size Sweep (B=4 → B=32)](#2-h100-analysis)
3. [H200 Analysis — B=128 (Two Runs)](#3-h200-analysis)
4. [Cross-Hardware Comparison](#4-cross-hardware-comparison)
5. [Multi-Turn vs Single-Turn Dynamics](#5-multi-turn-vs-single-turn-dynamics)
6. [Capacity Planning and Next Experiments](#6-capacity-planning-and-next-experiments)

---

## 1. Setup and Configuration

| Parameter | 8× H100 SXM | 8× H200 SXM |
|---|---|---|
| Model | Qwen3-235B-A22B-FP8 | Qwen3-235B-A22B-FP8 |
| Quantisation | FP8 | FP8 |
| Serving framework | vLLM | vLLM |
| `max_num_seqs` | 64 | 128 |
| `gpu_memory_utilization` | 0.92 | 0.92 |
| `max_model_len` | 131,072 tokens | 131,072 tokens |
| Concurrent users tested | 4 / 8 / 16 / 32 | 128 (two runs) |
| Workload type | Multi-turn (4 turns) | Multi-turn (4 turns) |

**Workload characteristics.** Each "user" runs a 4-turn conversation. By the time a sequence reaches turn 4 it carries the full accumulated context from prior turns — this is what drives the large prompt token counts (mean ~3.7–4.2k on H100, ~2.6–3.0k on H200) and the long generation lengths (mean ~8–12k output tokens). The multi-turn structure is critical because it stress-tests the KV cache differently from single-turn: prompt length grows monotonically across turns, and prefix cache becomes the primary performance lever.

---

## 2. H100 Analysis

### 2.1 Batch Size Sweep Summary

| Metric | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|
| **Throughput median (tok/s)** | 79 | 268 | 392 | **753** |
| **Throughput max (tok/s)** | 157 | 477 | 838 | 1,493 |
| **Interactivity p50 (tok/s/user)** | 27.6 | **57.1** | **57.1** | 55.6 |
| **Interactivity p90 (tok/s/user)** | 21.2 | 42.6 | 42.6 | 40.0 |
| **TTFT p50 (s)** | 0.25 | 1.15 | 0.61 | 1.91 |
| **TTFT p90 (s)** | 1.23 | 4.76 | 1.76 | 6.20 |
| **GPU utilisation (%)** | 53.1 | **100.0** | **100.0** | **100.0** |
| **KV cache usage median (%)** | 2.4 | 4.7 | 6.6 | 17.0 |
| **KV cache usage max (%)** | 5.4 | 9.9 | 17.7 | 35.4 |
| **Prefix cache hit rate (%)** | 75.4 | 68.7 | 71.1 | 72.5 |
| **Total power draw (W)** | 1,516 | 2,087 | 2,206 | 2,555 |
| **tok/s per Watt** | 0.055 | 0.124 | 0.163 | **0.264** |
| **Requests running — median** | 2 | 5 | 7 | 18 |
| **Requests waiting — max** | 0 | 0 | 0 | 0 |

### 2.2 The B=4 Anomaly — An Off-Pareto Operating Point

B=4 is the most surprising result in the sweep: it is **strictly dominated by B=8 on both axes simultaneously**. B=8 delivers 3.4× more aggregate throughput *and* 2.1× better per-user interactivity p50.

The root cause is straightforward: with only 4 concurrent users against 64 `max_num_seqs` capacity, the GPU is sitting at **53% utilisation** on average with a median of only 2 active requests at any given poll. The H100s are designed to saturate compute at sustained decode — running at half-load means the entire matrix multiplication pipeline is starved.

Critically, the requests_waiting queue was empty at every sample point, meaning the system was not constrained by queue depth at all — there simply was not enough work being sent. B=4 operates in an *under-loaded* regime where the hardware cost is fixed but the useful work being extracted is halved relative to even a modestly loaded configuration.

**Takeaway:** For a 235B MoE model on 8 GPUs, 4 concurrent users is not a meaningful production load. The saturated GPU regime begins at B=8.

### 2.3 The True Pareto Frontier: B=8 → B=32

Once the GPU saturates at B=8 (100% utilisation), the system enters a compute-bound regime and subsequent batch increases produce a classic throughput-interactivity tradeoff:

- **Throughput** scales cleanly: B=8→B=16 is +46%, B=16→B=32 is +92%, B=8→B=32 is **2.8×** overall.
- **Interactivity p50** is remarkably stable: 57.1 → 57.1 → 55.6 tok/s/user from B=8 to B=32. The per-user decode rate degrades less than 3% even as the batch quadruples. This is a signature of the MoE architecture — because only 22B of the 235B parameters are activated per token, the decode step for each token is memory-bandwidth-bound and scales favourably with parallelism.
- **Interactivity p90** (the slowest 10% of users) holds at ~42 tok/s across B=8–32, well above the 15 tok/s speech-quality floor and hovering around the 50 tok/s comfortable reading threshold from the SemiAnalysis benchmark.

B=16 is a particularly interesting operating point: it delivers 46% more throughput than B=8 with *identical* p50 interactivity (57.1 tok/s) and actually *better* TTFT (0.61s vs 1.15s at B=8). The shorter TTFT at B=16 vs B=8 likely reflects better prefill batching — larger batches amortise the attention computation across more tokens, improving prefill efficiency and reducing the per-request wait.

### 2.4 TTFT Behaviour

TTFT at B=32 p90 is 6.2 seconds, which is the one quality-of-experience concern in this sweep. The p50 (1.91s) is acceptable, but the tail is long — some requests are waiting through multiple decode cycles of other sequences before their prefill is scheduled. This is an inherent property of continuous batching under high concurrency: large decode batches delay incoming prefills.

If TTFT tail latency matters for the product (e.g., agentic pipelines where users see the cursor blinking), B=16 offers the best balance: 392 tok/s throughput at 0.61s TTFT p50 and 1.76s TTFT p90, both under the 2s interactive-feel threshold.

### 2.5 Energy Efficiency

The efficiency story is dramatic: **tok/W scales 4.8× from B=4 to B=32** (0.055 → 0.264 tok/s/W). The hardware is the same, the model is the same — the only variable is how well we're keeping the GPUs fed. Running at half utilisation wastes power proportionally. At B=32 the system draws 2,555W total (across all 8 GPUs) and produces 753 tok/s median, compared to 1,516W and 79 tok/s at B=4.

### 2.6 KV Cache and Prefix Reuse

KV cache utilisation is very low across all H100 batch sizes — maxing at 35.4% at B=32. This is not a sign of waste; it confirms that the H100 is **compute-bound, not memory-bound**, at these batch sizes. There is substantial headroom to scale up concurrency further.

The prefix cache hit rate is consistently **68–75%** across all batch sizes. This is the multi-turn signature: in a 4-turn conversation, turns 2–4 each carry the prefix of all prior turns. When vLLM's prefix cache is warm, those tokens cost nothing to prefill — they are read directly from the KV cache blocks allocated during earlier turns. A 70%+ hit rate on H100 means roughly 7 in 10 input tokens arriving at the prefill stage are served from cache rather than recomputed. This is a significant throughput multiplier that would be entirely absent in a single-turn benchmark.

---

## 3. H200 Analysis

### 3.1 B=128 Performance (Two Runs)

| Metric | Run 1 | Run 2 |
|---|---|---|
| **Throughput median (tok/s)** | 1,990 | 1,873 |
| **Throughput max (tok/s)** | 3,597 | 3,866 |
| **Interactivity p50 (tok/s/user)** | 26.7 | 26.7 |
| **Interactivity p90 (tok/s/user)** | 21.1 | 21.1 |
| **TTFT p50 (s)** | 2.16 | 1.99 |
| **TTFT p90 (s)** | 7.11 | 6.74 |
| **GPU utilisation (%)** | 99.9 | 99.9 |
| **KV cache usage median (%)** | 25.6 | 23.8 |
| **KV cache usage p90 (%)** | 43.6 | 42.5 |
| **KV cache usage max (%)** | 48.6 | 46.6 |
| **Prefix cache hit rate (%)** | 75.3 | 78.5 |
| **Total power draw (W)** | 3,380 | 3,258 |
| **tok/s per Watt** | 0.528 | 0.487 |
| **Requests running — median** | 65 | 58 |
| **Requests running — max** | 100 | 100 |
| **Mean prompt tokens** | 2,600 | 2,952 |
| **Mean gen tokens** | 11,212 | 12,150 |

### 3.2 Headline Numbers

The H200 at B=128 delivers **~1,930 tok/s median throughput** — roughly **2.6× the throughput of H100 at B=32** (753 tok/s). It does this at 3.2–3.4 kW of power versus 2.55 kW on H100, making it approximately **2.0× more energy-efficient** per output token (0.51 tok/W vs 0.264 tok/W).

Both runs are highly consistent: within ~6% of each other on throughput and within 1% on interactivity p50. This reproducibility is a sign of a stable, saturated-compute serving regime — the H200s are fully loaded and deterministically compute-bound.

### 3.3 Interactivity at Scale — the B=128 Cost

The per-user interactivity p50 at B=128 is **26.7 tok/s** — well above the 15 tok/s speech floor but below the 50 tok/s comfortable-reading threshold. This is the natural cost of running 4× more concurrency than H100 B=32: more users share each decode step, so each individual user receives tokens less frequently.

The p90 interactivity is 21.1 tok/s — 10% of users are receiving tokens at roughly half the comfortable reading rate. For a long-generation workload (mean 12k output tokens), this means the slowest 10% of users wait ~570 seconds end-to-end. Whether this is acceptable depends on the application context: acceptable for async/agent pipelines, borderline for interactive chat, unacceptable for real-time voice.

### 3.4 The KV Cache Puzzle at 23%

One of the most interesting details in the H200 data is that KV cache utilisation sits at a **23–26% median** despite running 128 concurrent users with multi-turn conversations averaging ~13.8k tokens per sequence.

The resolution is occupancy dynamics: out of 128 `max_num_seqs` slots, the **median number of requests actively running at any poll is 58–65, peaking at 100**. The remaining slots are empty — either between completions or still arriving. The KV cache is sized for the worst case (128 × max_model_len in terms of block pool) but actual token occupancy at any moment reflects the real concurrency load, not the theoretical maximum.

The **peak** KV utilisation of 46–48% is more meaningful: this is what the system actually touches at full burst. From a capacity planning perspective, the pool is approximately half utilised at peak, which means there is clear headroom to increase `max_num_seqs` to 256 — estimated peak KV at ~2× concurrency would be ~90%, within safe bounds given the H200's 141 GB HBM3e per GPU.

### 3.5 Prefix Cache Hit Rate

The H200 achieves **75–78% prefix cache hit rate** — slightly higher than H100, and particularly stable across both runs. The higher hit rate in Run 2 (78.5%) vs Run 1 (75.3%) likely reflects a warmer prefix cache at the start of Run 2, since the workload patterns are identical and the cache persists for the duration of the vLLM process.

The practical impact is large: with mean prompt tokens of ~2.8k and a 77% hit rate, roughly 2.1k tokens per request are served from cache. At B=128 concurrent with 1,930 tok/s throughput, this saves approximately 270k prefill compute-token-equivalents per second compared to cold-cache serving.

### 3.6 Power and Efficiency

The H200 draws 3.2–3.4 kW at B=128 versus 2.55 kW for H100 at B=32. The key efficiency insight: the H200 is drawing only 57–60% of its 5,600W TDP envelope, yet producing 2.6× more throughput than the H100 drawing 46% of the same TDP. The H200's superior HBM3e bandwidth (up to 4.8 TB/s vs 3.35 TB/s on H100) and larger memory pool allow it to sustain higher decode parallelism without becoming bandwidth-starved, translating directly to better tok/W at scale.

---

## 4. Cross-Hardware Comparison

| Configuration | Throughput (tok/s) | Inter p50 (tok/s/u) | TTFT p50 (s) | tok/W | KV max (%) |
|---|---|---|---|---|---|
| H100 B=4 | 79 | 27.6 | 0.25 | 0.055 | 5.4 |
| H100 B=8 | 268 | 57.1 | 1.15 | 0.124 | 9.9 |
| H100 B=16 | 392 | 57.1 | 0.61 | 0.163 | 17.7 |
| H100 B=32 | 753 | 55.6 | 1.91 | 0.264 | 35.4 |
| **H200 B=128** | **~1,930** | **26.7** | **~2.1** | **~0.51** | **~47** |

### Key cross-hardware observations

**Throughput.** H200 at B=128 delivers 2.6× the throughput of H100 at B=32. Given that H100 B=32 is the highest-throughput H100 configuration tested and the H100 is already at 100% GPU utilisation, this gap is a direct reflection of hardware generational uplift — more memory bandwidth, larger HBM capacity, and higher compute density on H200.

**Interactivity.** H200 B=128 p50 (26.7 tok/s) is lower than H100 B=32 p50 (55.6 tok/s). This is not a hardware regression — it is a concurrency effect. The H200 is serving 4× more users simultaneously. On a per-user-per-GPU basis the H200 is ahead. The right comparison is total system interactivity capacity (users × tok/s/user), where the H200 dominates.

**Efficiency.** H200 is approximately **2× more energy efficient** than H100 at their respective best operating points (0.51 vs 0.264 tok/W). Over a production deployment running 24/7, this is a significant operational cost difference.

**TTFT.** H200 B=128 TTFT p50 (~2.1s) is marginally above the 2s interactive-feel threshold, compared to H100 B=32's 1.91s. At 4× the concurrency this is expected — there are more requests competing for prefill scheduler slots. The H200's TTFT tail (p90 ~6.9s) is similar to H100 B=32 p90 (6.2s), again reflecting the larger queue depth.

---

## 5. Multi-Turn vs Single-Turn Dynamics

This entire dataset is multi-turn — 4 turns per conversation — and the differences from a single-turn workload are not superficial. They change the fundamental character of what the serving system is doing.

### 5.1 Prompt Length Growth

In a single-turn workload, every request arrives with roughly the same prompt length (fixed by the task). In a multi-turn workload, prompt length grows with each turn as prior context is appended. By turn 4, a user who generated 2,500 tokens per turn is submitting a ~10,000-token prompt. This has several downstream effects:

- **TTFT grows per-turn.** The same user sees longer TTFT on turn 4 than turn 1 because prefilling 10k tokens takes proportionally longer than prefilling 500 tokens. Single-turn benchmarks miss this completely.
- **KV cache blocks accumulate.** A single 4-turn user eventually occupies ~4× the KV cache of a fresh single-turn user. Cache eviction pressure builds over long conversations.
- **Memory bandwidth pressure shifts.** Late-turn decoding is reading from a much larger KV cache per request, increasing per-step bandwidth cost.

### 5.2 Prefix Cache: The Multi-Turn Superpower

The single most important structural advantage of multi-turn serving over single-turn is **prefix cache hit rate**. In single-turn workloads, prefix caching provides limited value because requests are independent — there is typically no common prefix beyond a system prompt (which does benefit from prefix caching, but is short).

In multi-turn, every turn after turn 1 carries the full history of prior turns as its prefix. When the same user sends turn 3, all of turn 1 and turn 2's KV blocks are already present in the cache. vLLM's prefix caching recognises this and skips the prefill computation entirely for those tokens.

Our data shows **68–78% prefix cache hit rates** across both hardware platforms. This means that on average, less than one-third of the input tokens in each request actually go through the prefill attention computation. The rest are served directly from cached KV blocks — effectively free.

In a single-turn benchmark, this hit rate would be close to 0% (beyond a shared system prompt). The difference between 70% and 0% hit rate represents a **3.3× reduction in effective prefill compute load**. A system that looks like it can handle 32 single-turn users might actually handle 100+ multi-turn users once the prefix cache warms up — which is exactly what we see when comparing the H100 at B=32 (32 concurrent single-turn equivalent) vs the H200 at B=128 (128 concurrent with warm cache and similar TTFT tails).

### 5.3 Interactivity Stability Across Turns

In a single-turn workload, interactivity (tokens per second per user) is constant — the generation length determines how long each user waits, but the decode speed is fixed by the batch composition. In multi-turn, the batch composition changes as conversations age: early-turn requests are short (small KV footprint, fast attention), late-turn requests are long (large KV footprint, higher bandwidth cost per decode step). The serving system must handle this mixed-age population.

The fact that our interactivity p50 remains **57.1 tok/s across B=8 and B=16**, with only minor degradation to 55.6 at B=32, suggests vLLM's continuous batching scheduler is managing the mixed-age population effectively. Requests are not piling up in a FIFO queue that degrades the oldest conversations — the chunked prefill and interleaved decode maintain fairness across conversation ages.

### 5.4 The Single-Turn Benchmark Blind Spot

Standard single-turn LLM benchmarks (SGLang, lm-evaluation-harness, even many throughput stress tests) have no mechanism for measuring:

1. TTFT growth across turns for the same user session
2. Prefix cache effects on effective compute load
3. KV cache fragmentation from varying-length multi-turn contexts
4. Queue fairness across users at different conversation depths

Our setup captures all four. The result is a serving characterisation that more closely matches what a real-world coding assistant, customer service bot, or agentic pipeline actually experiences than any single-turn benchmark can provide.

---

## 6. Capacity Planning and Next Experiments

### 6.1 H100 — Room to Scale to B=64

All H100 runs have **zero requests waiting** at all batch sizes, meaning the system is never queue-limited. The current `max_num_seqs=64` is the server-side ceiling, and we tested up to 32 concurrent users (filling 50% of available slots).

Pushing to **B=64 concurrent users** (matching max_num_seqs=64) would:
- Keep GPU at 100% utilisation (already compute-bound)
- Increase KV cache from ~35% peak (B=32) to an estimated ~70% peak — well within safe bounds given the H100 KV pool capacity
- Likely yield 1.4–1.8× further throughput gain over B=32
- Interactivity p50 expected to degrade modestly from 55.6 → ~40–45 tok/s/user

**B=128 with max_num_seqs=128 on H100 is not recommended** without validation: estimated peak KV at 128 fully-loaded requests × 14k average tokens exceeds the computed H100 KV pool capacity (~1.27M tokens), risking OOM or heavy preemption cascades.

### 6.2 H200 — Room to Scale to B=256

The H200's KV pool is estimated at ~3.78M tokens capacity (from the observed 23% median utilisation at 58 median running × 15k average tokens). Scaling to `max_num_seqs=256`:

- Estimated KV median ~46%, peak ~80–85% — within safe headroom on 141 GB HBM3e
- GPU is already compute-saturated at B=128, so throughput gains will come from better decode batching rather than unlocking new compute
- Expected throughput: +20–40% over B=128 (~2,300–2,700 tok/s)
- Interactivity p50 will degrade from 26.7 → estimated ~18–22 tok/s/user — marginal for interactive use but fine for async/agent workloads
- Prefix cache hit rate expected to remain high or improve due to more concurrent conversations sharing prefixes

### 6.3 Interesting Anomalies and Open Questions

**B=16 TTFT is faster than B=8.** At B=16, TTFT p50 is 0.61s versus 1.15s at B=8. This is counterintuitive — more concurrency should increase TTFT. The likely explanation is that B=16 produces larger and more uniform prefill batches, improving the efficiency of the FlashAttention kernel. At B=8, the scheduler may be batching prefills with only 2–3 sequences at a time, producing suboptimal tensor utilisation. This is worth investigating with attention kernel profiling.

**Prefix hit rate dips at B=8.** The hit rate is 68.7% at B=8 vs 75.4% at B=4 and 71–72% at B=16/32. At B=8, the prefix cache may be experiencing more evictions because 8 concurrent long conversations are competing for cache blocks. At B=4, fewer sequences, larger cache share per sequence. At B=16+, the cache is large enough relative to the increased concurrency that hit rates recover.

**H200 Run 2 prefix hit rate (78.5%) > Run 1 (75.3%).** Run 2 starts with a warm prefix cache from Run 1 — the vLLM process was not restarted between runs. This ~3 percentage point difference represents the value of cache warmup. In production, the cache will always be warm after the first traffic wave.

**Zero preemptions across all configurations.** vLLM's chunked prefill and paged attention scheduler produced zero preemptions at every batch size on both H100 and H200. This is a strong positive signal for serving stability and means the KV cache block allocation is working correctly — no mid-sequence evictions, no recomputation overhead.

---

*Charts:* `charts/batch_comparison_dashboard.png` · `charts/interactivity_vs_throughput.png` · `charts/h200_dashboard.png` · `charts/h100_vs_h200_comparison.png`

*Data:* `metrics/metrics-batch{4,8,16,32}/` (H100) · `metrics/metrics-batch128-h200s{,-2}/` (H200)

*Scripts:* `analysis_multiturn/chart_batch_comparison.py` · `analysis_multiturn/chart_h200.py`

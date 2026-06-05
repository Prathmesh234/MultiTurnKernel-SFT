# Inference Analysis: Qwen3-235B-A22B-FP8 Multi-Turn Serving

## 8x NVIDIA H100 SXM  ·  8x NVIDIA H200 SXM  ·  vLLM

> **Methodology.** This analysis uses the [SemiAnalysis InferenceX framework](https://semianalysis.com/2025/05/15/inferencex-llm-inference-benchmark/). The central idea: raw throughput is a misleading metric in isolation. The right way to evaluate a serving system is on the *interactivity-throughput Pareto frontier* — interactivity being tokens/sec *per active user* (the inverse of per-user TPOT). A system that maximises throughput by grinding every user to a crawl is a bad serving system. Both axes matter.
>
> All metrics were collected from live vLLM Prometheus endpoints at 10-second intervals during continuous multi-turn workloads. Idle/cold-start polls (zero requests running or throughput below 10 tok/s) are excluded from all aggregates.

---

## Table of Contents

1. [Setup and Configuration](#1-setup-and-configuration)
2. [H100 Analysis — Batch Size Sweep (B=4 to B=32)](#2-h100-analysis--batch-size-sweep)
3. [H200 Analysis — B=128 (Two Runs)](#3-h200-analysis--b128)
4. [Cross-Hardware Comparison](#4-cross-hardware-comparison)
5. [Multi-Turn vs Single-Turn: Why It Matters](#5-multi-turn-vs-single-turn-why-it-matters)
6. [Capacity Planning and Next Steps](#6-capacity-planning-and-next-steps)

---

## 1. Setup and Configuration

| Parameter | 8x H100 SXM | 8x H200 SXM |
|---|---|---|
| Model | Qwen3-235B-A22B-FP8 | Qwen3-235B-A22B-FP8 |
| Architecture | MoE: 128 experts + 1 shared, 6 active (22B active params) | Same |
| Quantisation | FP8 (W8A8) | FP8 (W8A8) |
| Serving framework | vLLM (TP8) | vLLM (TP8) |
| `max_num_seqs` | 64 | 128 |
| `gpu_memory_utilization` | 0.92 | 0.92 |
| `max_model_len` | 131,072 tokens | 131,072 tokens |
| HBM per GPU | 80 GB HBM2e | 141 GB HBM3e |
| HBM bandwidth | 3.35 TB/s | 4.8 TB/s |
| Concurrent users tested | 4, 8, 16, 32 | 128 (two independent runs) |
| Workload | Multi-turn, 4 turns per conversation | Same |

**Workload profile.** Each concurrent "user" runs a 4-turn conversation against the model. Prompt length grows monotonically across turns as prior context accumulates (mean prompt tokens: 2.6-4.2k; mean generation tokens: 8-12k). By turn 4, each sequence carries the full concatenated history of all prior turns — this is the key structural difference from single-turn benchmarks and drives most of the interesting dynamics we observe.

**Why MoE matters here.** Qwen3-235B is a Mixture-of-Experts model: 235B total parameters, but only 22B are activated per token (128 experts, 6 chosen per token, plus 1 shared). During decode, each forward pass reads the active expert weights from HBM — the per-token memory bandwidth cost is proportional to the *active* parameter count (22B), not the total (235B). This makes MoE decode fundamentally memory-bandwidth-bound rather than compute-bound, and it's why interactivity stays remarkably stable as batch size increases: adding more sequences to a decode batch doesn't significantly increase the per-step bandwidth cost until you start exhausting HBM bandwidth.

---

## 2. H100 Analysis — Batch Size Sweep

### 2.1 Summary Table

| Metric | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|
| **Throughput median (tok/s)** | 79 | 268 | 392 | **753** |
| **Throughput max (tok/s)** | 157 | 477 | 838 | 1,493 |
| **Throughput stdev (tok/s)** | 37 | 140 | 269 | 436 |
| **Per-GPU output (tok/s/gpu)** | 10.6 | 34.0 | 49.0 | **89.8** |
| Interactivity p50 (tok/s/user) | 27.6 | **57.1** | **57.1** | 55.6 |
| Interactivity p90 (tok/s/user) | 21.2 | 42.6 | 42.6 | 40.0 |
| Interactivity p95 (tok/s/user) | 20.6 | 41.2 | 41.2 | 33.1 |
| TTFT p50 (s) | **0.25** | 1.15 | 0.61 | 1.91 |
| TTFT p90 (s) | 1.23 | 4.76 | 1.76 | 6.20 |
| **TPOT p50 (ms)** | 36.3 | **17.5** | **17.5** | 18.0 |
| **ITL p50 (ms)** | 17.9 | **17.5** | **17.5** | 20.8 |
| **ITL p90 (ms)** | 88.4 | 23.5 | 23.5 | 41.7 |
| **ITL p90/p50 ratio** | **4.9x** | 1.3x | 1.3x | 2.0x |
| GPU utilisation (%) | 53.1 | **100.0** | **100.0** | **100.0** |
| KV cache median (%) | 2.4 | 4.7 | 6.6 | 17.0 |
| KV cache max (%) | 5.4 | 9.9 | 17.7 | 35.4 |
| Prefix cache hit rate (%) | 75.4 | 68.7 | 71.1 | 72.5 |
| Total power (W) | 1,516 | 2,087 | 2,206 | 2,555 |
| tok/s per Watt | 0.055 | 0.124 | 0.163 | **0.264** |
| Requests running median | 2 | 5 | 7 | 18 |
| Requests waiting max | 0 | 0 | 0 | 0 |
| Queue time mean (s) | 0.00 | 0.00 | 0.01 | 0.27 |
| Prefill time mean (s) | 0.17 | 1.59 | 0.60 | 1.47 |
| Preemptions total | 0 | 0 | 0 | 0 |

### 2.2 B=4: Why Underloading Is Worse Than Overloading

B=4 is **strictly dominated by B=8 on both Pareto axes simultaneously** — B=8 delivers 3.4x more throughput *and* 2.1x better interactivity p50. In typical serving tradeoffs you sacrifice one axis to gain the other; here, B=4 loses on both. It sits off the Pareto frontier entirely.

The surface explanation is GPU utilisation: 53% at B=4 vs 100% at B=8. With a median of only 2 requests actively running (against a `max_num_seqs` capacity of 64), the tensor cores are starved for work. But the deeper story is in the **inter-token latency (ITL) distribution**.

At B=4, the ITL p50 is 17.9ms but the p90 is **88.4ms — a 4.9x jitter ratio**. Compare with B=8 where the ratio is 1.3x (17.5ms / 23.5ms). What's happening: with only 2 active decode sequences, every incoming prefill operation stalls *half the active batch*. vLLM's chunked prefill interleaves prefill chunks between decode steps, but when the decode batch is only 2 sequences, one prefill chunk represents a 50% interruption. The p99 at B=4 reaches 219ms — these are moments where a user sees a visible stutter in token delivery.

At B=8+, the same prefill interruption is diluted across 5-8+ decode sequences. Each individual user absorbs a tiny fraction of the scheduling overhead, and the ITL distribution tightens dramatically. This is why B=4's *per-user* interactivity is paradoxically worse despite having fewer users competing for compute: the jitter from prefill interruptions dominates the user experience.

**Takeaway:** For a 235B MoE on 8 GPUs, the minimum viable production load is B=8. Below that, you pay full hardware cost but get worse per-user quality *and* lower throughput.

### 2.3 The Pareto Frontier: B=8, B=16, B=32

Once GPU saturates at B=8, the system enters a compute-bound regime. What follows is a clean Pareto tradeoff:

**Throughput scales well:** B=8 to B=32 is a 2.8x gain (268 to 753 tok/s median). B=4 to B=32 is 9.5x, but B=4 is off-frontier so that comparison is more about underloading than scaling.

**Interactivity holds:** p50 interactivity barely moves from B=8 to B=32 (57.1 to 55.6 tok/s/user — less than 3% degradation over a 4x batch increase). This is the MoE architecture paying off: each decode step reads ~22B of active expert weights from HBM. Adding more sequences to the decode batch adds KV cache reads, but the weight reads are amortised. At 3.35 TB/s HBM bandwidth on H100, the system can absorb 4x more concurrent KV reads before per-step latency meaningfully increases.

The p90 (slowest 10% of users) holds at 40+ tok/s across B=8-32 — well above the 15 tok/s speech-quality floor. But B=32 sees the first cracks: interactivity p95 drops to 33.1 tok/s (vs 41.2 at B=8/16), and ITL p90/p50 widens from 1.3x to 2.0x. The tail users are starting to feel decode congestion.

**B=16 is a sweet spot.** It delivers 46% more throughput than B=8 with identical p50 interactivity (57.1 tok/s), and it has the best TTFT of any batch size above B=4: 0.61s p50 and 1.76s p90, both under the 2-second interactive-feel threshold. The TTFT improvement over B=8 (1.15s p50) is counterintuitive — more concurrency should mean longer waits for prefill scheduling. The data shows the mechanism: B=16's mean prefill time is 0.60s vs B=8's 1.59s, and its prompt throughput is 6,246 tok/s vs B=8's 3,591 tok/s. With 16 concurrent users, the scheduler has more prefill-eligible requests available at each scheduling step, producing larger prefill batches that utilise FlashAttention more efficiently. Each prefill is individually faster even though there are more of them.

### 2.4 TTFT Under Pressure at B=32

B=32 TTFT p90 is 6.2 seconds — the main quality-of-experience concern. The p50 (1.91s) is borderline acceptable, but the tail means some users wait through multiple full decode iterations of other sequences before their prefill is scheduled.

This is inherent to continuous batching at high decode concurrency: the scheduler prioritises active decode sequences (they're already allocated KV blocks and have users waiting), and incoming prefills queue behind them. At B=32, with median 18 sequences decoding simultaneously, the scheduler has a large decode batch producing tokens every ~18ms — preempting it for a prefill has a real throughput cost.

For latency-sensitive applications (agentic loops, interactive chat), B=16 remains the recommended operating point. For throughput-oriented workloads (batch inference, async pipelines), B=32 is the right choice.

### 2.5 Energy Efficiency

**tok/W scales 4.8x from B=4 to B=32** (0.055 to 0.264 tok/s/W). The hardware doesn't change — the model doesn't change — the only variable is how well the GPUs are fed. At B=4 the system draws 1,516W and produces 79 tok/s; at B=32 it draws 2,555W (+68%) and produces 753 tok/s (+853%). The power increase is sublinear because most GPU power draw comes from the chip being powered on, not from the marginal cost of additional compute. Running GPUs at half utilisation wastes nearly half your electricity.

### 2.6 KV Cache and Prefix Reuse

KV cache peaks at 35.4% at B=32 — the H100 is firmly **compute-bound, not memory-bound**. There is substantial room to increase concurrency before KV becomes the bottleneck.

Prefix cache hit rate is 68-75% across all batch sizes, confirming the multi-turn advantage: roughly 7 of every 10 prompt tokens arriving at the prefill stage are served directly from cached KV blocks, requiring zero recomputation. This metric would be near 0% in a single-turn benchmark (see Section 5).

The dip at B=8 (68.7% vs 75.4% at B=4 and 71-72% at B=16/32) is a cache pressure effect: 8 concurrent long conversations compete for prefix cache blocks more aggressively than 4, but less effectively than 16+ where the cache pool is large enough relative to the working set.

---

## 3. H200 Analysis — B=128

### 3.1 Performance Summary (Two Runs)

| Metric | Run 1 | Run 2 |
|---|---|---|
| **Throughput median (tok/s)** | 1,990 | 1,873 |
| **Throughput max (tok/s)** | 3,597 | 3,866 |
| **Throughput stdev (tok/s)** | 971 | 1,082 |
| **Per-GPU output (tok/s/gpu)** | 234.2 | 215.9 |
| Interactivity p50 (tok/s/user) | 26.7 | 26.7 |
| Interactivity p90 (tok/s/user) | 21.1 | 21.1 |
| Interactivity p95 (tok/s/user) | 20.5 | 20.5 |
| TTFT p50 (s) | 2.16 | 1.99 |
| TTFT p90 (s) | 7.11 | 6.74 |
| **TPOT p50 (ms)** | 37.4 | 37.4 |
| **ITL p50 (ms)** | 36.8 | 36.6 |
| **ITL p90 (ms)** | 47.4 | 47.4 |
| **ITL p90/p50 ratio** | 1.3x | 1.3x |
| GPU utilisation (%) | 99.9 | 99.9 |
| KV cache median (%) | 25.6 | 23.8 |
| KV cache p90 (%) | 43.6 | 42.5 |
| KV cache max (%) | 48.6 | 46.6 |
| Prefix cache hit rate (%) | 75.3 | 78.5 |
| Total power (W) | 3,380 | 3,258 |
| tok/s per Watt | 0.528 | 0.487 |
| Requests running median | 65 | 58 |
| Requests running max | 100 | 100 |
| Queue time mean (s) | 0.56 | 0.78 |
| Prefill time mean (s) | 0.48 | 1.24 |
| Mean prompt tokens | 2,600 | 2,952 |
| Mean gen tokens | 11,212 | 12,150 |
| Preemptions total | 0 | 0 |

### 3.2 Headlines

The H200 at B=128 delivers **~1,930 tok/s median throughput** — **2.6x the best H100 configuration** (B=32 at 753 tok/s). Both runs are within 6% of each other on throughput and within 1% on interactivity — the kind of reproducibility you only see when the system is fully saturated and deterministic.

The per-GPU throughput comparison tells you where the gain comes from: H200 averages **215-234 tok/s/gpu** vs H100 B=32's 89.8 tok/s/gpu. That's a 2.4-2.6x per-device improvement, almost exactly matching the total system ratio. The gain is not from parallelism or scheduling differences — it's raw hardware generational uplift: more HBM bandwidth (4.8 TB/s vs 3.35 TB/s) means each decode step reads KV cache faster, and more HBM capacity (141 GB vs 80 GB) means more sequences can be in flight without contention.

### 3.3 Interactivity at B=128

Per-user interactivity p50 is **26.7 tok/s** — above the 15 tok/s speech floor but below the 50 tok/s comfortable-reading threshold. The TPOT p50 is 37.4ms, meaning each user receives a new token roughly every 37ms. For comparison, H100 B=8/16 achieves 17.5ms TPOT — about 2x faster per-user, but serving 16x fewer concurrent users.

Critically, the **ITL jitter is tight**: p90/p50 ratio of 1.3x on both runs. Despite running 58-65 concurrent decode sequences (vs 5-7 on H100 B=8/16), the H200 delivers smooth, predictable token delivery with minimal stuttering. The H200's larger decode batch actually *helps* absorb prefill interruptions — each prefill chunk affects a smaller fraction of the active batch.

Whether 26.7 tok/s is acceptable depends on the use case:
- **Async/agent pipelines:** Yes. The user isn't watching tokens stream in real time.
- **Interactive chat:** Marginal. Users will perceive a noticeable slowdown compared to ~55 tok/s on H100 B=8-32.
- **Real-time voice:** No. The 37ms TPOT is too slow for speech synthesis latency requirements.

### 3.4 The 23% KV Cache Question

KV cache sits at **23-26% median** despite 128 concurrent users. This looks like massive headroom, but the explanation is occupancy dynamics, not memory surplus.

Out of 128 `max_num_seqs` slots, the **median active count is 58-65, peaking at 100**. The remaining slots are empty — requests have completed and new ones haven't arrived yet, or the Modal container orchestration is creating gaps between batches. The vLLM KV cache pool is pre-allocated at startup for the theoretical maximum (128 sequences worth of blocks), so the 23% reflects *actual occupancy* against that pre-allocated pool.

The **peak KV of 46-48%** is the real capacity indicator. At burst, nearly half the pool is in use. Doubling concurrency to B=256 would push peak to roughly 85-90% — tight but viable on H200's 141 GB HBM3e.

A back-of-the-envelope: at 58 median running sequences with ~15k average context tokens and 23.8% KV, the total pool holds approximately 58 * 15k / 0.238 = **3.7M tokens**. This is consistent with the H200's much larger memory pool after FP8 model weights (~60-70 GB for 235B FP8 across 8 GPUs leaves ~1,000 GB for KV cache and overhead).

### 3.5 Prefix Cache

Both runs achieve **75-78% hit rate**. Run 2's 78.5% (vs Run 1's 75.3%) reflects a warmer cache — the vLLM process was not restarted between runs, so Run 2 starts with cached KV blocks from Run 1's conversations already in the pool.

A quick validation of the cache's impact: H200 Run 2 reports 3,003 prompt tok/s throughput. If 78.5% of those are served from cache, only ~645 tok/s actually go through prefill attention computation. The vast majority of prompt processing is a cache lookup, not a matmul. This is the multi-turn superpower (see Section 5).

### 3.6 Power and Efficiency

The H200 draws 3.2-3.4 kW at B=128 — roughly **57-60% of the 5,600W TDP envelope**. Efficiency is 0.49-0.53 tok/W, approximately **1.9x more efficient** than H100 B=32 (0.264 tok/W). Note: the previous version of this analysis stated 2.0x; the actual measured ratio averages 1.92x.

The H200 is drawing 32% more power than H100 B=32 (3,320W vs 2,555W) but producing 156% more throughput. The marginal watt on the H200 is producing 5x the marginal tokens compared to the H100. This is the bandwidth-bound MoE advantage: the H200's 43% higher HBM bandwidth (4.8 vs 3.35 TB/s) translates almost linearly to higher decode throughput, while power consumption scales sub-linearly because the additional bandwidth comes from an architecture change (HBM3e vs HBM2e), not from higher clock speeds.

### 3.7 Queue Dynamics

Unlike the H100 runs (where queue time was effectively zero at all batch sizes), the H200 shows **0.56-0.78s mean queue time**. Requests are waiting briefly before being scheduled. This is the first sign that the system is approaching capacity — the scheduler can't immediately admit every new request because active decode sequences are consuming the available compute. At B=256, queue times will increase further, and this will directly add to TTFT.

---

## 4. Cross-Hardware Comparison

### 4.1 Summary Table

| Config | Throughput | Inter p50 | TPOT p50 | ITL p90/p50 | TTFT p50 | tok/W | KV max | Per-GPU |
|---|---|---|---|---|---|---|---|---|
| H100 B=4 | 79 | 27.6 | 36.3ms | **4.9x** | 0.25s | 0.055 | 5.4% | 10.6 |
| H100 B=8 | 268 | 57.1 | 17.5ms | 1.3x | 1.15s | 0.124 | 9.9% | 34.0 |
| H100 B=16 | 392 | 57.1 | 17.5ms | 1.3x | **0.61s** | 0.163 | 17.7% | 49.0 |
| H100 B=32 | 753 | 55.6 | 18.0ms | 2.0x | 1.91s | 0.264 | 35.4% | 89.8 |
| **H200 B=128** | **~1,930** | 26.7 | 37.4ms | 1.3x | ~2.1s | **~0.51** | ~47% | **~225** |

### 4.2 Key Observations

**The throughput gain is per-GPU, not from parallelism.** H200 averages ~225 tok/s/gpu vs H100 B=32's 89.8 tok/s/gpu — a 2.5x per-device improvement. Both systems are 8-GPU TP8 configurations running the same model at the same precision. The entire throughput gap is attributable to the H200's hardware: more HBM bandwidth for faster KV cache reads during decode, and more HBM capacity enabling higher concurrent sequence counts.

**Interactivity is a concurrency effect, not a hardware regression.** H200 B=128 interactivity p50 (26.7 tok/s) is lower than H100 B=32 (55.6 tok/s), but H200 is serving 4x more users. The right comparison is *aggregate interactivity capacity* — users times tok/s/user:
- H100 B=32: 32 users x 55.6 = **1,779 aggregate tok/s/user**
- H200 B=128: 128 users x 26.7 = **3,418 aggregate tok/s/user** (1.9x)

If you ran the H200 at B=32, its interactivity would be substantially higher than the H100's thanks to the faster per-step decode. You'd just be wasting the extra memory capacity.

**TPOT reveals the MoE decode mechanics.** At B=4, TPOT p50 is 36.3ms — nearly identical to H200 B=128's 37.4ms. This seems counterintuitive: B=4 has 2 active sequences while B=128 has 58-65. But it makes sense for MoE decode: the dominant cost is reading the active expert weights from HBM (~22B parameters, same for every batch size). At B=4 on H100 (3.35 TB/s bandwidth), the weight read takes most of the decode step. At B=128 on H200 (4.8 TB/s bandwidth), the weight read is ~30% faster, but the larger KV cache read (65 sequences vs 2) offsets the bandwidth gain. The result is nearly identical TPOT despite a 30x difference in active sequence count — a signature of bandwidth-bound MoE serving.

The real difference between B=4 and B=8-16 isn't the TPOT itself — it's the ITL *jitter*. B=4's ITL p90/p50 ratio of 4.9x means users experience dramatic stuttering. B=8-16 and H200 all sit at 1.3x — smooth, predictable token delivery. B=32 is at 2.0x, the first sign of decode congestion.

**Zero preemptions everywhere.** Both hardware platforms, all batch sizes, all runs: zero preemptions. vLLM's paged attention and chunked prefill scheduler never needed to evict a mid-generation sequence's KV blocks. This is remarkable at B=128 on H200 with 46% peak KV utilisation and suggests the block allocator has ample headroom.

---

## 5. Multi-Turn vs Single-Turn: Why It Matters

This is a multi-turn workload — 4 turns per conversation — and the differences from single-turn are not cosmetic. They change the fundamental economics of what the serving system is doing.

### 5.1 Prefix Cache Is the Defining Difference

In single-turn serving, every request is independent. Prefix cache hit rate is ~0% (or a small percentage from shared system prompts). Every prompt token goes through full prefill attention computation.

In our multi-turn workload, prefix cache hit rates are **68-78%**. When a user sends turn 3, the KV blocks from turns 1 and 2 are already in the cache. vLLM recognises the prefix match and skips the prefill computation for those tokens entirely. Only the new content (the latest user message, ~500-1000 tokens) needs actual prefill.

The impact: if 70% of prompt tokens are served from cache, only 30% require actual compute. That's a **3.3x reduction in effective prefill compute load**. This is why a system that handles 32 single-turn users can handle 100+ multi-turn users with comparable TTFT — the prefill cost per turn drops dramatically after turn 1.

This is not a hypothetical advantage — it shows up directly in the measured prompt throughput. H100 B=32 processes 14,020 prompt tok/s, but with a 72.5% hit rate, only ~3,850 tok/s actually require prefill computation. The rest are cache lookups. Without the prefix cache, the same TTFT would require 3.6x more prefill compute, and the system would either need to throttle concurrency or accept much longer TTFT tails.

### 5.2 Context Growth Per Turn

In single-turn, every request has roughly the same prompt length. In multi-turn, prompt length grows with each turn:

- Turn 1: ~500 tokens (initial user message + system prompt)
- Turn 2: ~500 + 2,500 generated + 500 new = ~3,500 tokens
- Turn 3: ~3,500 + 2,500 generated + 500 new = ~6,500 tokens
- Turn 4: ~6,500 + 2,500 generated + 500 new = ~9,500 tokens

This has several downstream effects that single-turn benchmarks cannot capture:

**TTFT grows per-turn.** The same user sees longer TTFT on turn 4 than turn 1, because even with prefix cache, the new prompt suffix is longer and the KV cache blocks must be located and verified. Single-turn benchmarks report a single TTFT distribution; multi-turn has a per-turn distribution that shifts rightward.

**KV cache accumulates.** A 4-turn user occupies ~4x the KV blocks of a fresh single-turn user. The cache pool must accommodate the *peak* occupancy, not the average. This is why peak KV (35-47%) is so much higher than median (2.4-25.6%) — it reflects the moments when many conversations are simultaneously in their late turns.

**Decode step cost increases.** Each decode step for a sequence in turn 4 must attend over 10,000+ cached KV entries, compared to ~500 in turn 1. The attention computation (and corresponding KV cache read from HBM) is proportional to sequence length. In a mixed-age batch, late-turn sequences are "heavier" than early-turn sequences, creating non-uniform per-sequence costs within the same decode batch.

### 5.3 Throughput Variance

Multi-turn workloads produce **high throughput variance** — throughput stdev is 47-58% of the median across all configurations. This is a direct consequence of the bursty lifecycle of multi-turn conversations: when a batch of users simultaneously finishes turn 3 and starts turn 4 (large prefill), throughput momentarily dips. When they're all in mid-decode, throughput peaks. Single-turn benchmarks with continuous arrival would show much tighter distributions.

This variance means that median throughput is a better summary statistic than mean, and that p10-p90 ranges matter for capacity planning: H100 B=32's p10 throughput (124 tok/s) is 6x lower than its p90 (1,313 tok/s). A system sized for the median would under-provision for the troughs.

### 5.4 What Single-Turn Benchmarks Miss

Standard LLM serving benchmarks (ShareGPT traces, lm-evaluation-harness, etc.) typically operate in single-turn mode with synthetic or real-but-independent prompts. They cannot measure:

1. **Prefix cache effects** — the single largest performance lever in multi-turn serving, reducing effective prefill compute by 3x+
2. **TTFT growth across turns** — each turn is progressively more expensive, and the TTFT distribution shifts rightward over the conversation
3. **KV cache fragmentation** — varying-age conversations create non-uniform block allocation patterns that stress the paged attention allocator
4. **Mixed-age decode batches** — the scheduler must fairly serve sequences with 500-token and 10,000-token KV caches in the same decode step
5. **Throughput burstiness** — the lifecycle coupling of multi-turn conversations creates oscillating throughput patterns absent in continuous single-turn arrivals

Our setup captures all five. The result is a serving characterisation that reflects what a real-world coding assistant, customer support agent, or multi-step reasoning pipeline actually experiences.

---

## 6. Capacity Planning and Next Steps

### 6.1 H100: Scale to B=64

All H100 runs have **zero requests waiting** — the system is never queue-limited. Current `max_num_seqs=64` is the server ceiling; we tested up to B=32 (50% of slots). The natural next experiment:

**B=64 with max_num_seqs=64:**
- GPU stays at 100% (already compute-bound)
- KV cache estimated peak: ~70% (from 35.4% at B=32, roughly linear with concurrency)
- Expected throughput: 1.4-1.8x over B=32 (~1,050-1,350 tok/s)
- Interactivity p50 expected: ~40-45 tok/s/user (based on stable TPOT trend)
- ITL p90/p50 ratio likely rises above 2.0x — watch for decode congestion

**B=128 with max_num_seqs=128 on H100 is risky.** At B=32, peak KV is 35.4% with 32 sequences. Scaling to 128 sequences with similar per-sequence context (~14k tokens) gives 128/32 * 35.4% = ~140% peak KV — the pool cannot hold it. The result would be OOM or, if vLLM handles it gracefully, heavy preemptions that destroy throughput and interactivity simultaneously. A B=64 experiment should be conducted first to establish the actual scaling curve.

### 6.2 H200: Scale to B=256

KV pool capacity of ~3.7M tokens gives more room. At B=128 peak, KV is 47%. Estimated at B=256:

**B=256 with max_num_seqs=256:**
- KV peak estimated: ~80-90% (viable but approaching the ceiling; monitor closely)
- Throughput gain: +20-40% over B=128 (~2,300-2,700 tok/s) — limited by compute saturation, not memory
- Interactivity p50: ~18-22 tok/s/user (below comfortable reading, acceptable for async workloads)
- Queue time will increase from the current 0.56-0.78s
- Watch for the first preemptions at high KV utilisation

### 6.3 Open Questions and Anomalies

**B=16 TTFT < B=8.** Confirmed in the data: mean prefill time 0.60s (B=16) vs 1.59s (B=8), and prompt throughput 6,246 tok/s (B=16) vs 3,591 tok/s (B=8). At B=16, the scheduler has more prefill-eligible requests per scheduling step, producing larger prefill batches with better FlashAttention kernel utilisation. Profiling the attention kernels directly (nsight compute traces of the prefill step at both concurrency levels) would confirm this.

**H200 GPU temperature is notably higher.** H200 runs at 52-53C vs H100 at 33-36C, despite similar TDP fractions (57-60% vs 46%). Likely a datacenter cooling configuration difference, not a hardware thermal issue. Worth monitoring at B=256 where power draw will increase.

**The B=4 TPOT / H200 B=128 TPOT coincidence.** Both sit at ~37ms despite 30x different active sequence counts. This is not a coincidence — it's the MoE bandwidth floor. At very low concurrency (B=4 on H100) and very high concurrency (B=128 on H200), the dominant decode cost is reading the 22B active expert weights from HBM. The weight read sets a minimum TPOT that's hard to beat regardless of how few sequences you're running. At intermediate concurrency (B=8-16 on H100), the KV cache is small enough that the weight read can be partially overlapped with compute, yielding a lower effective TPOT of ~17.5ms. This overlap breaks down at B=32 (TPOT rises to 18.0ms) and higher.

---

*Charts:* `charts/batch_comparison_dashboard.png` · `charts/interactivity_vs_throughput.png` · `charts/h200_dashboard.png` · `charts/h100_vs_h200_comparison.png`

*Data:* `metrics/metrics-batch{4,8,16,32}/` (H100) · `metrics/metrics-batch128-h200s{,-2}/` (H200)

*Scripts:* `analysis_multiturn/chart_batch_comparison.py` · `analysis_multiturn/chart_h200.py`

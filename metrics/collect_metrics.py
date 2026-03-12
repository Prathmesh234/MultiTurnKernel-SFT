"""
vLLM Comprehensive Metrics Collector
=====================================
Polls vLLM's Prometheus /metrics endpoint at a fixed interval and writes
every snapshot to metrics/metrics.jsonl (one JSON object per line).

Additionally stores the raw Prometheus scrape text in metrics/prometheus.json
as ground truth for every single metric vLLM exposes.

This is a READ-ONLY observer — it never touches the vLLM server's inference
path and can run safely alongside the orchestrator.

Metrics collected (inspired by SemiAnalysis InferenceX v2):
─────────────────────────────────────────────────────────────
  THROUGHPUT
    • Output token throughput (total, per-GPU, per-request)
    • Prompt/prefill token throughput (total, per-GPU)
    • Iteration-level tokens per engine step

  LATENCY (InferenceX-style)
    • TTFT  — Time To First Token (histogram: p50/p90/p95/p99/mean)
    • TPOT  — Time Per Output Token (histogram: p50/p90/p95/p99/mean)
    • ITL   — Inter-Token Latency (histogram: p50/p90/p95/p99/mean)
    • E2E   — End-to-End Request Latency (histogram: p50/p90/p95/p99/mean)
    • Queue time, Prefill time, Decode time, Inference time (histograms)
    • Interactivity (tok/s/user) — inverse of TPOT

  KV CACHE & MEMORY
    • KV cache usage %
    • Prefix cache hit rate (local + external)
    • Multi-modal cache hit rate
    • Preemption count (KV cache pressure indicator)

  GPU HARDWARE (nvidia-smi)
    • Per-GPU: utilization %, memory used/free/total, temperature,
      power draw, SM clock, PCIe gen/width
    • Aggregate: avg/total GPU util, memory, power
    • NVLink bandwidth per link

  SYSTEM
    • CPU model, core count
    • System RAM total/used/available

  ENERGY EFFICIENCY (InferenceX-style)
    • Tokens per Watt (output tokens / total GPU power draw)
    • Picojoules per token estimate

  SERVER CONFIG
    • Model name, quant, TP/EP, max_model_len, etc.
    • Workload type (non-disaggregated)

Usage:
    uv run --no-sync python metrics/collect_metrics.py                   # every 10s
    uv run --no-sync python metrics/collect_metrics.py --interval 30     # every 30s
    uv run --no-sync python metrics/collect_metrics.py --url http://host:port
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package not found. Install with: uv pip install requests")
    sys.exit(1)


# ── defaults ──────────────────────────────────────────────────────────────────
VLLM_BASE_URL = "http://localhost:8000"
DEFAULT_BATCH_SIZE = 64

DEFAULT_INTERVAL = 10  # seconds between polls

# ── server configuration snapshot ─────────────────────────────────────────────
# Mirrors the flags used in serve_qwen3_235b.sh
SERVER_CONFIG = {
    "model": "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
    "model_type": "MoE (235B total / 22B active, 128 experts top-8)",
    "quantization": "FP8",
    "tensor_parallel_size": 8,
    "expert_parallel": True,
    "max_model_len": 131072,
    "gpu_memory_utilization": 0.92,
    "max_num_seqs": 64,
    "swap_space_gb": 16,
    "reasoning_parser": "qwen3",
    "prefix_caching": True,
    "dtype": "auto",
    "vllm_use_v1": True,
    "workload_type": "non-disaggregated (prefill + decode co-located)",
    "serving_framework": "vLLM",
    "host": "0.0.0.0",
    "port": 8000,
}


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM INFO (collected once at startup)
# ══════════════════════════════════════════════════════════════════════════════

def collect_system_info() -> dict:
    """One-time snapshot of the host system."""
    info: dict[str, Any] = {}

    # CPU info
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
        result = subprocess.run(["nproc"], capture_output=True, text=True, timeout=5)
        info["cpu_cores"] = int(result.stdout.strip())
    except Exception:
        info["cpu_model"] = "unknown"
        info["cpu_cores"] = 0

    # RAM
    try:
        result = subprocess.run(
            ["free", "-b"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if line.startswith("Mem:"):
                parts = line.split()
                info["ram_total_bytes"] = int(parts[1])
                info["ram_used_bytes"] = int(parts[2])
                info["ram_available_bytes"] = int(parts[6]) if len(parts) > 6 else None
                break
    except Exception:
        pass

    # GPU topology
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True, timeout=10,
        )
        info["gpu_topology"] = result.stdout.strip()
    except Exception:
        info["gpu_topology"] = "unavailable"

    # NVLink info (per-GPU link bandwidth)
    try:
        result = subprocess.run(
            ["nvidia-smi", "nvlink", "--status"],
            capture_output=True, text=True, timeout=10,
        )
        links: dict[str, list] = {}
        current_gpu = ""
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("GPU"):
                current_gpu = line.split(":")[0].strip()
                links[current_gpu] = []
            elif line.startswith("Link") and "GB/s" in line:
                bw = re.search(r'([\d.]+)\s*GB/s', line)
                if bw:
                    links[current_gpu].append(float(bw.group(1)))
        nvlink_summary = {}
        for gpu, bws in links.items():
            if bws:
                nvlink_summary[gpu] = {
                    "num_links": len(bws),
                    "per_link_gbps": bws[0] if bws else 0,
                    "total_bw_gbps": round(sum(bws), 2),
                }
        info["nvlink"] = nvlink_summary
    except Exception:
        info["nvlink"] = "unavailable"

    # GPU hardware info (static — collected once)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,pci.bus_id,"
                "memory.total,pcie.link.gen.max,pcie.link.width.max",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        gpu_hw = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpu_hw.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "driver_version": parts[2],
                    "pci_bus_id": parts[3],
                    "memory_total_mib": float(parts[4]),
                    "pcie_gen_max": parts[5],
                    "pcie_width_max": parts[6],
                })
        info["gpus"] = gpu_hw
        info["num_gpus"] = len(gpu_hw)
        if gpu_hw:
            info["gpu_type"] = f"{gpu_hw[0]['name']} x{len(gpu_hw)} (NVLink)"
            info["total_gpu_memory_mib"] = sum(g["memory_total_mib"] for g in gpu_hw)
    except Exception:
        info["gpus"] = []
        info["num_gpus"] = 0

    # Linux kernel
    try:
        result = subprocess.run(["uname", "-r"], capture_output=True, text=True, timeout=5)
        info["kernel"] = result.stdout.strip()
    except Exception:
        info["kernel"] = "unknown"

    return info


# ══════════════════════════════════════════════════════════════════════════════
#  PROMETHEUS PARSER
# ══════════════════════════════════════════════════════════════════════════════

_METRIC_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}\s+([\d.eE+\-NnAa]+)'
)
_METRIC_NOLABEL_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([\d.eE+\-NnAa]+)'
)


def parse_prometheus_full(text: str) -> dict:
    """
    Parse Prometheus text exposition format into a nested dict:
        { metric_name: { "__help__": str, "__type__": str,
                         label_str_or___value__: float, ... } }
    Preserves HELP and TYPE metadata for ground truth storage.
    """
    metrics: dict[str, dict] = {}
    current_name = ""

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# HELP"):
            parts = line.split(None, 3)
            if len(parts) >= 4:
                current_name = parts[2]
                metrics.setdefault(current_name, {})["__help__"] = parts[3]
            continue
        if line.startswith("# TYPE"):
            parts = line.split(None, 4)
            if len(parts) >= 4:
                metrics.setdefault(parts[2], {})["__type__"] = parts[3]
            continue
        if line.startswith("#"):
            continue

        m = _METRIC_RE.match(line)
        if m:
            name, labels_str, value_str = m.group(1), m.group(2), m.group(3)
        else:
            m2 = _METRIC_NOLABEL_RE.match(line)
            if not m2:
                continue
            name, labels_str, value_str = m2.group(1), "", m2.group(2)

        try:
            value = float(value_str)
        except ValueError:
            continue

        metrics.setdefault(name, {})
        key = labels_str if labels_str else "__value__"
        metrics[name][key] = value

    return metrics


def flatten_vllm_metrics(raw: dict) -> dict:
    """
    Extract all vllm:* metrics into a clean flat dict.
    For single-value metrics, unwrap to just the value.
    Preserves histogram buckets as nested dicts.
    """
    out = {}
    for name, label_dict in raw.items():
        if not name.startswith("vllm:"):
            continue
        short = name[len("vllm:"):]
        # Strip metadata keys
        data = {k: v for k, v in label_dict.items()
                if k not in ("__help__", "__type__")}
        if len(data) == 1:
            out[short] = next(iter(data.values()))
        else:
            out[short] = data
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  HISTOGRAM PERCENTILE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_histogram_stats(raw: dict, metric_prefix: str) -> dict | None:
    """
    Given the raw parsed Prometheus dict, extract histogram stats for a
    vllm metric. Returns percentiles (p50, p90, p95, p99), mean, count, sum.

    metric_prefix: e.g. "vllm:time_to_first_token_seconds"
    """
    bucket_key = f"{metric_prefix}_bucket"
    count_key = f"{metric_prefix}_count"
    sum_key = f"{metric_prefix}_sum"

    if bucket_key not in raw:
        return None

    bucket_data = raw[bucket_key]
    count_data = raw.get(count_key, {})
    sum_data = raw.get(sum_key, {})

    # Parse buckets into sorted (le, cumulative_count) pairs
    buckets = []
    for label_str, cum_count in bucket_data.items():
        if label_str in ("__help__", "__type__"):
            continue
        le_match = re.search(r'le="([^"]+)"', label_str)
        if le_match:
            le_val = le_match.group(1)
            if le_val == "+Inf":
                le_float = float("inf")
            else:
                try:
                    le_float = float(le_val)
                except ValueError:
                    continue
            buckets.append((le_float, cum_count))

    if not buckets:
        return None

    buckets.sort(key=lambda x: x[0])

    # Get total count and sum
    total_count = 0
    total_sum = 0.0
    for k, v in count_data.items():
        if k not in ("__help__", "__type__"):
            total_count = int(v)
            break
    for k, v in sum_data.items():
        if k not in ("__help__", "__type__"):
            total_sum = float(v)
            break

    if total_count == 0:
        return {"count": 0, "sum": 0.0, "mean": 0.0,
                "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

    mean = total_sum / total_count if total_count > 0 else 0.0

    def percentile(p: float) -> float:
        target = total_count * p
        prev_le = 0.0
        prev_count = 0.0
        for le, cum in buckets:
            if le == float("inf"):
                return prev_le  # best estimate
            if cum >= target:
                # Linear interpolation within bucket
                bucket_width = le - prev_le
                bucket_count = cum - prev_count
                if bucket_count == 0:
                    return le
                fraction = (target - prev_count) / bucket_count
                return prev_le + fraction * bucket_width
            prev_le = le
            prev_count = cum
        return buckets[-2][0] if len(buckets) > 1 else 0.0

    return {
        "count": total_count,
        "sum": round(total_sum, 6),
        "mean": round(mean, 6),
        "p50": round(percentile(0.50), 6),
        "p90": round(percentile(0.90), 6),
        "p95": round(percentile(0.95), 6),
        "p99": round(percentile(0.99), 6),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GPU STATS (per-poll)
# ══════════════════════════════════════════════════════════════════════════════

def collect_gpu_stats() -> list[dict]:
    """Query nvidia-smi for per-GPU runtime stats."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,"
                "utilization.memory,memory.total,memory.used,memory.free,"
                "power.draw,power.limit,clocks.current.sm,clocks.current.memory,"
                "pcie.link.gen.current,pcie.link.width.current,"
                "encoder.stats.sessionCount,encoder.stats.averageFps",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 14:
                continue

            def safe_float(s: str) -> float | None:
                try:
                    return float(s)
                except (ValueError, TypeError):
                    return None

            def safe_int(s: str) -> int | None:
                try:
                    return int(s)
                except (ValueError, TypeError):
                    return None

            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "temperature_c": safe_float(parts[2]),
                "utilization_gpu_pct": safe_float(parts[3]),
                "utilization_mem_pct": safe_float(parts[4]),
                "memory_total_mib": safe_float(parts[5]),
                "memory_used_mib": safe_float(parts[6]),
                "memory_free_mib": safe_float(parts[7]),
                "power_draw_w": safe_float(parts[8]),
                "power_limit_w": safe_float(parts[9]),
                "sm_clock_mhz": safe_float(parts[10]),
                "mem_clock_mhz": safe_float(parts[11]),
                "pcie_gen_current": safe_int(parts[12]),
                "pcie_width_current": safe_int(parts[13]),
                "encoder_sessions": safe_int(parts[14]) if len(parts) > 14 else None,
                "encoder_avg_fps": safe_float(parts[15]) if len(parts) > 15 else None,
            })
        return gpus
    except Exception as e:
        return [{"error": str(e)}]


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_total(val) -> float:
    """Extract a scalar total from a vllm_flat value that might be a dict
    (when the Prometheus counter has labels like finished_reason)."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        # Sum all numeric label values (e.g. stop + length + abort)
        return sum(v for v in val.values() if isinstance(v, (int, float)))
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  DELTA TRACKER (for per-second rates from monotonic counters)
# ══════════════════════════════════════════════════════════════════════════════

class DeltaTracker:
    """
    Tracks counter deltas between consecutive polls to compute rates.
    Prometheus counters are monotonically increasing, so we need diffs
    to get per-interval rates.
    """

    def __init__(self):
        self.prev_values: dict[str, float] = {}
        self.prev_ts: float | None = None

    def update(self, vllm_flat: dict, unix_ts: float) -> dict:
        """Return per-second rates for key counters."""
        rates: dict[str, float | None] = {}

        counters = [
            "prompt_tokens_total",
            "generation_tokens_total",
            "request_success_total",
            "num_preemptions_total",
            "prefix_cache_hits_total",
            "prefix_cache_queries_total",
            "external_prefix_cache_hits_total",
            "external_prefix_cache_queries_total",
        ]

        if self.prev_ts is not None:
            dt = unix_ts - self.prev_ts
            if dt > 0:
                for c in counters:
                    cur = _safe_total(vllm_flat.get(c, 0))
                    prev = self.prev_values.get(c, 0)
                    delta = cur - prev
                    rate_name = c.replace("_total", "") + "_per_sec"
                    rates[rate_name] = round(delta / dt, 3)
            else:
                for c in counters:
                    rates[c.replace("_total", "") + "_per_sec"] = None
        else:
            for c in counters:
                rates[c.replace("_total", "") + "_per_sec"] = None

        # Save current values for next delta
        self.prev_ts = unix_ts
        for c in counters:
            self.prev_values[c] = _safe_total(vllm_flat.get(c, 0))

        return rates


# ══════════════════════════════════════════════════════════════════════════════
#  DERIVED / SUMMARY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def derive_inferencex_metrics(
    vllm_flat: dict,
    raw: dict,
    gpus: list[dict],
    rates: dict,
    num_gpus: int,
) -> dict:
    """
    Compute InferenceX-style derived metrics (SemiAnalysis article metrics).

    These mirror the key metrics discussed in the InferenceX v2 article:
    - Throughput (total, per-GPU)
    - Interactivity (tok/s/user)
    - TTFT, TPOT, ITL, E2E latency distributions
    - Energy efficiency (tokens/W, pJ/token)
    - Prefix cache efficiency
    - KV cache pressure
    """
    derived: dict[str, Any] = {}

    # ── Throughput ────────────────────────────────────────────────────────
    gen_rate = rates.get("generation_tokens_per_sec")
    prompt_rate = rates.get("prompt_tokens_per_sec")

    derived["output_throughput_tok_per_sec"] = gen_rate
    derived["prompt_throughput_tok_per_sec"] = prompt_rate

    if gen_rate is not None and num_gpus > 0:
        derived["output_throughput_tok_per_sec_per_gpu"] = round(gen_rate / num_gpus, 3)
    else:
        derived["output_throughput_tok_per_sec_per_gpu"] = None

    if prompt_rate is not None and num_gpus > 0:
        derived["prompt_throughput_tok_per_sec_per_gpu"] = round(prompt_rate / num_gpus, 3)
    else:
        derived["prompt_throughput_tok_per_sec_per_gpu"] = None

    # Total (cumulative counters — may be dicts when they carry labels)
    derived["total_output_tokens"] = int(_safe_total(vllm_flat.get("generation_tokens_total", 0)))
    derived["total_prompt_tokens"] = int(_safe_total(vllm_flat.get("prompt_tokens_total", 0)))
    derived["total_requests_succeeded"] = int(_safe_total(vllm_flat.get("request_success_total", 0)))
    # Break out request_success by reason if available
    rs = vllm_flat.get("request_success_total")
    if isinstance(rs, dict):
        derived["requests_by_finish_reason"] = {}
        for label_str, count in rs.items():
            reason_match = re.search(r'finished_reason="([^"]+)"', label_str)
            if reason_match:
                derived["requests_by_finish_reason"][reason_match.group(1)] = int(count)
    else:
        derived["requests_by_finish_reason"] = None
    derived["preemptions_total"] = int(_safe_total(vllm_flat.get("num_preemptions_total", 0)))

    # ── Latency Histograms (InferenceX key metrics) ──────────────────────
    histogram_metrics = {
        "ttft": "vllm:time_to_first_token_seconds",
        "tpot": "vllm:request_time_per_output_token_seconds",
        "itl": "vllm:inter_token_latency_seconds",
        "e2e_latency": "vllm:e2e_request_latency_seconds",
        "queue_time": "vllm:request_queue_time_seconds",
        "prefill_time": "vllm:request_prefill_time_seconds",
        "decode_time": "vllm:request_decode_time_seconds",
        "inference_time": "vllm:request_inference_time_seconds",
        "request_prompt_tokens": "vllm:request_prompt_tokens",
        "request_generation_tokens": "vllm:request_generation_tokens",
        "request_max_gen_tokens": "vllm:request_max_num_generation_tokens",
        "request_max_tokens_param": "vllm:request_params_max_tokens",
        "iteration_tokens": "vllm:iteration_tokens_total",
        "prefill_kv_computed_tokens": "vllm:request_prefill_kv_computed_tokens",
    }

    for key, metric_name in histogram_metrics.items():
        stats = extract_histogram_stats(raw, metric_name)
        if stats:
            derived[key] = stats
        else:
            derived[key] = None

    # ── Interactivity (tok/s/user) — inverse of TPOT ─────────────────────
    tpot_stats = derived.get("tpot")
    if tpot_stats and isinstance(tpot_stats, dict):
        for pct in ["p50", "p90", "p95", "p99", "mean"]:
            val = tpot_stats.get(pct, 0)
            if val and val > 0:
                derived[f"interactivity_{pct}_tok_per_sec_per_user"] = round(1.0 / val, 3)
            else:
                derived[f"interactivity_{pct}_tok_per_sec_per_user"] = None
    else:
        for pct in ["p50", "p90", "p95", "p99", "mean"]:
            derived[f"interactivity_{pct}_tok_per_sec_per_user"] = None

    # ── Request concurrency ──────────────────────────────────────────────
    derived["requests_running"] = int(_safe_total(vllm_flat.get("num_requests_running", 0)))
    derived["requests_waiting"] = int(_safe_total(vllm_flat.get("num_requests_waiting", 0)))
    # Effective batch size = running requests (approximation)
    derived["effective_batch_size"] = derived["requests_running"]

    # ── KV Cache ─────────────────────────────────────────────────────────
    kv_usage = _safe_total(vllm_flat.get("kv_cache_usage_perc", 0))
    derived["kv_cache_usage_pct"] = round(kv_usage * 100, 4)

    # ── Prefix Cache ─────────────────────────────────────────────────────
    hits = _safe_total(vllm_flat.get("prefix_cache_hits_total", 0))
    queries = _safe_total(vllm_flat.get("prefix_cache_queries_total", 0))
    derived["prefix_cache_hit_rate_pct"] = round(hits / queries * 100, 4) if queries > 0 else 0.0
    derived["prefix_cache_hits_total"] = int(hits)
    derived["prefix_cache_queries_total"] = int(queries)

    # External prefix cache
    ext_hits = _safe_total(vllm_flat.get("external_prefix_cache_hits_total", 0))
    ext_queries = _safe_total(vllm_flat.get("external_prefix_cache_queries_total", 0))
    derived["external_prefix_cache_hit_rate_pct"] = round(ext_hits / ext_queries * 100, 4) if ext_queries > 0 else 0.0

    # Cached / recomputed prompt tokens
    derived["prompt_tokens_cached_total"] = _safe_total(vllm_flat.get("prompt_tokens_cached_total", 0))
    derived["prompt_tokens_recomputed_total"] = _safe_total(vllm_flat.get("prompt_tokens_recomputed_total", 0))

    # ── GPU Aggregate Stats ──────────────────────────────────────────────
    if gpus and "error" not in gpus[0]:
        n = len(gpus)
        derived["avg_gpu_utilization_pct"] = round(
            sum(g.get("utilization_gpu_pct", 0) or 0 for g in gpus) / n, 2
        )
        derived["max_gpu_utilization_pct"] = max(
            g.get("utilization_gpu_pct", 0) or 0 for g in gpus
        )
        derived["avg_gpu_memory_used_mib"] = round(
            sum(g.get("memory_used_mib", 0) or 0 for g in gpus) / n, 1
        )
        derived["total_gpu_memory_used_mib"] = round(
            sum(g.get("memory_used_mib", 0) or 0 for g in gpus), 1
        )
        derived["total_gpu_memory_total_mib"] = round(
            sum(g.get("memory_total_mib", 0) or 0 for g in gpus), 1
        )
        derived["gpu_memory_utilization_pct"] = round(
            derived["total_gpu_memory_used_mib"] / derived["total_gpu_memory_total_mib"] * 100, 2
        ) if derived["total_gpu_memory_total_mib"] > 0 else 0.0

        derived["avg_gpu_temperature_c"] = round(
            sum(g.get("temperature_c", 0) or 0 for g in gpus) / n, 1
        )
        derived["max_gpu_temperature_c"] = max(
            g.get("temperature_c", 0) or 0 for g in gpus
        )

        total_power = sum(
            g.get("power_draw_w", 0) or 0 for g in gpus
        )
        derived["total_power_draw_w"] = round(total_power, 2)
        derived["avg_power_draw_per_gpu_w"] = round(total_power / n, 2)

        total_power_limit = sum(
            g.get("power_limit_w", 0) or 0 for g in gpus
        )
        derived["total_power_limit_w"] = round(total_power_limit, 2)

        # SM / memory clocks
        derived["avg_sm_clock_mhz"] = round(
            sum(g.get("sm_clock_mhz", 0) or 0 for g in gpus) / n, 1
        )
        derived["avg_mem_clock_mhz"] = round(
            sum(g.get("mem_clock_mhz", 0) or 0 for g in gpus) / n, 1
        )

        # ── Energy Efficiency (InferenceX-style) ─────────────────────────
        if gen_rate is not None and total_power > 0:
            derived["output_tokens_per_watt"] = round(gen_rate / total_power, 6)
            # Picojoules per token = (total_power_W) / (gen_rate_tok/s) * 1e12
            derived["picojoules_per_output_token"] = round(
                (total_power / gen_rate) * 1e12, 2
            ) if gen_rate > 0 else None
            # Energy per token in millijoules
            derived["millijoules_per_output_token"] = round(
                (total_power / gen_rate) * 1e3, 4
            ) if gen_rate > 0 else None
        else:
            derived["output_tokens_per_watt"] = None
            derived["picojoules_per_output_token"] = None
            derived["millijoules_per_output_token"] = None
    else:
        derived["avg_gpu_utilization_pct"] = None
        derived["total_power_draw_w"] = None
        derived["output_tokens_per_watt"] = None

    # ── Preemption rate (KV cache pressure indicator) ────────────────────
    preempt_rate = rates.get("num_preemptions_per_sec")
    derived["preemptions_per_sec"] = preempt_rate

    # ── Multi-modal cache ────────────────────────────────────────────────
    mm_hits = _safe_total(vllm_flat.get("mm_cache_hits_total", 0))
    mm_queries = _safe_total(vllm_flat.get("mm_cache_queries_total", 0))
    derived["mm_cache_hit_rate_pct"] = round(mm_hits / mm_queries * 100, 4) if mm_queries > 0 else 0.0

    # ── Request success rate ─────────────────────────────────────────────
    derived["request_success_per_sec"] = rates.get("request_success_per_sec")

    # ── Prompt tokens by source (if available) ───────────────────────────
    prompt_by_source = vllm_flat.get("prompt_tokens_by_source_total")
    if isinstance(prompt_by_source, dict):
        derived["prompt_tokens_by_source"] = prompt_by_source
    else:
        derived["prompt_tokens_by_source"] = None

    return derived


# ══════════════════════════════════════════════════════════════════════════════
#  COLLECTION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def collect_once(
    session: requests.Session,
    base_url: str,
    delta_tracker: DeltaTracker,
    system_info: dict,
    num_gpus: int,
) -> tuple[dict | None, str | None]:
    """
    Fetch one snapshot. Returns (processed_record, raw_prometheus_text).
    Returns (None, None) on failure.
    """
    ts = datetime.now(timezone.utc)
    unix_ts = ts.timestamp()

    try:
        resp = session.get(f"{base_url}/metrics", timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"[{ts.isoformat()[:19]}] ⚠  metrics fetch failed: {e}", file=sys.stderr)
        return None, None

    raw_text = resp.text
    raw = parse_prometheus_full(raw_text)
    vllm_flat = flatten_vllm_metrics(raw)
    gpus = collect_gpu_stats()
    rates = delta_tracker.update(vllm_flat, unix_ts)
    derived = derive_inferencex_metrics(vllm_flat, raw, gpus, rates, num_gpus)

    # ── HTTP server metrics ──────────────────────────────────────────────
    http_metrics = {}
    for name, data in raw.items():
        if name.startswith("http_"):
            short = name
            cleaned = {k: v for k, v in data.items()
                       if k not in ("__help__", "__type__")}
            if len(cleaned) == 1:
                http_metrics[short] = next(iter(cleaned.values()))
            else:
                http_metrics[short] = cleaned

    # ── Process metrics ──────────────────────────────────────────────────
    process_metrics = {}
    for name in ["process_cpu_seconds_total", "process_resident_memory_bytes",
                 "process_virtual_memory_bytes", "process_open_fds",
                 "process_max_fds", "process_start_time_seconds"]:
        if name in raw:
            data = raw[name]
            for k, v in data.items():
                if k not in ("__help__", "__type__"):
                    process_metrics[name] = v
                    break

    # ── Engine sleep state ───────────────────────────────────────────────
    engine_state = {}
    if "vllm:engine_sleep_state" in raw:
        data = raw["vllm:engine_sleep_state"]
        for k, v in data.items():
            if k not in ("__help__", "__type__"):
                engine_state[k] = v

    # ── Cache config info ────────────────────────────────────────────────
    cache_config = {}
    if "vllm:cache_config_info" in raw:
        data = raw["vllm:cache_config_info"]
        for k, v in data.items():
            if k not in ("__help__", "__type__"):
                cache_config[k] = v

    record = {
        "timestamp": ts.isoformat(),
        "unix_ts": unix_ts,
        "collection_interval_s": None,  # filled in by caller
        "poll_index": None,             # filled in by caller

        # Server & system config (embedded in every record for self-contained analysis)
        "server_config": SERVER_CONFIG,
        "system_info": system_info,

        # InferenceX-style derived metrics (the main show)
        "inferencex_metrics": derived,

        # Per-poll rate counters
        "rates": rates,

        # All raw vLLM metrics (flat)
        "vllm_metrics": vllm_flat,

        # Per-GPU hardware stats
        "gpu_stats": gpus,

        # HTTP server metrics
        "http_metrics": http_metrics,

        # Process-level metrics
        "process_metrics": process_metrics,

        # Engine state
        "engine_state": engine_state,

        # Cache config
        "cache_config": cache_config,
    }

    return record, raw_text


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive vLLM metrics collector (InferenceX-style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL,
        help=f"Seconds between polls (default: {DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size label — auto-creates metrics/metrics-batch{{N}}/ subfolder (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--url", default=VLLM_BASE_URL,
        help=f"vLLM base URL (default: {VLLM_BASE_URL})",
    )
    args = parser.parse_args()

    batch_dir = Path(f"metrics/metrics-batch{args.batch_size}")
    metrics_path = batch_dir / f"metrics-batch{args.batch_size}.jsonl"
    prometheus_path = batch_dir / f"prometheus-batch{args.batch_size}.json"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect system info once ─────────────────────────────────────────
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│        vLLM Comprehensive Metrics Collector                │")
    print("│        (InferenceX-style — SemiAnalysis inspired)          │")
    print("└─────────────────────────────────────────────────────────────┘")
    print()
    print("Collecting system info...")
    system_info = collect_system_info()
    num_gpus = system_info.get("num_gpus", 8)

    print(f"  CPU     : {system_info.get('cpu_model', '?')}")
    print(f"  Cores   : {system_info.get('cpu_cores', '?')}")
    print(f"  RAM     : {system_info.get('ram_total_bytes', 0) / 1e9:.1f} GB")
    print(f"  GPUs    : {system_info.get('gpu_type', '?')}")
    print(f"  Kernel  : {system_info.get('kernel', '?')}")
    nvlink = system_info.get("nvlink", {})
    if isinstance(nvlink, dict) and nvlink:
        first_gpu = next(iter(nvlink.values()))
        print(f"  NVLink  : {first_gpu.get('num_links', '?')} links × "
              f"{first_gpu.get('per_link_gbps', '?')} GB/s = "
              f"{first_gpu.get('total_bw_gbps', '?')} GB/s per GPU")
    print()
    print(f"  Model   : {SERVER_CONFIG['model']}")
    print(f"  Quant   : {SERVER_CONFIG['quantization']}")
    print(f"  TP      : {SERVER_CONFIG['tensor_parallel_size']}")
    print(f"  EP      : {'enabled' if SERVER_CONFIG['expert_parallel'] else 'disabled'}")
    print(f"  Workload: {SERVER_CONFIG['workload_type']}")
    print()
    print(f"  Metrics → {metrics_path}")
    print(f"  Prometheus (raw) → {prometheus_path}")
    print(f"  Polling every {args.interval}s")
    print()
    print("Press Ctrl+C to stop.")
    print("─" * 62)

    session = requests.Session()
    delta_tracker = DeltaTracker()
    poll_count = 0

    # Initialize prometheus ground truth file (as a JSON array)
    # We append each scrape as an element
    prometheus_scrapes: list[dict] = []
    if prometheus_path.exists():
        try:
            with open(prometheus_path) as f:
                prometheus_scrapes = json.load(f)
            print(f"  Loaded {len(prometheus_scrapes)} existing Prometheus scrapes")
        except (json.JSONDecodeError, Exception):
            prometheus_scrapes = []

    while True:
        t_start = time.monotonic()
        record, raw_prometheus_text = collect_once(
            session, args.url, delta_tracker, system_info, num_gpus
        )

        if record is not None and raw_prometheus_text is not None:
            record["collection_interval_s"] = args.interval
            record["poll_index"] = poll_count

            # ── Write processed metrics to JSONL ─────────────────────────
            with open(metrics_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")

            # ── Write raw Prometheus to ground truth file ────────────────
            prometheus_scrapes.append({
                "timestamp": record["timestamp"],
                "unix_ts": record["unix_ts"],
                "poll_index": poll_count,
                "raw_text": raw_prometheus_text,
                "parsed": {
                    k: {
                        sk: sv for sk, sv in v.items()
                    } if isinstance(v, dict) else v
                    for k, v in parse_prometheus_full(raw_prometheus_text).items()
                },
            })
            with open(prometheus_path, "w") as f:
                json.dump(prometheus_scrapes, f, indent=2, default=str)

            # ── Pretty-print summary ─────────────────────────────────────
            d = record["inferencex_metrics"]
            r = record["rates"]

            gen_rate = d.get("output_throughput_tok_per_sec")
            prompt_rate = d.get("prompt_throughput_tok_per_sec")
            gen_str = f"{gen_rate:.1f}" if gen_rate is not None else "—"
            prompt_str = f"{prompt_rate:.1f}" if prompt_rate is not None else "—"

            ttft_str = "—"
            tpot_str = "—"
            interactivity_str = "—"
            if d.get("ttft") and isinstance(d["ttft"], dict):
                ttft_str = f"{d['ttft'].get('p50', 0)*1000:.0f}ms"
            if d.get("tpot") and isinstance(d["tpot"], dict):
                tpot_str = f"{d['tpot'].get('p50', 0)*1000:.1f}ms"
            if d.get("interactivity_p50_tok_per_sec_per_user") is not None:
                interactivity_str = f"{d['interactivity_p50_tok_per_sec_per_user']:.1f}"

            power_str = f"{d.get('total_power_draw_w', 0):.0f}W" if d.get("total_power_draw_w") else "—"
            eff_str = "—"
            if d.get("output_tokens_per_watt") is not None:
                eff_str = f"{d['output_tokens_per_watt']:.3f}"

            print(
                f"[{record['timestamp'][:19]}] "
                f"gen={gen_str} tok/s  prompt={prompt_str} tok/s │ "
                f"run={d['requests_running']} wait={d['requests_waiting']} │ "
                f"TTFT={ttft_str} TPOT={tpot_str} │ "
                f"interact={interactivity_str} tok/s/user │ "
                f"kv={d['kv_cache_usage_pct']:.1f}% "
                f"pfx={d['prefix_cache_hit_rate_pct']:.1f}% │ "
                f"gpu={d.get('avg_gpu_utilization_pct', '?')}% "
                f"{power_str} "
                f"eff={eff_str} tok/W"
            )
            poll_count += 1

        elapsed = time.monotonic() - t_start
        sleep_for = max(0, args.interval - elapsed)
        try:
            time.sleep(sleep_for)
        except KeyboardInterrupt:
            print()
            print("─" * 62)
            print(f"Stopped. {poll_count} snapshots collected.")
            print(f"  Metrics   : {metrics_path}")
            print(f"  Prometheus: {prometheus_path}")
            break


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Batch-size comparison charts for Qwen3-235B-A22B on 8x H100 — vLLM serving.
Inspired by SemiAnalysis InferenceX methodology: interactivity vs throughput
Pareto curves plus supporting efficiency, latency, and cache panels.

Usage:
    python analysis_multiturn/chart_batch_comparison.py
    python analysis_multiturn/chart_batch_comparison.py --out charts/
"""

import json
import argparse
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────────
SEMIANALYSIS_STYLE = {
    "figure.facecolor": "#0f0f13",
    "axes.facecolor": "#16161e",
    "axes.edgecolor": "#2a2a3a",
    "axes.labelcolor": "#c8c8d8",
    "axes.titlecolor": "#e8e8f0",
    "axes.grid": True,
    "grid.color": "#2a2a3a",
    "grid.linewidth": 0.6,
    "xtick.color": "#8888a0",
    "ytick.color": "#8888a0",
    "text.color": "#c8c8d8",
    "legend.facecolor": "#1e1e2a",
    "legend.edgecolor": "#3a3a4a",
    "legend.labelcolor": "#c8c8d8",
    "font.family": "sans-serif",
    "font.size": 11,
}

BATCH_COLORS = {
    4:  "#4e9af1",   # blue
    8:  "#f1c94e",   # amber
    16: "#4ef1a0",   # green
    32: "#f14e4e",   # red
    64: "#c44ef1",   # purple (future)
}

BATCH_SIZES = [4, 8, 16, 32]

METRIC_FILES = {
    4:  "metrics/metrics-batch4/metrics.jsonl",
    8:  "metrics/metrics-batch8/metrics-batch8.jsonl",
    16: "metrics/metrics-batch16/metrics-batch16.jsonl",
    32: "metrics/metrics-batch32/metrics-batch32.jsonl",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_inferencex(records: list[dict]) -> list[dict]:
    return [r["inferencex_metrics"] for r in records if "inferencex_metrics" in r]


def safe_avg(lst, key, min_val=0):
    vals = [x[key] for x in lst if isinstance(x.get(key), (int, float)) and x[key] > min_val]
    return statistics.mean(vals) if vals else None


def safe_median(lst, key, min_val=0):
    vals = [x[key] for x in lst if isinstance(x.get(key), (int, float)) and x[key] > min_val]
    return statistics.median(vals) if vals else None


def safe_nested(lst, outer_key, inner_key, min_val=0):
    vals = [x[outer_key][inner_key] for x in lst
            if isinstance(x.get(outer_key), dict) and isinstance(x[outer_key].get(inner_key), (int, float))
            and x[outer_key][inner_key] > min_val]
    return statistics.mean(vals) if vals else None


def percentile(data, p):
    if not data:
        return None
    data = sorted(data)
    idx = int(len(data) * p / 100)
    return data[min(idx, len(data) - 1)]


def load_all(root: Path) -> dict:
    """Load all batch-size datasets; return dict of {batch_size: list[im_dict]}."""
    result = {}
    for bs, rel_path in METRIC_FILES.items():
        full_path = root / rel_path
        if not full_path.exists():
            print(f"  [skip] batch={bs} — file not found: {full_path}")
            continue
        records = load_jsonl(str(full_path))
        im = extract_inferencex(records)
        result[bs] = im
        print(f"  batch={bs:>2}: {len(im)} records loaded")
    return result


# ── Aggregated summary per batch ───────────────────────────────────────────────

def summarise(im_list: list[dict]) -> dict:
    throughputs = [x["output_throughput_tok_per_sec"] for x in im_list
                   if isinstance(x.get("output_throughput_tok_per_sec"), (int, float))
                   and x["output_throughput_tok_per_sec"] > 5]

    active = [x for x in im_list if (x.get("requests_running") or 0) > 0]

    return {
        # throughput
        "throughput_median":  percentile(throughputs, 50),
        "throughput_p25":     percentile(throughputs, 25),
        "throughput_p75":     percentile(throughputs, 75),
        "throughput_max":     max(throughputs) if throughputs else None,
        # interactivity
        "interactivity_mean":  safe_avg(im_list, "interactivity_mean_tok_per_sec_per_user"),
        "interactivity_p50":   safe_avg(im_list, "interactivity_p50_tok_per_sec_per_user"),
        "interactivity_p90":   safe_avg(im_list, "interactivity_p90_tok_per_sec_per_user"),
        "interactivity_p95":   safe_avg(im_list, "interactivity_p95_tok_per_sec_per_user"),
        # latency
        "ttft_p50":     safe_nested(im_list, "ttft", "p50"),
        "ttft_p90":     safe_nested(im_list, "ttft", "p90"),
        "ttft_mean":    safe_nested(im_list, "ttft", "mean"),
        "tpot_p50":     safe_nested(im_list, "tpot", "p50"),
        "tpot_mean":    safe_nested(im_list, "tpot", "mean"),
        "e2e_p50":      safe_nested(im_list, "e2e_latency", "p50"),
        "e2e_mean":     safe_nested(im_list, "e2e_latency", "mean"),
        # cache
        "kv_cache_pct":         safe_avg(im_list, "kv_cache_usage_pct"),
        "prefix_hit_rate":      safe_avg(im_list, "prefix_cache_hit_rate_pct"),
        # gpu / efficiency
        "gpu_util":             safe_avg(im_list, "avg_gpu_utilization_pct"),
        "total_power_w":        safe_avg(im_list, "total_power_draw_w"),
        "tok_per_watt":         safe_avg(im_list, "output_tokens_per_watt"),
        "mj_per_tok":           safe_avg(im_list, "millijoules_per_output_token"),
        # per-request token counts (from active records)
        "mean_gen_tokens":      safe_nested(active, "request_generation_tokens", "mean"),
        "mean_prompt_tokens":   safe_nested(active, "request_prompt_tokens", "mean"),
        "requests_running_avg": safe_avg(active, "requests_running"),
        "requests_waiting_avg": safe_avg(active, "requests_waiting"),
    }


# ── Chart helpers ──────────────────────────────────────────────────────────────

def apply_style():
    plt.rcParams.update(SEMIANALYSIS_STYLE)


def label_bars(ax, bars, fmt="{:.0f}", color="#c8c8d8", fontsize=9, offset=3):
    for bar in bars:
        h = bar.get_height()
        if h and h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + offset,
                fmt.format(h),
                ha="center", va="bottom",
                fontsize=fontsize, color=color, fontweight="bold",
            )


def spine_off(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_annotation(ax, text, xy, xytext, color="#f1c94e"):
    ax.annotate(
        text, xy=xy, xytext=xytext,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        fontsize=8.5, color=color,
        bbox=dict(boxstyle="round,pad=0.3", fc="#1e1e2a", ec=color, lw=0.8),
    )


# ── Individual panels ──────────────────────────────────────────────────────────

def plot_interactivity_vs_throughput(ax, summaries, data_all):
    """
    Main InferenceX-style Pareto scatter.
    x-axis = interactivity (1/TPOT_p50, tok/s per user)
    y-axis = aggregate output throughput (tok/s)

    B=4 is a dominated / underloaded operating point — GPU only 53% utilized,
    avg 2.6 requests running, prefill preemptions degrade TPOT.
    True Pareto frontier runs from B=8 → B=16 → B=32.
    """
    # Separate dominated point (B=4) from Pareto frontier points (B>=8)
    DOMINATED = {4}

    pareto_xs, pareto_ys, pareto_bs = [], [], []
    dom_xs, dom_ys, dom_bs = [], [], []

    for bs in BATCH_SIZES:
        if bs not in summaries:
            continue
        s = summaries[bs]
        x = s["interactivity_p50"]
        y = s["throughput_median"]
        if not x or not y:
            continue
        if bs in DOMINATED:
            dom_xs.append(x); dom_ys.append(y); dom_bs.append(bs)
        else:
            pareto_xs.append(x); pareto_ys.append(y); pareto_bs.append(bs)

    # Pareto frontier line (B=8 → B=16 → B=32)
    order = np.argsort(pareto_xs)
    ax.plot([pareto_xs[i] for i in order], [pareto_ys[i] for i in order],
            color="#6060a0", linewidth=2.0, linestyle="--", zorder=1, alpha=0.9,
            label="Pareto frontier")

    # Pareto points
    for x, y, bs in zip(pareto_xs, pareto_ys, pareto_bs):
        ax.scatter(x, y, color=BATCH_COLORS[bs], s=260, zorder=5,
                   edgecolors="#ffffff44", linewidths=1.2)
        ax.text(x + 0.3, y + 18, f"B={bs}", color=BATCH_COLORS[bs],
                fontsize=10, fontweight="bold", va="bottom")

    # Dominated points (dimmed, dashed border)
    for x, y, bs in zip(dom_xs, dom_ys, dom_bs):
        ax.scatter(x, y, color=BATCH_COLORS[bs], s=200, zorder=4,
                   edgecolors="#f1c94e", linewidths=1.5,
                   alpha=0.6, marker="D")
        ax.text(x + 0.3, y + 18, f"B={bs}\n(under-\nloaded)",
                color=BATCH_COLORS[bs], fontsize=8, va="bottom", alpha=0.75)

    # Arrow from dominated point showing it's strictly dominated by B=8
    if dom_xs and pareto_xs:
        bx, by = dom_xs[0], dom_ys[0]   # B=4
        # find B=8 point
        b8_idx = pareto_bs.index(8) if 8 in pareto_bs else 0
        px, py = pareto_xs[b8_idx], pareto_ys[b8_idx]
        ax.annotate("",
            xy=(px - 0.5, py - 20), xytext=(bx + 1, by + 20),
            arrowprops=dict(arrowstyle="-|>", color="#f1c94e",
                            lw=1.4, connectionstyle="arc3,rad=0.25"),
            zorder=6)
        ax.text((bx + px) / 2 - 5, (by + py) / 2 + 60,
                "B=8 dominates B=4\n(↑ throughput  +  ↑ interactivity)",
                color="#f1c94e", fontsize=8.5, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#1e1e2a", ec="#f1c94e", lw=0.8))

    # 50 tok/s reference
    ax.axvline(50, color="#ffffff18", lw=1, linestyle=":")
    ax.text(50.2, 30, "50 tok/s/user\n(comfortable reading)", color="#ffffff45", fontsize=7.5)

    # Explain why B=4 is off-frontier in a text box
    ax.text(0.02, 0.97,
        "Why B=4 is off the frontier:\n"
        "GPU only 53% utilised (avg 2.6 req running).\n"
        "Frequent prefill preemptions inflate TPOT.\n"
        "B=8 wins on both axes → B=4 is dominated.",
        transform=ax.transAxes, fontsize=8, va="top", ha="left",
        color="#aaaacc",
        bbox=dict(boxstyle="round,pad=0.4", fc="#0f0f1a", ec="#3a3a6a", lw=0.8))

    ax.set_xlabel("Interactivity  (1/TPOT_p50,  tok/s per user)  ←  more responsive", fontsize=11)
    ax.set_ylabel("Aggregate Output Throughput  (tok/s)  ↑  more capacity", fontsize=11)
    ax.set_title("Interactivity  ↔  Throughput Tradeoff\nQwen3-235B-A22B-FP8 · 8× H100 · vLLM", fontsize=13, pad=10)
    spine_off(ax)

    add_annotation(ax,
        "Pareto starts at B=8\n(GPU saturated, 100% util)",
        (pareto_xs[0], pareto_ys[0]) if pareto_xs else (57, 264),
        (pareto_xs[0] - 7, pareto_ys[0] + 100) if pareto_xs else (50, 364),
    )
    if len(pareto_bs) >= 2:
        add_annotation(ax,
            "B=32: 9.5× more throughput\nvs B=4, interactivity intact",
            (pareto_xs[-1], pareto_ys[-1]),
            (pareto_xs[-1] - 8, pareto_ys[-1] - 120),
        )

    # ── Context table (top-left) — replaces the matplotlib legend ────────────
    if summaries:
        bs_cols = [bs for bs in BATCH_SIZES if bs in summaries]
        col_w = 7  # chars per column

        # Header row with colored batch labels (we'll render as plain text; color via bbox)
        header = f"{'Metric':<24s}" + "".join(f"{'B='+str(bs):>{col_w}}" for bs in bs_cols)
        divider = "─" * (24 + col_w * len(bs_cols))

        rows_def = [
            ("● Throughput (tok/s)",   "throughput_median",    "{:.0f}"),
            ("● Interactivity p50",    "interactivity_p50",    "{:.1f}"),
            ("  TTFT p50 (s)",         "ttft_p50",             "{:.2f}"),
            ("  Mean gen tokens",      "mean_gen_tokens",      "{:.0f}"),
            ("  Mean prompt tokens",   "mean_prompt_tokens",   "{:.0f}"),
            ("  Req running (avg)",    "requests_running_avg", "{:.1f}"),
            ("  GPU util (%)",         "gpu_util",             "{:.0f}"),
            ("  KV cache (%)",         "kv_cache_pct",         "{:.1f}"),
            ("  Prefix hit rate (%)",  "prefix_hit_rate",      "{:.1f}"),
            ("  tok/s per Watt",       "tok_per_watt",         "{:.3f}"),
        ]

        table_lines = [header, divider]
        for label, key, fmt in rows_def:
            vals = []
            for bs in bs_cols:
                v = summaries[bs].get(key)
                vals.append(fmt.format(v) if v is not None else "—")
            table_lines.append(f"{label:<24s}" + "".join(f"{v:>{col_w}}" for v in vals))

        # Color swatches row at top
        swatch_row = f"{'':24s}" + "".join(
            f"{'██':>{col_w}}" for _ in bs_cols
        )

        full_text = "\n".join(table_lines)
        ax.text(0.01, 0.99, full_text,
                transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
                fontfamily="monospace", color="#b8b8d0",
                bbox=dict(boxstyle="round,pad=0.5", fc="#0b0b16", ec="#2a2a4a", lw=0.9, alpha=0.95))

        # Draw small colored squares next to batch column headers using ax.transAxes
        # We place them as small patches aligned with the text
        # Approximate x positions based on character width in axes coords
        char_w_ax = 0.016   # approximate width per monospace char in axes fraction
        start_x = 0.01 + 24 * char_w_ax
        line_h = 0.028      # approx height per line in axes fraction
        top_y = 0.99 - 0.012
        for i, bs in enumerate(bs_cols):
            cx = start_x + (i * col_w + col_w // 2 - 1) * char_w_ax
            rect = mpatches.FancyBboxPatch(
                (cx, top_y - 0.008), 0.022, 0.013,
                boxstyle="round,pad=0.001",
                facecolor=BATCH_COLORS[bs], edgecolor="none",
                transform=ax.transAxes, zorder=10, alpha=0.9,
            )
            ax.add_patch(rect)


def plot_ttft(ax, summaries):
    """Time-to-First-Token p50 and p90 bar chart."""
    bs_list = [bs for bs in BATCH_SIZES if bs in summaries]
    x = np.arange(len(bs_list))
    w = 0.35

    p50s = [summaries[bs]["ttft_p50"] or 0 for bs in bs_list]
    p90s = [summaries[bs]["ttft_p90"] or 0 for bs in bs_list]

    b1 = ax.bar(x - w / 2, p50s, w, label="p50", color=[BATCH_COLORS[bs] for bs in bs_list], alpha=0.9)
    b2 = ax.bar(x + w / 2, p90s, w, label="p90",
                color=[BATCH_COLORS[bs] for bs in bs_list], alpha=0.45, hatch="//", edgecolor="none")

    label_bars(ax, b1, fmt="{:.1f}s", offset=0.05)
    label_bars(ax, b2, fmt="{:.1f}s", offset=0.05)

    ax.set_xticks(x)
    ax.set_xticklabels([f"B={bs}" for bs in bs_list])
    ax.set_ylabel("TTFT (seconds)  ↓ lower is better")
    ax.set_title("Time to First Token")
    ax.legend(fontsize=9)
    spine_off(ax)

    # Reference line: ~2s is comfortable for interactive use
    ax.axhline(2.0, color="#f1c94e", lw=1, linestyle="--", alpha=0.6)
    ax.text(len(bs_list) - 0.5, 2.1, "2s threshold", color="#f1c94e", fontsize=8, ha="right")


def plot_throughput_bars(ax, summaries):
    """Median output throughput with IQR error bars."""
    bs_list = [bs for bs in BATCH_SIZES if bs in summaries]
    x = np.arange(len(bs_list))
    medians = [summaries[bs]["throughput_median"] or 0 for bs in bs_list]
    p25 = [summaries[bs]["throughput_p25"] or 0 for bs in bs_list]
    p75 = [summaries[bs]["throughput_p75"] or 0 for bs in bs_list]

    err_lo = [m - l for m, l in zip(medians, p25)]
    err_hi = [h - m for m, h in zip(medians, p75)]

    bars = ax.bar(x, medians, color=[BATCH_COLORS[bs] for bs in bs_list], alpha=0.9,
                  yerr=[err_lo, err_hi], error_kw=dict(ecolor="#ffffff60", capsize=5, lw=1.5))

    for bar, val in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 15,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9,
                color="#c8c8d8", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"B={bs}" for bs in bs_list])
    ax.set_ylabel("Output Throughput (tok/s)  ↑ higher is better")
    ax.set_title("Aggregate Output Throughput\n(median ± IQR)")
    spine_off(ax)


def plot_interactivity_bands(ax, summaries):
    """
    Interactivity percentile bands (mean, p50, p90) as grouped bars.
    Shows how per-user responsiveness degrades at higher concurrency.
    """
    bs_list = [bs for bs in BATCH_SIZES if bs in summaries]
    x = np.arange(len(bs_list))
    w = 0.25

    means = [summaries[bs]["interactivity_mean"] or 0 for bs in bs_list]
    p50s  = [summaries[bs]["interactivity_p50"] or 0 for bs in bs_list]
    p90s  = [summaries[bs]["interactivity_p90"] or 0 for bs in bs_list]

    ax.bar(x - w, means, w, label="Mean",   color=[BATCH_COLORS[bs] for bs in bs_list], alpha=0.95)
    ax.bar(x,     p50s,  w, label="p50",    color=[BATCH_COLORS[bs] for bs in bs_list], alpha=0.65)
    ax.bar(x + w, p90s,  w, label="p90 (slowest users)",
           color=[BATCH_COLORS[bs] for bs in bs_list], alpha=0.35)

    # Speech-quality reference
    ax.axhline(15, color="#f14e4e", lw=1, linestyle="--", alpha=0.7)
    ax.text(len(bs_list) - 0.3, 16, "15 tok/s — speech minimum", color="#f14e4e", fontsize=8, ha="right")
    ax.axhline(50, color="#4ef1a0", lw=1, linestyle="--", alpha=0.7)
    ax.text(len(bs_list) - 0.3, 51, "50 tok/s — comfortable reading", color="#4ef1a0", fontsize=8, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([f"B={bs}" for bs in bs_list])
    ax.set_ylabel("Tokens / sec / user  ↑ more responsive")
    ax.set_title("Per-User Interactivity\n(Mean / p50 / p90)")
    ax.legend(fontsize=9, loc="lower left")
    spine_off(ax)


def plot_gpu_util_and_power(ax1, ax2, summaries):
    """GPU utilisation and power draw side-by-side."""
    bs_list = [bs for bs in BATCH_SIZES if bs in summaries]
    x = np.arange(len(bs_list))
    colors = [BATCH_COLORS[bs] for bs in bs_list]

    utils = [summaries[bs]["gpu_util"] or 0 for bs in bs_list]
    bars1 = ax1.bar(x, utils, color=colors, alpha=0.9)
    label_bars(ax1, bars1, fmt="{:.0f}%", offset=0.5)
    ax1.set_ylim(0, 115)
    ax1.axhline(100, color="#ffffff30", lw=0.8, linestyle="--")
    ax1.set_xticks(x); ax1.set_xticklabels([f"B={bs}" for bs in bs_list])
    ax1.set_ylabel("Avg GPU Utilisation (%)")
    ax1.set_title("GPU Utilisation")
    spine_off(ax1)

    powers = [summaries[bs]["total_power_w"] or 0 for bs in bs_list]
    bars2 = ax2.bar(x, powers, color=colors, alpha=0.85)
    label_bars(ax2, bars2, fmt="{:.0f}W", offset=5)
    ax2.set_xticks(x); ax2.set_xticklabels([f"B={bs}" for bs in bs_list])
    ax2.set_ylabel("Total System Power Draw (W)")
    ax2.set_title("Power Draw  (8× H100)")
    # max power line: 8 × 700W
    ax2.axhline(5600, color="#f14e4e", lw=1, linestyle="--", alpha=0.6)
    ax2.text(0, 5650, "TDP limit (8×700W)", color="#f14e4e", fontsize=8)
    spine_off(ax2)


def plot_efficiency(ax, summaries):
    """Output tokens per watt — higher is better."""
    bs_list = [bs for bs in BATCH_SIZES if bs in summaries]
    x = np.arange(len(bs_list))

    tpw = [summaries[bs]["tok_per_watt"] or 0 for bs in bs_list]
    bars = ax.bar(x, tpw, color=[BATCH_COLORS[bs] for bs in bs_list], alpha=0.9)

    for bar, val in zip(bars, tpw):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                color="#c8c8d8", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"B={bs}" for bs in bs_list])
    ax.set_ylabel("Output tok/s / Watt  ↑ more efficient")
    ax.set_title("Energy Efficiency")
    spine_off(ax)

    # Annotation: efficiency scales with batch
    if len(tpw) >= 2 and tpw[0] and tpw[-1]:
        ratio = tpw[-1] / tpw[0]
        ax.text(len(bs_list) / 2 - 0.5, max(tpw) * 0.7,
                f"{ratio:.1f}× more efficient\nat B=32 vs B=4",
                color="#f1c94e", fontsize=9, ha="center",
                bbox=dict(boxstyle="round", fc="#1e1e2a", ec="#f1c94e", lw=0.8))


def plot_cache(ax1, ax2, summaries):
    """KV cache utilisation and prefix cache hit rate."""
    bs_list = [bs for bs in BATCH_SIZES if bs in summaries]
    x = np.arange(len(bs_list))
    colors = [BATCH_COLORS[bs] for bs in bs_list]

    kv = [summaries[bs]["kv_cache_pct"] or 0 for bs in bs_list]
    bars1 = ax1.bar(x, kv, color=colors, alpha=0.9)
    label_bars(ax1, bars1, fmt="{:.1f}%", offset=0.2)
    ax1.set_xticks(x); ax1.set_xticklabels([f"B={bs}" for bs in bs_list])
    ax1.set_ylabel("KV Cache Usage (%)")
    ax1.set_title("KV Cache Utilisation")
    spine_off(ax1)

    phr = [summaries[bs]["prefix_hit_rate"] or 0 for bs in bs_list]
    bars2 = ax2.bar(x, phr, color=colors, alpha=0.85)
    label_bars(ax2, bars2, fmt="{:.1f}%", offset=0.3)
    ax2.set_xticks(x); ax2.set_xticklabels([f"B={bs}" for bs in bs_list])
    ax2.set_ylabel("Prefix Cache Hit Rate (%)")
    ax2.set_title("Prefix Cache Hit Rate\n(multi-turn context reuse)")
    ax2.set_ylim(0, 100)
    spine_off(ax2)


def plot_e2e_latency(ax, summaries):
    """E2E request latency p50 and mean."""
    bs_list = [bs for bs in BATCH_SIZES if bs in summaries]
    x = np.arange(len(bs_list))
    w = 0.35

    p50s  = [summaries[bs]["e2e_p50"] or 0 for bs in bs_list]
    means = [summaries[bs]["e2e_mean"] or 0 for bs in bs_list]

    b1 = ax.bar(x - w / 2, p50s,  w, label="p50",  color=[BATCH_COLORS[bs] for bs in bs_list], alpha=0.9)
    b2 = ax.bar(x + w / 2, means, w, label="mean", color=[BATCH_COLORS[bs] for bs in bs_list], alpha=0.45)

    label_bars(ax, b1, fmt="{:.0f}s", offset=1)
    label_bars(ax, b2, fmt="{:.0f}s", offset=1)

    ax.set_xticks(x)
    ax.set_xticklabels([f"B={bs}" for bs in bs_list])
    ax.set_ylabel("E2E Latency (seconds)  ↓ lower is better")
    ax.set_title("End-to-End Request Latency\n(p50 and mean)")
    ax.legend(fontsize=9)
    spine_off(ax)


# ── Main dashboard ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--out", default="analysis_multiturn/charts", help="Output directory")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metrics…")
    data_all = load_all(root)
    if not data_all:
        print("No data found. Check METRIC_FILES paths.")
        return

    summaries = {bs: summarise(im) for bs, im in data_all.items()}

    print("\nSummary table:")
    print(f"  {'Metric':<35}  " + "  ".join(f"B={bs:>2}" for bs in BATCH_SIZES if bs in summaries))
    rows = [
        ("Throughput median (tok/s)",   "throughput_median", "{:.0f}"),
        ("Throughput max (tok/s)",       "throughput_max",    "{:.0f}"),
        ("Interactivity mean (tok/s/u)", "interactivity_mean", "{:.1f}"),
        ("Interactivity p50 (tok/s/u)",  "interactivity_p50",  "{:.1f}"),
        ("Interactivity p90 (tok/s/u)",  "interactivity_p90",  "{:.1f}"),
        ("TTFT p50 (s)",                 "ttft_p50",    "{:.2f}"),
        ("TTFT p90 (s)",                 "ttft_p90",    "{:.2f}"),
        ("E2E latency p50 (s)",          "e2e_p50",     "{:.0f}"),
        ("KV cache usage (%)",           "kv_cache_pct", "{:.1f}"),
        ("Prefix hit rate (%)",          "prefix_hit_rate", "{:.1f}"),
        ("GPU utilisation (%)",          "gpu_util",    "{:.1f}"),
        ("Total power (W)",              "total_power_w", "{:.0f}"),
        ("tok/s per Watt",               "tok_per_watt", "{:.4f}"),
    ]
    for label, key, fmt in rows:
        vals = [fmt.format(summaries[bs][key]) if summaries[bs].get(key) is not None else "  —  "
                for bs in BATCH_SIZES if bs in summaries]
        print(f"  {label:<35}  " + "  ".join(f"{v:>8}" for v in vals))

    apply_style()

    # ── Figure 1: Main dashboard (2×4 grid) ───────────────────────────────────
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor("#0f0f13")

    # Title
    fig.suptitle(
        "Qwen3-235B-A22B-FP8 · 8× NVIDIA H100 · vLLM  ·  Batch Size Sweep",
        fontsize=16, fontweight="bold", color="#e8e8f0", y=0.98,
    )
    fig.text(
        0.5, 0.955,
        "Methodology inspired by SemiAnalysis InferenceX — interactivity vs throughput Pareto analysis",
        ha="center", fontsize=10, color="#8888a0",
    )

    gs = fig.add_gridspec(3, 4, hspace=0.52, wspace=0.38,
                          left=0.06, right=0.97, top=0.93, bottom=0.05)

    # Row 0: main Pareto spans first 2 cols; throughput bars; interactivity bands
    ax_pareto  = fig.add_subplot(gs[0, :2])
    ax_tput    = fig.add_subplot(gs[0, 2])
    ax_inter   = fig.add_subplot(gs[0, 3])

    # Row 1: TTFT | E2E latency | GPU util | power
    ax_ttft    = fig.add_subplot(gs[1, 0])
    ax_e2e     = fig.add_subplot(gs[1, 1])
    ax_gutil   = fig.add_subplot(gs[1, 2])
    ax_power   = fig.add_subplot(gs[1, 3])

    # Row 2: efficiency | kv cache | prefix hit rate | (summary text)
    ax_eff     = fig.add_subplot(gs[2, 0])
    ax_kv      = fig.add_subplot(gs[2, 1])
    ax_prefix  = fig.add_subplot(gs[2, 2])
    ax_notes   = fig.add_subplot(gs[2, 3])

    # Populate
    plot_interactivity_vs_throughput(ax_pareto, summaries, data_all)
    plot_throughput_bars(ax_tput, summaries)
    plot_interactivity_bands(ax_inter, summaries)
    plot_ttft(ax_ttft, summaries)
    plot_e2e_latency(ax_e2e, summaries)
    plot_gpu_util_and_power(ax_gutil, ax_power, summaries)
    plot_efficiency(ax_eff, summaries)
    plot_cache(ax_kv, ax_prefix, summaries)

    # Notes panel
    ax_notes.set_facecolor("#0f0f13")
    spine_off(ax_notes)
    ax_notes.set_xticks([]); ax_notes.set_yticks([])

    bs_present = [bs for bs in BATCH_SIZES if bs in summaries]
    notes_lines = [
        "Key Findings",
        "",
        "▸ B=4 is off the Pareto frontier",
        "  GPU: 53% util, avg 2.6 req",
        "  running. Prefill preemptions",
        "  inflate TPOT → B=8 wins",
        "  on BOTH throughput & interactivity",
        "",
        "▸ True Pareto: B=8 → B=32",
        "  GPU saturates at B=8 (100%)",
        "  Throughput scales; interactivity",
        "  erodes slowly with queue depth",
        "",
        "▸ Throughput scaling",
    ]
    if len(bs_present) >= 2:
        b8_t = summaries.get(8, {}).get("throughput_median")
        b32_t = summaries.get(32, {}).get("throughput_median")
        if b8_t and b32_t:
            notes_lines.append(f"  {b32_t/b8_t:.1f}× gain: B=8 → B=32")

    notes_lines += [
        "",
        "▸ Interactivity (B=8-32)",
        "  p90 stays >40 tok/s/user",
        "  above speech threshold",
        "  even at max batch",
        "",
        "▸ Prefix cache hit rate ~70%",
        "  stable across all batches",
        "  (multi-turn context reuse)",
        "",
        "▸ Efficiency",
        "  tok/W scales 4.7× from",
        "  B=4 → B=32 (GPU util gain)",
    ]

    ax_notes.text(0.05, 0.97, "\n".join(notes_lines),
                  transform=ax_notes.transAxes,
                  fontsize=8.5, color="#c8c8d8",
                  va="top", ha="left",
                  fontfamily="monospace",
                  bbox=dict(boxstyle="round", fc="#16161e", ec="#3a3a4a", lw=0.8))

    # Legend
    legend_patches = [mpatches.Patch(color=BATCH_COLORS[bs], label=f"Batch size {bs}")
                      for bs in BATCH_SIZES if bs in summaries]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(legend_patches),
               fontsize=10, framealpha=0.4, bbox_to_anchor=(0.5, 0.0))

    out_path = out_dir / "batch_comparison_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n✓ Saved dashboard → {out_path}")
    plt.close(fig)

    # ── Figure 2: Standalone Pareto (cleaner, shareable) ──────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    fig2.patch.set_facecolor("#0f0f13")
    ax2.set_facecolor("#16161e")

    plot_interactivity_vs_throughput(ax2, summaries, data_all)

    # Legend is embedded in the context table inside plot_interactivity_vs_throughput

    fig2.tight_layout()
    out_path2 = out_dir / "interactivity_vs_throughput.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    print(f"✓ Saved Pareto chart → {out_path2}")
    plt.close(fig2)

    print("\nDone.")


if __name__ == "__main__":
    main()

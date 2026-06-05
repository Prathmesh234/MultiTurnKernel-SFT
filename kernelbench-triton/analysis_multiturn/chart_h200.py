#!/usr/bin/env python3
"""
H200 analysis charts for Qwen3-235B-A22B on 8x H200 — vLLM serving.

Produces two figures:
  1. h200_dashboard.png            — B=128 metrics dashboard (Run 1 vs Run 2)
  2. h100_vs_h200_comparison.png   — Cross-hardware comparison: H100 best (B=32) vs H200 (B=128)

Usage:
    uv run analysis_multiturn/chart_h200.py
    uv run analysis_multiturn/chart_h200.py --out charts/
"""

import json
import argparse
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "#f7f6f2",
    "axes.facecolor": "#efeeea",
    "axes.edgecolor": "#c8c4bc",
    "axes.labelcolor": "#2a2a3a",
    "axes.titlecolor": "#1a1a28",
    "axes.grid": True,
    "grid.color": "#d8d4cc",
    "grid.linewidth": 0.6,
    "xtick.color": "#4a4a5a",
    "ytick.color": "#4a4a5a",
    "text.color": "#2a2a3a",
    "legend.facecolor": "#f0efe9",
    "legend.edgecolor": "#c0bbb0",
    "legend.labelcolor": "#2a2a3a",
    "font.family": "sans-serif",
    "font.size": 11,
}

# H200 run colors (lighter)
RUN_COLORS = {
    "h200_run1": "#c4b5fd",   # light violet
    "h200_run2": "#6ee7b7",   # light emerald
}

# Cross-hardware colors (lighter)
HW_COLORS = {
    "H100 B=4":  "#7eb8f7",
    "H100 B=8":  "#f7d97e",
    "H100 B=16": "#7ef7b8",
    "H100 B=32": "#f78e8e",
    "H200 B=128 R1": "#c4b5fd",
    "H200 B=128 R2": "#6ee7b7",
}

# H100 metric files (relative to project root)
H100_METRIC_FILES = {
    4:  "metrics/metrics-batch4/metrics.jsonl",
    8:  "metrics/metrics-batch8/metrics-batch8.jsonl",
    16: "metrics/metrics-batch16/metrics-batch16.jsonl",
    32: "metrics/metrics-batch32/metrics-batch32.jsonl",
}

H200_METRIC_FILES = {
    "run1": "metrics/metrics-batch128-h200s/metrics-batch128-h200s.jsonl",
    "run2": "metrics/metrics-batch128-h200s-2/metrics-batch128-h200s-2.jsonl",
}


# ── Data helpers ───────────────────────────────────────────────────────────────

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


def safe_nested(lst, outer_key, inner_key, min_val=0):
    vals = [x[outer_key][inner_key] for x in lst
            if isinstance(x.get(outer_key), dict)
            and isinstance(x[outer_key].get(inner_key), (int, float))
            and x[outer_key][inner_key] > min_val]
    return statistics.mean(vals) if vals else None


def percentile(data, p):
    if not data:
        return None
    data = sorted(data)
    idx = int(len(data) * p / 100)
    return data[min(idx, len(data) - 1)]


def summarise(im_list: list[dict], min_throughput: float = 10.0) -> dict:
    # Drop idle/startup polls where the engine has no active requests.
    # This removes Modal container cold-start gaps and between-batch idle periods
    # that would otherwise drag down all averages toward zero.
    im_list = [
        x for x in im_list
        if (x.get("requests_running") or 0) > 0
        and (x.get("output_throughput_tok_per_sec") or 0) >= min_throughput
    ]

    throughputs = [x["output_throughput_tok_per_sec"] for x in im_list
                   if isinstance(x.get("output_throughput_tok_per_sec"), (int, float))]
    active = im_list  # already filtered to active records above
    return {
        "throughput_median":     percentile(throughputs, 50),
        "throughput_p25":        percentile(throughputs, 25),
        "throughput_p75":        percentile(throughputs, 75),
        "throughput_max":        max(throughputs) if throughputs else None,
        "interactivity_mean":    safe_avg(im_list, "interactivity_mean_tok_per_sec_per_user"),
        "interactivity_p50":     safe_avg(im_list, "interactivity_p50_tok_per_sec_per_user"),
        "interactivity_p90":     safe_avg(im_list, "interactivity_p90_tok_per_sec_per_user"),
        "interactivity_p95":     safe_avg(im_list, "interactivity_p95_tok_per_sec_per_user"),
        "ttft_p50":              safe_nested(im_list, "ttft", "p50"),
        "ttft_p90":              safe_nested(im_list, "ttft", "p90"),
        "ttft_mean":             safe_nested(im_list, "ttft", "mean"),
        "tpot_p50":              safe_nested(im_list, "tpot", "p50"),
        "tpot_mean":             safe_nested(im_list, "tpot", "mean"),
        "e2e_p50":               safe_nested(im_list, "e2e_latency", "p50"),
        "e2e_mean":              safe_nested(im_list, "e2e_latency", "mean"),
        "kv_cache_pct":          safe_avg(im_list, "kv_cache_usage_pct"),
        "prefix_hit_rate":       safe_avg(im_list, "prefix_cache_hit_rate_pct"),
        "gpu_util":              safe_avg(im_list, "avg_gpu_utilization_pct"),
        "total_power_w":         safe_avg(im_list, "total_power_draw_w"),
        "tok_per_watt":          safe_avg(im_list, "output_tokens_per_watt"),
        "mj_per_tok":            safe_avg(im_list, "millijoules_per_output_token"),
        "mean_gen_tokens":       safe_nested(active, "request_generation_tokens", "mean"),
        "mean_prompt_tokens":    safe_nested(active, "request_prompt_tokens", "mean"),
        "requests_running_avg":  safe_avg(active, "requests_running"),
        "requests_waiting_avg":  safe_avg(active, "requests_waiting"),
    }


# ── Chart helpers ──────────────────────────────────────────────────────────────

def apply_style():
    plt.rcParams.update(STYLE)


def spine_off(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)


def label_bars(ax, bars, fmt="{:.0f}", color="#2a2a3a", fontsize=9, offset=3):
    for bar in bars:
        h = bar.get_height()
        if h and h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + offset,
                fmt.format(h), ha="center", va="bottom",
                fontsize=fontsize, color=color, fontweight="bold",
            )


# ── Figure 1: H200 B=128 Dashboard ────────────────────────────────────────────

def make_h200_dashboard(summaries_h200: dict, out_dir: Path):
    """
    Single-batch (B=128) dashboard comparing Run 1 vs Run 2 on 8× H200.
    Shows all key metrics side by side for the two runs.
    """
    runs = [k for k in ["run1", "run2"] if k in summaries_h200]
    labels = {"run1": "Run 1", "run2": "Run 2"}
    colors = {"run1": RUN_COLORS["h200_run1"], "run2": RUN_COLORS["h200_run2"]}

    x = np.arange(len(runs))
    w = 0.5

    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor("#f7f6f2")
    fig.suptitle(
        "Qwen3-235B-A22B-FP8 · 8× NVIDIA H200 · vLLM  ·  Batch Size B=128",
        fontsize=16, fontweight="bold", color="#1a1a28", y=0.98,
    )
    fig.text(
        0.5, 0.955,
        "Two independent runs — interactivity, throughput, latency, power, cache",
        ha="center", fontsize=10, color="#555566",
    )

    gs = fig.add_gridspec(3, 4, hspace=0.52, wspace=0.38,
                          left=0.06, right=0.97, top=0.93, bottom=0.05)

    run_colors_list = [colors[r] for r in runs]
    run_labels_list = [labels[r] for r in runs]

    # ── Row 0: Summary table | Throughput | Interactivity bands ──────────────
    ax_summary = fig.add_subplot(gs[0, :2])
    ax_tput    = fig.add_subplot(gs[0, 2])
    ax_inter   = fig.add_subplot(gs[0, 3])

    # Summary table panel
    ax_summary.set_facecolor("#f7f6f2")
    spine_off(ax_summary)
    ax_summary.set_xticks([]); ax_summary.set_yticks([])

    col_w = 12
    header = f"{'Metric':<32s}" + "".join(f"{'Run '+r[-1]:>{col_w}}" for r in runs)
    divider = "─" * (32 + col_w * len(runs))
    rows_def = [
        ("Throughput median (tok/s)",   "throughput_median",     "{:.0f}"),
        ("Throughput max (tok/s)",       "throughput_max",        "{:.0f}"),
        ("Interactivity mean (tok/s/u)", "interactivity_mean",    "{:.1f}"),
        ("Interactivity p50 (tok/s/u)",  "interactivity_p50",     "{:.1f}"),
        ("Interactivity p90 (tok/s/u)",  "interactivity_p90",     "{:.1f}"),
        ("TTFT p50 (s)",                 "ttft_p50",              "{:.2f}"),
        ("TTFT p90 (s)",                 "ttft_p90",              "{:.2f}"),
        ("E2E latency p50 (s)",          "e2e_p50",               "{:.0f}"),
        ("KV cache usage (%)",           "kv_cache_pct",          "{:.1f}"),
        ("Prefix hit rate (%)",          "prefix_hit_rate",       "{:.1f}"),
        ("GPU utilisation (%)",          "gpu_util",              "{:.1f}"),
        ("Total power (W)",              "total_power_w",         "{:.0f}"),
        ("tok/s per Watt",               "tok_per_watt",          "{:.4f}"),
        ("Req running (avg)",            "requests_running_avg",  "{:.1f}"),
        ("Mean gen tokens",              "mean_gen_tokens",       "{:.0f}"),
    ]
    table_lines = [header, divider]
    for label, key, fmt in rows_def:
        vals = []
        for r in runs:
            v = summaries_h200[r].get(key)
            vals.append(fmt.format(v) if v is not None else "—")
        table_lines.append(f"{label:<32s}" + "".join(f"{v:>{col_w}}" for v in vals))

    ax_summary.text(
        0.03, 0.95, "\n".join(table_lines),
        transform=ax_summary.transAxes,
        fontsize=9, va="top", ha="left",
        fontfamily="monospace", color="#2a2a3a",
        bbox=dict(boxstyle="round,pad=0.6", fc="#e8e7e0", ec="#aaa8a0", lw=1.0),
    )
    ax_summary.set_title("B=128 · 8× H200 Performance Summary", fontsize=12, pad=10)

    # Throughput bars
    medians  = [summaries_h200[r]["throughput_median"] or 0 for r in runs]
    p25s     = [summaries_h200[r]["throughput_p25"]    or 0 for r in runs]
    p75s     = [summaries_h200[r]["throughput_p75"]    or 0 for r in runs]
    err_lo   = [m - l for m, l in zip(medians, p25s)]
    err_hi   = [h - m for m, h in zip(medians, p75s)]
    bars = ax_tput.bar(x, medians, w, color=run_colors_list, alpha=0.9,
                       yerr=[err_lo, err_hi],
                       error_kw=dict(ecolor="#00000050", capsize=6, lw=1.5))
    for bar, val in zip(bars, medians):
        ax_tput.text(bar.get_x() + bar.get_width() / 2, val + 20,
                     f"{val:.0f}", ha="center", va="bottom", fontsize=10,
                     color="#2a2a3a", fontweight="bold")
    ax_tput.set_xticks(x); ax_tput.set_xticklabels(run_labels_list)
    ax_tput.set_ylabel("Output Throughput (tok/s)  ↑")
    ax_tput.set_title("Aggregate Throughput\n(median ± IQR)")
    spine_off(ax_tput)

    # Interactivity percentile bands
    means_i = [summaries_h200[r]["interactivity_mean"] or 0 for r in runs]
    p50s_i  = [summaries_h200[r]["interactivity_p50"]  or 0 for r in runs]
    p90s_i  = [summaries_h200[r]["interactivity_p90"]  or 0 for r in runs]
    bw = 0.22
    ax_inter.bar(x - bw, means_i, bw, label="Mean",  color=run_colors_list, alpha=0.95)
    ax_inter.bar(x,      p50s_i,  bw, label="p50",   color=run_colors_list, alpha=0.65)
    ax_inter.bar(x + bw, p90s_i,  bw, label="p90",   color=run_colors_list, alpha=0.35)
    ax_inter.axhline(15, color="#cc3333", lw=1, linestyle="--", alpha=0.7)
    ax_inter.text(len(runs) - 0.5, 16, "15 tok/s — speech min", color="#cc3333", fontsize=8, ha="right")
    ax_inter.axhline(50, color="#228855", lw=1, linestyle="--", alpha=0.7)
    ax_inter.text(len(runs) - 0.5, 51, "50 tok/s — comfortable", color="#228855", fontsize=8, ha="right")
    ax_inter.set_xticks(x); ax_inter.set_xticklabels(run_labels_list)
    ax_inter.set_ylabel("tok/s/user  ↑")
    ax_inter.set_title("Per-User Interactivity\n(Mean / p50 / p90)")
    ax_inter.legend(fontsize=9, loc="lower left")
    spine_off(ax_inter)

    # ── Row 1: TTFT | E2E | GPU util | Power ──────────────────────────────────
    ax_ttft  = fig.add_subplot(gs[1, 0])
    ax_e2e   = fig.add_subplot(gs[1, 1])
    ax_gutil = fig.add_subplot(gs[1, 2])
    ax_power = fig.add_subplot(gs[1, 3])

    # TTFT
    p50s_t = [summaries_h200[r]["ttft_p50"] or 0 for r in runs]
    p90s_t = [summaries_h200[r]["ttft_p90"] or 0 for r in runs]
    b1 = ax_ttft.bar(x - 0.18, p50s_t, 0.35, label="p50", color=run_colors_list, alpha=0.9)
    b2 = ax_ttft.bar(x + 0.18, p90s_t, 0.35, label="p90", color=run_colors_list, alpha=0.45, hatch="//", edgecolor="none")
    label_bars(ax_ttft, b1, fmt="{:.1f}s", offset=0.05)
    label_bars(ax_ttft, b2, fmt="{:.1f}s", offset=0.05)
    ax_ttft.axhline(2.0, color="#b06000", lw=1, linestyle="--", alpha=0.6)
    ax_ttft.text(len(runs) - 0.5, 2.1, "2s threshold", color="#b06000", fontsize=8, ha="right")
    ax_ttft.set_xticks(x); ax_ttft.set_xticklabels(run_labels_list)
    ax_ttft.set_ylabel("TTFT (s)  ↓")
    ax_ttft.set_title("Time to First Token")
    ax_ttft.legend(fontsize=9)
    spine_off(ax_ttft)

    # E2E latency
    e2e_p50  = [summaries_h200[r]["e2e_p50"]  or 0 for r in runs]
    e2e_mean = [summaries_h200[r]["e2e_mean"] or 0 for r in runs]
    b3 = ax_e2e.bar(x - 0.18, e2e_p50,  0.35, label="p50",  color=run_colors_list, alpha=0.9)
    b4 = ax_e2e.bar(x + 0.18, e2e_mean, 0.35, label="mean", color=run_colors_list, alpha=0.45)
    label_bars(ax_e2e, b3, fmt="{:.0f}s", offset=1)
    label_bars(ax_e2e, b4, fmt="{:.0f}s", offset=1)
    ax_e2e.set_xticks(x); ax_e2e.set_xticklabels(run_labels_list)
    ax_e2e.set_ylabel("E2E Latency (s)  ↓")
    ax_e2e.set_title("End-to-End Request Latency\n(p50 and mean)")
    ax_e2e.legend(fontsize=9)
    spine_off(ax_e2e)

    # GPU util
    utils = [summaries_h200[r]["gpu_util"] or 0 for r in runs]
    bars_g = ax_gutil.bar(x, utils, w, color=run_colors_list, alpha=0.9)
    label_bars(ax_gutil, bars_g, fmt="{:.0f}%", offset=0.5)
    ax_gutil.set_ylim(0, 115)
    ax_gutil.axhline(100, color="#00000025", lw=0.8, linestyle="--")
    ax_gutil.set_xticks(x); ax_gutil.set_xticklabels(run_labels_list)
    ax_gutil.set_ylabel("Avg GPU Utilisation (%)")
    ax_gutil.set_title("GPU Utilisation")
    spine_off(ax_gutil)

    # Power
    powers = [summaries_h200[r]["total_power_w"] or 0 for r in runs]
    bars_p = ax_power.bar(x, powers, w, color=run_colors_list, alpha=0.85)
    label_bars(ax_power, bars_p, fmt="{:.0f}W", offset=20)
    ax_power.set_xticks(x); ax_power.set_xticklabels(run_labels_list)
    ax_power.set_ylabel("Total Power Draw (W)")
    ax_power.set_title("Power Draw  (8× H200)")
    # H200 TDP: 8 × 700W = 5600W (same envelope as H100 SXM)
    ax_power.axhline(5600, color="#cc3333", lw=1, linestyle="--", alpha=0.6)
    ax_power.text(0, 5650, "TDP limit (8×700W)", color="#cc3333", fontsize=8)
    spine_off(ax_power)

    # ── Row 2: efficiency | kv cache | prefix hit rate | findings ─────────────
    ax_eff    = fig.add_subplot(gs[2, 0])
    ax_kv     = fig.add_subplot(gs[2, 1])
    ax_prefix = fig.add_subplot(gs[2, 2])
    ax_notes  = fig.add_subplot(gs[2, 3])

    # Efficiency
    tpw = [summaries_h200[r]["tok_per_watt"] or 0 for r in runs]
    bars_e = ax_eff.bar(x, tpw, w, color=run_colors_list, alpha=0.9)
    for bar, val in zip(bars_e, tpw):
        ax_eff.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                    color="#2a2a3a", fontweight="bold")
    ax_eff.set_xticks(x); ax_eff.set_xticklabels(run_labels_list)
    ax_eff.set_ylabel("tok/s / Watt  ↑")
    ax_eff.set_title("Energy Efficiency")
    spine_off(ax_eff)

    # KV cache
    kv = [summaries_h200[r]["kv_cache_pct"] or 0 for r in runs]
    bars_k = ax_kv.bar(x, kv, w, color=run_colors_list, alpha=0.9)
    label_bars(ax_kv, bars_k, fmt="{:.1f}%", offset=0.3)
    ax_kv.set_xticks(x); ax_kv.set_xticklabels(run_labels_list)
    ax_kv.set_ylabel("KV Cache Usage (%)")
    ax_kv.set_title("KV Cache Utilisation")
    spine_off(ax_kv)

    # Prefix hit rate
    phr = [summaries_h200[r]["prefix_hit_rate"] or 0 for r in runs]
    bars_ph = ax_prefix.bar(x, phr, w, color=run_colors_list, alpha=0.85)
    label_bars(ax_prefix, bars_ph, fmt="{:.1f}%", offset=0.5)
    ax_prefix.set_ylim(0, 100)
    ax_prefix.set_xticks(x); ax_prefix.set_xticklabels(run_labels_list)
    ax_prefix.set_ylabel("Prefix Hit Rate (%)")
    ax_prefix.set_title("Prefix Cache Hit Rate\n(multi-turn context reuse)")
    spine_off(ax_prefix)

    # Notes
    ax_notes.set_facecolor("#f7f6f2")
    spine_off(ax_notes)
    ax_notes.set_xticks([]); ax_notes.set_yticks([])

    r1 = summaries_h200.get("run1", {})
    r2 = summaries_h200.get("run2", {})

    notes = ["Key Findings — 8× H200  B=128", ""]
    t1 = r1.get("throughput_median"); t2 = r2.get("throughput_median")
    if t1: notes.append(f"▸ Run 1 throughput: {t1:.0f} tok/s")
    if t2: notes.append(f"▸ Run 2 throughput: {t2:.0f} tok/s")
    if t1 and t2:
        notes.append(f"  Run consistency: {min(t1,t2)/max(t1,t2)*100:.0f}%")
    notes += [""]
    i1 = r1.get("interactivity_p50"); i2 = r2.get("interactivity_p50")
    if i1: notes.append(f"▸ Interactivity p50 R1: {i1:.1f} tok/s/u")
    if i2: notes.append(f"▸ Interactivity p50 R2: {i2:.1f} tok/s/u")
    notes += [""]
    g1 = r1.get("gpu_util"); g2 = r2.get("gpu_util")
    if g1: notes.append(f"▸ GPU util R1: {g1:.0f}%")
    if g2: notes.append(f"▸ GPU util R2: {g2:.0f}%")
    notes += [""]
    p1 = r1.get("prefix_hit_rate"); p2 = r2.get("prefix_hit_rate")
    if p1: notes.append(f"▸ Prefix hit rate R1: {p1:.1f}%")
    if p2: notes.append(f"▸ Prefix hit rate R2: {p2:.1f}%")
    notes += [
        "",
        "▸ H200 vs H100 SXM:",
        "  141 GB HBM3e vs 80 GB HBM2e",
        "  3.35 TB/s vs 3.35 TB/s BW",
        "  B=128 saturates memory →",
        "  high KV cache utilisation",
        "  expected at large batch.",
        "",
        "▸ See h100_vs_h200_comparison.png",
        "  for cross-hardware Pareto.",
    ]

    ax_notes.text(0.05, 0.97, "\n".join(notes),
                  transform=ax_notes.transAxes,
                  fontsize=8.5, color="#2a2a3a",
                  va="top", ha="left", fontfamily="monospace",
                  bbox=dict(boxstyle="round", fc="#efeeea", ec="#aaa8a0", lw=0.8))

    # Legend
    legend_patches = [
        mpatches.Patch(color=colors["run1"], label="B=128 Run 1"),
        mpatches.Patch(color=colors["run2"], label="B=128 Run 2"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               fontsize=10, framealpha=0.4, bbox_to_anchor=(0.5, 0.0))

    out_path = out_dir / "h200_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✓ Saved H200 dashboard → {out_path}")
    plt.close(fig)


# ── Figure 2: H100 vs H200 Cross-Hardware Comparison ──────────────────────────

def make_cross_hardware_comparison(summaries_h100: dict, summaries_h200: dict, out_dir: Path):
    """
    Cross-hardware Pareto + side-by-side bar comparison.
    H100 batch sweep (B=4,8,16,32) vs H200 B=128 (combined runs).
    """
    # Combine H200 runs for a merged summary
    h200_combined = {}
    all_keys = set()
    for r in summaries_h200.values():
        all_keys.update(r.keys())
    for key in all_keys:
        vals = [summaries_h200[r][key] for r in summaries_h200 if summaries_h200[r].get(key) is not None]
        h200_combined[key] = statistics.mean(vals) if vals else None

    apply_style()
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor("#f7f6f2")
    fig.suptitle(
        "Qwen3-235B-A22B-FP8 · 8× H100 (B=4–32) vs 8× H200 (B=128) · vLLM",
        fontsize=16, fontweight="bold", color="#1a1a28", y=0.98,
    )
    fig.text(
        0.5, 0.955,
        "Cross-hardware Pareto analysis — interactivity vs throughput",
        ha="center", fontsize=10, color="#555566",
    )

    gs = fig.add_gridspec(2, 4, hspace=0.52, wspace=0.38,
                          left=0.06, right=0.97, top=0.93, bottom=0.06)

    ax_pareto = fig.add_subplot(gs[:, :2])   # Pareto spans both rows, left half
    ax_tput   = fig.add_subplot(gs[0, 2])
    ax_inter  = fig.add_subplot(gs[0, 3])
    ax_ttft   = fig.add_subplot(gs[1, 2])
    ax_eff    = fig.add_subplot(gs[1, 3])

    # ── Pareto panel ──────────────────────────────────────────────────────────
    H100_BATCH_SIZES = [4, 8, 16, 32]
    H100_COLORS = {4: "#7eb8f7", 8: "#f7d97e", 16: "#7ef7b8", 32: "#f78e8e"}
    H200_COLOR  = "#c4b5fd"
    H200_R1_COLOR = RUN_COLORS["h200_run1"]
    H200_R2_COLOR = RUN_COLORS["h200_run2"]

    # H100 Pareto frontier (B=8→16→32 — B=4 dominated)
    DOMINATED = {4}
    pareto_xs, pareto_ys, pareto_bs = [], [], []
    dom_xs,   dom_ys,   dom_bs   = [], [], []

    for bs in H100_BATCH_SIZES:
        if bs not in summaries_h100:
            continue
        s = summaries_h100[bs]
        x_val = s.get("interactivity_p50")
        y_val = s.get("throughput_median")
        if not x_val or not y_val:
            continue
        if bs in DOMINATED:
            dom_xs.append(x_val); dom_ys.append(y_val); dom_bs.append(bs)
        else:
            pareto_xs.append(x_val); pareto_ys.append(y_val); pareto_bs.append(bs)

    # Draw H100 Pareto frontier line
    if len(pareto_xs) >= 2:
        order = np.argsort(pareto_xs)
        ax_pareto.plot([pareto_xs[i] for i in order], [pareto_ys[i] for i in order],
                       color="#7070b0", linewidth=2.0, linestyle="--", zorder=1, alpha=0.8,
                       label="H100 Pareto frontier")

    # H100 Pareto points
    for xi, yi, bs in zip(pareto_xs, pareto_ys, pareto_bs):
        ax_pareto.scatter(xi, yi, color=H100_COLORS[bs], s=280, zorder=5,
                          edgecolors="#00000033", linewidths=1.2, marker="o")
        ax_pareto.text(xi + 0.4, yi + 25, f"H100 B={bs}",
                       color=H100_COLORS[bs], fontsize=10, fontweight="bold")

    # H100 dominated B=4
    for xi, yi, bs in zip(dom_xs, dom_ys, dom_bs):
        ax_pareto.scatter(xi, yi, color=H100_COLORS[bs], s=200, zorder=4,
                          edgecolors="#b06000", linewidths=1.5, alpha=0.5, marker="D")
        ax_pareto.text(xi + 0.4, yi + 25, f"H100 B={bs}\n(under-loaded)",
                       color=H100_COLORS[bs], fontsize=8, alpha=0.65)

    # H200 individual runs
    h200_run_styles = [
        ("run1", H200_R1_COLOR, "^", "H200 B=128 Run 1"),
        ("run2", H200_R2_COLOR, "s", "H200 B=128 Run 2"),
    ]
    h200_pts = []
    for run_key, color, marker, run_label in h200_run_styles:
        if run_key not in summaries_h200:
            continue
        s = summaries_h200[run_key]
        xi = s.get("interactivity_p50")
        yi = s.get("throughput_median")
        if xi and yi:
            ax_pareto.scatter(xi, yi, color=color, s=340, zorder=6,
                              edgecolors="#00000055", linewidths=1.5, marker=marker)
            ax_pareto.text(xi + 0.4, yi + 25, run_label,
                           color=color, fontsize=10, fontweight="bold")
            h200_pts.append((xi, yi))

    # Line connecting H200 runs if both present
    if len(h200_pts) == 2:
        ax_pareto.plot([h200_pts[0][0], h200_pts[1][0]],
                       [h200_pts[0][1], h200_pts[1][1]],
                       color=H200_COLOR, linewidth=1.5, linestyle=":", alpha=0.7,
                       label="H200 B=128 runs")

    # H200 combined centroid
    xc = h200_combined.get("interactivity_p50")
    yc = h200_combined.get("throughput_median")
    if xc and yc:
        ax_pareto.scatter(xc, yc, color=H200_COLOR, s=420, zorder=7,
                          edgecolors="#ffffff", linewidths=2.0, marker="*")

    # Reference lines
    ax_pareto.axvline(50, color="#00000020", lw=1, linestyle=":")
    ax_pareto.text(50.2, ax_pareto.get_ylim()[0] + 30 if ax_pareto.get_ylim()[0] else 30,
                   "50 tok/s/user", color="#777788", fontsize=7.5)

    # Annotation: H200 vs H100 best
    if xc and yc and pareto_xs:
        h100_best_tput = max(pareto_ys) if pareto_ys else 0
        h100_best_inter = pareto_xs[np.argmax(pareto_ys)] if pareto_ys else 0
        if h100_best_tput:
            gain = yc / h100_best_tput
            ax_pareto.annotate(
                f"H200 B=128: {gain:.1f}× throughput\nvs H100 B=32",
                xy=(xc, yc), xytext=(xc - 12, yc - 200),
                arrowprops=dict(arrowstyle="->", color=H200_COLOR, lw=1.3),
                fontsize=9, color=H200_COLOR,
                bbox=dict(boxstyle="round,pad=0.3", fc="#fffbf0", ec=H200_COLOR, lw=0.8),
            )

    ax_pareto.set_xlabel("Interactivity  (1/TPOT_p50,  tok/s per user)  ←  more responsive", fontsize=11)
    ax_pareto.set_ylabel("Aggregate Output Throughput  (tok/s)  ↑  more capacity", fontsize=11)
    ax_pareto.set_title("Cross-Hardware Pareto: Interactivity ↔ Throughput\n"
                        "H100 batch sweep vs H200 B=128", fontsize=13, pad=10)
    spine_off(ax_pareto)
    ax_pareto.legend(fontsize=9, loc="upper left")

    # ── Throughput comparison bar ──────────────────────────────────────────────
    hw_labels = [f"H100 B={bs}" for bs in H100_BATCH_SIZES if bs in summaries_h100]
    hw_labels += ["H200 B=128\n(avg)"]
    hw_vals   = [summaries_h100[bs]["throughput_median"] or 0 for bs in H100_BATCH_SIZES if bs in summaries_h100]
    hw_vals   += [h200_combined.get("throughput_median") or 0]
    hw_colors = [H100_COLORS[bs] for bs in H100_BATCH_SIZES if bs in summaries_h100] + [H200_COLOR]

    xb = np.arange(len(hw_labels))
    bars_t = ax_tput.bar(xb, hw_vals, 0.6, color=hw_colors, alpha=0.9)
    for bar, val in zip(bars_t, hw_vals):
        ax_tput.text(bar.get_x() + bar.get_width() / 2, val + 15,
                     f"{val:.0f}", ha="center", va="bottom", fontsize=8.5,
                     color="#2a2a3a", fontweight="bold")
    ax_tput.set_xticks(xb); ax_tput.set_xticklabels(hw_labels, fontsize=8)
    ax_tput.set_ylabel("Throughput (tok/s)  ↑")
    ax_tput.set_title("Throughput Across Hardware")
    spine_off(ax_tput)

    # ── Interactivity p50 bar ─────────────────────────────────────────────────
    inter_vals = [summaries_h100[bs]["interactivity_p50"] or 0 for bs in H100_BATCH_SIZES if bs in summaries_h100]
    inter_vals += [h200_combined.get("interactivity_p50") or 0]
    bars_i = ax_inter.bar(xb, inter_vals, 0.6, color=hw_colors, alpha=0.9)
    for bar, val in zip(bars_i, inter_vals):
        ax_inter.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                      f"{val:.1f}", ha="center", va="bottom", fontsize=8.5,
                      color="#2a2a3a", fontweight="bold")
    ax_inter.axhline(50, color="#228855", lw=1, linestyle="--", alpha=0.7)
    ax_inter.text(len(hw_labels) - 1.5, 51, "50 tok/s comfortable", color="#228855", fontsize=7.5)
    ax_inter.set_xticks(xb); ax_inter.set_xticklabels(hw_labels, fontsize=8)
    ax_inter.set_ylabel("Interactivity p50 (tok/s/user)  ↑")
    ax_inter.set_title("Per-User Interactivity p50")
    spine_off(ax_inter)

    # ── TTFT comparison ───────────────────────────────────────────────────────
    ttft_vals = [summaries_h100[bs]["ttft_p50"] or 0 for bs in H100_BATCH_SIZES if bs in summaries_h100]
    ttft_vals += [h200_combined.get("ttft_p50") or 0]
    bars_tf = ax_ttft.bar(xb, ttft_vals, 0.6, color=hw_colors, alpha=0.9)
    label_bars(ax_ttft, bars_tf, fmt="{:.1f}s", offset=0.05, fontsize=8)
    ax_ttft.axhline(2.0, color="#b06000", lw=1, linestyle="--", alpha=0.6)
    ax_ttft.text(len(hw_labels) - 1.5, 2.1, "2s threshold", color="#b06000", fontsize=7.5)
    ax_ttft.set_xticks(xb); ax_ttft.set_xticklabels(hw_labels, fontsize=8)
    ax_ttft.set_ylabel("TTFT p50 (s)  ↓")
    ax_ttft.set_title("Time to First Token p50")
    spine_off(ax_ttft)

    # ── Efficiency comparison ─────────────────────────────────────────────────
    eff_vals = [summaries_h100[bs]["tok_per_watt"] or 0 for bs in H100_BATCH_SIZES if bs in summaries_h100]
    eff_vals += [h200_combined.get("tok_per_watt") or 0]
    bars_ef = ax_eff.bar(xb, eff_vals, 0.6, color=hw_colors, alpha=0.9)
    for bar, val in zip(bars_ef, eff_vals):
        ax_eff.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5,
                    color="#2a2a3a", fontweight="bold")
    ax_eff.set_xticks(xb); ax_eff.set_xticklabels(hw_labels, fontsize=8)
    ax_eff.set_ylabel("tok/s / Watt  ↑")
    ax_eff.set_title("Energy Efficiency")
    spine_off(ax_eff)

    out_path = out_dir / "h100_vs_h200_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✓ Saved cross-hardware comparison → {out_path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--out",  default="analysis_multiturn/charts", help="Output directory")
    args = parser.parse_args()

    root    = Path(args.root).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_style()

    # ── Load H200 data ────────────────────────────────────────────────────────
    print("Loading H200 metrics…")
    summaries_h200 = {}
    for run_key, rel_path in H200_METRIC_FILES.items():
        full = root / rel_path
        if not full.exists():
            print(f"  [skip] {run_key} — not found: {full}")
            continue
        records = load_jsonl(str(full))
        im = extract_inferencex(records)
        summaries_h200[run_key] = summarise(im)
        print(f"  {run_key}: {len(im)} records loaded")

    if not summaries_h200:
        print("No H200 data found.")
        return

    # ── Load H100 data ────────────────────────────────────────────────────────
    print("\nLoading H100 metrics…")
    summaries_h100 = {}
    for bs, rel_path in H100_METRIC_FILES.items():
        full = root / rel_path
        if not full.exists():
            print(f"  [skip] batch={bs} — not found: {full}")
            continue
        records = load_jsonl(str(full))
        im = extract_inferencex(records)
        summaries_h100[bs] = summarise(im)
        print(f"  batch={bs:>2}: {len(im)} records loaded")

    # ── Print H200 summary ────────────────────────────────────────────────────
    print("\nH200 Summary:")
    rows = [
        ("Throughput median (tok/s)",   "throughput_median",  "{:.0f}"),
        ("Throughput max (tok/s)",       "throughput_max",     "{:.0f}"),
        ("Interactivity mean",           "interactivity_mean", "{:.1f}"),
        ("Interactivity p50",            "interactivity_p50",  "{:.1f}"),
        ("TTFT p50 (s)",                 "ttft_p50",           "{:.2f}"),
        ("GPU utilisation (%)",          "gpu_util",           "{:.1f}"),
        ("Total power (W)",              "total_power_w",      "{:.0f}"),
        ("tok/s per Watt",               "tok_per_watt",       "{:.4f}"),
        ("KV cache usage (%)",           "kv_cache_pct",       "{:.1f}"),
        ("Prefix hit rate (%)",          "prefix_hit_rate",    "{:.1f}"),
    ]
    run_keys = [k for k in ["run1", "run2"] if k in summaries_h200]
    print(f"  {'Metric':<35}  " + "  ".join(f"{'Run '+r[-1]:>8}" for r in run_keys))
    for label, key, fmt in rows:
        vals = []
        for r in run_keys:
            v = summaries_h200[r].get(key)
            vals.append(fmt.format(v) if v is not None else "  —  ")
        print(f"  {label:<35}  " + "  ".join(f"{v:>8}" for v in vals))

    # ── Generate charts ───────────────────────────────────────────────────────
    print()
    make_h200_dashboard(summaries_h200, out_dir)
    if summaries_h100:
        make_cross_hardware_comparison(summaries_h100, summaries_h200, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

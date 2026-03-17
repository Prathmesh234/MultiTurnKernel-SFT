#!/usr/bin/env python3
"""
Summary analysis of Qwen3 multi-turn kernel generation traces.

Produces a high-level report:
  - Overall success / failure rates
  - Per-turn correctness & speedup progression
  - Error categorization across turns
  - Speedup tiers (fast_0 / fast_1 / fast_2) breakdown
  - Per-kernel turn-by-turn outcome table
"""

import json
import sys
import os
from collections import Counter, defaultdict


def load_traces(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def classify_error(error_str: str | None) -> str:
    if not error_str:
        return "none"
    e = error_str.lower()
    if "cuda" in e and ("out of memory" in e or "oom" in e):
        return "CUDA OOM"
    if "illegal memory access" in e or "illegal mem" in e:
        return "Illegal Memory Access"
    if "triton" in e and "compile" in e:
        return "Triton Compilation"
    if "assertion" in e:
        return "Assertion Error"
    if "runtimeerror" in e and "size" in e and "match" in e:
        return "Shape Mismatch"
    if "runtimeerror" in e and "boolean value" in e:
        return "Boolean Ambiguity"
    if "typeerror" in e:
        return "Type Error"
    if "syntaxerror" in e:
        return "Syntax Error"
    if "attributeerror" in e:
        return "Attribute Error"
    if "timeout" in e:
        return "Timeout"
    if "correctness" in e:
        return "Wrong Result"
    return "Other Runtime Error"


def pct(num, den):
    return f"{num/den*100:.1f}%" if den else "N/A"


def bar(value, width=30):
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


# ── Report sections ──────────────────────────────

def section_overview(data):
    total = len(data)
    correct_final = sum(1 for d in data if d["final_result"]["correctness"])
    stop_reasons = Counter(d["stop_reason"] for d in data)
    levels = Counter(d.get("level") or "unknown" for d in data)
    turn_counts = Counter(d["num_turns"] for d in data)

    print("=" * 70)
    print("  OVERVIEW — Qwen3 Multi-Turn Kernel Traces")
    print("=" * 70)
    print(f"  Total problems attempted : {total}")
    print(f"  Final correct kernels    : {correct_final}/{total} ({pct(correct_final, total)})")
    print(f"  Final incorrect kernels  : {total - correct_final}/{total} ({pct(total - correct_final, total)})")
    print()
    print(f"  Stop reasons:")
    for reason, cnt in stop_reasons.most_common():
        print(f"    {reason:30s} : {cnt}")
    print()
    print(f"  Difficulty levels:")
    for level, cnt in sorted(levels.items(), key=lambda x: str(x[0])):
        print(f"    Level {str(level):5s} : {cnt}")
    print()
    print(f"  Turn count distribution:")
    for turns, cnt in sorted(turn_counts.items()):
        print(f"    {turns} turn(s) : {cnt} problems")
    print()


def section_per_turn_progression(data):
    max_turns = max(d["num_turns"] for d in data)

    print("=" * 70)
    print("  PER-TURN PROGRESSION")
    print("=" * 70)

    for turn_num in range(1, max_turns + 1):
        samples_at_turn = []
        for d in data:
            for t in d["turns"]:
                if t["turn"] == turn_num:
                    samples_at_turn.append((d, t))

        if not samples_at_turn:
            continue

        n = len(samples_at_turn)
        correct = sum(1 for _, t in samples_at_turn if t["result"]["correctness"])
        speedups = [t["result"]["speedup"] for _, t in samples_at_turn
                    if t["result"]["correctness"] and t["result"]["speedup"] > 0]
        errors = [classify_error(t["result"].get("error")) for _, t in samples_at_turn
                  if not t["result"]["correctness"]]

        print(f"\n  ── Turn {turn_num} ({n} samples) ──")
        print(f"  Correct    : {correct}/{n} ({pct(correct, n)})  {bar(correct/n if n else 0)}")

        if speedups:
            print(f"  Speedup    : avg={sum(speedups)/len(speedups):.2f}x  "
                  f"min={min(speedups):.2f}x  max={max(speedups):.2f}x  "
                  f"(over {len(speedups)} correct)")
        else:
            print(f"  Speedup    : no correct kernels this turn")

        if errors:
            for err, cnt in Counter(errors).most_common(5):
                print(f"    {err:30s} : {cnt}")

    print(f"\n  ── Turn-over-Turn Recovery ──")
    recovered = 0
    for d in data:
        turns = sorted(d["turns"], key=lambda t: t["turn"])
        prev_correct = False
        for t in turns:
            curr_correct = t["result"]["correctness"]
            if not prev_correct and curr_correct and t["turn"] > 1:
                recovered += 1
                print(f"    {d['name'] or d['sample_key']:40s} : fixed at turn {t['turn']}")
            prev_correct = curr_correct
    if recovered == 0:
        print("    No problems recovered from error in later turns.")
    print()


def section_speedup_tiers(data):
    print("=" * 70)
    print("  SPEEDUP TIERS (fast_0 / fast_1 / fast_2)")
    print("=" * 70)
    print("  fast_0 = correct  |  fast_1 = faster than ref  |  fast_2 = significantly faster")
    print()

    max_turns = max(d["num_turns"] for d in data)
    print(f"  {'Turn':<8}  {'fast_0':>12}  {'fast_1':>12}  {'fast_2':>12}")
    print("  " + "-" * 50)

    for turn_num in range(1, max_turns + 1):
        results = [t["result"] for d in data for t in d["turns"] if t["turn"] == turn_num]
        if not results:
            continue
        n = len(results)
        f0 = sum(1 for r in results if r["fast_0"])
        f1 = sum(1 for r in results if r["fast_1"])
        f2 = sum(1 for r in results if r["fast_2"])
        print(f"  Turn {turn_num:<3}  {f0:>3}/{n} ({pct(f0,n):>5})  "
              f"{f1:>3}/{n} ({pct(f1,n):>5})  {f2:>3}/{n} ({pct(f2,n):>5})")

    print()
    n = len(data)
    f0 = sum(1 for d in data if d["final_result"]["fast_0"])
    f1 = sum(1 for d in data if d["final_result"]["fast_1"])
    f2 = sum(1 for d in data if d["final_result"]["fast_2"])
    print(f"  {'Final':<10}{f0:>3}/{n} ({pct(f0,n):>5})  "
          f"{f1:>3}/{n} ({pct(f1,n):>5})  {f2:>3}/{n} ({pct(f2,n):>5})")
    print()


def section_error_analysis(data):
    print("=" * 70)
    print("  ERROR ANALYSIS")
    print("=" * 70)

    all_errors = []
    for d in data:
        for t in d["turns"]:
            r = t["result"]
            if not r["correctness"] and r.get("error"):
                all_errors.append({
                    "category": classify_error(r["error"]),
                    "kernel": d["name"] or d["sample_key"],
                    "turn": t["turn"],
                })

    if not all_errors:
        print("  No errors found.")
        return

    print(f"\n  Total error occurrences: {len(all_errors)}\n")
    cat_counts = Counter(e["category"] for e in all_errors)
    print(f"  {'Error Category':<30s}  {'Count':>5}  {'Share':>6}")
    print("  " + "-" * 45)
    for cat, cnt in cat_counts.most_common():
        print(f"  {cat:<30s}  {cnt:>5}  {pct(cnt, len(all_errors)):>6}")

    print(f"\n  ── Persistent Errors (same category across ≥2 turns) ──")
    kernel_error_turns = defaultdict(lambda: defaultdict(list))
    for e in all_errors:
        kernel_error_turns[e["kernel"]][e["category"]].append(e["turn"])

    found = False
    for kernel, cats in sorted(kernel_error_turns.items()):
        for cat, turns in cats.items():
            if len(turns) >= 2:
                found = True
                print(f"    {kernel:40s} | {cat:25s} | turns {turns}")
    if not found:
        print("    None found.")
    print()


def section_per_kernel_table(data):
    print("=" * 70)
    print("  PER-KERNEL SUMMARY TABLE")
    print("=" * 70)

    for d in sorted(data, key=lambda x: (not x["final_result"]["correctness"], x["sample_key"])):
        name = d["name"] or d["sample_key"]
        fr = d["final_result"]
        status = "PASS" if fr["correctness"] else "FAIL"
        speedup_str = f"{fr['speedup']:.2f}x" if fr["speedup"] > 0 else "  —  "
        tiers = "/".join(t for t in ["f0","f1","f2"]
                        if fr[f"fast_{t[1]}"]) or "—"

        print(f"\n  {name} [{status}] final_speedup={speedup_str} tiers=[{tiers}] "
              f"turns={d['num_turns']} stop={d['stop_reason']}")

        for t in sorted(d["turns"], key=lambda x: x["turn"]):
            r = t["result"]
            icon = "✓" if r["correctness"] else "✗"
            sp = f"{r['speedup']:.2f}x" if r["speedup"] > 0 else "  —  "
            err = classify_error(r.get("error")) if not r["correctness"] else ""
            print(f"    T{t['turn']}: {icon}  speedup={sp:>8s}  {err}")
    print()


# ── Main ─────────────────────────────────────────

def main():
    trace_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "traces", "reasoning_traces_multiturn.json"
    )

    if not os.path.exists(trace_file):
        print(f"Error: trace file not found: {trace_file}")
        sys.exit(1)

    data = load_traces(trace_file)
    print(f"\nLoaded {len(data)} traces from {trace_file}\n")

    section_overview(data)
    section_per_turn_progression(data)
    section_speedup_tiers(data)
    section_error_analysis(data)
    section_per_kernel_table(data)

    print("=" * 70)
    print("  Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

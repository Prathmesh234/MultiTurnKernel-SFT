#!/usr/bin/env python3
"""
Deep audit: inspect per-turn progression for suspicious patterns.

Checks:
1. Traces where correctness flips without meaningful code changes
2. Traces where speedup jumps dramatically between turns
3. Turns where the kernel got shorter but "more correct" (possible simplification hack)
4. Traces where ALL turns are identical code
5. Traces with final_result correctness=True but final_triton_code is empty/None
6. Cross-check: does final_triton_code match the last turn's triton_code?
"""

import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict

TRACES_FILE = Path(__file__).resolve().parent.parent / "traces" / "reasoning_traces_correct_multiturn.json"


def load_traces():
    with open(TRACES_FILE) as f:
        return json.load(f)


def normalize_code(code: str) -> str:
    """Normalize whitespace for comparison."""
    if not code:
        return ""
    return re.sub(r'\s+', ' ', code.strip())


def code_similarity(a: str, b: str) -> float:
    """Return 0-1 similarity ratio between two code strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize_code(a), normalize_code(b)).ratio()


def check_correctness_flip_no_change(trace: dict) -> list[str]:
    """Flag traces where correctness flips but code barely changed."""
    issues = []
    turns = trace.get("turns", [])
    for i in range(1, len(turns)):
        prev = turns[i-1]
        curr = turns[i]
        prev_correct = prev.get("result", {}).get("correctness", False)
        curr_correct = curr.get("result", {}).get("correctness", False)

        if prev_correct != curr_correct:
            sim = code_similarity(prev.get("triton_code", ""), curr.get("triton_code", ""))
            if sim > 0.95:
                issues.append(
                    f"Turn {prev.get('turn')}->{curr.get('turn')}: correctness flipped "
                    f"({prev_correct}->{curr_correct}) but code similarity={sim:.2f}"
                )
    return issues


def check_speedup_jump(trace: dict) -> list[str]:
    """Flag traces with dramatic speedup jumps between turns."""
    issues = []
    turns = trace.get("turns", [])
    for i in range(1, len(turns)):
        prev_spd = turns[i-1].get("result", {}).get("speedup", 0) or 0
        curr_spd = turns[i].get("result", {}).get("speedup", 0) or 0
        if prev_spd > 0 and curr_spd > 0 and curr_spd / prev_spd > 10:
            issues.append(
                f"Turn {turns[i-1].get('turn')}->{turns[i].get('turn')}: "
                f"speedup jumped {prev_spd:.2f}x -> {curr_spd:.2f}x ({curr_spd/prev_spd:.1f}x increase)"
            )
    return issues


def check_shorter_but_correct(trace: dict) -> list[str]:
    """Flag traces where code got much shorter but became correct."""
    issues = []
    turns = trace.get("turns", [])
    for i in range(1, len(turns)):
        prev = turns[i-1]
        curr = turns[i]
        prev_len = len(prev.get("triton_code", "") or "")
        curr_len = len(curr.get("triton_code", "") or "")
        prev_correct = prev.get("result", {}).get("correctness", False)
        curr_correct = curr.get("result", {}).get("correctness", False)

        if not prev_correct and curr_correct and prev_len > 0 and curr_len > 0:
            reduction = 1 - (curr_len / prev_len)
            if reduction > 0.5:
                issues.append(
                    f"Turn {prev.get('turn')}->{curr.get('turn')}: code shrank {reduction*100:.0f}% "
                    f"({prev_len}->{curr_len} chars) and became correct"
                )
    return issues


def check_all_turns_identical(trace: dict) -> list[str]:
    """Flag traces where all turns produced identical code."""
    turns = trace.get("turns", [])
    if len(turns) < 2:
        return []
    codes = [normalize_code(t.get("triton_code", "") or "") for t in turns]
    if all(c == codes[0] for c in codes) and codes[0]:
        return [f"All {len(turns)} turns produced identical code"]
    return []


def check_empty_final_code(trace: dict) -> list[str]:
    """Flag traces marked correct but with no final code."""
    result = trace.get("final_result", {})
    code = trace.get("final_triton_code") or ""
    if result.get("correctness") and not code.strip():
        return ["Marked correct but final_triton_code is empty"]
    return []


def check_final_code_mismatch(trace: dict) -> list[str]:
    """Check if final_triton_code matches the last turn's code."""
    turns = trace.get("turns", [])
    if not turns:
        return []
    last_turn_code = turns[-1].get("triton_code") or ""
    final_code = trace.get("final_triton_code") or ""
    if last_turn_code and final_code:
        sim = code_similarity(last_turn_code, final_code)
        if sim < 0.9:
            return [f"final_triton_code differs from last turn's code (similarity={sim:.2f})"]
    return []


def check_decreasing_quality(trace: dict) -> list[str]:
    """Flag traces where later turns are worse than earlier ones (possible instability)."""
    turns = trace.get("turns", [])
    if len(turns) < 3:
        return []

    # Check if first turn was correct but later turns were not
    first_correct = turns[0].get("result", {}).get("correctness", False)
    last_correct = turns[-1].get("result", {}).get("correctness", False)
    if first_correct and not last_correct:
        return [f"Turn 1 was correct but final turn {turns[-1].get('turn')} was not"]
    return []


def main():
    data = load_traces()
    print(f"Deep audit of {len(data)} traces from {TRACES_FILE.name}")
    print("=" * 80)

    checks = {
        "correctness_flip_no_change": check_correctness_flip_no_change,
        "speedup_jump": check_speedup_jump,
        "shorter_but_correct": check_shorter_but_correct,
        "all_turns_identical": check_all_turns_identical,
        "empty_final_code": check_empty_final_code,
        "final_code_mismatch": check_final_code_mismatch,
        "decreasing_quality": check_decreasing_quality,
    }

    all_findings = defaultdict(list)
    flagged_traces = {}

    for trace in data:
        key = trace["sample_key"]
        trace_issues = {}

        for check_name, check_fn in checks.items():
            issues = check_fn(trace)
            if issues:
                trace_issues[check_name] = issues
                all_findings[check_name].append((key, issues))

        if trace_issues:
            flagged_traces[key] = trace_issues

    # Summary
    print(f"\n{'CHECK':<35} {'FLAGGED':>8}  {'% OF TOTAL':>10}")
    print("-" * 60)
    for check_name in checks:
        count = len(all_findings.get(check_name, []))
        pct = f"{100*count/len(data):.1f}%" if count > 0 else "-"
        print(f"  {check_name:<33} {count:>8}  {pct:>10}")

    print(f"\n{'='*80}")
    print(f"TOTAL FLAGGED: {len(flagged_traces)} / {len(data)} ({100*len(flagged_traces)/len(data):.1f}%)")
    print(f"{'='*80}")

    # Detailed
    if flagged_traces:
        print(f"\nDETAILED FINDINGS")
        print("=" * 80)
        for check_name in checks:
            findings = all_findings.get(check_name, [])
            if not findings:
                continue
            print(f"\n--- {check_name.upper()} ({len(findings)} traces) ---")
            for key, issues in findings:
                for issue in issues:
                    print(f"  [{key}] {issue}")

    # Save
    report_path = TRACES_FILE.parent.parent / "analysis_multiturn" / "audit_deep_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "total_traces": len(data),
        "flagged_count": len(flagged_traces),
        "summary": {name: len(findings) for name, findings in all_findings.items()},
        "flagged_traces": {
            key: {check: issues for check, issues in checks_found.items()}
            for key, checks_found in flagged_traces.items()
        },
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    main()

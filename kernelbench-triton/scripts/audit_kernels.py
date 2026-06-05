#!/usr/bin/env python3
"""
Audit Triton kernels for reward hacking / suspicious patterns.

Checks:
1. Kernels that call torch.* / F.* / nn.* (should be pure Triton)
2. Kernels that import forbidden modules
3. Empty or trivially short kernels
4. Kernels that just return input unchanged (identity/passthrough)
5. Hardcoded constants / magic numbers in outputs
6. Kernels wrapping torch.compile or torch ops instead of real Triton
7. Kernels that copy the PyTorch reference verbatim
8. Kernels with no @triton.jit decorator (not real Triton kernels)
9. Kernels with suspiciously high speedup (may indicate no-op)
10. Kernels marked correct but with no actual computation
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

TRACES_FILE = Path(__file__).resolve().parent.parent / "traces" / "reasoning_traces_correct_multiturn.json"

def load_traces():
    with open(TRACES_FILE) as f:
        return json.load(f)


def check_torch_calls(code: str) -> list[str]:
    """Find torch.* / F.* / nn.* calls inside @triton.jit kernels."""
    issues = []
    # Find all @triton.jit decorated functions
    jit_pattern = re.compile(r'@triton\.jit.*?(?=\n(?:@|class |def (?!.*triton))|\Z)', re.DOTALL)
    jit_blocks = jit_pattern.findall(code)

    for block in jit_blocks:
        torch_calls = re.findall(r'\btorch\.\w+', block)
        f_calls = re.findall(r'\bF\.\w+', block)
        nn_calls = re.findall(r'\bnn\.\w+', block)
        if torch_calls:
            issues.append(f"torch calls in @triton.jit: {torch_calls}")
        if f_calls:
            issues.append(f"F.* calls in @triton.jit: {f_calls}")
        if nn_calls:
            issues.append(f"nn.* calls in @triton.jit: {nn_calls}")
    return issues


def check_torch_in_wrapper(code: str) -> list[str]:
    """Find torch ops used in the wrapper that do the actual computation (not just tensor setup)."""
    issues = []
    # Look for computation-heavy torch ops in the wrapper function
    # These are fine: torch.empty, torch.zeros, torch.ones (allocation)
    # These are suspicious: torch.matmul, torch.conv, torch.relu, etc.

    suspicious_ops = [
        r'torch\.matmul', r'torch\.mm', r'torch\.bmm',
        r'torch\.conv', r'torch\.linear',
        r'torch\.relu', r'torch\.sigmoid', r'torch\.tanh', r'torch\.softmax',
        r'torch\.layer_norm', r'torch\.batch_norm', r'torch\.group_norm',
        r'torch\.max_pool', r'torch\.avg_pool',
        r'F\.relu', r'F\.sigmoid', r'F\.tanh', r'F\.softmax',
        r'F\.conv', r'F\.linear', r'F\.layer_norm', r'F\.batch_norm',
        r'F\.gelu', r'F\.silu', r'F\.leaky_relu',
        r'torch\.nn\.functional',
        r'torch\.einsum',
        r'torch\.addmm',
    ]

    # Get wrapper function body (after @triton.jit blocks)
    # Look for the forward/call method or triton_kernel_wrapper
    wrapper_match = re.search(r'def (?:forward|call|triton_kernel_wrapper)\b.*?(?=\nclass |\Z)', code, re.DOTALL)
    if wrapper_match:
        wrapper_code = wrapper_match.group()
        for pattern in suspicious_ops:
            matches = re.findall(pattern, wrapper_code)
            if matches:
                issues.append(f"Suspicious torch op in wrapper: {matches}")
    return issues


def check_forbidden_imports(code: str) -> list[str]:
    """Check for forbidden imports."""
    issues = []
    forbidden = ['torchvision', 'transformers', 'scipy', 'sklearn', 'tensorflow', 'jax']
    for mod in forbidden:
        if re.search(rf'\bimport\s+{mod}\b|\bfrom\s+{mod}\b', code):
            issues.append(f"Forbidden import: {mod}")
    return issues


def check_no_triton_jit(code: str) -> list[str]:
    """Check if there's no actual @triton.jit kernel."""
    if not code or not code.strip():
        return ["Empty kernel code"]
    if '@triton.jit' not in code and '@triton.autotune' not in code:
        return ["No @triton.jit or @triton.autotune decorator found"]
    return []


def check_trivially_short(code: str) -> list[str]:
    """Flag suspiciously short kernels."""
    if not code:
        return ["No code"]
    lines = [l for l in code.strip().split('\n') if l.strip() and not l.strip().startswith('#')]
    if len(lines) < 5:
        return [f"Only {len(lines)} non-empty, non-comment lines"]
    return []


def check_identity_kernel(code: str) -> list[str]:
    """Check if kernel just copies input to output (identity/passthrough)."""
    issues = []
    jit_pattern = re.compile(r'@triton\.jit\s+def\s+\w+\(.*?\):\s*(.*?)(?=\n(?:@|class |def )|\Z)', re.DOTALL)
    for match in jit_pattern.finditer(code):
        body = match.group(1)
        body_lines = [l.strip() for l in body.split('\n') if l.strip() and not l.strip().startswith('#')]
        # If kernel body is just load + store with no computation
        has_computation = False
        for line in body_lines:
            # Skip loads, stores, masks, offsets
            if any(kw in line for kw in ['tl.load', 'tl.store', 'tl.arange', 'tl.program_id',
                                          'mask', 'offset', 'BLOCK', 'stride', 'n_elements']):
                continue
            if '=' in line and not line.startswith('tl.store'):
                has_computation = True
                break
        if not has_computation and len(body_lines) > 0:
            issues.append("Kernel may be identity (load + store only, no computation)")
    return issues


def check_hardcoded_outputs(code: str) -> list[str]:
    """Check for hardcoded constant outputs."""
    issues = []
    # Look for patterns like: output[:] = 0, output.fill_(x), tl.store(... , 0.0)
    if re.search(r'tl\.store\([^,]+,\s*(0\.0|1\.0|0|1|-1)\s*[,)]', code):
        issues.append("Possible hardcoded constant in tl.store")
    if re.search(r'\.fill_\s*\(', code):
        issues.append("Uses .fill_() — possible hardcoded output")
    return issues


def check_torch_compile_wrapper(code: str) -> list[str]:
    """Check if the 'Triton kernel' is actually just torch.compile wrapping PyTorch."""
    issues = []
    if 'torch.compile' in code:
        issues.append("Uses torch.compile — may not be a real Triton kernel")
    if '@torch.jit.script' in code:
        issues.append("Uses torch.jit.script instead of Triton")
    return issues


def check_reference_copy(triton_code: str, pytorch_code: str) -> list[str]:
    """Check if the Triton code just embeds the PyTorch reference."""
    issues = []
    if not triton_code or not pytorch_code:
        return issues

    # Normalize whitespace for comparison
    def normalize(s):
        return re.sub(r'\s+', ' ', s.strip())

    norm_triton = normalize(triton_code)
    norm_pytorch = normalize(pytorch_code)

    # Check if the PyTorch forward method is embedded verbatim
    # Extract forward body from pytorch code
    fwd_match = re.search(r'def forward\(self.*?\):(.*?)(?=\ndef |\nclass |\Z)', pytorch_code, re.DOTALL)
    if fwd_match:
        fwd_body = normalize(fwd_match.group(1))
        if len(fwd_body) > 50 and fwd_body in norm_triton:
            issues.append("PyTorch forward() body appears verbatim in Triton code")

    return issues


def check_suspicious_speedup(trace: dict) -> list[str]:
    """Flag suspiciously high speedups that might indicate a no-op."""
    issues = []
    result = trace.get("final_result", {})
    speedup = result.get("speedup", 0)
    if speedup > 100:
        issues.append(f"Extremely high speedup ({speedup:.1f}x) — possible no-op kernel")
    return issues


def check_no_triton_ops(code: str) -> list[str]:
    """Check if @triton.jit kernel uses no Triton-specific operations."""
    issues = []
    jit_blocks = re.findall(r'@triton\.jit.*?(?=\n(?:@triton|class |def (?!.*\())|\Z)', code, re.DOTALL)
    for block in jit_blocks:
        triton_ops = re.findall(r'tl\.\w+', block)
        if not triton_ops:
            issues.append("@triton.jit kernel uses no tl.* operations")
    return issues


def main():
    data = load_traces()
    print(f"Auditing {len(data)} traces from {TRACES_FILE.name}")
    print("=" * 80)

    checks = {
        "torch_in_jit": check_torch_calls,
        "torch_in_wrapper": check_torch_in_wrapper,
        "forbidden_imports": check_forbidden_imports,
        "no_triton_jit": check_no_triton_jit,
        "trivially_short": check_trivially_short,
        "identity_kernel": check_identity_kernel,
        "hardcoded_output": check_hardcoded_outputs,
        "torch_compile": check_torch_compile_wrapper,
        "no_triton_ops": check_no_triton_ops,
    }

    # Track findings
    all_findings = defaultdict(list)  # check_name -> [(sample_key, issues)]
    flagged_traces = {}  # sample_key -> {check: issues}

    for trace in data:
        key = trace["sample_key"]
        code = trace.get("final_triton_code") or ""
        pytorch_code = trace.get("pytorch_code", "")
        trace_issues = {}

        # Run code-based checks
        for check_name, check_fn in checks.items():
            issues = check_fn(code)
            if issues:
                trace_issues[check_name] = issues
                all_findings[check_name].append((key, issues))

        # Run trace-level checks
        ref_issues = check_reference_copy(code, pytorch_code)
        if ref_issues:
            trace_issues["reference_copy"] = ref_issues
            all_findings["reference_copy"].append((key, ref_issues))

        speedup_issues = check_suspicious_speedup(trace)
        if speedup_issues:
            trace_issues["suspicious_speedup"] = speedup_issues
            all_findings["suspicious_speedup"].append((key, speedup_issues))

        if trace_issues:
            flagged_traces[key] = trace_issues

    # Print summary
    print(f"\n{'CHECK':<25} {'FLAGGED':>8}  {'% OF TOTAL':>10}")
    print("-" * 50)

    all_check_names = list(checks.keys()) + ["reference_copy", "suspicious_speedup"]
    for check_name in all_check_names:
        count = len(all_findings.get(check_name, []))
        pct = f"{100*count/len(data):.1f}%" if count > 0 else "-"
        print(f"  {check_name:<23} {count:>8}  {pct:>10}")

    print(f"\n{'='*80}")
    print(f"TOTAL FLAGGED TRACES: {len(flagged_traces)} / {len(data)} ({100*len(flagged_traces)/len(data):.1f}%)")
    print(f"CLEAN TRACES: {len(data) - len(flagged_traces)} / {len(data)} ({100*(len(data)-len(flagged_traces))/len(data):.1f}%)")
    print(f"{'='*80}")

    # Print detailed findings
    if flagged_traces:
        print(f"\n{'='*80}")
        print("DETAILED FINDINGS")
        print(f"{'='*80}")

        # Group by severity — most concerning first
        severity_order = [
            "torch_in_jit", "torch_compile", "reference_copy", "forbidden_imports",
            "no_triton_jit", "no_triton_ops", "suspicious_speedup",
            "torch_in_wrapper", "identity_kernel", "hardcoded_output", "trivially_short",
        ]

        for check_name in severity_order:
            findings = all_findings.get(check_name, [])
            if not findings:
                continue
            print(f"\n--- {check_name.upper()} ({len(findings)} traces) ---")
            for key, issues in findings:
                for issue in issues:
                    print(f"  [{key}] {issue}")

    # Save full report to JSON
    report_path = TRACES_FILE.parent.parent / "analysis_multiturn" / "audit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "total_traces": len(data),
        "flagged_count": len(flagged_traces),
        "clean_count": len(data) - len(flagged_traces),
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

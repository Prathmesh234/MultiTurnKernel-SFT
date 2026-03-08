"""
Simple utilities for extracting input info from PyTorch code.
"""

import json
import tempfile
import importlib.util
import sys
import os
from pathlib import Path


def extract_inputs(pytorch_code: str):
    """
    Execute get_inputs() from PyTorch code and return the tensor list.
    
    Returns:
        List of tensors from get_inputs()
    """
    import torch
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(pytorch_code)
        temp_file = f.name
    
    try:
        spec = importlib.util.spec_from_file_location("temp_module", temp_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["temp_module"] = module
        spec.loader.exec_module(module)
        
        if hasattr(module, "get_inputs"):
            inputs = module.get_inputs()
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            return inputs
        else:
            raise ValueError("PyTorch code must define 'get_inputs' function")
    finally:
        try:
            os.unlink(temp_file)
        except:
            pass
        if "temp_module" in sys.modules:
            del sys.modules["temp_module"]


FORBIDDEN_IMPORTS = {
    "torchvision", "torchaudio", "torchtext", "transformers",
    "scipy", "sklearn", "cv2", "PIL", "pandas",
}

FORBIDDEN_IN_KERNEL = [
    "torch.relu", "torch.sigmoid", "torch.tanh", "torch.softmax",
    "torch.sum", "torch.mean", "torch.matmul", "torch.mm",
    "torch.bmm", "torch.nn.", "F.relu", "F.sigmoid",
]


def pre_validate_triton_code(triton_code: str) -> str | None:
    """
    Quick local checks on generated Triton code before sending to Modal.

    Returns None if the code looks okay, or an error string describing
    what's wrong. This saves a Modal call for obviously broken code.
    """
    # 1. Must define triton_kernel_wrapper
    if "def triton_kernel_wrapper" not in triton_code:
        return "Missing required function 'triton_kernel_wrapper'"

    # 2. Must have at least one @triton.jit kernel
    if "@triton.jit" not in triton_code:
        return "No @triton.jit kernel found"

    # 3. Check for imports that won't exist on Modal
    for lib in FORBIDDEN_IMPORTS:
        if f"import {lib}" in triton_code or f"from {lib}" in triton_code:
            return f"Forbidden import: '{lib}' is not available on the benchmark server"

    # 4. Check for torch ops used inside @triton.jit kernels
    #    Split on @triton.jit to isolate kernel bodies from the wrapper.
    #    After @triton.jit, the first "def " is the kernel itself.
    #    The next unindented "def " marks the end of the kernel.
    parts = triton_code.split("@triton.jit")
    for kernel_part in parts[1:]:  # skip everything before first @triton.jit
        lines = kernel_part.split("\n")
        kernel_body = []
        found_kernel_def = False
        for line in lines:
            if not found_kernel_def:
                if line.startswith("def "):
                    found_kernel_def = True
                kernel_body.append(line)
            else:
                # Next top-level def = end of kernel
                if line.startswith("def "):
                    break
                kernel_body.append(line)
        kernel_text = "\n".join(kernel_body)

        for forbidden in FORBIDDEN_IN_KERNEL:
            if forbidden in kernel_text:
                return (f"Found '{forbidden}' inside @triton.jit kernel. "
                        f"Use tl.* operations only inside kernels.")

    # 5. Basic syntax check
    try:
        compile(triton_code, "<triton_code>", "exec")
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"

    return None


def load_solved_keys(*trace_files: str) -> set[str]:
    """
    Load sample_keys that have already been solved from one or more trace JSON
    files (completed and/or failed multi-turn output files).

    Skips files that don't exist or are unreadable without raising.

    Usage:
        solved = load_solved_keys(OUTPUT_FILE_MULTITURN, OUTPUT_FILE_MULTITURN_FAILED)
        if sample_key in solved:
            continue  # skip — already done
    """
    solved: set[str] = set()
    for path in trace_files:
        p = Path(path)
        if not p.exists():
            continue
        try:
            with open(p) as f:
                traces = json.load(f)
            keys = {t["sample_key"] for t in traces if t.get("sample_key")}
            solved |= keys
            print(f"  [resume] {len(keys)} solved keys loaded from {p.name}")
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [resume] Warning: could not read {p}: {e}")
    return solved


def get_shapes(pytorch_code: str):
    """
    Get shapes from get_inputs().
    
    Returns:
        List of shapes (tuples)
    """
    inputs = extract_inputs(pytorch_code)
    return [tuple(t.shape) for t in inputs if hasattr(t, 'shape')]

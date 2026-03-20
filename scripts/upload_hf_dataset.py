#!/usr/bin/env python3
"""
Upload reasoning_traces_multiturn_qwen.json to HuggingFace as a dataset.
Usage:
    uv run --with datasets --with huggingface_hub scripts/upload_hf_dataset.py
"""

import json
from pathlib import Path
from datasets import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACES_FILE = PROJECT_ROOT / "traces" / "reasoning_traces_multiturn_qwen.json"
HF_REPO = "ppbhatt500/kernelbook-triton-multiturn-reasoning-traces"

EXCLUDE_COLS = {"level", "name", "problem_id", "timestamp"}

def main():
    print(f"Loading {TRACES_FILE} ...")
    with open(TRACES_FILE) as f:
        raw = json.load(f)

    print(f"Loaded {len(raw)} traces")

    records = []
    for entry in raw:
        record = {k: v for k, v in entry.items() if k not in EXCLUDE_COLS}
        # Serialize nested fields to JSON strings so HF handles them cleanly
        for key in ("turns", "full_messages"):
            if key in record and not isinstance(record[key], str):
                record[key] = json.dumps(record[key])
        records.append(record)

    ds = Dataset.from_list(records)
    print(ds)

    print(f"\nPushing to {HF_REPO} ...")
    ds.push_to_hub(HF_REPO, private=False)
    print("Done!")

if __name__ == "__main__":
    main()

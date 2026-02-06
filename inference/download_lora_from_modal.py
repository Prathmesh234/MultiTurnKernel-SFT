"""
Download the LoRA checkpoint from Modal volume to local disk.

Prerequisites:
    pip install modal
    modal setup  # authenticate with Modal

Usage:
    python download_lora_from_modal.py
"""

import argparse
import os
import modal

VOLUME_NAME = "arcee-vol"
# Based on AGENTS.md and README.md findings
LORA_PATH_IN_VOLUME = "models/trinity-triton-sft/checkpoint-40"
DEFAULT_OUTPUT_DIR = "./sft-reasoning/checkpoint-40"


def download_lora(volume_name: str, lora_path: str, output_dir: str):
    """Download LoRA files from a Modal volume to local disk."""
    print(f"Connecting to Modal volume: {volume_name}")
    try:
        vol = modal.Volume.from_name(volume_name)
    except Exception as e:
        print(f"ERROR: Could not connect to volume '{volume_name}': {e}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Listing files in {lora_path}/ ...")
    try:
        entries = list(vol.listdir(lora_path, recursive=True))
    except Exception as e:
        print(f"ERROR: Could not list files in '{lora_path}': {e}")
        return

    if not entries:
        print(f"ERROR: No files found at '{lora_path}' in volume '{volume_name}'.")
        return

    print(f"Found {len(entries)} entries. Downloading...")

    for entry in entries:
        remote_path = entry.path
        rel_path = os.path.relpath(remote_path, lora_path)
        local_path = os.path.join(output_dir, rel_path)

        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            os.makedirs(local_path, exist_ok=True)
            continue

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"  Downloading: {rel_path}")
        try:
            with open(local_path, "wb") as f:
                for chunk in vol.read_file(remote_path):
                    f.write(chunk)
        except Exception as e:
            print(f"  FAILED to download {rel_path}: {e}")

    print(f"\nDone! LoRA checkpoint downloaded to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LoRA checkpoint from Modal volume")
    parser.add_argument(
        "--volume-name", default=VOLUME_NAME,
        help=f"Modal volume name (default: {VOLUME_NAME})",
    )
    parser.add_argument(
        "--lora-path", default=LORA_PATH_IN_VOLUME,
        help=f"Path inside volume (default: {LORA_PATH_IN_VOLUME})",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Local output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    download_lora(args.volume_name, args.lora_path, args.output_dir)

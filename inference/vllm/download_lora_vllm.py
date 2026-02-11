#!/usr/bin/env python3
"""
Download Trinity-Mini LoRA adapter from Modal volume.

This downloads the vLLM-compatible adapter that only targets attention layers:
- q_proj, k_proj, v_proj, o_proj (NOT MoE expert layers)

From: arcee-vol/models/trinity-triton-sft-vllm
To: ./sft-reasoning/trinity-triton-sft-vllm/
"""

import os
from pathlib import Path
import modal
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Configuration from environment variables with defaults
MODAL_VOLUME_NAME = os.getenv("MODAL_VOLUME_NAME", "arcee-vol")
REMOTE_ADAPTER_PATH = os.getenv("TRINITY_ADAPTER_PATH_REMOTE", "models/trinity-triton-sft-vllm")
LOCAL_ADAPTER_PATH = os.getenv("TRINITY_ADAPTER_PATH_LOCAL", "./sft-reasoning/trinity-triton-sft-vllm")

# Connect to Modal volume
volume = modal.Volume.from_name(MODAL_VOLUME_NAME)

# Create Modal app for downloading
app = modal.App("download-lora-vllm")

@app.function(volumes={"/vol": volume})
def download_checkpoint():
    """Download the vLLM-compatible LoRA checkpoint from Modal volume."""
    import os
    import shutil
    from pathlib import Path

    source_path = os.path.join("/vol", REMOTE_ADAPTER_PATH)
    local_dest = LOCAL_ADAPTER_PATH

    print(f"Looking for checkpoint at: {source_path}")

    # Check if source exists
    if not os.path.exists(source_path):
        print(f"ERROR: Checkpoint not found at {source_path}")
        print("Available paths in /vol/models:")
        if os.path.exists("/vol/models"):
            for item in os.listdir("/vol/models"):
                print(f"  - {item}")
        return {"error": "Checkpoint not found"}

    # List files in checkpoint
    print(f"\nFiles in checkpoint:")
    for root, dirs, files in os.walk(source_path):
        level = root.replace(source_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')

    # Prioritize final_model if it exists
    if os.path.exists(os.path.join(source_path, "final_model")):
        final_checkpoint = "final_model"
        full_source_path = os.path.join(source_path, final_checkpoint)
        print(f"\nUsing: {final_checkpoint}")
    else:
        # Fallback to highest numbered checkpoint
        checkpoints = [d for d in os.listdir(source_path) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoint_nums = [int(c.split("-")[1]) for c in checkpoints]
            final_checkpoint = f"checkpoint-{max(checkpoint_nums)}"
            full_source_path = os.path.join(source_path, final_checkpoint)
            print(f"\nUsing highest checkpoint: {final_checkpoint}")
        else:
            full_source_path = source_path
            final_checkpoint = "base"
            print(f"\nUsing base directory")

    return {
        "source": full_source_path,
        "checkpoint_name": final_checkpoint,
        "files": [f for f in os.listdir(full_source_path) if os.path.isfile(os.path.join(full_source_path, f))] if os.path.exists(full_source_path) else []
    }

@app.local_entrypoint()
def main():
    """Main entry point to download the checkpoint."""
    import os
    import shutil
    from pathlib import Path

    print("=" * 70)
    print("Downloading Trinity-Mini vLLM LoRA Adapter from Modal")
    print("=" * 70)

    # First, check what's available
    info = download_checkpoint.remote()

    if "error" in info:
        print(f"\nERROR: {info['error']}")
        return

    print(f"\nCheckpoint info:")
    print(f"  Source: {info['source']}")
    print(f"  Checkpoint: {info['checkpoint_name']}")
    print(f"  Files: {', '.join(info['files'][:5])}{'...' if len(info['files']) > 5 else ''}")

    # Now download using Modal's volume.read_file
    local_dest = os.path.join(LOCAL_ADAPTER_PATH, info['checkpoint_name'])
    Path(local_dest).mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading to: {local_dest}")

    # Download each file
    with volume.batch_upload() as batch:
        for filename in info['files']:
            remote_path = os.path.join("/", REMOTE_ADAPTER_PATH, info['checkpoint_name'], filename)
            local_path = os.path.join(local_dest, filename)

            print(f"  Downloading {filename}...", end=" ", flush=True)
            try:
                # Read from volume
                content = volume.read_file(remote_path)
                # Write locally
                with open(local_path, 'wb') as f:
                    f.write(content)
                print("done")
            except Exception as e:
                print(f"FAILED ({e})")

    print(f"\nDownload complete!")
    print(f"Checkpoint saved to: {local_dest}")
    print("\nNext step: Run 'bash inference/vllm/serve_trinity_uv.sh' to start the vLLM server")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Dump the exact conversation a model sees for a single kernel journey.

Usage:
  python kernel_journey.py                        # first correct kernel
  python kernel_journey.py --kernel "9_Matmul"   # filter by name substring
  python kernel_journey.py --output my.txt        # custom output file
"""

import json
import os
import sys
import argparse


def load_traces(path):
    with open(path) as f:
        return json.load(f)


def write_journey(d, out):
    sep = "=" * 80
    fr = d["final_result"]

    out.write(f"{sep}\n")
    out.write(f"KERNEL: {d.get('name') or d['sample_key']}\n")
    out.write(f"Turns: {d['num_turns']}  |  Final: {'CORRECT' if fr['correctness'] else 'FAILED'}  |  Speedup: {fr['speedup']:.4f}x\n")
    out.write(f"{sep}\n\n")

    for msg in d.get("full_messages", []):
        out.write(f"{'─' * 80}\n")
        out.write(msg["content"].strip() + "\n\n")

    # full_messages stops before the final assistant response — append it
    last_turn = sorted(d["turns"], key=lambda t: t["turn"])[-1]
    final_response = last_turn.get("full_completion") or last_turn.get("triton_code") or ""
    if final_response:
        out.write(f"{'─' * 80}\n")
        out.write(final_response.strip() + "\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default=None)
    parser.add_argument("--kernel", default=None, help="Filter by name substring")
    parser.add_argument("--output", default="journey.txt")
    parser.add_argument("--all-failed", action="store_true",
                        help="Pick a kernel where every turn failed")
    args = parser.parse_args()

    trace_file = args.trace or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "traces", "reasoning_traces_multiturn.json"
    )

    if not os.path.exists(trace_file):
        print(f"Trace file not found: {trace_file}")
        sys.exit(1)

    data = load_traces(trace_file)

    if args.kernel:
        substr = args.kernel.lower()
        data = [d for d in data
                if substr in (d.get("name") or "").lower()
                or substr in d["sample_key"].lower()]
    elif args.all_failed:
        # All turns failed, hit max turns
        data = [d for d in data
                if all(not t["result"]["correctness"] for t in d["turns"])]
    else:
        # Default: failed turn 1 but became correct on a later turn
        def failed_then_correct(d):
            turns = sorted(d["turns"], key=lambda t: t["turn"])
            return (
                len(turns) >= 2
                and not turns[0]["result"]["correctness"]
                and any(t["result"]["correctness"] for t in turns[1:])
            )
        data = [d for d in data if failed_then_correct(d)]

    if not data:
        print("No matching kernels found.")
        sys.exit(0)

    kernel = data[0]

    with open(args.output, "w") as f:
        write_journey(kernel, f)

    print(f"Written journey for '{kernel.get('name') or kernel['sample_key']}' to {args.output}")


if __name__ == "__main__":
    main()

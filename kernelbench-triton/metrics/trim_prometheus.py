"""
trim_prometheus.py
==================
Produces a compact, git-friendly summary of prometheus.json.

What gets stripped per scrape:
  - raw_text          (59 KB/scrape — the full Prometheus text dump)
  - _bucket lines     (histogram bucket arrays — huge, rarely needed)
  - _created gauges   (redundant creation-time markers)

What survives:
  - timestamp / unix_ts / poll_index
  - Every vllm:* metric's _sum, _count, and plain gauge values
  - All non-histogram vllm metrics as-is
  - python_* and process_* metrics (tiny, useful for process health)

Typical size reduction: ~107 KB → ~4 KB per scrape  (~96% smaller)

Usage:
    python metrics/trim_prometheus.py                          # default paths
    python metrics/trim_prometheus.py --last 20               # keep last 20 scrapes only
    python metrics/trim_prometheus.py --input metrics/prometheus.json \\
                                      --output metrics/prometheus_compact.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_INPUT  = "metrics/prometheus.json"
DEFAULT_OUTPUT = "metrics/prometheus_compact.json"

# Label keys we keep from parsed metric dicts (anything else is a bucket line or _created)
_SKIP_SUFFIXES = (
    "_bucket",   # histogram buckets — too many rows, not needed for summary
    "_created",  # redundant float timestamp injected by prometheus client
)

# Gauge/counter values we always want to preserve even if unlabelled
_ALWAYS_KEEP_PREFIXES = (
    "vllm:",
    "process_",
    "python_",
)


def _is_bucket_or_created(label_key: str) -> bool:
    """True if this label_key string is a histogram bucket or _created entry."""
    # bucket entries look like: 'engine="0",le="0.04",model_name="..."'
    if 'le="' in label_key:
        return True
    return False


def compact_parsed(parsed: dict) -> dict:
    """
    Strip bucket entries and _created metrics from a parsed Prometheus dict.
    Keep: scalar gauges, counters, histogram _sum / _count.
    """
    compacted = {}

    for metric_name, label_dict in parsed.items():
        # Drop metrics whose name ends with _bucket or _created
        lower = metric_name.lower()
        if any(lower.endswith(s) for s in _SKIP_SUFFIXES):
            continue

        # Only keep metrics we care about
        if not any(metric_name.startswith(p) for p in _ALWAYS_KEEP_PREFIXES):
            # Keep http_* too (small)
            if not metric_name.startswith("http_"):
                continue

        if not isinstance(label_dict, dict):
            compacted[metric_name] = label_dict
            continue

        # Strip per-bucket rows (le="..." labels)
        clean = {}
        for k, v in label_dict.items():
            if k in ("__help__", "__type__"):
                continue  # drop metadata keys — already in README
            if _is_bucket_or_created(k):
                continue
            clean[k] = v

        if clean:
            # If only one value and it's keyed "__value__", unwrap scalar
            if list(clean.keys()) == ["__value__"]:
                compacted[metric_name] = clean["__value__"]
            else:
                compacted[metric_name] = clean

    return compacted


def compact_scrape(scrape: dict) -> dict:
    """Turn one full scrape record into a compact record."""
    return {
        "timestamp":  scrape.get("timestamp"),
        "unix_ts":    scrape.get("unix_ts"),
        "poll_index": scrape.get("poll_index"),
        "metrics":    compact_parsed(scrape.get("parsed", {})),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Trim prometheus.json to a compact, git-friendly format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",  default=DEFAULT_INPUT,
                        help=f"Source file (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Compact output file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--last",   type=int, default=0,
                        help="Keep only the last N scrapes (0 = keep all)")
    parser.add_argument("--inplace", action="store_true",
                        help="Overwrite --input with the compacted version")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output) if not args.inplace else src

    if not src.exists():
        print(f"ERROR: {src} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {src} …", end=" ", flush=True)
    with open(src) as f:
        scrapes = json.load(f)
    print(f"{len(scrapes)} scrapes loaded")

    original_size = src.stat().st_size

    if args.last > 0 and len(scrapes) > args.last:
        print(f"Keeping last {args.last} scrapes (dropping {len(scrapes) - args.last} older ones)")
        scrapes = scrapes[-args.last:]

    print("Compacting …", end=" ", flush=True)
    compact = [compact_scrape(s) for s in scrapes]
    print("done")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(compact, f, indent=2, default=str)

    compact_size = dst.stat().st_size
    reduction = (1 - compact_size / original_size) * 100 if original_size else 0

    print()
    print(f"  Input  : {src}  ({original_size / 1024:.1f} KB)")
    print(f"  Output : {dst}  ({compact_size / 1024:.1f} KB)")
    print(f"  Saved  : {(original_size - compact_size) / 1024:.1f} KB  ({reduction:.1f}% reduction)")
    print(f"  Scrapes: {len(compact)}")
    if compact:
        n_metrics = len(compact[0].get("metrics", {}))
        print(f"  Metrics per scrape: {n_metrics}")


if __name__ == "__main__":
    main()

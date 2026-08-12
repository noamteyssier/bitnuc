#!/usr/bin/env python3

"""Builds the README performance table from criterion benchmark output.

Reads the mean time estimates written by `cargo bench` and prints a markdown
table of throughput per sequence length. Throughput uses the benchmark's byte
counts (1 byte per base), so GB/s == input bytes / mean ns.

Usage:
    RUSTFLAGS="-C target-cpu=native" cargo bench --bench simd_comparison -- coding_2bit
    uv run scripts/perf_table.py
"""

import json
from pathlib import Path

CRITERION = Path(__file__).parent.parent / "target" / "criterion"

GROUPS = {
    "encode": CRITERION / "encoding" / "encoding_2bit",
    "decode": CRITERION / "decoding" / "decoding_2bit",
}


def mean_ns(group: Path, size: int) -> float:
    estimates = group / str(size) / "new" / "estimates.json"
    with open(estimates) as handle:
        return json.load(handle)["mean"]["point_estimate"]


def sizes_of(group: Path) -> set[int]:
    return {int(p.name) for p in group.iterdir() if p.name.isdigit()}


def main() -> None:
    # intersect so stale result dirs from alternative size lists are ignored
    sizes = sorted(set.intersection(*(sizes_of(group) for group in GROUPS.values())))
    if not sizes:
        raise SystemExit("no benchmark results found; run cargo bench first")

    print("| bp | encode (GB/s) | decode (GB/s) |")
    print("| ---: | ---: | ---: |")
    for size in sizes:
        gbs = {name: size / mean_ns(group, size) for name, group in GROUPS.items()}
        print(f"| {size} | {gbs['encode']:.1f} | {gbs['decode']:.1f} |")


if __name__ == "__main__":
    main()

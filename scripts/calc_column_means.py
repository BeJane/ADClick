#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    default_csv = repo_root / "/home/Jingqi/AD/SimpleClickResLang/logs/prompt_index_01/mvtec_zero_conv_clsprompt/others/003/result_20.csv"
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_csv

    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        return 1

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("CSV has no header.", file=sys.stderr)
            return 1

        sums: dict[str, float] = {}
        counts: dict[str, int] = {}

        for row in reader:
            for key, value in row.items():
                if value is None:
                    continue
                value = value.strip()
                if not value:
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue
                sums[key] = sums.get(key, 0.0) + number
                counts[key] = counts.get(key, 0) + 1

    if not sums:
        print("No numeric columns found.", file=sys.stderr)
        return 1

    print(f"CSV: {csv_path}")
    print("Column means:")
    for key in reader.fieldnames:
        if key in sums:
            mean = sums[key] / counts[key]
            print(f"{key}: {mean:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

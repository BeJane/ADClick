#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mean and standard deviation for numeric columns across CSV files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="CSV files, glob patterns, or directories containing CSV files.",
    )
    parser.add_argument(
        "--pattern",
        default="result_5_mean.csv",
        help="Filename pattern to search for inside input directories. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the aggregated statistics CSV.",
    )
    parser.add_argument(
        "--expected-files",
        type=int,
        default=40,
        help="Fail if the number of discovered CSV files does not match this value.",
    )
    parser.add_argument(
        "--std-mode",
        choices=("population", "sample"),
        default="population",
        help="Standard deviation mode. Default: %(default)s",
    )
    return parser.parse_args()


def discover_csv_files(inputs: Iterable[str], pattern: str) -> list[Path]:
    csv_files: list[Path] = []
    for raw_path in inputs:
        expanded = glob.glob(str(Path(raw_path).expanduser()), recursive=True)
        if expanded:
            for match in sorted(Path(p).resolve() for p in expanded):
                if match.is_file():
                    csv_files.append(match)
                elif match.is_dir():
                    csv_files.extend(sorted(p.resolve() for p in match.rglob(pattern) if p.is_file()))
            continue

        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {raw_path}")
        if path.is_file():
            csv_files.append(path)
            continue
        csv_files.extend(sorted(p.resolve() for p in path.rglob(pattern) if p.is_file()))

    unique_files = sorted(dict.fromkeys(csv_files))
    if not unique_files:
        raise FileNotFoundError("No CSV files were found from the provided inputs.")
    return unique_files


def read_csv_rows(csv_files: list[Path]) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames: list[str] | None = None
    rows: list[dict[str, str]] = []

    for csv_file in csv_files:
        with csv_file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {csv_file}")
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(
                    f"Header mismatch in {csv_file}\n"
                    f"Expected: {fieldnames}\n"
                    f"Actual:   {reader.fieldnames}"
                )
            rows.extend(reader)

    if fieldnames is None:
        raise ValueError("No CSV headers found.")
    if not rows:
        raise ValueError("No data rows found in the input CSV files.")
    return fieldnames, rows


def compute_column_stats(
    fieldnames: list[str], rows: list[dict[str, str]], std_mode: str
) -> list[dict[str, str]]:
    stats_rows: list[dict[str, str]] = []

    for key in fieldnames:
        values: list[float] = []
        for row in rows:
            value = row.get(key, "")
            if value is None:
                continue
            value = value.strip()
            if not value:
                continue
            try:
                values.append(float(value))
            except ValueError:
                continue

        if not values:
            continue

        count = len(values)
        mean = sum(values) / count
        if std_mode == "sample":
            if count < 2:
                raise ValueError(f"Column '{key}' has fewer than 2 numeric values for sample std.")
            variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        else:
            variance = sum((value - mean) ** 2 for value in values) / count
        std = math.sqrt(variance)

        stats_rows.append(
            {
                "column": key,
                "count": str(count),
                "mean": f"{mean:.6f}",
                "std": f"{std:.6f}",
            }
        )

    if not stats_rows:
        raise ValueError("No numeric columns were found.")
    return stats_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["column", "count", "mean", "std"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    try:
        csv_files = discover_csv_files(args.inputs, args.pattern)
        if args.expected_files is not None and len(csv_files) != args.expected_files:
            raise ValueError(
                f"Expected {args.expected_files} files, but found {len(csv_files)} files."
            )
        fieldnames, rows = read_csv_rows(csv_files)
        stats_rows = compute_column_stats(fieldnames, rows, args.std_mode)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"files={len(csv_files)} rows={len(rows)} std_mode={args.std_mode}")
    print("column,count,mean,std")
    for row in stats_rows:
        print(f"{row['column']},{row['count']},{row['mean']},{row['std']}")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        write_csv(output_path, stats_rows)
        print(f"Wrote statistics CSV to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

DEFAULT_EXPECTED_DATASETS = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge subset result CSV files and compute mean values for numeric columns."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="CSV files, glob patterns, or directories containing result CSV files.",
    )
    parser.add_argument(
        "--pattern",
        default="result_5.csv",
        help="Filename pattern to search for inside input directories. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        help="Path to the merged CSV output.",
    )
    parser.add_argument(
        "--mean-output",
        help="Path to the single-row mean CSV output. Default: <output_stem>_mean.csv",
    )
    parser.add_argument(
        "--expected-files",
        type=int,
        default=None,
        help="Fail if the number of discovered CSV files does not match this value.",
    )
    parser.add_argument(
        "--batch-by-shot",
        action="store_true",
        help="Batch merge files grouped by <shot>/others/<subset>/<filename>.",
    )
    parser.add_argument(
        "--batch-output-dir",
        default="merged",
        help="Output directory name created under each <shot>/others directory in batch mode.",
    )
    parser.add_argument(
        "--expected-group-size",
        type=int,
        default=2,
        help="Expected number of files in each batch group. Default: %(default)s",
    )
    parser.add_argument(
        "--skip-dataset-check",
        action="store_true",
        help="Disable validation that merged rows contain the expected 15 dataset names.",
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
    merged_rows: list[dict[str, str]] = []

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
            merged_rows.extend(reader)

    if fieldnames is None:
        raise ValueError("No CSV headers found.")
    if not merged_rows:
        raise ValueError("No data rows found in the input CSV files.")
    return fieldnames, merged_rows


def validate_dataset_names(rows: list[dict[str, str]], csv_files: list[Path]) -> None:
    observed = sorted({row.get("dataset", "").strip() for row in rows if row.get("dataset", "").strip()})
    expected = sorted(DEFAULT_EXPECTED_DATASETS)

    if not observed:
        raise ValueError("Merged rows do not contain a usable dataset column.")
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ValueError(
            "Dataset validation failed.\n"
            f"Files: {', '.join(str(p) for p in csv_files)}\n"
            f"Expected {len(expected)} datasets: {', '.join(expected)}\n"
            f"Observed {len(observed)} datasets: {', '.join(observed)}\n"
            f"Missing: {', '.join(missing) if missing else 'None'}\n"
            f"Extra: {', '.join(extra) if extra else 'None'}"
        )


def compute_numeric_means(fieldnames: list[str], rows: list[dict[str, str]]) -> dict[str, str]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    for row in rows:
        for key in fieldnames:
            value = row.get(key, "")
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
        raise ValueError("No numeric columns were found.")

    mean_row: dict[str, str] = {}
    for key in fieldnames:
        if key in counts:
            mean_row[key] = f"{sums[key] / counts[key]:.6f}"
        else:
            mean_row[key] = "mean" if key == fieldnames[0] else ""
    return mean_row


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_mean_output_path(output_path: Path, mean_output: str | None) -> Path:
    if mean_output:
        return Path(mean_output).expanduser().resolve()
    return output_path.with_name(f"{output_path.stem}_mean{output_path.suffix}")


def batch_group_key(csv_file: Path) -> tuple[Path, str]:
    if len(csv_file.parents) < 3:
        raise ValueError(f"Path is too shallow for batch grouping: {csv_file}")
    subset_dir = csv_file.parent
    others_dir = subset_dir.parent
    shot_dir = others_dir.parent
    if others_dir.name != "others":
        raise ValueError(f"Path is not under an others directory: {csv_file}")
    return shot_dir, csv_file.name


def run_single_merge(
    csv_files: list[Path],
    output_path: Path,
    mean_output_path: Path,
    validate_datasets: bool,
) -> None:
    fieldnames, merged_rows = read_csv_rows(csv_files)
    if validate_datasets:
        validate_dataset_names(merged_rows, csv_files)
    mean_row = compute_numeric_means(fieldnames, merged_rows)
    write_csv(output_path, fieldnames, merged_rows)
    write_csv(mean_output_path, fieldnames, [mean_row])


def run_batch_merge(
    csv_files: list[Path],
    batch_output_dir: str,
    expected_group_size: int,
    validate_datasets: bool,
) -> tuple[int, int]:
    grouped_files: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    for csv_file in csv_files:
        grouped_files[batch_group_key(csv_file)].append(csv_file)

    processed = 0
    skipped = 0

    for (shot_dir, filename), group in sorted(grouped_files.items()):
        group = sorted(group)
        if len(group) != expected_group_size:
            skipped += 1
            print(
                f"Skip {shot_dir.name}/{filename}: expected {expected_group_size} files, found {len(group)}",
                file=sys.stderr,
            )
            continue

        output_dir = shot_dir / "others" / batch_output_dir
        output_path = output_dir / filename
        mean_output_path = output_dir / f"{Path(filename).stem}_mean{Path(filename).suffix}"
        run_single_merge(group, output_path, mean_output_path, validate_datasets=validate_datasets)
        print(f"Merged {len(group)} files into: {output_path}")
        print(f"Wrote column means to: {mean_output_path}")
        processed += 1

    return processed, skipped


def main() -> int:
    args = parse_args()
    validate_datasets = not args.skip_dataset_check

    try:
        csv_files = discover_csv_files(args.inputs, args.pattern)
        if args.expected_files is not None and len(csv_files) != args.expected_files:
            raise ValueError(
                f"Expected {args.expected_files} files, but found {len(csv_files)} files."
            )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.batch_by_shot:
        try:
            processed, skipped = run_batch_merge(
                csv_files=csv_files,
                batch_output_dir=args.batch_output_dir,
                expected_group_size=args.expected_group_size,
                validate_datasets=validate_datasets,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if processed == 0:
            print("No complete batch groups were processed.", file=sys.stderr)
            return 1
        print(f"Processed {processed} batch groups. Skipped {skipped} incomplete groups.")
        return 0

    if not args.output:
        print("--output is required when --batch-by-shot is not used.", file=sys.stderr)
        return 1

    try:
        output_path = Path(args.output).expanduser().resolve()
        mean_output_path = build_mean_output_path(output_path, args.mean_output)
        run_single_merge(
            csv_files,
            output_path,
            mean_output_path,
            validate_datasets=validate_datasets,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Merged {len(csv_files)} files into: {output_path}")
    print(f"Wrote column means to: {mean_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
